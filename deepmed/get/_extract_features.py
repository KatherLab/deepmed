from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from PIL import Image
from fastai.layers import AdaptiveConcatPool2d
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from deepmed.utils import factory
from typing import Callable, Generator, Iterable, Iterator, Optional, Sequence, TypeVar
from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner
from torch import nn
from fastai.vision.augment import RandomCrop, Resize
from fastai.vision.models import resnet18
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import re
import torch
from fastdownload import FastDownload
from fastai.data.external import fastai_cfg
import fastai
from ..types import PathLike, Task


def _extract(
        project_dir: Path,
        tile_dir: PathLike,
        feat_dir: Optional[PathLike] = None,
        arch: Callable[[bool], nn.Module] = resnet18,
        num_workers: int = 32 if os.name == 'posix' else 0,
        **kwargs
) -> Iterator[Task]:
    tile_dir = Path(tile_dir)
    feat_dir = Path(feat_dir) if feat_dir is not None else project_dir

    feat_dir.mkdir(exist_ok=True)
    yield ExtractTask(path=feat_dir,
                      requirements=[],
                      slides=list(tile_dir.iterdir()),
                      arch=arch,
                      num_workers=num_workers)


Extract = factory(_extract)


def PretrainedModel(url, arch=resnet18) -> nn.Module:
    model = arch(pretrained=False)
    d = FastDownload(fastai_cfg(), module=fastai.data, base='~/.fastai')
    path = d.download(url)
    checkpoint = torch.load(path)

    missing = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert set(missing.missing_keys) == {'fc.weight', 'fc.bias'}

    return lambda pretrained: model


@dataclass
class ExtractTask(Task):
    slides: Iterable[Path]
    arch: Callable[[bool], nn.Module]
    num_workers: int

    def do_work(self):
        for slides in (slide_pbar := tqdm(list(batch(self.slides, n=256)), leave=False)):
            learn = feature_extractor(
                arch=self.arch, num_workers=self.num_workers, item_tfms=Resize(224))
            slide_pbar.set_description(slides[0].name)
            do_slides(slides, learn, self.path)


def do_slides(slides: Iterable[Path], learn: Learner, feat_dir: Path):
    dfs = []
    for slide in slides:
        if (feat_dir/f'{slide.name}.h5').exists():
            continue
        slide_df = pd.DataFrame(
            list(slide.glob('*.jpg')), columns=['path'])
        slide_df['slide'] = slide
        if slide_df.empty:
            continue
        dfs.append(slide_df)

    if not dfs:
        return
    df = pd.concat(dfs).reset_index()

    test_dl = learn.dls.test_dl(df)
    preds, _ = learn.get_preds(dl=test_dl, act=nn.Identity())

    for slide, data in df.groupby('slide'):
        coords = np.array(list(data.path.map(_get_coords)))
        outpath = feat_dir/f'{slide.name}.h5'
        with h5py.File(outpath, 'w') as f:
            f.create_dataset('feats', dtype='float32', data=preds[data.index])
            f.create_dataset('coords', dtype='int', data=coords)


T = TypeVar('T')


def batch(sequence: Sequence[T], n: int) -> Iterable[Sequence[T]]:
    l = len(sequence)
    for ndx in range(0, l, n):
        yield sequence[ndx:min(ndx + n, l)]


def _get_coords(filename: PathLike) -> Optional[np.ndarray]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None


def feature_extractor(
        arch: Callable[[bool], nn.Module], num_workers: int, **kwargs
) -> Learner:
    dblock = DataBlock(
        blocks=ImageBlock,
        get_x=ColReader('path'),
        **kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tilepath = Path(tempdir)/'tile.jpg'
        Image.new('RGB', (224, 224)).save(tilepath)
        df = pd.DataFrame([tilepath], columns=['path'])
        dls = dblock.dataloaders(df, num_workers=num_workers)

    learn = cnn_learner(dls, arch, n_out=2,
                        loss_func=CrossEntropyLossFlat(),
                        custom_head=nn.Sequential(AdaptiveConcatPool2d(),
                                                  nn.Flatten()))

    return learn
