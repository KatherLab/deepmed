# TODO having a train / deploy split for this is kind of silly... Try getting rid of it somehow!

import logging
from pathlib import Path
from fastai.layers import AdaptiveConcatPool2d

from fastai.losses import CrossEntropyLossFlat
import torch
from deepmed.utils import exists_and_has_size, factory
from typing import Callable, Optional
from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner
from torch import nn
from fastai.vision.augment import Resize
from fastai.vision.models import resnet18
import pandas as pd
from .types import GPUTask, PathLike


def _extract(
        task: GPUTask, /,
        arch: Callable[[bool], nn.Module] = resnet18,
        tfms: Optional[Callable] = Resize(size=224),
        num_workers: int = 0,
        feature_save_path: Optional[PathLike] = None) -> None:
    logger = logging.getLogger(str(task.path))

    train_df = task.train_df

    if train_df is None:
        logger.warning('Cannot extract features: no training set given!')
        return None
    elif exists_and_has_size(features_path := task.path/'features.csv.zip'):
        logger.warning(
            f'{features_path} already exists, skipping deployment...')
        return None

    # don't regenerate features we already have
    feature_save_path = Path(feature_save_path) if feature_save_path else task.path/'features'
    slides = train_df.tile_path.map(lambda x: Path(x).parent.name)
    already_existing_slide_features = [x.stem for x in feature_save_path.glob('*.pt')]
    if (n := sum(slides.isin(already_existing_slide_features))):
        logger.warning(f'Skipping feature extraction for {n} already existing slides.')
        train_df = train_df[~slides.isin(already_existing_slide_features)]
        slides = train_df.tile_path.map(lambda x: Path(x).parent.name)

    dblock = DataBlock(blocks=(ImageBlock),
                       get_x=ColReader('tile_path'),
                       batch_tfms=tfms)

    dls = dblock.dataloaders(train_df, num_workers=num_workers)

    learn = cnn_learner(dls, arch, n_out=2, path=task.path,
                        loss_func=CrossEntropyLossFlat())
    learn.model = nn.Sequential(learn.model[:-1], AdaptiveConcatPool2d())

    test_dl = learn.dls.test_dl(train_df)
    feats, _ = learn.get_preds(dl=test_dl, act=nn.Identity())
    feats = feats.squeeze()

    feature_save_path.mkdir(exist_ok=True)
    for slide in slides.unique():
        slide_feats = feats[slides == slide]
        torch.save(slide_feats, feature_save_path/f'{slide}.pt')

    return None


Extract = factory(_extract)
