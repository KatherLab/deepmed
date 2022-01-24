from functools import partial
import os
import random
import shutil
import logging
from typing import Callable, Iterable, Mapping, Optional
from dataclasses import dataclass, field
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.data.block import CategoryBlock, DataBlock, RegressionBlock, TransformBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.learner import Learner, load_learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.learner import create_head
from fastcore.foundation import L
from tqdm import tqdm

import torch
import h5py
import numpy as np
import pandas as pd

from .types import GPUTask
from .utils import is_continuous
from .get import DatasetType

__all__ = ['Train', 'get_h5s']


def get_h5s(
        dataset_type: DatasetType, cohorts_df: pd.DataFrame,
        max_tile_nums: Mapping[DatasetType, int] = {DatasetType.TRAIN: 128,
                                                    DatasetType.VALID: 256,
                                                    DatasetType.TEST: 512},
        resample_each_epoch: bool = False,
        logger: logging.Logger = logging,
) -> pd.DataFrame:
    """Create df containing patient, tiles, other data."""
    cohorts_df.slide_path = cohorts_df.slide_path.map(lambda p: p.parent/f'{p.name}.h5')
    cohorts_df = cohorts_df[cohorts_df.slide_path.map(lambda p: p.exists())]

    tiles_dfs = []
    for slide_path in tqdm(cohorts_df.slide_path):
        with h5py.File(slide_path, 'r') as f:
            tiles = [(slide_path, i)
                    for i in range(len(f['feats']))]
        if (tile_num := max_tile_nums.get(dataset_type)):
            tiles = random.sample(tiles, min(len(tiles), tile_num))
        tiles_df = pd.DataFrame(tiles, columns=['slide_path', 'i'])

        tiles_dfs.append(tiles_df)

    tiles_df = pd.concat(tiles_dfs)
    tiles_df = cohorts_df.merge(tiles_df, on='slide_path').reset_index()

    logger.info(
        f'Found {len(cohorts_df)} tiles for {len(cohorts_df["PATIENT"].unique())} patients')

    # if we want the training procedure to resample a slide's tiles every epoch,
    # we have to supply a slide path instead of the tile path
    if dataset_type == DatasetType.TRAIN and resample_each_epoch:
        tiles_df.i = -1

    return tiles_df


def load_feats(args: L):
    path, i = args
    with h5py.File(path, 'r') as f:
        # check if all features stem from the same extractor
        #h5_checksum = f.attrs['extractor-checksum']
        #assert self.extractor_checksum == h5_checksum, \
        #     f'feature extractor mismatch for {path} ' \
        #     f'(expected {self.extractor_checksum:08x}, got {h5_checksum:08x})'
        if i == -1:
            return torch.from_numpy(f['feats'][np.random.randint(len(f['feats']))])
        else:
            return torch.from_numpy(f['feats'][i])


@dataclass
class Train:
    """Trains a single model.

    Args:
        batch_size:  The number of training samples used through the network during one forward and backward pass.
        task:  The task to train a model for.
        arch:  The architecture of the model to train.
        max_epochs:  The absolute maximum number of epochs to train.
        lr:  The initial learning rate.
        num_workers:  The number of workers to use in the data loaders.  Set to
            0 on windows!
        tfms:  Transforms to apply to the data.
        metrics:  The metrics to calculate on the validation set each epoch.
        patience:  The number of epochs without improvement before stopping the
            training.
        monitor:  The metric to monitor for early stopping.

    Returns:
        The trained model.

    If the training is interrupted, it will be continued from the last model
    checkpoint.
    """
    batch_size: int = 64
    max_epochs: int = 32
    lr: float = 2e-3
    num_workers: int = (32 if os.name == 'posix' else 0)
    metrics: Iterable[Callable] = field(default_factory=list)
    patience: int = 3
    monitor: str = 'valid_loss'

    def __call__(self, task: GPUTask) -> Optional[Learner]:
        logger = logging.getLogger(str(task.path))

        if (model_path := task.path/'export.pkl').exists():
            logger.warning(f'{model_path} already exists! using old model...')
            return load_learner(model_path)

        target_label, train_df, result_dir = task.target_label, task.train_df, task.path

        if train_df is None:
            logger.warning('Cannot train: no training set given!')
            return None

        y_block = RegressionBlock if is_continuous(train_df[target_label]) else CategoryBlock
        dblock = DataBlock(blocks=(TransformBlock(item_tfms=load_feats), y_block),
                           get_x=ColReader(['slide_path', 'i']),
                           get_y=ColReader(target_label),
                           splitter=ColSplitter('is_valid'))
        dls = dblock.dataloaders(
            train_df, bs=self.batch_size, num_workers=self.num_workers)

        logger.debug(
            f'Class counts in training set: {train_df[~train_df.is_valid][target_label].value_counts()}')
        logger.debug(
            f'Class counts in validation set: {train_df[train_df.is_valid][target_label].value_counts()}')

        if is_continuous(train_df[target_label]):
            loss_func = None
        else:
            counts = torch.tensor(train_df[~train_df.is_valid][target_label].value_counts())
            weight = counts.sum() / counts
            weight /= weight.sum()
            loss_func = CrossEntropyLossFlat(weight=weight.cuda())
            logger.debug(f'{dls.vocab = }, {weight = }')

        n_feats = dls.one_batch()[0].shape[-1]
        head = create_head(n_feats, dls.c, concat_pool=False)[2:]

        learn = Learner(
            dls, head,
            path=result_dir,
            loss_func=loss_func,
            metrics=self.metrics)

        # save the features' extractor in the model so we can trace it back later
        with h5py.File(train_df.slide_path.iloc[0]) as f:
            learn.extractor_checksum = f.attrs['extractor-checksum']

        cbs = [
            SaveModelCallback(
                monitor=self.monitor, fname=f'best_{self.monitor}', reset_on_fit=False),
            SaveModelCallback(every_epoch=True, with_opt=True,
                              reset_on_fit=False),
            EarlyStoppingCallback(
                monitor=self.monitor, min_delta=0.001, patience=self.patience, reset_on_fit=False),
            CSVLogger(append=True)]

        learn.fit_one_cycle(n_epoch=self.max_epochs, lr_max=self.lr, cbs=cbs)

        learn.export()
        shutil.rmtree(result_dir/'models')
        return learn