import shutil
import os
import logging
import random
from typing import Callable, Iterable, Optional, List
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback, TrackerCallback
from fastai.data.block import CategoryBlock, DataBlock, RegressionBlock, TransformBlock
from fastai.data.transforms import ColReader, ColSplitter, IntToFloatTensor
from fastai.learner import Learner, load_learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImage
from fastai.vision.learner import cnn_learner
from torchvision.models import resnet18

import torch
import pandas as pd
from torch import nn

from .types import GPUTask
from .utils import is_continuous

__all__ = ['Train']


@lru_cache(10000)
def get_tile_list(slide_dir: Path) -> List[Path]:
    return list(slide_dir.glob('*.jpg'))


def get_tile(tile_path) -> PILImage:
    """Gets a tile.

    If tile_path points to a file, the file is loaded directly.  If it's a
    directory, a random file will be sampled."""
    # Don't specify arg types! Otherwise fastai will do some whack dispatching
    # and this function will not be called
    tile_path = Path(tile_path)
    if tile_path.is_dir():
        tile_path = random.choice(get_tile_list(tile_path))

    return PILImage.create(tile_path)


TileBlock = TransformBlock(type_tfms=get_tile, batch_tfms=IntToFloatTensor)


@dataclass
class Train:
    """Trains a single model.

    Args:
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
    arch: Callable[[bool], nn.Module] = resnet18
    batch_size: int = 64
    max_epochs: int = 32
    lr: float = 2e-3
    num_workers: int = 0
    tfms: Optional[Callable] = field(
        default_factory=lambda: aug_transforms(
            flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224))
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
            logger.debug('Cannot train: no training set given!')
            return None

        y_block = RegressionBlock if is_continuous(train_df[target_label]) else CategoryBlock

        dblock = DataBlock(blocks=(TileBlock, y_block),
                           get_x=ColReader('tile_path'),
                           get_y=ColReader(target_label),
                           splitter=ColSplitter('is_valid'),
                           batch_tfms=self.tfms)
        dls = dblock.dataloaders(
            train_df, bs=self.batch_size, num_workers=self.num_workers)

        target_col_idx = train_df[~train_df.is_valid].columns.get_loc(
            target_label)

        logger.debug(
            f'Class counts in training set: {train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()}')
        logger.debug(
            f'Class counts in validation set: {train_df[train_df.is_valid].iloc[:, target_col_idx].value_counts()}')

        if is_continuous(train_df[target_label]):
            loss_func = None
        else:
            counts = train_df[~train_df.is_valid].iloc[:,target_col_idx].value_counts()
            counts = torch.tensor([counts[k] for k in dls.vocab])
            weights = 1 - (counts / sum(counts))
            loss_func = CrossEntropyLossFlat(weight=weights.cuda())
            logger.debug(f'{dls.vocab = }, {weights = }')

        learn = cnn_learner(
            dls, self.arch,
            path=result_dir,
            loss_func=loss_func,
            metrics=self.metrics)

        cbs = [
            SaveModelCallback(
                monitor=self.monitor, fname=f'best_{self.monitor}', reset_on_fit=False),
            SaveModelCallback(every_epoch=True, with_opt=True,
                              reset_on_fit=False),
            EarlyStoppingCallback(
                monitor=self.monitor, min_delta=0.001, patience=self.patience, reset_on_fit=False),
            CSVLogger(append=True)]

        if (result_dir/'models'/f'best_{self.monitor}.pth').exists():
            _fit_from_checkpoint(
                learn=learn, result_dir=result_dir, lr=self.lr/2, max_epochs=self.max_epochs, cbs=cbs,
                monitor=self.monitor, logger=logger)
        else:
            learn.fine_tune(epochs=self.max_epochs, base_lr=self.lr, cbs=cbs)

        learn.export()
        shutil.rmtree(result_dir/'models')
        return learn


def _fit_from_checkpoint(
        learn: Learner, result_dir: Path, lr: float, max_epochs: int, cbs: Iterable[Callable],
        monitor: str, logger) -> None:
    logger.info('Continuing from checkpoint...')

    # get best performance so far
    history_df = pd.read_csv(result_dir/'history.csv')
    scores = pd.to_numeric(history_df[monitor], errors='coerce')
    high_score = scores.min() if 'loss' in monitor or 'error' in monitor else scores.max()
    logger.info(f'Best {monitor} up to checkpoint: {high_score}')

    # update tracker callback's high scores
    for cb in cbs:
        if isinstance(cb, TrackerCallback):
            cb.best = high_score

    # load newest model
    name = max((result_dir/'models').glob('model_*.pth'),
               key=os.path.getctime).stem
    learn.load(name, with_opt=True, strict=True)

    remaining_epochs = max_epochs - int(name.split('_')[1])
    logger.info(f'{remaining_epochs = }')
    learn.unfreeze()
    learn.fit_one_cycle(remaining_epochs, slice(
        lr/100, lr), pct_start=.3, div=5., cbs=cbs)