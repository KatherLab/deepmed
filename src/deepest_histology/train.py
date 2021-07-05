import shutil
import os
from typing import Callable, Iterable
import logging
from pathlib import Path
from fastai.callback.tracker import TrackerCallback

import torch
import pandas as pd
from torch import nn

from fastai.vision.all import (
    Learner, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter, resnet18, cnn_learner,
    BalancedAccuracy, SaveModelCallback, EarlyStoppingCallback, CSVLogger, CrossEntropyLossFlat,
    aug_transforms)

from .utils import log_defaults
from .types import Run

__all__ = ['train']


@log_defaults
def train(
        run: Run, /,
        arch: Callable[[bool], nn.Module] = resnet18,
        batch_size: int = 64,
        max_epochs: int = 10,
        lr: float = 2e-3,
        num_workers: int = 0,
        tfms: Callable = aug_transforms(
            flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224),
        metrics: Iterable[Callable] = [BalancedAccuracy()],
        patience: int = 3,
        monitor: str = 'valid_loss') -> Learner:
    """Trains a single model.

    Args:
        run:  The run to train a model for.
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
    target_label, train_df, result_dir = run.target, run.train_df, run.directory

    assert train_df is not None, 'Cannot train: no training set given!'


    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=ColReader('tile_path'),
                       get_y=ColReader(target_label),
                       splitter=ColSplitter('is_valid'),
                       batch_tfms=tfms)
    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    target_col_idx = train_df[~train_df.is_valid].columns.get_loc(target_label)

    run.logger.debug(f'Class counts in training set: {train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()}')
    run.logger.debug(f'Class counts in validation set: {train_df[train_df.is_valid].iloc[:, target_col_idx].value_counts()}')

    counts = train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()

    counts = torch.tensor([counts[k] for k in dls.vocab])
    weights = 1 - (counts / sum(counts))

    run.logger.debug(f'{dls.vocab = }, {weights = }')

    learn = cnn_learner(
        dls, arch,
        path=result_dir,
        loss_func=CrossEntropyLossFlat(weight=weights.cuda()),
        metrics=metrics)

    cbs = [
        SaveModelCallback(monitor=monitor, fname=f'best_{monitor}', reset_on_fit=False),
        SaveModelCallback(every_epoch=True, with_opt=True, reset_on_fit=False),
        EarlyStoppingCallback(
            monitor=monitor, min_delta=0.001, patience=patience, reset_on_fit=False),
        CSVLogger(append=True)]

    if (result_dir/'models'/f'best_{monitor}.pth').exists():
        _fit_from_checkpoint(
            learn=learn, result_dir=result_dir, lr=lr/2, max_epochs=max_epochs, cbs=cbs,
            monitor=monitor, logger=run.logger)
    else:
        learn.fine_tune(epochs=max_epochs, base_lr=lr, cbs=cbs)

    learn.export()

    shutil.rmtree(result_dir/'models')

    return learn


def _fit_from_checkpoint(
        learn: Learner, result_dir: Path, lr: float, max_epochs: int, cbs: Iterable[Callable],
        monitor: str, logger) \
        -> None:
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
    name = max((result_dir/'models').glob('model_*.pth'), key=os.path.getctime).stem
    learn.load(name, with_opt=True, strict=True)

    remaining_epochs = max_epochs - int(name.split('_')[1])
    logger.info(f'{remaining_epochs = }')
    learn.unfreeze()
    learn.fit_one_cycle(remaining_epochs, slice(lr/100, lr), pct_start=.3, div=5., cbs=cbs)