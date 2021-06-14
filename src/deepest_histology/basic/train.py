from typing import Callable
import logging
from pathlib import Path

import torch
import pandas as pd
from torch import nn
from torch import optim
from sklearn import preprocessing
from tqdm import tqdm

from fastai.vision.all import (
    Optimizer, Adam, Learner, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter,
    resnet18, cnn_learner, BalancedAccuracy, RocAucBinary, SaveModelCallback, EarlyStoppingCallback,
    CSVLogger, CrossEntropyLossFlat, aug_transforms)

from ..utils import log_defaults


logger = logging.getLogger(__name__)


@log_defaults
def train(target_label: str, train_df: pd.DataFrame, result_dir: Path,
          arch: Callable[[bool], nn.Module] = resnet18,
          batch_size: int = 64,
          max_epochs: int = 10,
          opt: Optimizer = Adam,
          patience: int = 3,
          num_workers: int = 0,
          device: torch.cuda._device_t = None,
          tfms: Callable = aug_transforms(
              flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224),
          metrics: Iterable[Callable] = [BalancedAccuracy(), RocAucBinary()],
          monitor: str = 'valid_loss',
          **kwargs) -> Learner:

    if device:
        torch.cuda.set_device(device)

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=ColReader('tile_path'),
                       get_y=ColReader(target_label),
                       splitter=ColSplitter('is_valid'),
                       batch_tfms=tfms)
    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    target_col_idx = train_df[~train_df.is_valid].columns.get_loc(target_label)

    logger.debug(f'Class counts in training set: {train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()}')
    logger.debug(f'Class counts in validation set: {train_df[train_df.is_valid].iloc[:, target_col_idx].value_counts()}')

    counts = train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()

    counts = torch.tensor([counts[k] for k in dls.vocab])
    weights = 1 - (counts / sum(counts))

    logger.info(f'{dls.vocab = }, {weights = }')

    learn = cnn_learner(
        dls, arch,
        path=result_dir,
        loss_func=CrossEntropyLossFlat(weight=weights.cuda()),
        metrics=metrics,
        opt_func=opt)

    logger.info('Searching for best LR.')
    lr = learn.lr_find().lr_min
    logger.info(f'{lr = }.')
    learn.fine_tune(epochs=max_epochs, base_lr=lr,
                    cbs=[SaveModelCallback(monitor=monitor),
                         EarlyStoppingCallback(monitor=monitor, min_delta=0.001,
                                               patience=patience),
                         CSVLogger()])
    learn.export()

    return learn
