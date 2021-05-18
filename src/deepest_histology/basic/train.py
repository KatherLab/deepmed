from typing import Callable
import time
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from torch import nn
from torch import optim
from sklearn import preprocessing
from tqdm import tqdm

from fastai.vision.all import (
    Optimizer, Adam, Learner, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter,
    resnet18, cnn_learner, BalancedAccuracy, SaveModelCallback, EarlyStoppingCallback, CSVLogger,
    aug_transforms)

from ..utils import log_defaults


logger = logging.getLogger(__name__)


@log_defaults
def train(target_label: str, train_df: pd.DataFrame, result_dir: Path,
          arch: Callable[[bool], nn.Module] = resnet18,
          batch_size: int = 64,
          max_epochs: int = 10,
          opt: Optimizer = Adam,
          lr: float = 2e-3,
          patience: int = 3,
          num_workers: int = 0,
          device: torch.cuda._device_t = None,
          tfms: Callable = aug_transforms(
              flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224),
          **kwargs) -> Learner:

    if device:
        torch.cuda.set_device(device)

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=ColReader('tile_path'),
                       get_y=ColReader(target_label),
                       splitter=ColSplitter('is_valid'),
                       batch_tfms=tfms)
    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    learn = cnn_learner(
        dls, arch, path=result_dir, metrics=[BalancedAccuracy()], opt_func=opt)

    learn.fine_tune(epochs=max_epochs,
                    base_lr=lr,
                    cbs=[SaveModelCallback(monitor='balanced_accuracy_score'),
                         SaveModelCallback(every_epoch=True),
                         EarlyStoppingCallback(monitor='balanced_accuracy_score', min_delta=0.001,
                                               patience=patience),
                         CSVLogger()])
    learn.export()

    return learn
