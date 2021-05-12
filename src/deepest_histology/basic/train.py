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
    Optimizer, Adam, Learner, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter, Resize,
    resnet18, cnn_learner, BalancedAccuracy, SaveModelCallback, EarlyStoppingCallback, CSVLogger)

from ..utils import log_defaults


logger = logging.getLogger(__name__)


@log_defaults
def train(target_label: str, train_df: pd.DataFrame, result_dir: Path,
          arch: Callable[[bool], nn.Module] = resnet18,
          batch_size: int = 64,
          image_size: int = 224,
          max_epochs: int = 10,
          opt: Optimizer = Adam,
          base_lr: float = 2e-3,
          patience: int = 3,
          num_workers: int = 0,
          device: torch.cuda._device_t = None,
          **kwargs) -> Learner:

    if device:
        torch.cuda.set_device(device)

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=ColReader('tile_path'),
                       get_y=ColReader(target_label),
                       splitter=ColSplitter('is_valid'),
                       item_tfms=Resize(image_size))
    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    learn = cnn_learner(
        dls, resnet18, path=result_dir, metrics=[BalancedAccuracy()], cbs=CSVLogger())

    learn.fine_tune(epochs=max_epochs,
                    base_lr=base_lr,
                    cbs=[SaveModelCallback(monitor='valid_loss'),
                         SaveModelCallback(every_epoch=True),
                         EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1,
                                               patience=patience)])

    learn.export()

    return learn
