import shutil
import os
import math
import logging
from typing import Callable, Iterable, Optional
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Union, Optional
from fastai.callback.tracker import TrackerCallback
from functools import partial
from fastai.data.transforms import CategoryMap

import torch
import pandas as pd
from torch import nn

from fastcore.foundation import L
from fastai.vision.all import (
    Learner, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter, resnet18,
    BalancedAccuracy, SaveModelCallback, EarlyStoppingCallback, CSVLogger, CrossEntropyLossFlat,
    aug_transforms, load_learner, create_body, AdaptiveConcatPool2d, Flatten, create_head, params,
    num_features_model, TransformBlock, RegressionSetup, delegates, create_cnn_model, Adam,
    defaults, model_meta, store_attr, get_c, cast, apply_init, ifnone)
from fastai.vision.learner import _add_norm, _default_meta

from .utils import log_defaults
from .types import GPURun

__all__ = ['train']


class MultiInputModel(nn.Module):
    """A model which takes tabular information in addition to an image.

    In some cases, there may be additinal information available which may aid in
    classification.  This model extends a CNN by feeding this information into
    the in addition to the image features calcuated by the convolutional layers.
    """
    def __init__(
            self, arch, n_out: int, n_additional: int, n_in: int = 3, init=nn.init.kaiming_normal_,
            pretrained: bool = True, cut=None) -> None:
        super().__init__()

        meta = model_meta.get(arch, _default_meta)
        body = create_body(arch, n_in, pretrained, ifnone(cut, meta['cut']))
        self.cnn_feature_extractor = nn.Sequential(body, AdaptiveConcatPool2d(), Flatten())

        nf_body = num_features_model(nn.Sequential(*body.children()))
        # throw away pooling / flattenting layers
        self.head = create_head(nf_body*2 + n_additional, n_out, concat_pool=False)[2:]
        if init is not None: apply_init(self.head, init)


    def forward(self, img, *tab):
        img_feats = self.cnn_feature_extractor(img)

        if tab:
            stack_val = torch.stack((tab),axis=1)
            tensor_stack = cast(stack_val, torch.Tensor)

            features = torch.cat([img_feats, tensor_stack], dim=1)
        else:
            features = img_feats
        return self.head(features)


def multi_input_splitter(model, base_splitter):
    #TODO HIER HABE ICH AUFGEHOERT
    return [*base_splitter(model.cnn_feature_extractor)[:-1], params(model.head)]


@dataclass
class Normalize:
    mean: float
    std: float

    def __call__(self, x):
        x = float(x)
        return (x - self.mean)/self.std if not math.isnan(x) else 0


@delegates(create_cnn_model)
def multi_input_learner(
        dls, arch, normalize=True, n_out=None, n_additional=0, pretrained=True,
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None,
        path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True,
        moms=(0.95,0.85,0.95),
        # other model args
        **kwargs):
    # adapted from fastai.vision.learner.cnn_learner

    meta = model_meta.get(arch, _default_meta)
    if normalize: _add_norm(dls, meta, pretrained)

    if n_out is None: n_out = L(get_c(dls))[-1]
    assert n_out, \
        "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    model = MultiInputModel(
        arch, n_out=n_out, n_additional=n_additional, pretrained=pretrained, **kwargs)

    splitter=ifnone(splitter, meta['split'])
    splitter=partial(multi_input_splitter, base_splitter=splitter)
    learn = Learner(
        dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter,
        cbs=cbs, metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias,
        train_bn=train_bn, moms=moms)

    if pretrained: learn.freeze()

    # keep track of args for loggers
    store_attr('arch, normalize, n_out, pretrained', self=learn, **kwargs)
    return learn


@dataclass
class Category:
    name: str
    vocab: Optional[Iterable[str]] = None

    @property
    def block(self) -> CategoryBlock:
        return CategoryBlock(vocab=self, sort=False, add_na=True)
    
    def __str__(self) -> str:
        return self.name


@log_defaults
def train(
        run: GPURun, /,
        arch: Callable[[bool], nn.Module] = resnet18,
        batch_size: int = 64,
        max_epochs: int = 10,
        lr: float = 2e-3,
        num_workers: int = 0,
        tfms: Callable = aug_transforms(
            flip_vert=True, max_rotate=360, max_zoom=1, max_warp=0, size=224),
        metrics: Iterable[Callable] = [BalancedAccuracy()],
        patience: int = 3,
        monitor: str = 'valid_loss',
        conts: Iterable[str] = [],
        cats: Iterable[Union[str, Category]] = [],
        ) -> Optional[Learner]:
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
    logger = logging.getLogger(str(run.directory))

    if (model_path := run.directory/'export.pkl').exists():
        logger.warning(f'{model_path} already exists! using old model...')
        return load_learner(model_path, cpu=False)

    target_label, train_df, result_dir = run.target, run.train_df, run.directory

    if train_df is None:
        logger.debug('Cannot train: no training set given!')
        return None

    for col in conts:
        train_df[col] = train_df[col].astype(float)
        
    conts = [cont for cont in conts if cont != target_label]
    cats = [cat for cat in cats if str(cat) != target_label]

    cont_blocks = [
        TransformBlock(type_tfms=[Normalize(mean=mean, std=std), RegressionSetup()])
        for label in conts
        for mean, std in [(train_df[label].mean(), train_df[label].std())]]
    cat_blocks = [
        CategoryBlock(add_na=True) if isinstance(cat, str)
        else cat.block
        for cat in cats]

    dblock = DataBlock(
        blocks=(
            ImageBlock,
            *cont_blocks,
            *cat_blocks,
            CategoryBlock),
        getters=(
            ColReader('tile_path'),
            *(ColReader(name) for name in conts),
            *(ColReader(name) for name in cats),
            ColReader(target_label),
        ),
        splitter=ColSplitter('is_valid'),
        batch_tfms=tfms)

    dls = dblock.dataloaders(train_df, bs=batch_size, num_workers=num_workers)

    target_col_idx = train_df[~train_df.is_valid].columns.get_loc(target_label)

    logger.debug(
        'Class counts in training set: '
        f'{dict(train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts())}')
    logger.debug(
        'Class counts in validation set: '
        f'{dict(train_df[train_df.is_valid].iloc[:, target_col_idx].value_counts())}')

    counts = train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()

    vocab = dls.vocab if isinstance(dls.vocab, CategoryMap) else dls.vocab[-1]
    counts = torch.tensor([counts[k] for k in vocab])
    weights = 1 - (counts / sum(counts))

    logger.debug(f'{dls.vocab = }, {weights = }')

    learn = multi_input_learner(
        dls, arch,
        n_additional=len(conts)+len(cats),
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
            monitor=monitor, logger=logger)
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