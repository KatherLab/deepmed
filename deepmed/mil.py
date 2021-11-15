from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.data.block import CategoryBlock, DataBlock, RegressionBlock, TransformBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.learner import Learner, load_learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.learner import create_head
from fastcore.transform import Transform
import h5py
import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple, Union

from deepmed.types import GPUTask
from deepmed.utils import is_continuous


def _to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat((bag_samples,
                             torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))


class MILBagTransform(Transform):
    def __init__(self, valid_files: Iterable[os.PathLike], max_bag_size: int = 512) -> None:
        self.max_train_bag_size = max_bag_size
        self.valid = {fn: self._draw(fn) for fn in tqdm(valid_files, leave=False)}

    def encodes(self, fn):# -> Tuple[torch.Tensor, int]:
        if not isinstance(fn, (Path, str)):
            return fn

        return self.valid.get(fn, self._draw(fn))

    def _draw(self, fn: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        with h5py.File(fn, 'r') as f:
            feats = torch.from_numpy(f['feats'][:])
        return _to_fixed_size_bag(feats, bag_size=self.max_train_bag_size)


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight.

    Taken from arXiv:1802.04712
    """
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1))


class GatedAttention(nn.Module):
    """A network calculating an embedding's importance weight.

    Taken from arXiv:1802.04712
    """

    def __init__(self, n_in: int, n_latent: Optional[int] = None) -> None:
        super().__init__()
        n_latent = n_latent or (n_in + 1) // 2

        self.fc1 = nn.Linear(n_in, n_latent)
        self.gate = nn.Linear(n_in, n_latent)
        self.fc2 = nn.Linear(n_latent, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(h)) * torch.sigmoid(self.gate(h)))


class MILModel(nn.Module):
    def __init__(
        self, n_feats: int, n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        with_attention_scores: bool = False,
    ) -> None:
        """

        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(
            nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256)# GatedAttention(512)
        self.head = head or create_head(
            256, n_out, concat_pool=False, lin_ftrs=[])[1:]

        self.with_attention_scores = with_attention_scores

    def forward(self, bags_and_lens):
        bags, lens = bags_and_lens
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        embeddings = self.encoder(bags)

        masked_attention_scores = self._masked_attention_scores(
            embeddings, lens)
        weighted_embedding_sums = (
            masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores

    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.

        Returns:
            A tensor containing
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size)
               .repeat(bs, 1)
               .to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask,
            attention_scores,
            torch.full_like(attention_scores, -1e10))
        return torch.softmax(masked_attention, dim=1)


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
    max_bag_size: int = 512
    batch_size: int = 32
    max_epochs: int = 64
    lr: Optional[float] = 1e-3
    num_workers: int = 0
    metrics: Iterable[Callable] = field(default_factory=list)
    patience: int = 12
    monitor: str = 'valid_loss'

    def __call__(self, task: GPUTask) -> Optional[Learner]:
        logger = logging.getLogger(str(task.path))

        if (model_path := task.path/'export.pkl').exists():
            logger.warning(f'{model_path} already exists! using old model...')
            return load_learner(model_path)

        target_label, train_df = task.target_label, task.train_df

        if train_df is None:
            logger.warning('Cannot train: no training set given!')
            return None

        # create dataloader
        y_block = RegressionBlock if is_continuous(
            train_df[target_label]) else CategoryBlock

        train_df.slide_path = train_df.slide_path.map(Path)
        mil_tfm = MILBagTransform(train_df[~train_df.is_valid].slide_path, self.max_bag_size)
        dblock = DataBlock(blocks=(TransformBlock, y_block),
                           get_x=ColReader('slide_path'),
                           get_y=ColReader(target_label),
                           splitter=ColSplitter('is_valid'),
                           item_tfms=mil_tfm)
        dls = dblock.dataloaders(
            train_df, bs=self.batch_size, num_workers=self.num_workers)

        target_col_idx = train_df[~train_df.is_valid].columns.get_loc(target_label)

        logger.debug(
            f'Class counts in training set: {train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()}')
        logger.debug(
            f'Class counts in validation set: {train_df[train_df.is_valid].iloc[:, target_col_idx].value_counts()}')

        # create weighted loss function in case of categorical data
        if is_continuous(train_df[target_label]):
            loss_func = None
        else:
            counts = train_df[~train_df.is_valid].iloc[:, target_col_idx].value_counts()
            weight = counts.sum() / counts
            weight /= weight.sum()
            # reorder according to vocab
            weight = torch.tensor(list(map(weight.get, dls.vocab)), dtype=torch.float32)
            loss_func = CrossEntropyLossFlat(weight=weight.cuda())
            logger.info(f'{dls.vocab = }, {weight = }')

        feat_no = dls.one_batch()[0][0].shape[-1]
        learn = Learner(dls, MILModel(feat_no, dls.c),
                        path=task.path, loss_func=loss_func, metrics=self.metrics)

        # find learning rate if necessary
        if not self.lr:
            logger.info('searching learning rate...')
            suggested_lrs = learn.lr_find()
            logger.info(f'{suggested_lrs = }')
            self.lr = suggested_lrs.valley

        # finally: train!
        cbs = [
            SaveModelCallback(
                monitor=self.monitor, fname=f'best_{self.monitor}', reset_on_fit=False),
            SaveModelCallback(every_epoch=True, with_opt=True,
                              reset_on_fit=False),
            EarlyStoppingCallback(
                monitor=self.monitor, min_delta=0.001, patience=self.patience, reset_on_fit=False),
            CSVLogger(append=True)]

        learn.fit_one_cycle(n_epoch=self.max_epochs, lr_max=self.lr, cbs=cbs)

        # make bag size "realistically big" for deployment
        mil_tfm.max_bag_size = max(_bag_lens(train_df[train_df.is_valid].slide_path))
        dls.valid.bs = 1

        learn.export()
        shutil.rmtree(task.path/'models')
        return learn


def _bag_lens(h5_files: Iterable[os.PathLike]) -> Iterable[int]:
    lens = []
    for fn in h5_files:
        with h5py.File(fn, 'r') as f:
            lens.append(len(f['feats']))
    return lens