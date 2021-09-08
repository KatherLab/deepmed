import logging
import re
from typing import Optional

import pandas as pd
from fastai.vision.all import Learner, CategoryMap
from typing import Iterable

from .types import GPUTask
from .utils import log_defaults, is_continuous, factory

__all__ = ['Deploy']


@log_defaults
def _deploy(learn: Learner, task: GPUTask) -> Optional[pd.DataFrame]:
    logger = logging.getLogger(str(task.path))

    if task.test_df is None:
        logger.warning('No testing set found! Skipping deployment...')
        return None
    elif (preds_path := task.path/'predictions.csv.zip').exists():
        logger.warning(f'{preds_path} already exists, skipping deployment...')
        return pd.read_csv(preds_path, low_memory=False)

    test_df, target_label = task.test_df, task.target_label

    vocab = learn.dls.vocab
    if not isinstance(vocab, CategoryMap):
        vocab = vocab[-1]

    test_df = _discretize_if_necessary(
        test_df=test_df, target_label=target_label, vocab=vocab)

    # restrict testing classes to those known by the model
    if not (known_idx := test_df[target_label].isin(vocab)).all():
        unknown_classes = test_df[target_label][~known_idx].unique()
        logger.warning(
            f'classes unknown to model in test data: {unknown_classes}!  Dropping them...')
        test_df = test_df[known_idx]

    test_dl = learn.dls.test_dl(test_df)
    # inner needed so we don't jump GPUs
    # FIXME What does `inner` actually _do_? Is this harmful?
    scores, _, class_preds = learn.get_preds(
        dl=test_dl, inner=True, with_decoded=True)

    # class-wise scores
    for class_, i in vocab.o2i.items():
        test_df[f'{target_label}_{class_}'] = scores[:, i]

    # class prediction (i.e. the class w/ the highest score for each tile)
    test_df[f'{target_label}_pred'] = vocab.map_ids(class_preds)

    test_df.to_csv(preds_path, index=False, compression='zip')

    return test_df


def _discretize_if_necessary(test_df: pd.DataFrame, target_label: str, vocab: Iterable[str]) -> pd.DataFrame:
    # check if this target was discretized for training and discretize testing set if necessary
    interval = re.compile(
        r'^\[([+-]?\d+\.?\d*(?:e[+-]?\d+|)|-inf),([+-]?\d+\.?\d*(?:e[+-]?\d+|)|inf)\)$')
    if is_continuous(test_df[target_label]) and \
            all(interval.match(class_) is not None for class_ in vocab):

        # extract thresholds from vocab
        threshs = [*(interval.match(class_).groups()[0]  # type: ignore
                     for class_ in vocab), 'inf']
        threshs = sorted(threshs, key=float)

        def interval_label(x):
            """Discretizes data into ``[lower,upper)`` classes."""
            for l, h in zip(threshs, threshs[1:]):
                # we only transform the values here, because we want h, l to be
                # *exactly* as in the training set
                if float(l) <= x and x < float(h):
                    return f'[{l},{h})'
            raise RuntimeError('unreachable!')

        test_df[target_label] = test_df[target_label].map(interval_label)

    return test_df


Deploy = factory(_deploy)
