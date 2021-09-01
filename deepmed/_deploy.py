import logging
from typing import Optional

import pandas as pd
from fastai.vision.all import Learner, CategoryMap

from .types import GPUTask
from .utils import log_defaults

__all__ = ['deploy']


@log_defaults
def deploy(learn: Learner, task: GPUTask) -> Optional[pd.DataFrame]:
    logger = logging.getLogger(str(task.path))

    if task.test_df is None:
        logger.warning('No testing set found! Skipping deployment...')
        return None
    elif (preds_path := task.path/'predictions.csv.zip').exists():
        logger.warning(f'{preds_path} already exists, skipping deployment...')
        return pd.read_csv(preds_path, low_memory=False)

    test_df, target_label = task.test_df, task.target_label

    test_dl = learn.dls.test_dl(test_df)
    # inner needed so we don't jump GPUs
    #FIXME What does `inner` actually _do_? Is this harmful?
    scores, _, class_preds = learn.get_preds(dl=test_dl, inner=True, with_decoded=True)

    # class-wise scores
    vocab = learn.dls.vocab
    if not isinstance(vocab, CategoryMap):
        vocab = vocab[-1]
    for class_, i in vocab.o2i.items():
        test_df[f'{target_label}_{class_}'] = scores[:, i]

    # class prediction (i.e. the class w/ the highest score for each tile)
    test_df[f'{target_label}_pred'] = vocab.map_ids(class_preds)

    test_df.to_csv(preds_path, index=False, compression='zip')

    return test_df
