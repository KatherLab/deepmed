import logging
from typing import Optional

import pandas as pd
from fastai.vision.all import Learner

from .utils import log_defaults

__all__ = ['deploy']


@log_defaults
def deploy(learn: Learner, run) -> Optional[pd.DataFrame]:
    logger = logging.getLogger(str(run.directory))

    if run.test_df is None:
        logger.warning('No testing set found! Skipping deployment...')
        return None
    elif (preds_path := run.directory/'predictions.csv.zip').exists():
        logger.warning(f'{preds_path} already exists, skipping deployment...')
        return pd.read_csv(preds_path, low_memory=False)

    test_df, target_label = run.test_df, run.target

    test_dl = learn.dls.test_dl(test_df)
    # inner needed so we don't jump GPUs
    #FIXME What does `inner` actually _do_? Is this harmful?
    scores, _, class_preds = learn.get_preds(dl=test_dl, inner=True, with_decoded=True)

    # class-wise scores
    for class_, i in learn.dls.vocab.o2i.items():
        test_df[f'{target_label}_{class_}'] = scores[:, i]

    # class prediction (i.e. the class w/ the highest score for each tile)
    test_df[f'{target_label}_pred'] = learn.dls.vocab.map_ids(class_preds)

    test_df.to_csv(preds_path, index=False, compression='zip')

    return test_df