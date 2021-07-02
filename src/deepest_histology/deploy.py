from pathlib import Path

import pandas as pd
from fastai.vision.all import Learner

from .utils import log_defaults

__all__ = ['deploy']

__all__ = ['deploy']


@log_defaults
def deploy(learn: Learner, run) -> pd.DataFrame:
    test_df, target_label = run.test_df, run.target
    assert test_df is not None, 'Cannot deploy: no testing set given!'

    test_dl = learn.dls.test_dl(test_df)
    # inner needed so we don't jump GPUs
    #FIXME What does `inner` actually _do_? Is this harmful?
    scores, _, class_preds = learn.get_preds(dl=test_dl, inner=True, with_decoded=True)

    # class-wise scores
    for class_, i in learn.dls.vocab.o2i.items():
        test_df[f'{target_label}_{class_}'] = scores[:, i]

    # class prediction (i.e. the class w/ the highest score for each tile)
    test_df[f'{target_label}_pred'] = learn.dls.vocab.map_ids(class_preds)

    return test_df
