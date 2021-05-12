from typing import Optional
from pathlib import Path

import torch
import torchvision
import pandas as pd
from torch import nn
from sklearn import preprocessing
from tqdm import tqdm
from fastai.vision.all import Learner

from ..utils import log_defaults


@log_defaults
def deploy(learn: Learner, target_label: str, test_df: pd.DataFrame, result_dir: Path,
           device: torch.cuda._device_t = None, **kwargs) -> pd.DataFrame:
    if device:
        torch.cuda.set_device(device)

    test_dl = learn.dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)

    for class_, i in learn.dls.vocab.o2i.items():
        test_df[f'{target_label}_{class_}'] = preds[:, i]
    test_df

    return test_df
