from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
from enum import Enum, auto

import pandas as pd

import pandas as pd
from pathlib import Path
from typing import Optional

from deepmed.utils import is_continuous

from .types import Evaluator


class GroupMode(Enum):
    """Describes how to calculate grouped predictions (see Grouped)."""
    prediction_rate = auto()
    """The group class scores are set to the ratio of the elements' predictions."""
    mean = auto()
    """The group class scores are set to the mean of the elements' scores."""


@dataclass
class Grouped:
    """Calculates a metric with the data grouped on an attribute.

    It's not always meaningful to calculate metrics on the sample level. This
    function first accumulates the predictions according to another property of
    the sample (as specified in the clinical table), grouping samples with the
    same value together.  Furthermore, the result dir given to the result dir
    will be extended by a subdirectory named after the grouped-by property.
    """
    evaluator: Evaluator
    """Metric to evaluate on the grouped predictions."""
    mode: Optional[GroupMode] = None
    """Mode to group predictions."""
    by: str = 'PATIENT'
    """Label to group the predictions by."""

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) \
            -> Optional[pd.DataFrame]:
        group_dir = result_dir/self.by
        group_dir.mkdir(exist_ok=True)
        grouped_df = _group_df(preds_df, target_label, self.by, self.mode)
        if (df := self.evaluator(target_label, grouped_df, group_dir)) is not None:  # type: ignore
            columns = pd.MultiIndex.from_product([df.columns, [self.by]])
            return pd.DataFrame(df.values, index=df.index, columns=columns)

        return None


def _group_df(preds_df: pd.DataFrame, target_label: str, by: str, mode: Optional[GroupMode]) -> pd.DataFrame:
    grouped_df = preds_df.groupby(by).first()

    if mode is None:
        mode = (GroupMode.mean if is_continuous(preds_df[target_label])
                else GroupMode.prediction_rate)

    for class_ in preds_df[target_label].unique():
        if mode == GroupMode.prediction_rate:
            grouped_df[f'{target_label}_{class_}'] = (
                preds_df.groupby(by)[f'{target_label}_pred']
                .agg(lambda x: sum(x == class_) / len(x)))
        elif mode == GroupMode.mean:
            if is_continuous(preds_df[target_label]):
                grouped_df[f'{target_label}_score'] = \
                    preds_df.groupby(by)[f'{target_label}_score'].mean()
            else:
                raise NotImplementedError() #TODO
        else:
            raise ValueError(f'unexpected {mode=}')

    return grouped_df


@dataclass
class SubGrouped:
    """Calculates a metric for different subgroups."""
    evaluator: Evaluator
    by: str
    """The property to group by.

    The metric will be calculated seperately for each distinct label of this
    property.
    """

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) \
            -> Optional[pd.DataFrame]:
        dfs = []
        for group, group_df in preds_df.groupby(self.by):
            group_dir = result_dir/group
            group_dir.mkdir(parents=True, exist_ok=True)
            if (df := self.evaluator(target_label, group_df, group_dir)) is not None:  # type: ignore
                columns = pd.MultiIndex.from_product([df.columns, [group]])
                dfs.append(pd.DataFrame(
                    df.values, index=df.index, columns=columns))

        if dfs:
            return pd.concat(dfs)

        return None

@dataclass
class OnDiscretized:
    """Discretizes continuous values before passing it to an evaluator."""
    #TODO implement for arbitrary bin number
    evaluator: Evaluator

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) -> Optional[pd.DataFrame]:
        median = preds_df[target_label].median()
        discretized_df = preds_df.copy()
        median = discretized_df[target_label].median()

        discretized_df[target_label] = preds_df[target_label] > median
        discretized_df[f'{target_label}_pred'] = preds_df[f'{target_label}_score'] > median

        centered = discretized_df[f'{target_label}_score'] - median

        scaled_positives = (centered / centered.max() / 2 + .5)
        scaled_negatives = (-centered / centered.min() / 2 + .5)
        pos_scores = scaled_positives.where(centered > 0, scaled_negatives)

        discretized_df[f'{target_label}_True'] = pos_scores
        discretized_df[f'{target_label}_False'] = 1 - pos_scores

        return self.evaluator(target_label, discretized_df, result_dir)
