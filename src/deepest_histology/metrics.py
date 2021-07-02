"""Metrics to evaluate a model's performance with.

During evaluation, each metric will be called with three arguments:
 1. The target label for which the metric is to be evaluated.
 2. A predictions data frame, containing the complete testing set data frame and additional columns
    titled ``{target_label}_{class_}``, which contain the class scores as well as a column
    ``{target_label}_pred`` which contains a hard predicion for that item.
 3. A path the metric can store results to.

In general, metrics are implemented in two ways:
 1. As a function. Some of these functions may have additional arguments; these can be set using
    ``functools.partial``, e.g. ``partial(f1, min_tpr=.95)``.
 2. As a function object. These metrics usually encode meta-metrics, i.e. metrics which modify other
    metrics.

Metrics may return a ``DataFrame``, which will be written to the result directory inside the file
``stats.csv``.
"""

from dataclasses import dataclass
from typing import Optional, Iterable, Callable
from pathlib import Path
from enum import Enum, auto
from functools import lru_cache

import pandas as pd
from PIL import Image

import sklearn.metrics as skm

__all__ = [
    'Metric', 'Grouped', 'SubGrouped', 'aggregate_stats', 'f1', 'confusion_matrix', 'auroc',
    'count', 'top_tiles', 'roc', 'GroupMode']


Metric = Callable[[str, pd.DataFrame, Path], Optional[pd.DataFrame]]

class GroupMode(Enum):
    """Describes how to calculate grouped predictions (see Grouped)."""
    prediction_rate = auto()
    """The group class scores are set to the ratio of the elements' predictions."""
    mean = auto()
    """The group class scores are set to the mean of the elements' scores."""


@dataclass
class Grouped:
    """Calculates a metric with the data grouped on an attribute.

    It's not always meaningful to calculate metrics on the sample level. This function first
    accumulates the predictions according to another property of the sample (as specified in the
    clinical table), grouping samples with the same value together.  Furthermore, the result dir
    given to the result dir will be extended by a subdirectory named after the grouped-by property.
    """
    metric: Metric
    """Metric to evaluate on the grouped predictions."""
    mode: GroupMode = GroupMode.prediction_rate
    """Mode to group predictions."""
    by: str = 'PATIENT'
    """Label to group the predictions by."""

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) \
            -> Optional[pd.DataFrame]:
        group_dir = result_dir/self.by
        group_dir.mkdir(exist_ok=True)
        grouped_df = _group_df(preds_df, target_label, self.by, self.mode)
        if (df := self.metric(target_label, grouped_df, group_dir)) is not None:
            columns = pd.MultiIndex.from_product([df.columns, [self.by]])
            return pd.DataFrame(df.values, index=df.index, columns=columns)

        return None


@lru_cache(maxsize=64)
def _group_df(preds_df: pd.DataFrame, target_label: str, by: str, mode: GroupMode) -> pd.DataFrame:
    grouped_df = preds_df.groupby(by).first()
    for class_ in preds_df[target_label].unique():
        if mode == GroupMode.prediction_rate:
            grouped_df[f'{target_label}_{class_}'] = (
                    preds_df.groupby(by)[f'{target_label}_pred']
                            .agg(lambda x: sum(x == class_) / len(x)))
        elif mode == GroupMode.mean:
            grouped_df[f'{target_label}_{class_}'] = \
                    preds_df.groupby(by)[f'{target_label}_pred'].mean()

    return grouped_df


@dataclass
class SubGrouped:
    """Calculates a metric for different subgroups."""
    metric: Metric
    by: str
    """The property to group by.

    The metric will be calculated seperately for each distinct label of this property.
    """
    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) \
            -> Optional[pd.DataFrame]:
        dfs = []
        for group, group_df in preds_df.groupby(self.by):
            group_dir = result_dir/group
            group_dir.mkdir(parents=True, exist_ok=True)
            if (df := self.metric(target_label, group_df, group_dir)) is not None:
                columns = pd.MultiIndex.from_product([df.columns, [group]])
                dfs.append(pd.DataFrame(df.values, index=df.index, columns=columns))

        if dfs:
            return pd.concat(dfs)

        return None


def aggregate_stats(
        _target_label, _preds_df, result_dir: Path, group_levels: Iterable[int] = []) \
        -> pd.DataFrame:
    """Accumulates stats from subdirectories.

    By default, this function simply concatenates the contents of all the ``stats.csv`` files in
    ``result_dir``'s immediate subdirectories.  Each of the subdirectories' names will be added as
    to the index at its top level.

    This function may also aggregate over metrics: if the ``group_levels`` option is given, the
    stats will be grouped by the specified index levels.

    Args:
      group_levels: Iterable[int]:  (Default value = [])
    """
    # collect all parent stats dfs
    dfs = []
    df_paths = list(result_dir.glob('*/stats.csv'))
    for df_path in df_paths:
        header, index_col = _get_header_and_index_col(df_path)
        dfs.append(pd.read_csv(df_path, header=header, index_col=index_col))
    stats_df = pd.concat(dfs, keys=[path.parent.name for path in df_paths])

    if group_levels:
        # sum all labels which have 'count' in their topmost column level; calculate means,
        # confidence intervals for the rest
        count_labels = [col for col in stats_df.columns
                        if 'count' in (col[0] if isinstance(col, tuple) else col)]
        metric_labels = [col for col in stats_df.columns if col not in count_labels]

        # calculate count sums
        grouped = stats_df[count_labels].groupby(level=group_levels)
        counts = grouped.sum(min_count=1)

        # calculate means, confidence interval bounds
        grouped = stats_df[metric_labels].groupby(level=group_levels)
        means, ns, sems = grouped.mean(), grouped.count(), grouped.sem()
        l, h = st.t.interval(alpha=.95, df=ns-1, loc=means, scale=sems)
        confs = pd.DataFrame((h - l) / 2, index=means.index, columns=means.columns)

        # for some reason concat doesn't like it if one of the dfs is empty and we supply a key
        # nonetheless... so only generate the headers if needed
        keys = (([] if means.empty else ['mean', '95% conf']) +
                ([] if counts.empty else ['total']))
        stats_df = pd.concat([means, confs, counts], keys=keys, axis=1)

        # make mean, conf, total the lowest of the column levels
        stats_df = pd.DataFrame(
            stats_df.values, index=stats_df.index,
            columns=stats_df.columns.reorder_levels([*range(1, stats_df.columns.nlevels), 0]))

        # sort by every but the last (mean, 95%) columns so we get a nice hierarchical order
        stats_df = stats_df[sorted(stats_df.columns,
                            key=lambda x: x[:stats_df.columns.nlevels-1])]

    return stats_df


def _get_header_and_index_col(csv_path: Path):
    """Gets the range of header rows and index columns."""
    #FIXME bad, bad evil hack
    with open(csv_path) as f:
        index_no = f.readline().split(',').count('')
        header_no = next(i for i, line in enumerate(f) if line[0] != ',') + 1

    return (list(range(header_no)), list(range(index_no)))




def f1(target_label: str, preds_df: pd.DataFrame, _result_dir: Path,
       min_tpr: Optional[float] = None) \
       -> pd.DataFrame:
    """Calculates the F1 score.

    Args:
      min_tpr: Optional[float]:  (Default value = None)  If min_tpr is not given, a threshold which
        maximizes the F1 score is selected; otherwise the threshold which guarantees a tpr of at
        least min_tpr is used.
    """
    y_true = preds_df[target_label]

    stats = {}
    for class_ in y_true.unique():
        thresh = _get_thresh(target_label, preds_df, class_, min_tpr=min_tpr)

        stats[class_] = \
            skm.f1_score(y_true == class_, preds_df[f'{target_label}_{class_}'] >= thresh)

    return pd.DataFrame.from_dict(
        stats, columns=[f'f1 {min_tpr or "optimal"}'], orient='index')


def confusion_matrix(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        min_tpr: Optional[float] = None) \
        -> None:
    """Generates a confusion matrix for each class label.

    Args:
      min_tpr: Optional[float]:  (Default value = None)  The minimum true positive rate the
        confusion matrix shall have for each class.  If None, the true positive rate maximizing the
        F1 score will be calculated.
    """
    classes = preds_df[target_label].unique()
    if len(classes) == 2:
        for class_ in classes:
            thresh = _get_thresh(target_label, preds_df, pos_label=class_, min_tpr=min_tpr)
            y_true = preds_df[target_label] == class_
            y_pred = preds_df[f'{target_label}_{class_}'] >= thresh
            cm = skm.confusion_matrix(y_true, y_pred)
            disp = skm.ConfusionMatrixDisplay(
                confusion_matrix=cm,
                # FIXME this next line is horrible to read
                display_labels=(classes if class_ == classes[1] else list(reversed(classes))))
            disp.plot()
            plt.title(
                f'{target_label} ' +
                (f"({class_} TPR ≥ {min_tpr})" if min_tpr
                    else f"(Optimal {class_} F1 Score)"))
            plt.savefig(result_dir/
                        f'conf_matrix_{target_label}_{class_}_{min_tpr or "opt"}.svg')
            plt.close()
    else:   #TODO does this work?
        cm = skm.confusion_matrix(
            preds_df[target_label], preds_df[f'{target_label}_pred'], labels=classes)
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.title(f'{target_label}')
        plt.savefig(result_dir/f'conf_matrix_{target_label}.svg')
        plt.close()


def _get_thresh(target_label: str, preds_df: pd.DataFrame, pos_label: str,
                min_tpr: Optional[float] = None) -> float:
    """Calculates a classification threshold for a class.

    If `min_tpr` is given, the lowest threshold to guarantee the requested tpr is returned.  Else,
    the threshold optimizing the F1 score will be returned.

    Args:
      pos_label: str:  The class to optimize for.
      min_tpr: Optional[float]:  (Default value = None)  The minimum required true prositive rate,
        or the threshold which maximizes the F1 score if None.

    Returns:
      The optimal theshold.
    """
    fprs, tprs, threshs = skm.roc_curve(
        (preds_df[target_label] == pos_label)*1., preds_df[f'{target_label}_{pos_label}'])

    if min_tpr:
        return threshs[next(i for i, tpr in enumerate(tprs) if tpr >= min_tpr)]
    else:
        return max(
            threshs,
            key=lambda t: skm.f1_score(
                preds_df[target_label] == pos_label, preds_df[f'{target_label}_{pos_label}'] > t))


def auroc(target_label: str, preds_df: pd.DataFrame, _result_dir) -> pd.DataFrame:
    """Calculates the one-vs-rest AUROC for each class of the target label."""
    y_true = preds_df[target_label]
    df = pd.DataFrame.from_dict(
        {class_: [skm.roc_auc_score(y_true==class_, preds_df[f'{target_label}_{class_}'])]
                  for class_ in y_true.unique()},
        columns=['auroc'], orient='index')
    return df


def count(target_label: str, preds_df: pd.DataFrame, _result_dir) -> pd.DataFrame:
    """Calculates the number of testing instances for each class."""
    counts = preds_df[target_label].value_counts()
    return pd.DataFrame(counts.values, index=counts.index, columns=['count'])


def top_tiles(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        n_patients: int = 4, n_tiles: int = 4, patient_label: str = 'PATIENT') -> None:
    """Generates a grid of the best scoring tiles for each class.

    The function outputs a `n_patients` × `n_tiles` grid of tiles, where each row contains the
    `n_tiles` highest scoring tiles for one of the `n_patients` best-classified patients.
    """
    for class_ in preds_df[f'{target_label}_pred'].unique():
        plt.figure(figsize=(n_patients, n_tiles), dpi=300)
        # get patients with the best overall ratings for the label
        patients = (preds_df.groupby(patient_label)[target_label]
                            .agg(lambda x: sum(x == class_) / len(x)).nlargest(n_patients))
        for i, patient in enumerate(patients.keys()):
            # get the best tile for that patient
            tiles = (preds_df[preds_df[patient_label] == patient]
                        .nlargest(n=n_tiles, columns=f'{target_label}_{class_}').tile_path)
            for j, tile in enumerate(tiles):
                plt.subplot(n_patients, n_tiles, i*n_tiles + j+1)
                plt.axis('off')
                plt.imshow(Image.open(tile))
        plt.savefig(result_dir/f'{target_label}_{class_}_top_tiles.svg', bbox_inches='tight')
        plt.close()


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import scipy.stats as st

def _plot_roc(df: pd.DataFrame, target_label: str, pos_label: str, ax, conf: float = 0.95):
    # gracefully stolen from <https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html>
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold in sorted(df.fold.unique()):
        fold_df = df[df.fold == fold]
        fpr, tpr, _ = roc_curve((fold_df[target_label] == pos_label)*1., fold_df[f'{target_label}_{pos_label}'])

        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr=fpr,
                              tpr=tpr,
                              estimator_name=f'Fold {int(fold)}',
                              roc_auc=roc_auc)
        viz.plot(ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # type: ignore

    # calculate mean and conf intervals
    auc_mean = np.mean(aucs)
    auc_conf_limits = st.t.interval(alpha=conf, df=len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))
    auc_conf = (auc_conf_limits[1]-auc_conf_limits[0])/2

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {auc_mean:0.2f} $\\pm$ {auc_conf:0.2f})',
            lw=2, alpha=.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'{target_label}: {pos_label} ROC')
    ax.legend(loc="lower right")


def _plot_simple_roc(df: pd.DataFrame, target_label: str, pos_label: str, ax, conf: float = 0.95):
    # gracefully stolen from <https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html>
    fpr, tpr, _ = roc_curve((df[target_label] == pos_label)*1., df[f'{target_label}_{pos_label}'])

    roc_auc = auc(fpr, tpr)
    viz = RocCurveDisplay(fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc)
    viz.plot(ax=ax)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'{target_label}: {pos_label} ROC')
    ax.legend(loc="lower right")


def roc(target_label: str, preds_df: pd.DataFrame, result_dir: Path) -> None:
    """Creates a one-vs-all ROC curve plot for each class."""
    y_true = preds_df[target_label]
    for class_ in y_true.unique():
        fig, ax = plt.subplots()
        if 'fold' in preds_df:
            _plot_roc(preds_df, target_label, class_, ax=ax, conf=.95)
        else:
            _plot_simple_roc(preds_df, target_label, class_, ax=ax, conf=.95)

        fig.savefig(result_dir/f'roc_{target_label}_{class_}.svg')
        plt.close()
