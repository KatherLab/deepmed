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

from typing import Optional
from pathlib import Path

import pandas as pd

import sklearn.metrics as skm
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import scipy.stats as st

from ..utils import factory


def r2(target_label: str, preds_df: pd.DataFrame, _: Path) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {'score': [skm.r2_score(preds_df[target_label], preds_df[f'{target_label}_score'])]},
        columns=['r2'], orient='index')


def p_value(target_label: str, preds_df: pd.DataFrame, _result_dir: Path) -> pd.DataFrame:
    stats = {}
    for class_ in preds_df[target_label].unique():
        pos_scores = preds_df[f'{target_label}_{class_}'][preds_df[target_label] == class_]
        neg_scores = preds_df[f'{target_label}_{class_}'][preds_df[target_label] != class_]
        stats[class_] = [st.ttest_ind(pos_scores, neg_scores).pvalue]
    return pd.DataFrame.from_dict(stats, orient='index', columns=['p value'])


def _f1(target_label: str, preds_df: pd.DataFrame, _result_dir: Path,
        min_tpr: Optional[float] = None) \
        -> pd.DataFrame:
    """Calculates the F1 score.

    Args:
        min_tpr:  If min_tpr is not given, a threshold which maximizes the F1
        score is selected; otherwise, the threshold which guarantees a tpr of at
        least min_tpr is used.
    """
    y_true = preds_df[target_label]

    stats = {}
    for class_ in y_true.unique():
        thresh = _get_thresh(target_label, preds_df, class_, min_tpr=min_tpr)

        stats[class_] = \
            skm.f1_score(y_true == class_,
                         preds_df[f'{target_label}_{class_}'] >= thresh)

    return pd.DataFrame.from_dict(
        stats, columns=[f'f1 {min_tpr or "optimal"}'], orient='index')


F1 = factory(_f1)


def _confusion_matrix(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        min_tpr: Optional[float] = None) \
        -> None:
    """Generates a confusion matrix for each class label.

    Args:
        min_tpr:  The minimum true positive rate the confusion matrix shall have
            for each class.  If None, the true positive rate maximizing the F1
            score will be calculated.
    """
    classes = preds_df[target_label].unique()
    if len(classes) == 2:
        for class_ in classes:
            thresh = _get_thresh(target_label, preds_df,
                                 pos_label=class_, min_tpr=min_tpr)
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
            plt.savefig(result_dir /
                        f'conf_matrix_{target_label}_{class_}_{min_tpr or "opt"}.svg')
            plt.close()
    else:  # TODO does this work?
        cm = skm.confusion_matrix(
            preds_df[target_label], preds_df[f'{target_label}_pred'], labels=classes)
        disp = skm.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.title(f'{target_label}')
        plt.savefig(result_dir/f'conf_matrix_{target_label}.svg')
        plt.close()


ConfusionMatrix = factory(_confusion_matrix)


def _get_thresh(target_label: str, preds_df: pd.DataFrame, pos_label: str,
                min_tpr: Optional[float] = None) -> float:
    """Calculates a classification threshold for a class.

    If `min_tpr` is given, the lowest threshold to guarantee the requested tpr
    is returned.  Else, the threshold optimizing the F1 score will be returned.

    Args:
        pos_label: str:  The class to optimize for.
        min_tpr:  The minimum required true prositive rate, or the threshold
            which maximizes the F1 score if None.

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
        {class_: [skm.roc_auc_score(y_true == class_, preds_df[f'{target_label}_{class_}'])]
         for class_ in y_true.unique()},
        columns=['auroc'], orient='index')
    return df


def count(target_label: str, preds_df: pd.DataFrame, _result_dir) -> pd.DataFrame:
    """Calculates the number of testing instances for each class."""
    counts = preds_df[target_label].value_counts()
    return pd.DataFrame(counts.values, index=counts.index, columns=['count'])
