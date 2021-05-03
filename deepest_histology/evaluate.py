from typing import Iterable, Callable, Sequence, Any, Mapping
from pathlib import Path

import pandas as pd

from .experiment import Evaluator

import sklearn.metrics as skm


def Grouped(evaluate: Evaluator, by: str = 'PATIENT'):
    def grouped(target_label, preds_df, result_dir, **kwargs):
        grouped_df = preds_df.groupby(by).first()
        grouped_df.update(preds_df.groupby(by).mean())  #TODO

        results = evaluate(target_label, grouped_df, result_dir, **kwargs)
        if results:
            return { f'{eval_name}_{by}': val for eval_name, val in results.items() }

    return grouped


def SubGrouped(evaluate: Evaluator, by: str):
    def sub_grouped(target_label, preds_df, result_dir, **kwargs):
        return {
            f'{eval_name}_{group}': res
            for group, group_df in preds_df.groupby(by)
            for eval_name, res in evaluate(target_label, group_df, result_dir, **kwargs).items()
        }

    return sub_grouped


def accuracy(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    y_true = preds_df[target_label]
    y_pred = preds_df[f'{target_label}_pred']
    return skm.accuracy_score(y_true, y_pred)


def f1(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    y_true = preds_df[target_label]
    y_pred = preds_df[f'{target_label}_pred']
    #TODO what to do in multitarget case?
    return {
        f'{target_label}_{class_}_f1': skm.f1_score(y_true, y_pred, pos_label=class_)
        for class_ in y_true.unique()
    }


def auroc(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    y_true = preds_df[target_label]
    return {
        f'{target_label}_{class_}_auroc':
        skm.roc_auc_score(y_true==class_, preds_df[f'{target_label}_{class_}'])
        for class_ in y_true.unique()
    }


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm

def plot_roc(df: pd.DataFrame, target_label: str, pos_label: str, ax, conf: float = 0.95):
    # see <https://en.wikipedia.org/wiki/Confidence_interval#Basic_steps>
    z_star = -norm.ppf((1-conf)/2)

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
    auc_mean = auc(mean_fpr, mean_tpr)
    auc_dev = z_star * np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {auc_mean:0.2f} $\\pm$ {auc_dev:0.2f})',
            lw=2, alpha=.8)

    dev_tpr = z_star * np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + dev_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - dev_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=f'{conf*100:g}% Confidence Interval')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'{pos_label} ROC')
    ax.legend(loc="lower right")
    
    return auc_mean, auc_dev


def roc(target_label: str, preds_df: pd.DataFrame, result_dir: Path, **_kwargs) \
        -> None:
    y_true = preds_df[target_label]
    for class_ in y_true.unique():
        fig, ax = plt.subplots()
        _, _ = plot_roc(preds_df, target_label, class_, ax=ax, conf=.95)
        fig.savefig(result_dir/f'roc_{class_}.png')
