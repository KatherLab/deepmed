import shutil
from typing import Iterable, Callable, Sequence, Any, Mapping
from pathlib import Path

import pandas as pd
from PIL import Image

from .basic.evaluate import Evaluator

import sklearn.metrics as skm


def Grouped(evaluate: Evaluator, by: str = 'PATIENT'):
    def grouped(target_label, preds_df, result_dir, **kwargs):
        grouped_df = preds_df.groupby(by).first()
        for class_ in preds_df[target_label].unique():
            grouped_df[f'{target_label}_{class_}'] = (
                    preds_df.groupby(by)[f'{target_label}_pred']
                            .agg(lambda x: sum(x == class_) / len(x)))

        group_dir = result_dir/by
        group_dir.mkdir(exist_ok=True)
        results = evaluate(target_label, grouped_df, group_dir, **kwargs)
        if results:
            return { f'{eval_name}_{by}': val for eval_name, val in results.items() }

    return grouped


def SubGrouped(evaluate: Evaluator, by: str):
    def sub_grouped(target_label, preds_df, result_dir, **kwargs):
        results = {}
        for group, group_df in preds_df.groupby(by):
            group_dir = result_dir/group
            group_dir.mkdir(parents=True, exist_ok=True)
            if (group_results := evaluate(target_label, group_df, group_dir, **kwargs)):
                for eval_name, score in group_results.items():
                    results[f'{eval_name}_{group}'] = score
        return results

    return sub_grouped


def accuracy(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    y_true = preds_df[target_label]
    y_pred = preds_df[f'{target_label}_pred']
    return {f'{target_label}_accuracy': skm.accuracy_score(y_true, y_pred)}


def f1(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    y_true = preds_df[target_label]
    y_pred = preds_df[f'{target_label}_pred']
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


def count(target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
        -> Mapping[str, float]:
    """Calculates the number of training instances, both in total and per class."""
    return {f'{target_label}_count': len(preds_df), # total
            **{ # per class
                f'{target_label}_{class_}_count': len(preds_df[preds_df[target_label]==class_])
                for class_ in preds_df[target_label].unique()
            }
           }


def top_tiles(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        n_patients: int = 4, n_tiles: int = 4, patient_label: str = 'PATIENT', **kwargs) -> None:
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


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import scipy.stats as st

def plot_roc(df: pd.DataFrame, target_label: str, pos_label: str, ax, conf: float = 0.95):
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
    
    return auc_mean, auc_conf


def roc(target_label: str, preds_df: pd.DataFrame, result_dir: Path, **_kwargs) -> None:
    y_true = preds_df[target_label]
    for class_ in y_true.unique():
        fig, ax = plt.subplots()
        _, _ = plot_roc(preds_df, target_label, class_, ax=ax, conf=.95)
        fig.savefig(result_dir/f'roc_{target_label}_{class_}.svg')
