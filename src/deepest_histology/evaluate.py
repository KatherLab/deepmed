from dataclasses import dataclass
from typing import Mapping, Optional
from pathlib import Path

import pandas as pd
from PIL import Image

import sklearn.metrics as skm

from experiment import Evaluator

@dataclass
class Grouped:
    evaluate: Evaluator
    by: str = 'PATIENT'

    def __call__(self, target_label, preds_df, result_dir, **kwargs):
        grouped_df = preds_df.groupby(self.by).first()
        for class_ in preds_df[target_label].unique():
            grouped_df[f'{target_label}_{class_}'] = (
                    preds_df.groupby(self.by)[f'{target_label}_pred']
                            .agg(lambda x: sum(x == class_) / len(x)))

        group_dir = result_dir/self.by
        group_dir.mkdir(exist_ok=True)
        results = self.evaluate(target_label, grouped_df, group_dir, **kwargs)
        if results:
            return { f'{eval_name}_{self.by}': val for eval_name, val in results.items() }


@dataclass
class SubGrouped:
    evaluate: Evaluator
    by: str = 'PATIENT'
    def __call__(self, target_label, preds_df, result_dir, **kwargs):
        results = {}
        for group, group_df in preds_df.groupby(self.by):
            group_dir = result_dir/group
            group_dir.mkdir(parents=True, exist_ok=True)
            if (group_results := self.evaluate(target_label, group_df, group_dir, **kwargs)):
                for eval_name, score in group_results.items():
                    results[f'{eval_name}_{group}'] = score
        return results


@dataclass
class F1:
    """Calculates the F1 score.
    
    If min_tpr is not given, a threshold which maximizes the F1 score is selected; otherwise the
    threshold which guarantees a tpr of at least min_tpr is used.
    """
    min_tpr: Optional[float] = None

    def __call__(self, target_label: str, preds_df: pd.DataFrame, _result_dir: Path, **kwargs) \
            -> Mapping[str, float]:
        y_true = preds_df[target_label]

        stats = {}
        for class_ in y_true.unique():
            thresh = get_thresh(target_label, preds_df, class_, min_tpr=self.min_tpr)

            stats[f'{target_label}_{class_}_f1_{self.min_tpr or "opt"}'] = \
                skm.f1_score(y_true == class_,
                             preds_df[f'{target_label}_{class_}'] >= thresh)

        return stats


@dataclass
class ConfusionMatrix:
    min_tpr: Optional[float] = None

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path, **kwargs) \
            -> None:
        classes = preds_df[target_label].unique()
        if len(classes) == 2:
            for class_ in classes:
                thresh = get_thresh(target_label, preds_df, pos_label=class_, min_tpr=self.min_tpr)
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
                    (f"({class_} TPR ≥ {self.min_tpr})" if self.min_tpr
                     else f"(Optimal {class_} F1 Score)"))
                plt.savefig(result_dir/
                            f'conf_matrix_{target_label}_{class_}_{self.min_tpr or "opt"}.svg')
                plt.close()
        else:   #TODO does this work?
            cm = skm.confusion_matrix(
                preds_df[target_label], preds_df[f'{target_label}_pred'], labels=classes)
            disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot()
            plt.title(f'{target_label}')
            plt.savefig(result_dir/f'conf_matrix_{target_label}.svg')
            plt.close()


def get_thresh(target_label: str, preds_df: pd.DataFrame, pos_label: str,
        min_tpr: Optional[float] = None) -> float:
    """Calculates a classification threshold for a class.
    
    If `min_tpr` is given, the lowest threshold to guarantee the requested tpr is returned.  Else, 
    the threshold optimizing the F1 score will be returned.
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
        plt.close()


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

    
def plot_simple_roc(df: pd.DataFrame, target_label: str, pos_label: str, ax, conf: float = 0.95):
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


def roc(target_label: str, preds_df: pd.DataFrame, result_dir: Path, **_kwargs) -> None:
    y_true = preds_df[target_label]
    for class_ in y_true.unique():
        fig, ax = plt.subplots()
        if 'fold' in preds_df:
            plot_roc(preds_df, target_label, class_, ax=ax, conf=.95)
        else:
            plot_simple_roc(preds_df, target_label, class_, ax=ax, conf=.95)

        fig.savefig(result_dir/f'roc_{target_label}_{class_}.svg')
        plt.close()
