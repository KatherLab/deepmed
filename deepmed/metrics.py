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

import shutil
from dataclasses import dataclass
from typing import Optional, Iterable, Callable, Tuple, Union
from pathlib import Path
from enum import Enum, auto
from tqdm import tqdm

import pandas as pd
from PIL import Image

import sklearn.metrics as skm
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from scipy import interpolate
import logging
from openslide import OpenSlide
from matplotlib.patches import Patch


__all__ = [
    'Evaluator', 'Grouped', 'SubGrouped', 'aggregate_stats', 'f1', 'confusion_matrix', 'auroc',
    'count', 'top_tiles', 'roc', 'GroupMode', 'p_value', 'heatmap']


Evaluator = Callable[[str, pd.DataFrame, Path], Optional[pd.DataFrame]]

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
    mode: GroupMode = GroupMode.prediction_rate
    """Mode to group predictions."""
    by: str = 'PATIENT'
    """Label to group the predictions by."""

    def __call__(self, target_label: str, preds_df: pd.DataFrame, result_dir: Path) \
            -> Optional[pd.DataFrame]:
        group_dir = result_dir/self.by
        group_dir.mkdir(exist_ok=True)
        grouped_df = _group_df(preds_df, target_label, self.by, self.mode)
        if (df := self.evaluator(target_label, grouped_df, group_dir)) is not None: # type: ignore
            columns = pd.MultiIndex.from_product([df.columns, [self.by]])
            return pd.DataFrame(df.values, index=df.index, columns=columns)

        return None


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
            if (df := self.evaluator(target_label, group_df, group_dir)) is not None: # type: ignore
                columns = pd.MultiIndex.from_product([df.columns, [group]])
                dfs.append(pd.DataFrame(df.values, index=df.index, columns=columns))

        if dfs:
            return pd.concat(dfs)

        return None


def aggregate_stats(
        _target_label, _preds_df, result_dir: Path, group_levels: Iterable[int] = []) \
        -> pd.DataFrame:
    """Accumulates stats from subdirectories.

    By default, this function simply concatenates the contents of all the
    ``stats.csv`` files in ``result_dir``'s immediate subdirectories.  Each of
    the subdirectories' names will be added as to the index at its top level.

    This function may also aggregate over metrics: if the ``group_levels``
    option is given, the stats will be grouped by the specified index levels.
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


def p_value(target_label: str, preds_df: pd.DataFrame, _result_dir: Path) -> pd.DataFrame:
    stats = {}
    for class_ in preds_df[target_label].unique():
        pos_scores = preds_df[f'{target_label}_{class_}'][preds_df[target_label] == class_]
        neg_scores = preds_df[f'{target_label}_{class_}'][preds_df[target_label] != class_]
        stats[class_] = [st.ttest_ind(pos_scores, neg_scores).pvalue]
    return pd.DataFrame.from_dict(stats, orient='index', columns=['p value'])


def f1(target_label: str, preds_df: pd.DataFrame, _result_dir: Path,
       min_tpr: Optional[float] = None) \
       -> pd.DataFrame:
    """Calculates the F1 score.

    Args:
        min_tpr:  If min_tpr is not given, a threshold which maximizes the F1
        score is selected; otherwise the threshold which guarantees a tpr of at
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
        min_tpr:  The minimum true positive rate the confusion matrix shall have
            for each class.  If None, the true positive rate maximizing the F1
            score will be calculated.
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
        n_patients: int = 4, n_tiles: int = 4, patient_label: str = 'PATIENT',
        best_patients: bool = True, best_tiles: Optional[bool] = None,
        save_images: bool = False) -> None:
    """Generates a grid of the best scoring tiles for each class.

    The function outputs a `n_patients` × `n_tiles` grid of tiles, where each
    row contains the `n_tiles` highest scoring tiles for one of the `n_patients`
    best-classified patients.

    Args:
        best_patients:  Wether to select the best or worst n patients.
        best_tiles:  Whether to select the highest or lowest scoring tiles.  If
            set to ``None``, then the same as ``best_patients``.
        save_images:  Also save the tiles seperately.
    """
    # set `best_tiles` to `best_patients` if undefined
    best_tiles = best_tiles if best_tiles is not None else best_patients

    for class_ in preds_df[f'{target_label}_pred'].unique():
        # class_ == MSIH
        outdir = result_dir/_generate_tiles_fn(
                target_label, class_, best_patients, best_tiles, n_patients, n_tiles)
        outfile = outdir.with_suffix('.svg')
        if outfile.exists() and (outdir.exists() or not save_images):
            continue
        if save_images:
            outdir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(n_tiles, n_patients), dpi=600)
        # get patients with the best overall ratings for the label
        class_instance_df = preds_df[preds_df[target_label] == class_]
        patient_scores = \
            class_instance_df.groupby(patient_label)[f'{target_label}_pred'].agg(lambda x: sum(x == class_) / len(x))

        patients = (patient_scores.nlargest(n_patients) if best_patients
                    else patient_scores.nsmallest(n_patients))

        for i, patient in enumerate(patients.keys()):
            # get the best tile for that patient
            patient_tiles = preds_df[preds_df[patient_label] == patient]

            tiles = (patient_tiles.nlargest(n=n_tiles, columns=f'{target_label}_{class_}').tile_path
                     if best_tiles
                     else patient_tiles.nsmallest(n=n_tiles, columns=f'{target_label}_{class_}').tile_path)

            for j, tile in enumerate(tiles):
                if save_images:
                    shutil.copy(tile, outdir/Path(tile).name)
                if not outfile.exists():
                    plt.subplot(n_patients, n_tiles, i*n_tiles + j+1)
                    plt.axis('off')
                    plt.imshow(Image.open(tile))

        if not outfile.exists():
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def _generate_tiles_fn(
        target_label: str, class_: str, best_patients: bool, best_tiles: bool,
        n_patients: int, n_tiles: int) -> str:
    patient_str = f'{"best" if best_patients else "worst"}-{n_patients}-patients'
    tile_str = f'{"best" if best_tiles else "worst"}-{n_tiles}-tiles'

    return f'{target_label}_{class_}_{patient_str}_{tile_str}'

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
        outfile = result_dir/f'roc_{target_label}_{class_}.svg'
        if outfile.exists():
            continue

        fig, ax = plt.subplots()
        if 'fold' in preds_df:
            _plot_roc(preds_df, target_label, class_, ax=ax, conf=.95)
        else:
            _plot_simple_roc(preds_df, target_label, class_, ax=ax, conf=.95)

        fig.savefig(outfile)
        plt.close()


def heatmap(
        target_label: str, preds_df: pd.DataFrame, path: Path,
        colors=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]]),
        wsi_paths: Optional[Union[Path, str]] = [],
        wsi_suffixes: Iterable[str] = ['.svs', '.ndpi'], alpha: float = .5,
        superimpose: bool = False, format: str = '.svg') -> None:
    logger = logging.getLogger(str(path))
    outdir = path/'heatmaps'

    classes = sorted(preds_df[target_label].unique())
    score_labels = [f'{target_label}_{class_}' for class_ in classes]
    legend_elements = [
        Patch(facecolor=color, label=class_) for class_, color in zip(classes, colors)]

    for slide_name, tiles in preds_df.groupby('FILENAME'):
        true_label = tiles.iloc[0][target_label]
        try:
            plt.figure(dpi=600)
            slide_path = Path(tiles.tile_path.iloc[0]).parent
            map_coords = np.array([
                _get_coords(tile_path.name) for tile_path in slide_path.glob('*.jpg')])

            stride = _get_stride(map_coords)
            scaled_map_coords = map_coords // stride

            mask = np.zeros(scaled_map_coords.max(0) + 1)
            for coord in scaled_map_coords:
                mask[coord[0], coord[1]] = 1

            points = tiles.tile_path.map(lambda x: _get_coords(Path(x).name))
            points = np.array(list(points))

            points = points // stride

            values = tiles[score_labels].to_numpy()

            """Create a interpolated heatmap from points.
            The interpolated heatmap will be restricted to the area described by the given of points.
            Args:
                points: An (n, 2) array of coordinates.
                values: An (n, D) array of target activations, with each value in the range [0..1].
            Returns:
                An interpolated activation map.
            """
            assert points.shape[1] == 2, "expected points to have shape (_, 2)"
            assert points.shape[0] == values.shape[0], \
                   "expected points and values to have the same number of elements"
            # grid which will form the basis for our output image
            grid_x, grid_y = np.mgrid[0:scaled_map_coords[:,0].max()+1,
                                      0:scaled_map_coords[:,1].max()+1]

            # interpolate heatmap over grid
            activations = interpolate.griddata(points, values, (grid_x, grid_y))
            activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

            if not wsi_paths:
                heatmap = _visualize_activation_map(
                    activations.transpose(1, 0, 2), colors[:activations.shape[-1]])
                heatmap = heatmap.resize(np.multiply(heatmap.size, 8), resample=Image.NEAREST)
                plt.imshow(heatmap)
                plt.axis('off')
                legend = plt.legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
            else:
                # find a wsi file with the slide
                fn = next(filter(Path.exists,
                                 ((Path(wsi_path)/slide_name).with_suffix(suffix)
                                  for wsi_path in wsi_paths
                                  for suffix in wsi_suffixes)),
                                 None)

                if fn is None: continue
                slide = OpenSlide(str(fn))

                # get the first level smaller than max_size
                level = next((i for i, dims in enumerate(slide.level_dimensions)
                              if max(dims) <= 2400*2),
                             slide.level_count-1)
                thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
                covered_area_size = (
                        (map_coords.max(0)+stride) /
                        slide.level_downsamples[level]).astype(int)
                heatmap = _visualize_activation_map(
                    activations.transpose(1, 0, 2),
                    colors=colors[:activations.shape[-1]],
                    alpha=.5 if superimpose else 1)

                scaled_heatmap = Image.new('RGBA', thumb.size)
                scaled_heatmap.paste(
                    heatmap.resize(covered_area_size, resample=Image.NEAREST))

                if superimpose:
                    thumb.alpha_composite(
                        scaled_heatmap)
                    plt.imshow(thumb)
                    plt.axis('off')
                    legend = plt.legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(12,6), dpi=300)
                    axs[0].imshow(thumb)
                    axs[0].axis('off')
                    axs[1].imshow(scaled_heatmap)
                    axs[1].axis('off')
                    legend = axs[1].legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

            (outdir/true_label).mkdir(exist_ok=True, parents=True)
            fig.savefig((outdir/true_label/slide_name).with_suffix(format), bbox_extra_artists=[legend], bbox_inches='tight')
            plt.close('all')
        except Exception as exp:
            logger.exception(exp)


def _get_coords(filename: str) -> Optional[Tuple[int, int]]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', filename):
        return tuple(map(int, matches.groups()))


def _get_stride(coordinates: np.array) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)
    return stride


def _visualize_activation_map(activations: np.ndarray, colors: np.ndarray, alpha: float = 1.) -> Image:
    """Transforms an activation map into an RGBA image.
    Args:
        activations: An (h, w, D) array of activations.
        colors: A (D, 3) array mapping each of the target classes to a color.
    Returns:
        An interpolated activation map image. Regions which the algorithm assumes to be background
        will be transparent.
    """
    assert colors.shape[1] == 3, "expected color map to have three color channels"
    assert colors.shape[0] == activations.shape[2], "one color map entry per class required"

    # transform activation map into RGB map
    rgbmap = activations.dot(colors)

    # create RGBA image with non-zero activations being the foreground
    mask = activations.any(axis=2)
    im_data = (np.concatenate([rgbmap, np.expand_dims(mask * alpha, -1)], axis=2) * 255.5).astype(np.uint8)

    return Image.fromarray(im_data)