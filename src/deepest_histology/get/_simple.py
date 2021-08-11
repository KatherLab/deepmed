from multiprocessing.managers import SyncManager
import random
import logging
from typing import Iterable, Sequence, Iterator, Optional, Any, Union
from pathlib import Path
from numbers import Number

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm

from ..metrics import Evaluator
from ..utils import log_defaults
from .._experiment import Run, GPURun, EvalRun


logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def cohort(
        tiles_path: PathLike, clini_path: PathLike, slide_path: PathLike,
        patient_label: str = 'PATIENT', slidename_label: str = 'FILENAME') \
        -> pd.DataFrame:
    tiles_path, clini_path, slide_path = Path(tiles_path), Path(clini_path), Path(slide_path)

    clini_df = (
        pd.read_csv(clini_path, dtype=str) if clini_path.suffix == '.csv'
        else pd.read_excel(clini_path, dtype=str))
    slide_df = (
        pd.read_csv(slide_path, dtype=str) if slide_path.suffix == '.csv'
        else pd.read_excel(slide_path, dtype=str))

    cohort_df = clini_df.merge(slide_df, on=patient_label)
    cohort_df['tiles_path'] = tiles_path/cohort_df[slidename_label]

    logger.debug(f'#slides in {slide_path}: {len(slide_df)}')
    logger.debug(f'#patients in {clini_path}: {len(clini_df)}')
    logger.debug(f'#patients with slides for {tiles_path}: {len(cohort_df)}')

    return cohort_df


@log_defaults
def simple_run(
        project_dir: Path,
        manager: SyncManager,
        target_label: str,
        train_cohorts_df: Optional[pd.DataFrame] = None,
        test_cohorts_df: Optional[pd.DataFrame] = None,
        patient_label: str = 'PATIENT',
        balance: bool = True,
        max_tile_num: int = 250,
        seed: int = 0,
        valid_frac: float = .15,
        n_bins: int = 2,
        na_values: Iterable[Any] = [],
        min_support: int = 10,
        evaluators: Iterable[Evaluator] = []) \
        -> Iterator[Run]:
    """Creates runs for a basic test-deploy procedure.

    This function will generate a single training and / or deployment run.  Due
    to large in-patient similarities in slides it may be useful to only sample
    a limited number of tiles will from each patient.  The run will have:

    -   A training set, if ``train_cohorts`` is not empty. The training set will
        be balanced in such a way that each class is represented with a number
        of tiles equal to that of the smallest class if ``balanced`` is
        ``True``.
    -   A testing set, if ``test_cohorts`` is not empty.  The testing set may be
        unbalanced.

    If the target is continuous, it will be discretized.

    Args:
        project_dir:  Path to save project data to.
        train_cohorts_df:  The cohorts to use for training.
        test_cohorts_df:  The cohorts to test on.
        max_tile_num:  The maximum number of tiles to take from each patient.
        balance:  Whether the training set should be balanced.
        valid_frac:  The fraction of patients which will be reserved for
            validation during training.
        n_bins:  The number of bins to discretize continuous values into.
        na_values:  The class labels to consider as N/A values.
        min_support:  The minimum amount of class samples required for the class
            to be included in training.  Classes with less support are dropped.

    Yields:
        A single run to train and / or deploy a model on the given training and
        testing data.
    """
    logger = logging.getLogger(str(project_dir))

    eval_reqs = []
    if (preds_df_path := project_dir/'predictions.csv.zip').exists():
        logger.warning(f'{preds_df_path} already exists, skipping training/deployment!')

        yield EvalRun(
            directory=project_dir,
            target=target_label,
            done=manager.Event(),
            requirements=[],
            evaluators=evaluators)
    else:
        # training set
        if (train_df_path := project_dir/'training_set.csv.zip').exists():
            logger.warning(f'{train_df_path} already exists, using old training set!')
            train_df = pd.read_csv(train_df_path, dtype=str)
            train_df.is_valid = train_df.is_valid == 'True'
        elif train_cohorts_df is not None:
            train_df = _generate_train_df(
                train_cohorts_df, target_label, na_values, n_bins, min_support, logger, patient_label,
                valid_frac, max_tile_num, seed, train_df_path, balance)
        else:
            train_df = None

        # testing set
        if (test_df_path := project_dir/'testing_set.csv.zip').exists():
            # load old testing set if it exists
            logger.warning(f'{test_df_path} already exists, using old testing set!')
            test_df = pd.read_csv(test_df_path, dtype=str)
        elif test_cohorts_df is not None:
            logger.info(f'Searching for testing tiles')
            test_cohorts_df = _prepare_cohorts(
                test_cohorts_df, target_label, na_values, n_bins, min_support, logger)
            logger.info(f'Testing slide counts: {dict(test_cohorts_df[target_label].value_counts())}')
            test_df = _get_tiles(
                cohorts_df=test_cohorts_df, max_tile_num=max_tile_num, seed=seed, logger=logger)
            logger.info(f'Testing tiles: {dict(test_df[target_label].value_counts())}')

            train_df_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(test_df_path, index=False, compression='zip')
        else:
            test_df = None

        gpu_done = manager.Event()
        eval_reqs.append(gpu_done)
        yield GPURun(
            directory=project_dir,
            target=target_label,
            train_df=train_df,
            test_df=test_df,
            done=gpu_done)

        if test_df is not None:
            yield EvalRun(
                directory=project_dir,
                target=target_label,
                done=manager.Event(),
                requirements=eval_reqs,
                evaluators=evaluators)


def _generate_train_df(
        train_cohorts_df, target_label, na_values, n_bins, min_support, logger, patient_label,
        valid_frac, max_tile_num, seed, train_df_path, balance):
    train_cohorts_df = _prepare_cohorts(
        train_cohorts_df, target_label, na_values, n_bins, min_support, logger)

    if train_cohorts_df[target_label].nunique() < 2:
        logger.warning(f'Not enough classes for target {target_label}! skipping...')
        return

    logger.info(f'Training slide counts: {dict(train_cohorts_df[target_label].value_counts())}')

    # split off validation set
    patients = train_cohorts_df.groupby(patient_label)[target_label].first()
    _, valid_patients = train_test_split(
        patients.index, test_size=valid_frac, stratify=patients)
    train_cohorts_df['is_valid'] = train_cohorts_df[patient_label].isin(valid_patients)

    logger.info(f'Searching for training tiles')
    tiles_df = _get_tiles(
        cohorts_df=train_cohorts_df, max_tile_num=max_tile_num, seed=seed, logger=logger)

    logger.debug(
        f'Training tiles: {dict(tiles_df[~tiles_df.is_valid][target_label].value_counts())}')
    logger.debug(
        f'Validation tiles: {dict(tiles_df[tiles_df.is_valid][target_label].value_counts())}')

    if balance:
        train_cohorts_df = _balance_classes(
            tiles_df=tiles_df[~tiles_df.is_valid], target=target_label)
        valid_df = _balance_classes(tiles_df=tiles_df[tiles_df.is_valid], target=target_label)
        logger.info(f'Training tiles after balancing: {len(train_cohorts_df)}')
        logger.info(f'Validation tiles after balancing: {len(valid_df)}')
        train_df = pd.concat([train_cohorts_df, valid_df])
    else:
        train_df = tiles_df

    train_df_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_df_path, index=False, compression='zip')

    return train_df 


def _prepare_cohorts(
        cohorts_df: pd.DataFrame, target_label: str, na_values: Iterable[str], n_bins: int,
        min_support: int, logger: logging.Logger) -> pd.DataFrame:
    """Preprocesses the cohorts.

    Discretizes continuous targets and drops classes for which only few examples
    are present.
    """
    # remove N/As
    cohorts_df = cohorts_df[cohorts_df[target_label].notna()]
    for na_value in na_values:
        cohorts_df = cohorts_df[cohorts_df[target_label] != na_value]

    # discretize values if necessary
    if cohorts_df[target_label].nunique() > 10:
        try:
            cohorts_df[target_label] = cohorts_df[target_label].map(float)
            logger.info(f'Discretizing {target_label}')
            cohorts_df[target_label] = _discretize(cohorts_df[target_label].values, n_bins=n_bins)
        except ValueError:
            pass

    # drop classes with insufficient support
    class_counts = cohorts_df[target_label].value_counts()
    rare_classes = (class_counts[class_counts < min_support]).index
    cohorts_df = cohorts_df[~cohorts_df[target_label].isin(rare_classes)]

    return cohorts_df


def _discretize(xs: Sequence[Number], n_bins: int) -> Sequence[str]:
    """Returns a discretized version of a Sequence of continuous values."""
    unsqueezed = torch.tensor(xs).reshape(-1, 1)
    est = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode='ordinal').fit(unsqueezed)
    labels = [f'[-inf,{est.bin_edges_[0][1]})', # label for smallest class
                # labels for intermediate classes
                *(f'[{lower},{upper})'
                for lower, upper in zip(est.bin_edges_[0][1:], est.bin_edges_[0][2:-1])),
                f'[{est.bin_edges_[0][-2]}, inf)'] # label for largest class
    label_map = dict(enumerate(labels))
    discretized = est.transform(unsqueezed).reshape(-1).astype(int)
    return list(map(label_map.get, discretized)) # type: ignore


def _get_tiles(
        cohorts_df: pd.DataFrame, max_tile_num: int, seed: int, logger: logging.Logger) \
        -> pd.DataFrame:
    """Create df containing patient, tiles, other data."""
    random.seed(seed)   #FIXME doesn't work
    tiles_dfs = []
    for _, data in tqdm(cohorts_df.groupby('PATIENT')):
        tiles = [(tile_dir, file)
                 for tile_dir in data.tiles_path
                 if tile_dir.exists()
                 for file in tile_dir.iterdir()]
        tiles = random.sample(tiles, min(len(tiles), max_tile_num))
        tiles_df = pd.DataFrame(tiles, columns=['tiles_path', 'tile_path'])

        tiles_dfs.append(data.merge(tiles_df, on='tiles_path').drop(columns='tiles_path'))

    tiles_df = pd.concat(tiles_dfs).reset_index(drop=True)
    logger.info(f'Found {len(tiles_df)} tiles for {len(tiles_df["PATIENT"].unique())} patients')

    return tiles_df


def _balance_classes(tiles_df: pd.DataFrame, target: str) -> pd.DataFrame:
    smallest_class_count = min(tiles_df[target].value_counts())
    for label in tiles_df[target].unique():
        tiles_with_label = tiles_df[tiles_df[target] == label]
        to_keep = tiles_with_label.sample(n=smallest_class_count).index
        tiles_df = tiles_df[(tiles_df[target] != label) | (tiles_df.index.isin(to_keep))]

    return tiles_df
