from functools import partial
from multiprocessing.managers import SyncManager
import random
import logging
from typing import Iterable, Sequence, Iterator, Optional, Any, Union, Mapping
from pathlib import Path
from numbers import Number
from multiprocessing.synchronize import Semaphore

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch._C import _is_torch_function_enabled
from tqdm import tqdm
import numpy as np

from ..evaluators.types import Evaluator
from ..utils import exists_and_has_size, is_continuous, log_defaults
from .._experiment import Task, GPUTask, EvalTask

from .._train import Train
from .._deploy import Deploy
from ..types import Trainer, Deployer
from ..utils import factory


logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def cohort(
        tiles_path: PathLike, clini_path: PathLike, slide_path: PathLike,
        patient_label: str = 'PATIENT', slidename_label: str = 'FILENAME') \
        -> pd.DataFrame:
    """Creates a cohort df from a slide and a clini table.

    Args:
        tiles_path:  The path in which the slides' tiles are stored.  Each
            slides' tiles have to be stored in a directory in ``tiles_path``
            named after the slide.
        clini_path:  The path of the clinical table, either in csv or excel
            format.  The clinical table contains information on each patient.
            It also needs to contain a column titled ``patient_label`` (default:
            'PATIENT').
        slide_path:  A table in csv or excel format mapping slides (in a column
            ``slidename_label``) to patients (in a column ``patient_label``).
        patient_label:  Label to merge the clinical and slide tables on.
        slidename_label:  Column of the slide table containing the slide names.
    """
    tiles_path, clini_path, slide_path = Path(
        tiles_path), Path(clini_path), Path(slide_path)

    dtype = {patient_label: str, slidename_label: str}
    clini_df = (
        pd.read_csv(clini_path, dtype=dtype) if clini_path.suffix == '.csv'
        else pd.read_excel(clini_path, dtype=dtype))
    slide_df = (
        pd.read_csv(slide_path, dtype=dtype) if slide_path.suffix == '.csv'
        else pd.read_excel(slide_path, dtype=dtype))

    cohort_df = clini_df.merge(slide_df, on=patient_label)
    cohort_df = cohort_df.copy()    # for defragmentation
    cohort_df['slide_path'] = tiles_path/cohort_df[slidename_label]

    assert cohort_df.slide_path.map(Path.exists).any(), \
        f'none of the slide paths for "{slide_path}" exist!'

    logger.debug(f'#slides in {slide_path}: {len(slide_df)}')
    logger.debug(f'#patients in {clini_path}: {len(clini_df)}')
    logger.debug(f'#patients with slides for {tiles_path}: {len(cohort_df)}')

    return cohort_df


@log_defaults
def _simple_run(
        project_dir: Path,
        manager: SyncManager,
        target_label: str,
        capacities: Mapping[Union[int, str], Semaphore],
        train_cohorts_df: Optional[pd.DataFrame] = None,
        test_cohorts_df: Optional[pd.DataFrame] = None,
        patient_label: str = 'PATIENT',
        balance: bool = True,
        train: Trainer = Train(),
        deploy: Deployer = Deploy(),
        resample_each_epoch: bool = False,
        max_train_tile_num: Optional[int] = 128,
        max_valid_tile_num: Optional[int] = 256,
        max_test_tile_num: Optional[int] = 512,
        seed: int = 0,
        valid_frac: float = .2,
        n_bins: Optional[int] = 2,
        na_values: Iterable[Any] = [],
        min_support: int = 10,
        evaluators: Iterable[Evaluator] = [],
        max_class_count: Optional[Mapping[str, int]] = None,
        **kwargs,
) -> Iterator[Task]:
    """Creates tasks for a basic test-deploy procedure.

    This function will generate a single training and / or deployment task.
    Due to large in-patient similarities in slides it may be useful to only
    sample a limited number of tiles will from each patient.  The task will
    have:

    -   A training set, if ``train_cohorts`` is not empty. The training set
        will be balanced in such a way that each class is represented with a
        number of tiles equal to that of the smallest class if ``balanced``
        is ``True``.
    -   A testing set, if ``test_cohorts`` is not empty.  The testing set
        may be unbalanced.

    If the target is continuous, it will be discretized.

    Args:
        project_dir:  Path to save project data to.
        train_cohorts_df:  The cohorts to use for training.
        test_cohorts_df:  The cohorts to test on.
        resample_each_epoch:  Whether to resample the training tiles used
            from each slide each epoch.
        max_train_tile_num:  The maximum number of tiles per patient to use
            for training in each epoch or ``None`` for no subsampling.
        max_valid_tile_num:  The maximum number of validation tiles used in
            each epoch or ``None`` for no subsampling.
        max_valid_tile_num:  The maximum number of testing tiles used in
            each epoch or ``None`` for no subsampling.
        balance:  Whether the training set should be balanced.  Applies to
            categorical targets only.
        valid_frac:  The fraction of patients which will be reserved for
            validation during training.
        n_bins:  The number of bins to discretize continuous values into.
        na_values:  The class labels to consider as N/A values.
        min_support:  The minimum amount of class samples required for the
            class to be included in training.  Classes with less support are
            dropped.
        kwargs:  Other arguments to be passed to train.

    Yields:
        A task to train and / or deploy a model on the given training and
        testing data as well as an evaluation task.
    """
    logger = logging.getLogger(str(project_dir))

    eval_reqs = []
    if exists_and_has_size(preds_df_path := project_dir/'predictions.csv.zip'):
        logger.warning(
            f'{preds_df_path} already exists, skipping training/deployment!')

        yield EvalTask(
            path=project_dir,
            target_label=target_label,
            done=manager.Event(),
            requirements=[],
            evaluators=evaluators)
    else:
        # training set
        if exists_and_has_size(train_df_path := project_dir/'training_set.csv.zip'):
            logger.warning(
                f'{train_df_path} already exists, using old training set!')
            train_df = pd.read_csv(train_df_path, dtype={'is_valid': bool})
        elif train_cohorts_df is not None:
            train_df = _generate_train_df(
                train_cohorts_df, target_label, na_values, n_bins, min_support, logger,
                patient_label, valid_frac, seed, train_df_path, balance, max_class_count,
                resample_each_epoch, max_train_tile_num, max_valid_tile_num)
            # unable to generate a train df (e.g. because of insufficient data)
            if train_df is None:
                return
        else:
            train_df = None

        # testing set
        if exists_and_has_size(test_df_path := project_dir/'testing_set.csv.zip'):
            # load old testing set if it exists
            logger.warning(
                f'{test_df_path} already exists, using old testing set!')
            test_df = pd.read_csv(test_df_path)
        elif test_cohorts_df is not None:
            logger.info(f'Searching for testing tiles')
            test_cohorts_df = _prepare_cohorts(
                test_cohorts_df, target_label, na_values, n_bins=None, min_support=0, logger=logger)

            logger.info(f'Testing slide counts: {len(test_cohorts_df)}')
            test_df = _get_tiles(
                cohorts_df=test_cohorts_df, max_tile_num=max_test_tile_num, logger=logger)

            train_df_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(test_df_path, index=False, compression='zip')
        else:
            test_df = None

        assert train_df.is_valid.any(), f'no validation set!'

        gpu_done = manager.Event()
        eval_reqs.append(gpu_done)
        yield GPUTask(
            path=project_dir,
            target_label=target_label,
            requirements=[],
            done=gpu_done,
            train=partial(train, **kwargs),
            deploy=deploy,
            train_df=train_df,
            test_df=test_df,
            capacities=capacities)

        if test_df is not None:
            yield EvalTask(
                path=project_dir,
                target_label=target_label,
                done=manager.Event(),
                requirements=eval_reqs,
                evaluators=evaluators)


def _generate_train_df(
        train_cohorts_df: pd.DataFrame, target_label: str, na_values: Iterable,
        n_bins: Optional[int], min_support: int, logger, patient_label: str,
        valid_frac: float, seed: int, train_df_path: Path, balance: bool,
        max_class_count: Optional[Mapping[str, int]], resample_each_epoch: bool,
        max_train_tile_num: int, max_valid_tile_num: int
) -> Optional[pd.DataFrame]:
    train_cohorts_df = _prepare_cohorts(
        train_cohorts_df, target_label, na_values, n_bins, min_support, logger)

    if train_cohorts_df[target_label].nunique() < 2:
        logger.warning(
            f'Not enough classes for target {target_label}! skipping...')
        return

    if is_continuous(train_cohorts_df[target_label]):
        targets = train_cohorts_df[target_label]
        logger.info(
            f'Training slide count: {len(targets)} (mean={targets.mean()}, std={targets.std()})')
    else:
        logger.info(
            f'Training slide counts: {dict(train_cohorts_df[target_label].value_counts())}')

    # only use a subset of patients
    # (can be useful to compare behavior when training on different cohorts)
    if max_class_count is not None:
        patients_to_use = []
        for class_, count in max_class_count.items():
            class_patients = \
                train_cohorts_df[train_cohorts_df[target_label]
                                 == class_][patient_label].unique()
            patients_to_use.append(np.random.choice(
                class_patients, size=count, replace=False))
        train_cohorts_df = train_cohorts_df[train_cohorts_df[patient_label].isin(
            np.concatenate(patients_to_use))]

    # split off validation set
    patients = train_cohorts_df.groupby(patient_label)[target_label].first()
    if is_continuous(train_cohorts_df[target_label]):
        _, valid_patients = train_test_split(
            patients.index, test_size=valid_frac)
    else:
        _, valid_patients = train_test_split(
            patients.index, test_size=valid_frac, stratify=patients)

    train_cohorts_df['is_valid'] = train_cohorts_df[patient_label].isin(
        valid_patients)

    logger.info(f'Searching for training tiles')
    train_df = _get_tiles(
        cohorts_df=train_cohorts_df[~train_cohorts_df.is_valid],
        max_tile_num=max_train_tile_num, logger=logger)

    # if we want the training procedure to resample a slide's tiles every epoch,
    # we have to supply a slide path instead of the tile path
    if resample_each_epoch:
        train_df.tile_path = train_df.tile_path.map(lambda p: p.parent)

    valid_df = _get_tiles(
        cohorts_df=train_cohorts_df[train_cohorts_df.is_valid],
        max_tile_num=max_valid_tile_num, logger=logger)

    # restrict to classes present in training set
    train_classes = train_df[target_label].unique()
    valid_df = valid_df[valid_df[target_label].isin(train_classes)]

    logger.debug(
        f'Training tiles: {dict(train_df[target_label].value_counts())}')
    logger.debug(
        f'Validation tiles: {dict(valid_df[target_label].value_counts())}')

    if balance and not is_continuous(train_df[target_label]):
        train_df = _balance_classes(
            tiles_df=train_df, target=target_label)
        valid_df = _balance_classes(tiles_df=valid_df, target=target_label)
        logger.info(f'Training tiles after balancing: {len(train_df)}')
        logger.info(f'Validation tiles after balancing: {len(valid_df)}')

    train_df = pd.concat([train_df, valid_df])

    train_df_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_df_path, index=False, compression='zip')

    return train_df


def _prepare_cohorts(
        cohorts_df: pd.DataFrame, target_label: str, na_values: Iterable[str],
        n_bins: Optional[int], min_support: int, logger: logging.Logger
) -> pd.DataFrame:
    """Preprocesses the cohorts.

    Discretizes continuous targets and drops classes for which only few
    examples are present.
    """
    cohorts_df = cohorts_df.copy()
    if not is_continuous(cohorts_df[target_label]):
        cohorts_df[target_label] = cohorts_df[target_label].str.strip()

    # remove N/As
    cohorts_df = cohorts_df[cohorts_df[target_label].notna()]
    for na_value in na_values:
        cohorts_df = cohorts_df[cohorts_df[target_label] != na_value]

    if n_bins is not None and is_continuous(cohorts_df[target_label]):
        # discretize
        logger.info(f'Discretizing {target_label}')
        cohorts_df[target_label] = _discretize(
            cohorts_df[target_label].values, n_bins=n_bins)

    if not is_continuous(cohorts_df[target_label]):
        # drop classes with insufficient support
        class_counts = cohorts_df[target_label].value_counts()
        rare_classes = (class_counts[class_counts < min_support]).index
        cohorts_df = cohorts_df[~cohorts_df[target_label].isin(rare_classes)]

    # filter slides w/o tiles
    slides_with_tiles = cohorts_df.slide_path.map(
        lambda x: bool(next(x.glob('*.jpg'), False)))
    cohorts_df = cohorts_df[slides_with_tiles]

    return cohorts_df


def _discretize(xs: Sequence[Number], n_bins: int) -> Sequence[str]:
    """Returns a discretized version of a Sequence of continuous values."""
    unsqueezed = torch.tensor(xs).reshape(-1, 1)
    est = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal').fit(unsqueezed)
    labels = [f'[-inf,{est.bin_edges_[0][1]})',  # label for smallest class
              # labels for intermediate classes
              *(f'[{lower},{upper})'
                  for lower, upper in zip(est.bin_edges_[0][1:], est.bin_edges_[0][2:-1])),
              f'[{est.bin_edges_[0][-2]},inf)']  # label for largest class
    label_map = dict(enumerate(labels))
    discretized = est.transform(unsqueezed).reshape(-1).astype(int)
    return list(map(label_map.get, discretized))  # type: ignore


def _get_tiles(
        cohorts_df: pd.DataFrame, max_tile_num: Optional[int], logger: logging.Logger
) -> pd.DataFrame:
    """Create df containing patient, tiles, other data."""
    tiles_dfs = []
    for _, data in tqdm(cohorts_df.groupby('PATIENT')):
        tiles = [(tile_dir, file)
                 for tile_dir in data.slide_path
                 if tile_dir.exists()
                 for file in tile_dir.iterdir()]
        if max_tile_num is not None:
            tiles = random.sample(tiles, min(len(tiles), max_tile_num))
        tiles_df = pd.DataFrame(tiles, columns=['slide_path', 'tile_path'])

        tiles_dfs.append(data.merge(
            tiles_df, on='slide_path').drop(columns='slide_path'))

    tiles_df = pd.concat(tiles_dfs).reset_index(drop=True)
    logger.info(
        f'Found {len(tiles_df)} tiles for {len(tiles_df["PATIENT"].unique())} patients')

    return tiles_df


def _balance_classes(tiles_df: pd.DataFrame, target: str) -> pd.DataFrame:
    smallest_class_count = min(tiles_df[target].value_counts())
    for label in tiles_df[target].unique():
        tiles_with_label = tiles_df[tiles_df[target] == label]
        to_keep = tiles_with_label.sample(n=smallest_class_count).index
        tiles_df = tiles_df[(tiles_df[target] != label) |
                            (tiles_df.index.isin(to_keep))]

    return tiles_df


SimpleRun = factory(_simple_run)
