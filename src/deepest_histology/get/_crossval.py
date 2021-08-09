import logging
from multiprocessing.managers import SyncManager
from typing import Iterable, Iterator, Any
from pathlib import Path
from typing_extensions import Protocol

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._experiment import Run, EvalRun
from ._simple import _prepare_cohorts
from ..utils import log_defaults
from ..metrics import Evaluator


class CrossvalBaseRunGetter(Protocol):
    """The signature of a run getter which can be modified by ``crossval``."""
    def __call__(
            self, *args,
            project_dir: Path, manager: SyncManager, target_label: str,
            train_cohorts_df: pd.DataFrame, test_cohorts_df: pd.DataFrame, **kwargs) \
            -> Iterator[Run]:
        ...


@log_defaults
def crossval(
        get: CrossvalBaseRunGetter,
        project_dir: Path,
        target_label: str,
        cohorts_df: pd.DataFrame,
        manager: SyncManager,
        folds: int = 3,
        seed: int = 0,
        n_bins: int = 2,
        na_values: Iterable[Any] = [],
        min_support: int = 10,
        patient_label: str = 'PATIENT',
        crossval_evaluators: Iterable[Evaluator] = [],
        *args, **kwargs) \
        -> Iterator[Run]:
    """Generates cross validation runs for a single target.

    Args:
        get:  Getter to perform cross-validation with.
        project_dir:  Path to save project data to.
        cohorts_df:  The cohorts to perform cross-validation on.
        valid_frac:  The fraction of patients which will be reserved for
            validation during training.
        folds:  Number of subsets to split the training data into.
        n_bins:  The number of bins to discretize continuous values into.
        na_values:  The class labels to consider as N/A values.
        min_support:  The minimum amount of class samples required for the class
            per fold to be included in training.  Classes with less support are
            dropped.
        *args:  Arguments to pass to ``get``.
        *kwargs:  Keyword arguments to pass to ``get``.

    Yields:
        A run for each fold of the cross-validation.

    For each of the folds a new subdirectory will be created.  Each of the folds
    will be generated in a stratified fashion, meaning that the cohorts' class
    distribution will be maintained.
    """
    logger = logging.getLogger(str(project_dir))

    cohorts_df = _prepare_cohorts(
        cohorts_df, target_label, na_values, n_bins, min_support*folds//(folds-1), logger=logger)

    if cohorts_df[target_label].nunique() < 2:
        logger.warning(f'Not enough classes for target {target_label}! skipping...')
        return

    logger.debug(f'Slide target counts: {dict(cohorts_df[target_label].value_counts())}')

    folded_df = _create_folds(cohorts_df=cohorts_df, target_label=target_label, folds=folds,
                              seed=seed, patient_label=patient_label)

    # accumulate first to ensure training / testing set data is saved
    fold_runs = [
        run
        for fold in sorted(folded_df.fold.unique())
        for run in get( # type: ignore
            *args,
            project_dir=project_dir/f'fold_{fold}',
            target_label=target_label,
            manager=manager,
            train_cohorts_df=folded_df[folded_df.fold != fold],
            test_cohorts_df=folded_df[folded_df.fold == fold],
            **kwargs)
    ]
    for run in fold_runs:
        yield run

    yield EvalRun(
        directory=project_dir,
        target=target_label,
        requirements=[run.done for run in fold_runs if run.done is not None],
        evaluators=crossval_evaluators,
        done=manager.Event())


def _create_folds(
        cohorts_df: pd.DataFrame, target_label: str, folds: int, seed: int, patient_label: str) \
        -> pd.DataFrame:
    """Adds a ``fold`` column."""

    kf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    # Pepare our dataframe
    # We enumerate each fold; this way, the training set for the `k`th iteration can be easily
    # obtained through `df[df.fold != k]`. Additionally, we sample a validation set for early
    # stopping.
    patients = cohorts_df.groupby(patient_label)[target_label].first()
    cohorts_df['fold'] = 0
    for fold, (_, test_idx) \
            in enumerate(kf.split(patients.index, patients)):
        cohorts_df.loc[cohorts_df[patient_label].isin(patients.iloc[test_idx].index), 'fold'] = fold

    return cohorts_df
