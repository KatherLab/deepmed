#!/usr/bin/env python3
import logging
from zipfile import BadZipFile
from typing import Union, Iterable, Optional, Callable, Sequence, Tuple, Iterator
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, Manager
from multiprocessing.synchronize import Semaphore
from functools import partial

from fastai.vision.all import Learner, load_learner

import pandas as pd
from pandas.core.indexes.multi import MultiIndex
import torch

from .utils import Lazy
from .metrics import Metric


__all__ = [
    'Run', 'RunGetter', 'Trainer', 'Deployer', 'Coordinator' 'PathLike', 'do_experiment']


logger = logging.getLogger(__name__)


@dataclass
class Run:
    """A collection of data to train or test a model."""
    directory: Path
    """The directory to save data in for this run."""
    target: str
    """The name of the target to train or deploy on."""
    train_df: Optional[pd.DataFrame] = None
    """A dataframe mapping tiles to be used for training to their targets.

    It contains at least the following columns:
    - tile_path: Path
    - is_valid: bool:  whether the tile should be used for validation (e.g. for early stopping).
    - At least one target column with the name saved in the run's `target`.
    """
    test_df: Optional[pd.DataFrame] = None
    """A dataframe mapping tiles used for testing to their targets.

    It contains at least the following columns:
    - tile_path: Path
    """


RunGetter = Callable[..., Iterator[Run]]
"""A function which creates a series of runs."""

Trainer = Callable[..., Learner]
"""A function which trains a model.

Required kwargs:
    train_df: TrainDF:  A dataset specifying which tiles to train on.
    target_label: str:  The label to train on.
    result_dir:  A folder to write intermediate results to.

Returns:
    The trained model.
"""

Deployer = Callable[..., pd.DataFrame]
"""A function which deployes a model.

Required kwargs:
    model: Learner:  The model to test on.
    target_label: str:  The name to be given to the result column.
    test_df: TestDF:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.

Returns:
    `test_df`, but with at least an additional column for the target predictions.
"""


@dataclass
class Coordinator:
    """Defines how an experiment is to be performed."""
    get: RunGetter
    """A function which generates runs."""
    train: Optional[Trainer] = None
    """A function which trains a model for each of the runs."""
    deploy: Optional[Deployer] = None
    """A function which deploys a trained model to a test set, yielding predictions."""


PathLike = Union[str, Path]


def do_experiment(*,
        project_dir: PathLike,
        mode: Coordinator,
        model_path: Optional[PathLike] = None,
        num_concurrent_runs: int = 4,
        devices = [torch.cuda.current_device()],
        evaluator_groups: Sequence[Iterable[Metric]] = [],
        **kwargs) -> None:
    """Runs an experiement.

    Args:
        project_dir:  The directory to save project data in.
        mode:  How to perform the training / testing process.
        save_models:  Whether or not to save the resulting models.
    """

    assert num_concurrent_runs >= 1

    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True, parents=True)

    # add logfile handler
    file_handler = logging.FileHandler(project_dir/'logfile')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info('Getting runs')

    kwds = { 'mode': mode, 'model_path': model_path, 'project_dir': project_dir }

    with Manager() as manager:
        # semaphores which tell us which GPUs still have resources free
        # each gpu is a assumed to have the same capabilities
        capacities = [manager.Semaphore((num_concurrent_runs+len(devices)-1)//len(devices))  # type: ignore
                      for _ in devices]
        run_args = ({'run': run, 'devices': devices, 'capacities': capacities,  **kwds, **kwargs}
                     for run in mode.get(project_dir=project_dir, **kwargs))

        with Pool(num_concurrent_runs) as pool:
            # only use pool if we actually want to run multiple runs in parallel
            runs = filter(
                lambda x: x is not None,
                (pool.imap(_do_run_wrapper, run_args, chunksize=1) if num_concurrent_runs > 1
                 else (_do_run_wrapper(args) for args in run_args)))
            _evaluate_runs(runs, project_dir=project_dir, evaluator_groups=evaluator_groups)


def _evaluate_runs(
        runs: Iterator[Run], project_dir: Path, evaluator_groups: Sequence[Iterable[Callable]]) \
        -> None:
    """Calls evaluation functions for each run.

    Args:
        runs:  An iterator over the already completed runs.  This iterator has to traverse the runs
            in-order.
        project_dir:  The root directory of the experiment.
        evaluator_groups:  A sequence of collections of evaluation functions.

    TODO a more detailed description

    Assume we have the evaluator groups `[A, B, C]`.  Then the the evaluator groups will be invoked
    as follows:

        root/a/b
        root/a/c   -> C(b)
        root/a/d   -> C(c)
        root/e/f   -> C(d), B(a)
        root/e/g/h -> C(f)
        root/e/g/i
        root/e/j   -> C(g)
                   -> C(j), B(e), A(root)

    where B(a) means that all the evaluation functions in evaluator group B will be invoked on run
    a.
    """
    last_run = None

    for run in runs:
        run_dir_rel = run.directory.relative_to(project_dir)

        if last_run:
            first_differing_level = \
                next(i for i, (old, new) in enumerate(zip(last_run_dir_rel.parts,
                                                            run_dir_rel.parts))
                        if old != new)
            paths_and_evaluator_groups = list(zip([*reversed(last_run_dir_rel.parents),
                                                    last_run_dir_rel],
                                                    evaluator_groups))
            _run_evaluators(
                last_run.target, project_dir,
                paths_and_evaluator_groups[first_differing_level+1:])
        last_run, last_run_dir_rel = run, run_dir_rel
    else:
        paths_and_evaluator_groups = list(zip([*reversed(run_dir_rel.parents), run_dir_rel],
                                                evaluator_groups))
        _run_evaluators(run.target, project_dir, paths_and_evaluator_groups)


def _run_evaluators(
        target_label: str, project_dir: Path,
        paths_and_evaluator_groups: Sequence[Tuple[Path, Iterable[Metric]]]):
    for path, evaluators in reversed(paths_and_evaluator_groups):
        logger.info(f'Evaluating {path}')
        eval_dir = project_dir/path
        if not evaluators:
            continue

        preds_df = Lazy(partial(_get_preds_df, result_dir=eval_dir))

        #TODO rewrite this functionally (its nothing but a reduce operation)
        stats_df = None
        for evaluate in evaluators:
            if (df := evaluate(target_label, preds_df, eval_dir)) is not None:
                if stats_df is None:
                    stats_df = df
                else:
                    # make sure the two dfs have the same column level
                    levels = max(stats_df.columns.nlevels, df.columns.nlevels)
                    stats_df = _raise_df_column_level(stats_df, levels)
                    df = _raise_df_column_level(df, levels)
                    stats_df = stats_df.join(df)
        if stats_df is not None:
            stats_df.to_csv(eval_dir/'stats.csv')


def _raise_df_column_level(df, level):
    if df.columns.empty:
        columns = pd.MultiIndex.from_product([[]] * level)
    elif isinstance(df.columns, pd.MultiIndex):
        columns = pd.MultiIndex.from_tuples([col + (None,)*(level-df.columns.nlevels)
                                             for col in df.columns])
    else:
        columns = pd.MultiIndex.from_tuples([(col,) + (None,)*(level-df.columns.nlevels)
                                             for col in df.columns])

    return pd.DataFrame(df.values, index=df.index, columns=columns)


def _get_preds_df(result_dir: Path) -> pd.DataFrame:
    # load predictions
    if (preds_path := result_dir/'predictions.csv.zip').exists():
        try:
            preds_df = pd.read_csv(preds_path)
        except BadZipFile:
            # delete file and try to regenerate it
            preds_path.unlink()
            return _get_preds_df(result_dir)
    else:
        # create an accumulated predictions df if there isn't one already
        dfs = []
        #TODO do this recursively if needed
        for df_path in result_dir.glob('*/predictions.csv.zip'):
            df = pd.read_csv(df_path)
            # column which tells us which subset these predictions are from
            df[f'subset_{result_dir.name}'] = df_path.name
            dfs.append(df)

        preds_df = pd.concat(dfs)
        preds_df.to_csv(preds_path, index=False, compression='zip')

    return preds_df


def _do_run(run: Run, mode: Coordinator, model_path: Path, project_dir: Path,
           devices: Iterable, capacities: Iterable[Semaphore] = [], **kwargs) -> None:
    logger = logging.getLogger(str(run.directory.relative_to(project_dir)))
    logger.info(f'Starting run')

    _save_run_files(run, logger=logger)

    for device, capacity in zip(devices, capacities):
        # search for a free gpu
        if not capacity.acquire(blocking=False): continue   # type: ignore
        try:
            with torch.cuda.device(device):
                learn = (_train(train=mode.train, exp=run, logger=logger, **kwargs)
                        if mode.train and run.train_df is not None
                        else None)

                if mode.deploy and run.test_df is not None:
                    _deploy(deploy=mode.deploy, learn=learn, run=run, model_path=model_path, logger=logger,
                            **kwargs)

                break
        finally: capacity.release()
    else:
        raise RuntimeError('Could not find a free GPU!')



def _do_run_wrapper(kwds):
    #TODO explain!
    try:
        _do_run(**kwds)
        return kwds['run']
    except:
        logger.exception(f'Exception in run {kwds["run"]}!')
        return None


def _save_run_files(run: Run, logger) -> None:
    logger.info(f'Saving training/testing data for run {run.directory}...')
    run.directory.mkdir(exist_ok=True, parents=True)
    if run.train_df is not None and \
            not (training_set_path := run.directory/'training_set.csv.zip').exists():
        run.train_df.to_csv(training_set_path, index=False, compression='zip')
    if run.test_df is not None and \
            not (testing_set_path := run.directory/'testing_set.csv.zip').exists():
        run.test_df.to_csv(testing_set_path, index=False, compression='zip')


def _train(train: Trainer, exp: Run, logger, **kwargs) -> Learner:
    model_path = exp.directory/'export.pkl'
    if model_path.exists():
        logger.warning(f'{model_path} already exists, using old model!')
        return load_learner(model_path)

    logger.info('Starting training')
    learn = train(target_label=exp.target,
                  train_df=exp.train_df,
                  result_dir=exp.directory,
                  logger=logger,
                  **kwargs)

    return learn


def _deploy(deploy: Deployer, learn: Optional[Learner], run: Run, model_path: Optional[PathLike],
            logger, **kwargs) -> pd.DataFrame:
    preds_path = run.directory/'predictions.csv.zip'
    if preds_path.exists():
        logger.warning(f'{preds_path} already exists, using old predictions!')
        return pd.read_csv(preds_path)

    if not learn:
        logger.info('Loading model')
        learn = _load_learner_to_device(model_path or run.directory/'export.pkl')

    logger.info('Getting predictions')
    preds_df = deploy(learn=learn,
                      target_label=run.target,
                      test_df=run.test_df,
                      result_dir=run.directory,
                      logger=logger,
                      **kwargs)
    preds_df.to_csv(preds_path, index=False, compression='zip')

    return preds_df


def _load_learner_to_device(fname, device=None):
    """Loads a learner to a specific device."""
    device = torch.device(device or torch.cuda.current_device())
    res = torch.load(fname, map_location=device)
    res.dls.device = device
    if hasattr(res, 'to_fp32'): res = res.to_fp32()
    return res