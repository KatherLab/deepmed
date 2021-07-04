#!/usr/bin/env python3

import logging
from zipfile import BadZipFile
from typing import Iterable, Optional, Callable, Sequence, Tuple, Iterator
from pathlib import Path
from multiprocessing import Manager, Process
from multiprocessing.pool import ThreadPool
from multiprocessing.synchronize import Semaphore
from functools import partial

from fastai.vision.all import Learner, load_learner

import pandas as pd
import torch
import coloredlogs

from . import train, deploy
from .utils import Lazy
from .metrics import Metric
from .types import *


__all__ = ['do_experiment']


coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def do_experiment(
        project_dir: PathLike,
        get: RunGetter,
        train: Optional[Trainer] = train,
        deploy: Optional[Deployer] = deploy,
        model_path: Optional[PathLike] = None,
        num_concurrent_runs: int = 4,
        devices = [torch.cuda.current_device()],
        evaluator_groups: Sequence[Iterable[Metric]] = []) -> None:
    """Runs an experiement.

    Args:
        project_dir:  The directory to save project data in.
        get:  A function which generates runs.
        train:  A function training a model for a specific run.
        deploy:  A function deploying a trained model.
        num_concurrent_runs:  The maximum amount of runs to do at the same time.
            Useful for multi-GPU systems.
        devices:  The devices to use for training.
        evaluator_groups:  TODO
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

    with Manager() as manager:
        # semaphores which tell us which GPUs still have resources free
        # each gpu is a assumed to have the same capabilities
        capacities = [manager.Semaphore((num_concurrent_runs+len(devices)-1)//len(devices))  # type: ignore
                      for _ in devices]
        run_args = ({'run': run, 'train': train, 'deploy': deploy, 'devices': devices,
                     'capacities': capacities, 'model_path': model_path, 'project_dir': project_dir}
                     for run in get(project_dir))

        # We use a ThreadPool which starts processes so our launched processes are:
        #  1. Terminated after each training run so we don't leak resources
        #  2. We can spawn more processes in the launched subprocesses (not possible with Pool)
        with ThreadPool(num_concurrent_runs or 1) as pool:
            # only use pool if we actually want to run multiple runs in parallel
            runs = filter(
                lambda x: x is not None,
                (pool.imap(_do_run_wrapper, run_args, chunksize=1) if num_concurrent_runs >= 1
                 else (_do_run_wrapper(**args) for args in run_args)))
            _evaluate_runs(runs, project_dir=project_dir, evaluator_groups=evaluator_groups)


def _evaluate_runs(
        runs: Iterator[Run], project_dir: Path, evaluator_groups: Sequence[Iterable[Callable]]) \
        -> None:
    """Calls evaluation functions for each run.

    Args:
        runs:  An iterator over the already completed runs.  This iterator has
            to traverse the runs in-order.
        project_dir:  The root directory of the experiment.
        evaluator_groups:  A sequence of collections of evaluation functions.

    TODO a more detailed description

    Assume we have the evaluator groups `[A, B, C]`.  Then the the evaluator
    groups will be invoked as follows:

        root/a/b
        root/a/c   -> C(b)
        root/a/d   -> C(c)
        root/e/f   -> C(d), B(a)
        root/e/g/h -> C(f)
        root/e/g/i
        root/e/j   -> C(g)
                   -> C(j), B(e), A(root)

    where B(a) means that all the evaluation functions in evaluator group B will
    be invoked on run a.
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


def _do_run(
        run: Run, train: Optional[Trainer], deploy: Optional[Deployer], model_path: Path,
        devices: Iterable, capacities: Iterable[Semaphore] = []) \
        -> None:
    assert not (model_path and run.train_df is not None), \
        'Specified both a model to deploy on and a training procedure'

    run.logger.info(f'Starting run')

    for device, capacity in zip(devices, capacities):
        # search for a free gpu
        if not capacity.acquire(blocking=False): continue   # type: ignore
        try:
            with torch.cuda.device(device):
                learn = (_train(train=train, run=run)
                        if train and run.train_df is not None
                        else None)

                if deploy and run.test_df is not None:
                    _deploy(deploy=deploy, learn=learn, run=run, model_path=model_path)

                break
        finally: capacity.release()
    else:
        raise RuntimeError('Could not find a free GPU!')


def _do_run_wrapper(kwargs) -> Optional[Run]:
    """Starts a new process to train a model."""
    run = kwargs['run']
    # Starting a new process guarantees that the allocaded CUDA resources will
    # be released upon completion of training.
    p = Process(target=_do_run, kwargs=kwargs)
    p.start()
    p.join()

    if p.exitcode == 0:
        return run
    else:
        return None


def _train(train: Trainer, run: Run) -> Learner:
    model_path = run.directory/'export.pkl'
    if model_path.exists():
        run.logger.warning(f'{model_path} already exists, using old model!')
        return load_learner(model_path)

    run.logger.info('Starting training')
    learn = train(run)

    return learn


def _deploy(
    deploy: Deployer, learn: Optional[Learner], run: Run, model_path: Optional[PathLike]) -> pd.DataFrame:
    preds_path = run.directory/'predictions.csv.zip'
    if preds_path.exists():
        run.logger.warning(f'{preds_path} already exists, using old predictions!')
        return pd.read_csv(preds_path)

    if not learn:
        run.logger.info('Loading model')
        learn = _load_learner_to_device(model_path or run.directory/'export.pkl')

    run.logger.info('Getting predictions')
    preds_df = deploy(learn, run)
    preds_df.to_csv(preds_path, index=False, compression='zip')

    return preds_df


def _load_learner_to_device(fname, device=None):
    """Loads a learner to a specific device."""
    device = torch.device(device or torch.cuda.current_device())
    res = torch.load(fname, map_location=device)
    res.dls.device = device
    if hasattr(res, 'to_fp32'): res = res.to_fp32()
    return res