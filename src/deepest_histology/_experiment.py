#!/usr/bin/env python3

import logging
from typing import Iterable, Optional, Callable, Sequence, Tuple, Iterator
from pathlib import Path
from multiprocessing import Manager, Process
from multiprocessing.pool import ThreadPool
from multiprocessing.synchronize import Semaphore
from functools import partial, lru_cache

import pandas as pd
import torch

from ._train import train
from ._deploy import deploy
from .utils import Lazy
from .metrics import Evaluator
from .types import *


__all__ = ['do_experiment']


logger = logging.getLogger(__name__)


def do_experiment(
        project_dir: PathLike,
        get: RunGetter,
        train: Trainer = train,
        deploy: Deployer = deploy,
        num_concurrent_runs: int = 4,
        devices = [torch.cuda.current_device()],
        ) -> None:
    """Runs an experiement.

    Args:
        project_dir:  The directory to save project data in.
        get:  A function which generates runs.
        train:  A function training a model for a specific run.
        deploy:  A function deploying a trained model.
        num_concurrent_runs:  The maximum amount of runs to do at the same time.
            Useful for multi-GPU systems.
        devices:  The devices to use for training.
    """
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
        capacities = [
            manager.Semaphore(max(1, (num_concurrent_runs+len(devices)-1)//len(devices)))  # type: ignore
            for _ in devices]
        run_args = ({'run': run, 'train': train, 'deploy': deploy, 'devices': devices,
                     'capacities': capacities}
                     for run in get(project_dir=project_dir, manager=manager))

        # We use a ThreadPool which starts processes so our launched processes are:
        #  1. Terminated after each training run so we don't leak resources
        #  2. We can spawn more processes in the launched subprocesses (not possible with Pool)
        with ThreadPool(num_concurrent_runs or 1) as pool:
            # only use pool if we actually want to run multiple runs in parallel
            runs = (pool.imap(_do_run_wrapper, run_args, chunksize=1) if num_concurrent_runs >= 1
                    else (_do_run_wrapper(args, spawn_process=False) for args in run_args))
            for _ in runs:
                pass


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


@lru_cache(maxsize=4)
def _generate_preds_df(result_dir: Path) -> pd.DataFrame:
    # load predictions
    if (preds_path := result_dir/'predictions.csv.zip').exists():
        preds_df = pd.read_csv(preds_path)
    else:
        # create an accumulated predictions df if there isn't one already
        dfs = []
        for df_path in result_dir.glob('**/predictions.csv.zip'):
            df = pd.read_csv(df_path)
            # column which tells us which subset these predictions are from
            df[f'subset_{result_dir.name}'] = df_path.name
            dfs.append(df)

        preds_df = pd.concat(dfs)
        preds_df.to_csv(preds_path, index=False, compression='zip')

    return preds_df


def _do_run(
        run: Run, train: Trainer, deploy: Deployer, devices: Iterable,
        capacities: Iterable[Semaphore] = []) \
        -> None:
    logger = logging.getLogger(str(run.directory))

    for reqirement in run.requirements:
        reqirement.wait()

    logger.info(f'Starting run')

    for device, capacity in zip(devices, capacities):
        # search for a free gpu
        if not capacity.acquire(blocking=False): continue   # type: ignore
        try:
            with torch.cuda.device(device):
                learn = train(run)
                preds_df = deploy(learn, run) if learn else None

                break
        finally: capacity.release()
    else:
        raise RuntimeError('Could not find a free GPU!')

    logger.info('Evaluating')
    _evaluate(run=run, preds_df=preds_df)


def _evaluate(run: Run, preds_df: pd.DataFrame):
    if preds_df is None: preds_df = _generate_preds_df(run.directory)
    stats_df = None
    for evaluate in run.evaluators:
        if (df := evaluate(run.target, preds_df, run.directory)) is not None:
            if stats_df is None:
                stats_df = df
            else:
                # make sure the two dfs have the same column level
                levels = max(stats_df.columns.nlevels, df.columns.nlevels)
                stats_df = _raise_df_column_level(stats_df, levels)
                df = _raise_df_column_level(df, levels)
                stats_df = stats_df.join(df)
    if stats_df is not None:
        stats_df.to_csv(run.directory/'stats.csv')



def _do_run_wrapper(kwargs, spawn_process: bool = True) -> Optional[Run]:
    """Starts a new process to train a model."""
    run = kwargs['run']
    try:
        # Starting a new process guarantees that the allocaded CUDA resources will
        # be released upon completion of training.
        if spawn_process:
            p = Process(target=_do_run, kwargs=kwargs)
            p.start()
            p.join()
        else:
            _do_run(**kwargs)
    finally:
        run.done.set()