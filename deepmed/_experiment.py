#!/usr/bin/env python3

import logging
from typing import Mapping, Union, Optional
from pathlib import Path
from multiprocessing import Manager, Process
from multiprocessing.pool import ThreadPool

from .types import *


__all__ = ['do_experiment']


logger = logging.getLogger(__name__)


def do_experiment(
        project_dir: PathLike,
        get: TaskGetter,
        num_concurrent_tasks: Optional[int] = None,
        devices: Mapping[Union[str, int], int] = {0: 4},
        logfile: Optional[str] = 'logfile'
        ) -> None:
    """Runs an experiement.

    Args:
        project_dir:  The directory to save project data in.
        get:  A function which generates tasks.
        train:  A function training a model for a specific task.
        deploy:  A function deploying a trained model.
        num_concurrent_tasks:  The maximum amount of tasks to do at the same
            time.  If None, the number of tasks will grow with the number of
            available devices.  If 0, all jobs will be task in the main process
            (useful for debugging).
        devices:  The devices to use for training and the maximum number of
            models to be trained at once for each device.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True, parents=True)

    # add logfile handler
    if logfile is not None:
        file_handler = logging.FileHandler(f'{project_dir/"logfile"}')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    logger.info('Getting tasks')

    with Manager() as manager:
        # semaphores which tell us which GPUs still have resources available
        capacities = {
                device: manager.Semaphore(capacity)   # type: ignore
                for device, capacity in devices.items()}
        tasks = get(project_dir=project_dir, manager=manager, capacities=capacities)
        num_concurrent_tasks = \
                sum(devices.values())*3 if num_concurrent_tasks is None else num_concurrent_tasks

        # We use a ThreadPool which starts processes so our launched processes are:
        #  1. Terminated after each training task so we don't leak resources
        #  2. We can spawn more processes in the launched subprocesses (not possible with Pool)
        with ThreadPool(num_concurrent_tasks or 1) as pool:
            # only use pool if we actually want to task multiple tasks in parallel
            # for loop to consume iterator
            for _ in (pool.imap_unordered(_task_wrapper, tasks, chunksize=1)
                    if num_concurrent_tasks >= 1
                    else (task.run() for task in tasks)):  # type: ignore
                pass


def _task_wrapper(task: Task) -> None:
    """Starts a new process to train a model."""
    p = Process(target=task.run)
    p.start()
    p.join()
