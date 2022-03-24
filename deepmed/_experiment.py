#!/usr/bin/env python3

import logging
import threading
from typing import Mapping, Union, Optional
from pathlib import Path
from concurrent import futures
from fastcore.parallel import ThreadPoolExecutor

from .types import *


__all__ = ['do_experiment']


logger = logging.getLogger(__name__)


def do_experiment(
        project_dir: PathLike,
        get: TaskGetter,
        num_concurrent_tasks: Optional[int] = 0,
        devices: Mapping[Union[str, int], int] = {0: 4},
        logfile: Optional[str] = 'logfile',
        keep_going: bool = False) -> None:
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
        keep_going:  Whether to stop all runs on an exception.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True, parents=True)

    # add logfile handler
    if logfile is not None:
        file_handler = logging.FileHandler(f'{project_dir/"logfile"}')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s: %(levelname)s: %(name)s: %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    logger.info('Getting tasks')

    # semaphores which tell us which GPUs still have resources available
    capacities = {
        device: threading.Semaphore(capacity)   # type: ignore
        for device, capacity in devices.items()}
    tasks = get(project_dir=project_dir,
                capacities=capacities)

    try:
        if num_concurrent_tasks == 0:
            for task in tasks:
                task.run()
        else:
            with ThreadPoolExecutor(num_concurrent_tasks) as e:
                running = [e.submit(Task.run, task) for task in tasks]
                for future in futures.as_completed(running):
                    future.result() # consume results to trigger exceptions

    except Exception as e:
        if not keep_going:
            raise e