import logging
from multiprocessing.managers import SyncManager
from typing import Iterable, Iterator, Mapping
from pathlib import Path
from typing_extensions import Protocol

from .._experiment import Run, EvalRun
from ..utils import log_defaults
from ..metrics import Evaluator

import pandas as pd

#@log_defaults
def subgroup(
        get,
        *args,
        project_dir: Path,
        manager: SyncManager,
        target_label: str,
        cohorts_df: pd.DataFrame,
        subgroup_by: Mapping[str, Iterable[str]],
        subgroup_evaluators: Iterable[Evaluator] = [],
        **kwargs
        ) -> Iterator[Run]:
    tasks = (
        run
        for subgroup, classes in subgroup_by.items()
        for class_ in classes
        for run in get( # type: ignore
            *args,
            project_dir=project_dir/f'{subgroup}_{class_}',
            target_label=target_label,
            manager=manager,
            cohorts_df=cohorts_df[cohorts_df[subgroup] == class_],
            **kwargs))
    
    requirements = []
    for run in tasks:
        yield run
        requirements.append(run.done)

    yield EvalRun(
        directory=project_dir,
        target=target_label,
        requirements=requirements,
        evaluators=subgroup_evaluators,
        done=manager.Event())