from typing import Iterable, Iterator, Callable, Union
from pathlib import Path

import pandas as pd

from .._experiment import Task, EvalTask
from ..evaluators.types import Evaluator
from ..utils import factory


# @log_defaults
def _subgroup(
        get,
        *args,
        project_dir: Path,
        target_label: str,
        subgrouper: Callable[[pd.Series], Union[str, None]],
        subgroup_evaluators: Iterable[Evaluator] = [],
        cohort_df_arg_names: Iterable[str] = [
            'cohorts_df', 'train_cohorts_df', 'test_cohorts_df'],
        **kwargs) -> Iterator[Task]:
    """Splits a training data set into multiple subgroups.

    Args:
        train_cohorts_df:  Base data set to be split into subgroups.
        subgrouper:  A function mapping a sample of the training dataset onto a
            subgroup.  The function is given a row from the training dataset and
            has to return either a string describing the group name, or None if
            it shall be excluded from training.
        subgroup_evaluators:  A list of evaluators to be executed after all
            subgroup runs have been completed.
        cohort_df_arg_names:  The keys of cohort_dfs passed as kwargs to adapted
            task getters.
    """
    assert any(arg_name in kwargs for arg_name in cohort_df_arg_names), \
        f'none of {cohort_df_arg_names} given to `Subgroup()`!'
    groups = {
        cohorts_df_name: kwargs[cohorts_df_name].apply(subgrouper, axis=1)
        for cohorts_df_name in cohort_df_arg_names
        if cohorts_df_name in kwargs
    }
    assert groups, 'no subgroup instances found!'
    group_names = {
        x
        for gs in groups.values()
        for x in gs.unique()
        if x is not None
    }

    tasks = (
        task
        for group_name in group_names
        for task in get(  # type: ignore
            *args,
            project_dir=project_dir/group_name,
            target_label=target_label,
            **{**kwargs,
               **{
                   cohorts_df_name: kwargs[cohorts_df_name][gs == group_name]
                   for cohorts_df_name, gs in groups.items()
               }}))

    requirements = []
    for task in tasks:
        yield task
        requirements.append(task)

    yield EvalTask(
        path=project_dir,
        target_label=target_label,
        requirements=requirements,
        evaluators=subgroup_evaluators)

Subgroup = factory(_subgroup)