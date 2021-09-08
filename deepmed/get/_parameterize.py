from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any
from typing_extensions import Protocol
from ..types import Task, EvalTask
from ..evaluators.types import Evaluator
from ..utils import factory


class ParameterizeBaseTaskGetter(Protocol):
    """The signature of a task getter which can be modified by ``parameterize``."""

    def __call__(
            self, *args,
            project_dir: Path, manager: SyncManager, **kwargs) -> Iterator[Task]:
        ...


def _parameterize(
        get: ParameterizeBaseTaskGetter,
        *args,
        project_dir: Path,
        manager: SyncManager,
        parameterizations: Iterable[Mapping[str, Any]],
        parameterize_evaluators: Iterable[Evaluator] = [],
        **kwargs) -> Iterator[Task]:
    eval_reqirements = []
    for parameterization in parameterizations:
        path = project_dir/_make_dir_name(parameterization)

        for task in get(
                *args, project_dir=path, manager=manager,
                # overwrite default ``kwargs``` w/ parameterization ones, if they were given
                **{**kwargs, **parameterization}):
            eval_reqirements.append(task.done)
            yield task

    yield EvalTask(
        path=project_dir,
        target_label=None,  # TODO remove target label from eval task
        requirements=eval_reqirements,
        evaluators=parameterize_evaluators,
        done=manager.Event())


def _make_dir_name(parameters: Mapping[str, Any]) -> str:
    return '; '.join(f'{k}={v!r}' for k, v in sorted(parameters.items()))


Parameterize = factory(_parameterize)
