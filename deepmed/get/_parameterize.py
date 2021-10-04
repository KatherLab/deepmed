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
        parameterizations: Mapping[str, Mapping[str, Any]],
        parameterize_evaluators: Iterable[Evaluator] = [],
        **kwargs) -> Iterator[Task]:
    """Starts a family of runs with different parameterizations.

    Args:
        parameterizations:  A mapping from parameterization descriptions (i.e.
            descriptive names) to kwargs mappings.  For each element, ``get``
            will be invoked with these kwargs.
        parameterize_evaluators:  Evaluators to run at the end of all
            parameterized runs.
        kwargs:  Additional arguments to pass to each parameterized run.  If a
            keyword argument appears both in ``kwargs`` and in a
            parameterization, the parameterization's argument takes precedence.

    Yields:
        The tasks ``get`` would yield for each of the parameterizations.
    """
    eval_reqirements = []
    for name, parameterization in parameterizations:
        for task in get(
                *args, project_dir=project_dir/name, manager=manager,
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


Parameterize = factory(_parameterize)
