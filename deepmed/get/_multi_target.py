from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any
from typing_extensions import Protocol
from ..types import Task, EvalTask
from ..evaluators.types import Evaluator


class MultiTargetBaseTaskGetter(Protocol):
    """The signature of a task getter which can be modified by ``multi_target``."""
    def __call__(
            self, *args,
            project_dir: Path, manager: SyncManager, target_label: str, **kwargs) -> Iterator[Task]:
        ...


def multi_target(
    get: MultiTargetBaseTaskGetter,
    *args,
    project_dir: Path,
    manager: SyncManager,
    target_labels: Iterable[str],
    multi_target_evaluators: Iterable[Evaluator] = [],
    target_kwargs: Mapping[str, Mapping[str, Any]] = {},
    **kwargs) -> Iterator[Task]:
    """Adapts a `TaskGetter` into a multi-target one.

    Args:
        get:  The `TaskGetter` to adapt; it has to take at least one keyword
            argument `target_label`.
        project_dir:  The directory to save the tasks' results to.
        target_label:  The target labels to invoke ``get`` on.
        target_kwargs:  A dictionary of ``kwargs`` to pass to ``get`` for
            specific targets.  If there is overlap between the ``kwargs`` given
            to :func:`multi_target` and ``target_kwargs`` for a certain target,
            the ``target_kwargs`` one takes precedence.
        *args:  Additional arguments give to ``get``.
        **kwargs:  Additional keyword arguments to give to ``get``.

    Yields:
        The tasks which would be yielded by `get` for each of the target labels,
        in the order of the target labels.  The task directories are prepended
        by a the name of the target label.
    """
    eval_reqirements = []
    for target_label in target_labels:
        target_dir = project_dir/target_label

        for task in get(
                *args, project_dir=target_dir, manager=manager, target_label=target_label,
                **({**kwargs, **target_kwargs[target_label]}
                   if target_label in kwargs
                   else kwargs)):
            eval_reqirements.append(task.done)
            yield task

    yield EvalTask(
        path=project_dir,
        target_label=target_label,
        requirements=eval_reqirements,
        evaluators=multi_target_evaluators,
        done=manager.Event())