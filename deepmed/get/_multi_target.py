from pathlib import Path
from typing import Iterable, Iterator

from ..types import Task
from ..evaluators.types import Evaluator
from ..utils import factory
from ._parameterize import _parameterize, ParameterizeBaseTaskGetter


def _multi_target(
        get: ParameterizeBaseTaskGetter,
        *args,
        project_dir: Path,
        target_labels: Iterable[str],
        multi_target_evaluators: Iterable[Evaluator] = [],
        **kwargs) -> Iterator[Task]:
    """Adapts a `TaskGetter` into a multi-target one.

    Convenience wrapper around :func:``deepmed.Parameterize``.

    Args:
        get:  The `TaskGetter` to adapt; it has to take at least one keyword
            argument `target_label`.
        project_dir:  The directory to save the tasks' results to.
        target_label:  The target labels to invoke ``get`` on.
        *args:  Additional arguments give to ``get``.
        **kwargs:  Additional keyword arguments to give to ``get``.

    Yields:
        The tasks which would be yielded by `get` for each of the target labels,
        in the order of the target labels.  The task directories are prepended
        by a the name of the target label.
    """
    return _parameterize(
        get, *args, project_dir=project_dir,
        parameterizations={
            target_label: {'target_label': target_label}
            for target_label in target_labels},
        parameterize_evaluators=multi_target_evaluators,
        **kwargs)


MultiTarget = factory(_multi_target)
