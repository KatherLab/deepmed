from pathlib import Path


from fastai.vision.all import Learner
from fastai.vision.learner import load_learner

from .types import GPUTask
from .utils import factory

__all__ = ['Load']


def _load(
        task: GPUTask, /,
        project_dir: Path,
        training_project_dir: Path) \
        -> Learner:
    model_path = training_project_dir/task.path.relative_to(project_dir)/'export.pkl'
    return load_learner(model_path)

Load = factory(_load)
