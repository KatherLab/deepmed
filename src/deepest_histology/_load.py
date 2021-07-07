from pathlib import Path


from fastai.vision.all import Learner

from .types import Run
from ._experiment import _load_learner_to_device

__all__ = ['load']


def load(
        run: Run, /,
        project_dir: Path,
        training_project_dir: Path) \
        -> Learner:
    model_path = training_project_dir/run.directory.relative_to(project_dir)/'export.pkl'
    return _load_learner_to_device(model_path)