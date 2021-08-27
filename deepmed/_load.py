from pathlib import Path


from fastai.vision.all import Learner

from .types import GPURun
from fastai.vision.learner import load_learner

__all__ = ['load']


def load(
        run: GPURun, /,
        project_dir: Path,
        training_project_dir: Path) \
        -> Learner:
    model_path = training_project_dir/run.directory.relative_to(project_dir)/'export.pkl'
    return load_learner(model_path)