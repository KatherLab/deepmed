import logging
from typing import Optional, Callable, Iterator, Union
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from fastai.vision.all import Learner

__all__ = [
    'Run', 'RunGetter', 'Trainer', 'Deployer', 'PathLike']


@dataclass
class Run:
    """A collection of data to train or test a model."""
    directory: Path
    """The directory to save data in for this run."""
    target: str
    """The name of the target to train or deploy on."""
    train_df: Optional[pd.DataFrame] = None
    """A dataframe mapping tiles to be used for training to their
       targets.

    It contains at least the following columns:
    - tile_path: Path
    - is_valid: bool:  whether the tile should be used for validation (e.g. for
    early stopping).
    - At least one target column with the name saved in the run's `target`.
    """
    test_df: Optional[pd.DataFrame] = None
    """A dataframe mapping tiles used for testing to their targets.

    It contains at least the following columns:
    - tile_path: Path
    """

    logger: logging.Logger = logging.getLogger('deepest_histology')
    """Logger to write messages to."""


RunGetter = Callable[[Path], Iterator[Run]]
"""A function which creates a series of runs.

Args:
    project_dir:  The directory to save the run's data in.

Returns:
    An iterator over all runs.
"""

Trainer = Callable[[Run], Learner]
"""A function which trains a model.

Args:
    run:  The run to train.

Returns:
    The trained model.
"""

Deployer = Callable[[Learner, Run], pd.DataFrame]
"""A function which deployes a model.

Required kwargs:
    model:  The model to test on.
    target_label:  The name to be given to the result column.
    test_df:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.

Returns:
    `test_df`, but with at least an additional column for the target
    predictions.
"""


PathLike = Union[str, Path]