import logging

from typing import Optional, Callable, Iterator, Union
from typing_extensions import Protocol
from pathlib import Path
from dataclasses import dataclass, field
from threading import Event

import pandas as pd
from multiprocessing.managers import BaseManager
from multiprocessing.synchronize import Semaphore

from typing import Iterable
from pathlib import Path

import torch
import pandas as pd
from fastai.vision.all import Learner

from .metrics import Evaluator

__all__ = [
    'Run', 'GPURun', 'EvalRun', 'RunGetter', 'Trainer', 'Deployer', 'PathLike']


@dataclass
class Run:
    directory: Path
    """The directory to save data in for this run."""
    target: str
    """The name of the target to train or deploy on."""

    done: Event
    requirements: Iterable[Event] = field(default_factory=list)

    def __call__(self) -> None:
        for reqirement in self.requirements:
            reqirement.wait()


class RunGetter(Protocol):
    def __call__(self, project_dir: Path, manager: BaseManager) -> Iterator[Run]:
        """A function which creates a series of runs.

        Args:
            project_dir:  The directory to save the run's data in.

        Returns:
            An iterator over all runs.
        """
        ...


Trainer = Callable[[Run], Optional[Learner]]
"""A function which trains a model.

Args:
    run:  The run to train.

Returns:
    The trained model.
"""

Deployer = Callable[[Learner, Run], pd.DataFrame]
"""A function which deployes a model.

Writes the results to a file ``predictions.csv.zip`` in the run directory.

Args:
    model:  The model to test on.
    target_label:  The name to be given to the result column.
    test_df:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.
"""

PathLike = Union[str, Path]


@dataclass
class GPURun(Run):
    """A collection of data to train or test a model."""

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

    def __call__(
            self, train: Trainer, deploy: Deployer, devices: Iterable,
            capacities: Iterable[Semaphore] = [], **_) \
            -> None:

        super().__call__()
        logger = logging.getLogger(str(self.directory))
        logger.info(f'Starting GPU run')

        for device, capacity in zip(devices, capacities):
            # search for a free gpu
            if not capacity.acquire(blocking=False): continue   # type: ignore
            try:
                with torch.cuda.device(device):
                    learn = train(self)
                    deploy(learn, self) if learn else None

                    break
            except Exception as e:
                logger.exception(e)
                raise e
            finally: capacity.release()
        else:
            raise RuntimeError('Could not find a free GPU!')


@dataclass
class EvalRun(Run):
    evaluators: Iterable[Evaluator] = field(default_factory=list)

    def __call__(self, **_) -> None:
        super().__call__()
        logger = logging.getLogger(str(self.directory))
        logger.info('Evaluating')

        preds_df = _generate_preds_df(self.directory)
        stats_df = None
        for evaluate in self.evaluators:
            if (df := evaluate(self.target, preds_df, self.directory)) is not None:
                if stats_df is None:
                    stats_df = df
                else:
                    # make sure the two dfs have the same column level
                    levels = max(stats_df.columns.nlevels, df.columns.nlevels)
                    stats_df = _raise_df_column_level(stats_df, levels)
                    df = _raise_df_column_level(df, levels)
                    stats_df = stats_df.join(df)
        if stats_df is not None:
            stats_df.to_csv(self.directory/'stats.csv')


def _raise_df_column_level(df, level):
    if df.columns.empty:
        columns = pd.MultiIndex.from_product([[]] * level)
    elif isinstance(df.columns, pd.MultiIndex):
        columns = pd.MultiIndex.from_tuples([col + (None,)*(level-df.columns.nlevels)
                                             for col in df.columns])
    else:
        columns = pd.MultiIndex.from_tuples([(col,) + (None,)*(level-df.columns.nlevels)
                                             for col in df.columns])

    return pd.DataFrame(df.values, index=df.index, columns=columns)


def _generate_preds_df(result_dir: Path) -> pd.DataFrame:
    # load predictions
    if (preds_path := result_dir/'predictions.csv.zip').exists():
        preds_df = pd.read_csv(preds_path, low_memory=False)
    else:
        # create an accumulated predictions df if there isn't one already
        dfs = []
        for df_path in result_dir.glob('**/predictions.csv.zip'):
            df = pd.read_csv(df_path, low_memory=False)
            # column which tells us which subset these predictions are from
            df[f'subset_{result_dir.name}'] = df_path.name
            dfs.append(df)

        preds_df = pd.concat(dfs)
        preds_df.to_csv(preds_path, index=False, compression='zip')

    return preds_df