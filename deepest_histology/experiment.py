#!/usr/bin/env python3
import logging
from typing import Type, Sequence, Tuple, Callable, Optional, Any, Dict, TypeVar, Union, Iterable
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch


logger = logging.getLogger(__name__)

Model = TypeVar('Model')
"""An object which the Deployer can apply to a test set."""

TrainDF = Type[pd.DataFrame]
"""A pandas dataframe having at least the following columns:

- block_path: Path:  A path to each tile.   #TODO unify block/tile nomenclature
- is_valid: bool:  True if the tile should be used for validation (e.g. for early stopping).
- At least one target column.
"""

TestDF = Type[pd.DataFrame]
"""A pandas dataframe having at least the following columns:

- block_path: Path:  A path to each tile.
"""

#TODO doc
TilePredsDF = Type[pd.DataFrame]


@dataclass
class Run:
    directory: Path
    target: str
    train_df: Optional[TrainDF]
    test_df: Optional[TestDF]


RunGetter = Callable[..., Sequence[Run]]
"""A function which creates a series of runs."""

Trainer = Callable[..., Model]
"""A function which trains a model.

Required kwargs:
    train_df: TrainDF:  A dataset specifying which tiles to train on.
    target_label: str:  The label to train on.
    result_dir:  A folder to write intermediate results to.

Returns:
    The trained model.
"""

Deployer = Callable[..., TilePredsDF]
"""A function which deployes a model.

Required kwargs:
    model: Model:  The model to test on.
    target_label: str:  The name to be given to the result column.
    test_df: TestDF:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.

Returns:
    `test_df`, but with additional columns for the predictions. #TODO reword
"""


@dataclass
class Coordinator:
    get: RunGetter
    train: Optional[Trainer] = None
    deploy: Optional[Deployer] = None
    evaluate: Optional[Callable] = None


def do_experiment(*,
        project_dir: Union[str, Path],
        mode: Coordinator,
        save_models: bool = True,
        **kwargs) -> None:

    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True)

    # add logfile handler
    file_handler = logging.FileHandler(project_dir/'logfile')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info('Getting runs')
    runs = mode.get(project_dir=project_dir, **kwargs)
    create_experiment_dirs_(runs)

    for run in runs:
        logger.info(f'Starting run {run.directory}')
        
        model = (train_(train=mode.train, exp=run, save_models=save_models, **kwargs)
                 if mode.train and run.train_df is not None
                 else None)

        preds_df = (deploy_(deploy=mode.deploy, model=model, run=run, **kwargs)
                    if mode.deploy and run.test_df is not None
                    else None)

    if mode.evaluate:
        logger.info('Evaluating')
        preds_df = mode.evaluate(project_dir, **kwargs)


def create_experiment_dirs_(runs: Iterable[Run]) -> None:
    for exp in runs:
        exp.directory.mkdir(exist_ok=True, parents=True)
        if exp.train_df is not None:
            if (training_set_path := exp.directory/'training_set.csv').exists():
                logger.warning(f'{training_set_path} already exists, using old training set!')
                exp.train_df = pd.read_csv(training_set_path)
            else:
                exp.train_df.to_csv(exp.directory/'training_set.csv', index=False)
        if exp.test_df is not None:
            if (testing_set_path := exp.directory/'testing_set.csv').exists():
                logger.warning(f'{testing_set_path} already exists, using old testing set!')
                exp.test_df = pd.read_csv(testing_set_path)
            else:
                exp.test_df.to_csv(exp.directory/'testing_set.csv', index=False)


def train_(train: Trainer, exp: Run, save_models: bool, **kwargs) -> Model:
    model_path = exp.directory/'model.pt'
    if model_path.exists():
        logger.warning(f'{model_path} already exists, using old model!')
        return torch.load(model_path)

    logger.info('Starting training')
    model = train(target_label=exp.target,
                  train_df=exp.train_df,
                  result_dir=exp.directory,
                  **kwargs)
    if save_models:
        torch.save(model, model_path)

    return model


def deploy_(deploy: Deployer, model: Optional[Model], run: Run, **kwargs):
    preds_path = run.directory/'predictions.csv'
    if preds_path.exists():
        logger.warning(f'{preds_path} already exists, using old predictions!')
        return pd.read_csv(preds_path)

    if not model:
        logger.info('Loading model')
        model = torch.load(run.directory/'model.pt')

    logger.info('Getting predictions')
    preds_df = deploy(model=model,
                      target_label=run.target,
                      test_df=run.test_df,
                      result_dir=run.directory,
                      **kwargs)
    preds_df.to_csv(preds_path, index=False)

    return preds_df