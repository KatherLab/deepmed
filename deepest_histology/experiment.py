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

TestResultDF = Type[pd.DataFrame]
"""A pandas dataframe returned by a `Deployer`. It hase at least the following columns:

- A column `{target_label}_pred`, where `target_label` is the name of the inferred target containing
  the predictions of a model on a test set.
- For categorical targets, a column `{target_label}_{class}` for each class, given the probability
  that the test item is of that class
- All columns present in the TestDF given to the `Deployer`.
"""

#TODO doc
EvalDF = Type[pd.DataFrame]#['PATIENT', target_label, f'{target_label}_pred',
                            # *[f'{target_label}_{class[c]}' for c in classes], ...]

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

Deployer = Callable[..., EvalDF]
"""A function which deployes a model.

Required kwargs:
    model: Model:  The model to test on.
    target_label: str:  The name to be given to the result column.
    test_df: TestDF:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.

Returns:
    `test_df`, but with additional columns for the predictions. #TODO reword
"""

# target_label, preds_df, result_dir
Evaluator = Callable[[str, pd.DataFrame, Path], pd.DataFrame]


def do_experiment(*,
        project_dir: Union[str, Path],
        get_runs: RunGetter,
        train: Optional[Trainer] = None,
        deploy: Optional[Deployer] = None,
        evaluate: Optional[Any] = None,  #TODO
        **kwargs) -> None:

    project_dir = Path(project_dir)
    logger.info(f'Starting project {project_dir}')
    project_dir.mkdir(exist_ok=True)

    logger.info('Getting runs')
    runs = get_runs(project_dir=project_dir, **kwargs)
    create_experiment_dirs_(runs)

    for run in runs:
        logger.info(f'Starting experiment {run.directory}')
        
        model = (train_(train=train, exp=run, **kwargs)
                 if train and run.train_df is not None
                 else None)

        preds_df = (deploy_(deploy=deploy, model=model, run=run, **kwargs)
                    if deploy and run.test_df is not None
                    else None)

    logger.info('Evaluating')
    if evaluate:
        preds_df = evaluate(project_dir, **kwargs)


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


def train_(train: Trainer, exp: Run, **kwargs) -> Model:
    model_path = exp.directory/'model.pt'
    if model_path.exists():
        logger.warning(f'{model_path} already exists, using old model!')
        return torch.load(model_path)

    logger.info('Starting training')
    model = train(target_label=exp.target,
                  train_df=exp.train_df,
                  result_dir=exp.directory,
                  **kwargs)
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