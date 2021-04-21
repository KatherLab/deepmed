#!/usr/bin/env python3
import logging
from typing import Type, Sequence, Tuple, Callable, Optional, Any, Dict, TypeVar, Union, Iterable
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch


logger = logging.getLogger(__name__)

"""An object which the Deployer can apply to a test set."""
Model = TypeVar('Model')

"""A pandas dataframe having at least the following columns:

- block_path: Path:  A path to each tile.   #TODO unify block/tile nomenclature
- is_valid: bool:  True if the tile should be used for validation (e.g. for early stopping).
- At least one target column.
"""
TrainDF = Type[pd.DataFrame]

"""A pandas dataframe having at least the following columns:

- block_path: Path:  A path to each tile.
"""
TestDF = Type[pd.DataFrame]

"""A pandas dataframe returned by a `Deployer`. It hase at least the following columns:

- A column `{target_label}_pred`, where `target_label` is the name of the inferred target containing
  the predictions of a model on a test set.
- For categorical targets, a column `{target_label}_{class}` for each class, given the probability
  that the test item is of that class
- All columns present in the TestDF given to the `Deployer`.
"""
TestResultDF = Type[pd.DataFrame]

#TODO doc
EvalDF = Type[pd.DataFrame]#['PATIENT', target_label, f'{target_label}_pred',
                            # *[f'{target_label}_{class[c]}' for c in classes], ...]

@dataclass
class Run:
    directory: Path
    target: str
    train_df: Optional[TrainDF]
    test_df: Optional[TestDF]


"""A function which creates a series of runs."""
RunGetter = Callable[..., Sequence[Run]]

"""A function which trains a model.

Required kwargs:
    train_df: TrainDF:  A dataset specifying which tiles to train on.
    target_label: str:  The label to train on.
    result_dir:  A folder to write intermediate results to.

Returns:
    The trained model.
"""
Trainer = Callable[..., Model]

"""A function which deployes a model.

Required kwargs:
    model: Model:  The model to test on.
    target_label: str:  The name to be given to the result column.
    test_df: TestDF:  A dataframe specifying which tiles to deploy the model on.
    result_dir:  A folder to write intermediate results to.

Returns:
    `test_df`, but with additional columns for the predictions. #TODO reword
"""
Deployer = Callable[..., EvalDF]
# target_label, preds_df, result_dir
Evaluator = Callable[[str, pd.DataFrame, Path], pd.DataFrame]

def do_experiment(*,
        project_dir: Union[str, Path],
        get_runs: RunGetter,
        train: Optional[Trainer] = None,
        deploy: Optional[Deployer] = None,
        evaluate: Optional[Any] = None, #TODO
        **kwargs) -> None:

    project_dir = Path(project_dir)
    logger.info(f'starting project {project_dir}')
    project_dir.mkdir(exist_ok=True)

    logger.info('getting runs')
    runs = get_runs(project_dir=project_dir, **kwargs)
    create_experiment_dirs_(runs)

    for exp in runs:
        logger.info(f'starting experiment {exp.directory}')
        
        model = (train_(train=train, exp=exp, **kwargs)
                 if train and exp.train_df is not None
                 else None)

        preds_df = (deploy_(deploy=deploy, model=model, exp=exp, **kwargs)
                    if deploy and exp.test_df is not None
                    else None)

    logger.info(f'evaluating')
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

    logger.info('starting training')
    model = train(target_label=exp.target,
                  train_df=exp.train_df,
                  result_dir=exp.directory,
                  **kwargs)
    torch.save(model, model_path)

    return model


def deploy_(deploy: Deployer, model: Optional[Model], exp: Run, **kwargs):
    preds_path = exp.directory/'predictions.csv'
    if preds_path.exists():
        logger.warning(f'{preds_path} already exists, using old predictions!')
        return pd.read_csv(preds_path)

    if not model:
        logger.info('loading model')
        model = torch.load(exp.directory/'model.pt')

    logger.info('getting predictions')
    preds_df = deploy(model=model,
                      target_label=exp.target,
                      test_df=exp.test_df,
                      result_dir=exp.directory,
                      **kwargs)
    preds_df.to_csv(preds_path, index=False)

    return preds_df