import random
import logging
from typing import Iterable, Sequence, List, Optional, Callable, Any
from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ..experiment import Run
from ..config import Cohort
from ..basic.get_runs import concat_cohorts, get_tiles, balance_classes, discretize


logger = logging.getLogger(__name__)


#TODO log defaults
def create_runs(*,
        project_dir: Path,
        target_labels: Iterable[str],
        cohorts: Iterable[Cohort] = [],
        max_tile_num: int = 500,
        folds: int = 3,
        seed: int = 0,
        valid_frac: float = .1,
        n_bins: int = 2,
        na_values: Iterable[Any] = [],
        **kwargs) -> Sequence[Run]:

    runs = []

    for target_label in target_labels:
        existing_fold_dirs = [fold_dir
                              for fold_dir in (project_dir/target_label).iterdir()
                              if fold_dir.is_dir() and fold_dir.name.startswith('fold_')] \
                if (project_dir/target_label).is_dir() else None

        logger.info(f'For target {target_label}:')
        if existing_fold_dirs:
            logger.info(f'Using old training and testing set')
            for fold_dir in existing_fold_dirs:
                train_path = fold_dir/'training_set.csv.zip'
                test_path = fold_dir/'testing_set.csv.zip'
                runs.append(
                    Run(directory=fold_dir,
                        target=target_label,
                        train_df=pd.read_csv(train_path) if train_path.exists else None,
                        test_df=pd.read_csv(test_path) if test_path.exists else None))
        else:
            assert cohorts, 'No old training and testing sets found and no cohorts given!'
            cohorts_df = concat_cohorts(
                cohorts=cohorts, target_label=target_label, na_values=na_values)
                
            # discretize values if necessary
            if pd.api.types.is_numeric_dtype(cohorts_df[target_label]) and \
                    cohorts_df[target_label].nunique() > 10:
                cohorts_df[target_label] = discretize(cohorts_df[target_label], n_bins=n_bins)
                
            logger.info(f'Slide target counts: {dict(cohorts_df[target_label].value_counts())}')
            
            folded_df = create_folds(cohorts_df=cohorts_df, target_label=target_label, folds=folds,
                                    valid_frac=valid_frac, seed=seed)
            logger.info(f'Searching for tiles')
            tiles_df = get_tiles(cohorts_df=folded_df, max_tile_num=max_tile_num,
                                target=target_label, seed=seed)

            for fold in sorted(folded_df.fold.unique()):
                logger.info(f'For fold {fold}:')
                logger.info(f'Training tiles: {dict(tiles_df[(tiles_df.fold != fold) & ~tiles_df.is_valid][target_label].value_counts())}')
                logger.info(f'Validation tiles: {dict(tiles_df[(tiles_df.fold != fold) & tiles_df.is_valid][target_label].value_counts())}')
                train_df = balance_classes(
                    tiles_df=tiles_df[(tiles_df.fold != fold) & ~tiles_df.is_valid],
                    target=target_label)
                valid_df = balance_classes(
                    tiles_df=tiles_df[(tiles_df.fold != fold) & tiles_df.is_valid],
                    target=target_label)
                logger.info(f'{len(train_df)} training tiles')
                logger.info(f'{len(valid_df)} validation tiles')

                test_df = tiles_df[tiles_df.fold == fold]
                logger.info(f'{len(test_df)} testing tiles')
                assert not test_df.empty, 'Empty fold in cross validation!'

                runs.append(Run(directory=project_dir/target_label/f'fold_{fold}',
                                target=target_label,
                                train_df=pd.concat([train_df, valid_df]),
                                test_df=test_df))

    return runs


#TODO find better name
def load_runs(*,
        targets: Iterable[str],
        project_dir: Path,
        **kwargs) -> Sequence[Run]:

    runs = []
    for target in targets:
        for target_dir in (project_dir/target).iterdir():
            if target_dir.is_dir() and target_dir.name.startswith('fold_'):
                train_path = target_dir/'training_set.csv'
                test_path = target_dir/'testing_set.csv'
                runs.append(
                    Run(directory=target_dir,
                        target=target,
                        train_df=pd.read_csv(train_path) if train_path.exists else None,
                        test_df=pd.read_csv(test_path) if test_path.exists else None))

    return runs



# TODO define types for dfs
def create_folds(
        cohorts_df: pd.DataFrame, target_label, folds: int, valid_frac: float, seed: int) \
        -> pd.DataFrame:

    kf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    # Pepare our dataframe
    # We enumerate each fold; this way, the training set for the `k`th iteration can be easily
    # obtained through `df[df.fold != k]`. Additionally, we sample a validation set for early
    # stopping frotarget_labell.
    cohorts_df['fold'] = 0
    cohorts_df['is_valid'] = False
    for fold, (train_idx, test_idx) \
            in enumerate(kf.split(cohorts_df['PATIENT'], cohorts_df[target_label])):
        #FIXME: remove ugly iloc magic to prevent `SettingWithCopyWarning`
        cohorts_df.iloc[test_idx, cohorts_df.columns.get_loc('fold')] = fold
        test_df = cohorts_df.iloc[test_idx]
        cohorts_df.is_valid |= \
            cohorts_df.index.isin(test_df.sample(frac=valid_frac, random_state=seed).index)
    
    return cohorts_df
