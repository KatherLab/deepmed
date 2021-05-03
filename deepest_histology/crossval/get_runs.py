import random
import logging
from typing import Iterable, Sequence, List, Optional, Callable, Any
from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ..experiment import Run, TrainDF, TestDF, TilePredsDF
from ..config import Cohort


logger = logging.getLogger(__name__)


#TODO log defaults
def create_runs(*,
        project_dir: Path,
        target_labels: Iterable[str],
        cohorts: Iterable[Cohort],
        max_tile_num: int = 500,
        folds: int = 3,
        seed: int = 0,
        valid_frac: float = .1,
        **kwargs) -> Sequence[Run]:

    runs = []

    for target_label in target_labels:
        logger.info(f'For target {target_label}:')
        cohorts_df = concat_cohorts(cohorts=cohorts, target=target_label)
        folded_df = create_folds(cohorts_df=cohorts_df, target=target_label, folds=folds,
                                 valid_frac=valid_frac, seed=seed)
        logger.info(f'Searching for tiles')
        tiles_df = get_tiles(cohorts_df=folded_df, max_tile_num=max_tile_num,
                             target=target_label, seed=seed)

        for fold in sorted(folded_df.fold.unique()):
            logger.info(f'For fold {fold}:')
            train_df = balance_classes(tiles_df=tiles_df[tiles_df.fold != fold], target=target_label)
            logger.info(f'{len(train_df)} training tiles')
            test_df = tiles_df[tiles_df.fold == fold]
            logger.info(f'{len(test_df)} testing tiles')
            assert not test_df.empty, 'Empty fold in cross validation!'

            runs.append(Run(directory=project_dir/target_label/f'fold_{fold}',
                            target=target_label,
                            train_df=train_df,
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



def concat_cohorts(cohorts: Iterable[Cohort], target: str) -> pd.DataFrame:
    """Constructs a dataframe containing patient, slide and label data for multiple cohorts.
    
    Returns:
        A dataframe with the columns 'PATIENT', `target` and 'BLOCK_DIR'.
    """

    cohort_dfs: List[pd.DataFrame] = []

    for cohort in cohorts:
        clini_path, slide_path, tile_dir = cohort.clini_table, cohort.slide_table, cohort.tile_dir

        clini_df = (pd.read_csv(clini_path) if clini_path.suffix == '.csv'
                    else pd.read_excel(clini_path))
        slide_df = (pd.read_csv(slide_path) if slide_path.suffix == '.csv'
                    else pd.read_excel(slide_path))
        logger.info(f'#patients: {len(clini_df)}')
        logger.info(f'#slides: {len(slide_df)}')

        # filter n/a values
        clini_df[target] = clini_df[target].replace(' ', '')
        clini_df = clini_df[clini_df[target].notna()]
        for na_value in ['NA', 'NA ', 'NAN', 'N/A', 'na', 'n.a', 'N.A', 'UNKNOWN', 'x',
                        'NotAPPLICABLE', 'NOTPERFORMED', 'NotPerformed', 'Notassigned', 'excluded',
                        'exclide', '#NULL', 'PerformedButNotAvailable', 'x_', 'NotReported',
                        'notreported', 'INCONCLUSIVE']:
            clini_df = clini_df[clini_df[target] != na_value]

        logger.info(f'#slides after removing N/As: {len(clini_df)}')

        # strip patient ids, slide names
        clini_df['PATIENT'] = clini_df['PATIENT'].str.strip()
        slide_df['PATIENT'] = slide_df['PATIENT'].str.strip()
        slide_df['FILENAME'] = slide_df['FILENAME'].str.strip()

        # only keep patients which have slides
        cohort_df = clini_df.merge(slide_df, on='PATIENT')
        cohort_df['cohort'] = cohort.root_dir.name
        logger.info(f'#slides after removing slides without patient data: {len(cohort_df)}')

        # only keep slides which have tiles
        cohort_df['BLOCK_DIR'] = tile_dir/cohort_df['FILENAME']

        cohort_dfs.append(cohort_df)

    # merge cohort dfs
    cohorts_df = cohort_dfs[0]
    for cohort_df in cohort_dfs[1:]:
        # check for patient overlap between cohorts
        if shared := set(cohorts_df['PATIENT']) & set(cohort_df['PATIENT']):
            raise RuntimeError(f'Patient overlap between cohorts', shared)

        cohorts_df = pd.concat([cohorts_df, cohort_df])

    return cohort_df


#TODO df types
def get_tiles(cohorts_df: pd.DataFrame, max_tile_num: int, target: str, seed: int) -> pd.DataFrame:
    """Create df containing patient, tiles, other data."""
    random.seed(seed)   #FIXME doesn't work
    tiles_dfs = []
    for patient, data in tqdm(cohorts_df.groupby('PATIENT')):
        tiles = [file
                 for tile_dir in data['BLOCK_DIR']
                 if tile_dir.exists()
                 for file in tile_dir.iterdir()]
        tiles = random.sample(tiles, min(len(tiles), max_tile_num))
        
        tiles_dfs.append(data.drop(columns='BLOCK_DIR')
                             .merge(pd.Series(tiles, name='tile_path'), how='cross'))

    tiles_df = pd.concat(tiles_dfs).reset_index(drop=True)
    logger.info(f'Found {len(tiles_df)} tiles for {len(tiles_df["PATIENT"].unique())} patients')

    return tiles_df


def balance_classes(tiles_df: pd.DataFrame, target: str) -> pd.DataFrame:
    smallest_class_count = min(tiles_df[target].value_counts())
    for label in tiles_df[target].unique():
        tiles_with_label = tiles_df[tiles_df[target] == label]
        to_keep = tiles_with_label.sample(n=smallest_class_count).index
        tiles_df = tiles_df[(tiles_df[target] != label) | (tiles_df.index.isin(to_keep))]

    return tiles_df


# TODO define types for dfs
def create_folds(
        cohorts_df: pd.DataFrame, target: str, folds: int, valid_frac: float, seed: int) \
        -> pd.DataFrame:

    kf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    kf.get_n_splits(cohorts_df['PATIENT'], cohorts_df[target])

    # Pepare our dataframe
    # We enumerate each fold; this way, the training set for the `k`th iteration can be easily
    # obtained through `df[df.fold != k]`. Additionally, we sample a validation set for early
    # stopping from each fold.
    cohorts_df['fold'] = 0
    cohorts_df['is_valid'] = False
    for fold, (train_idx, test_idx) \
            in enumerate(kf.split(cohorts_df['PATIENT'], cohorts_df[target])):
        #FIXME: remove ugly iloc magic to prevent `SettingWithCopyWarning`
        cohorts_df.iloc[test_idx, cohorts_df.columns.get_loc('fold')] = fold
        test_df = cohorts_df.iloc[test_idx]
        cohorts_df.is_valid |= \
            cohorts_df.index.isin(test_df.sample(frac=valid_frac, random_state=seed).index)
    
    return cohorts_df