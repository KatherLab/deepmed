import logging
from typing import Iterable, Dict, Any
from pathlib import Path

import pandas as pd

from ..basic.evaluate import Evaluator


logger = logging.getLogger(__name__)


def evaluate(
    project_dir: Path,
    fold_evaluators: Iterable[Evaluator] = [],
    target_evaluators: Iterable[Evaluator] = [],
    **kwargs) -> None:

    for target_dir in project_dir.iterdir():
        logger.info(f'Evaluating target {target_dir.name}')
        if not target_dir.is_dir():
            continue

        target_label = target_dir.name

        subset_dfs = []
        subset_stats_dfs = []
        for fold_dir in target_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
                continue

            logger.info(f'Evaluating {fold_dir.name}')
            if not (preds_path := fold_dir/'predictions.csv.zip').exists():
                logger.warning(f'Could not find predictions at {preds_path}')
                continue

            subset_df = pd.read_csv(preds_path)

            # generate a column containing the predicted class labels (i.e. the class with the
            # highest score)
            if f'{target_label}_pred' not in subset_df:
                cols = [column
                        for column in subset_df.columns
                        if column.startswith(f'{target_label}_')]

                subset_df[f'{target_label}_pred'] = \
                    subset_df[cols].idxmax(axis=1).map(lambda c: c[len(target_label)+1:])

            subset_df['subset'] = fold_dir.name
            subset_dfs.append(subset_df)

            stats: Dict[str, Any] = {}
            for evaluate in fold_evaluators:
                stats.update(evaluate(target_label, subset_df, fold_dir, **kwargs) or {})

            subset_stats_dfs.append(pd.DataFrame([stats], index=[fold_dir.name]))

        # accumulate simple statistics over folds
        stats_df = pd.concat(subset_stats_dfs)
        mean, std = stats_df.mean(), stats_df.std()
        stats_df.loc['mean'] = mean
        stats_df.loc['std'] = std
        stats_df.to_csv(target_dir/'results.csv', index=False)

        subsets_df = pd.concat(subset_dfs)
        stats = {}
        for evaluate in target_evaluators:
            stats.update(evaluate(target_label, subsets_df, target_dir, **kwargs) or {})

        #TODO accumulate over targets