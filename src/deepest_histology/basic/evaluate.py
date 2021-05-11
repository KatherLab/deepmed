import logging
from typing import Iterable, Dict, Mapping, Union, Any, Callable
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


Evaluator = Callable[..., Union[None, Mapping[str, Any]]]


def evaluate(
    project_dir: Path,
    target_evaluators: Iterable[Evaluator] = [],
    **kwargs) -> None:

    stats_dfs = []
    for target_dir in project_dir.iterdir():
        if not target_dir.is_dir():
            continue

        target_label = target_dir.name

        if not (preds_path := target_dir/'predictions.csv.zip').exists():
            logger.warning(f'Could not find predictions at {preds_path}')
            continue

        preds_df = pd.read_csv(preds_path)

        # generate a column containing the predicted class labels (i.e. the class with the highest 
        # score)
        if f'{target_label}_pred' not in preds_df:
            cols = [column
                    for column in preds_df.columns
                    if column.startswith(f'{target_label}_')]

            preds_df[f'{target_label}_pred'] = \
                preds_df[cols].idxmax(axis=1).map(lambda c: c[len(target_label)+1:])

        stats: Dict[str, Any] = {}
        for evaluate in target_evaluators:
            stats.update(evaluate(target_label, preds_df, target_dir, **kwargs) or {})

        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(target_dir/'results.csv', index=False)
        stats_dfs.append(stats_df)

    if stats_dfs:
        pd.concat(stats_dfs, axis=1).to_csv(project_dir/'results.csv', index=False)
    else:
        logger.info('Nothing to evaluate')