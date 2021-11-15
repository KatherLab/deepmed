#!/usr/bin/env python3
from deepmed.experiment_imports import *
from deepmed import mil

def get_bags(
        cohorts_df: pd.DataFrame, max_tile_num: int, logger: logging.Logger
) -> pd.DataFrame:
    cohorts_df.slide_path = cohorts_df['slide_path'].map(lambda f: f.parent/f'{f.name}.h5')
    return cohorts_df[cohorts_df.slide_path.map(lambda f: f.exists())]

# this is a tiny toy data set; do not expect any good results from this
cohorts_df = cohort(
    tiles_path='h5/path',
    clini_path='clini/path.csv',
    slide_path='slide/path.csv')


def main():
    do_experiment(
        project_dir='mil',
        get=get.Crossval(
            get.SimpleRun(),
            cohorts_df=cohorts_df,
            target_label='ISHLT_2004_rej',
            valid_frac=.2,
            folds=48,
            balance=False,
            get_items=get_bags,
            crossval_evaluators=[AggregateStats(label='fold', over=['fold'])],
            evaluators=[auroc, p_value],
            train=mil.Train(batch_size=8, metrics=RocAucBinary()),
        ),
        num_concurrent_tasks=0)


if __name__ == '__main__':
    main()
