#!/usr/bin/env python3
from deepmed.experiment_imports import *


cohorts_df = cohort(
    tiles_path='/path/to/features-7171845d',
    clini_path='/path/to/clini.xlsx',
    slide_path='/path/to/slide.csv')

cohorts_df = cohorts_df[~cohorts_df['ISHLT_2004_rej'].isna()]


def main():
    do_experiment(
        project_dir='crossval_mil',
        get=get.Crossval(
            get.SimpleRun(),
            cohorts_df=cohorts_df,
            target_label='ISHLT_2004_rej',
            balance=False,
            get_items=mil.get_h5s,
            train=mil.Train(metrics=[RocAucBinary()]),
            evaluators=[auroc],
            crossval_evaluators=[AggregateStats()],
        ),
        keep_going=False)


if __name__ == '__main__':
    main()
