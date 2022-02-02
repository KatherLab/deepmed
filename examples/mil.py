#!/usr/bin/env python3
from deepmed.experiment_imports import *


cohorts_df = cohort(
    tiles_path='/path/to/features',
    clini_path='/path/to/clini.xlsx',
    slide_path='/path/to/slide.csv')


def main():
    do_experiment(
        project_dir='crossval_mil',
        get=get.Crossval(
            get.SimpleRun(),
            cohorts_df=cohorts_df,
            target_label='ISHLT_2004_rej',
            evaluators=[auroc],
            crossval_evaluators=[AggregateStats()],
            # The next three lines are different from normal training
            get_items=mil.get_h5s,
            train=mil.Train(),
            balance=False,
        ),
        keep_going=False)


if __name__ == '__main__':
    main()
