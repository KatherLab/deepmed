#!/usr/bin/env python3
from deepmed.experiment_imports import *


# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')

train_cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def main():
    do_experiment(
        project_dir='parameterize',
        get=get.Parameterize(
            get.Crossval(),
            get.SimpleRun(),
            cohorts_df=train_cohorts_df,
            target_label='TMB (nonsynonymous)',
            parameterizations=[
                {'folds': folds,
                 'train': Train(patience=patience, batch_size=bs, max_epochs=1)}
                for patience in [5]#, 8]
                for bs in [64]#, 128]
                for folds in [3]#, 5]
            ],
            #evaluators=[Grouped(count)],
            #crossval_evaluators=[AggregateStats(over=[0])],
            #parameterize_evaluators=[AggregateStats()],
        ),
    )


if __name__ == '__main__':
    main()
