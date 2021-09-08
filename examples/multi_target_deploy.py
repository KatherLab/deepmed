#!/usr/bin/env python3
"""Deploys previously trained models on a new data set.

This example assumes that you have previously trained some models using e.g. the
``multi_target_train`` example.
"""

from deepmed.experiment_imports import *

# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')

test_cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def main():
    project_dir = 'multi_target_deploy'

    do_experiment(
        project_dir=project_dir,
        get=get.MultiTarget(
            get.SimpleRun(),
            test_cohorts_df=test_cohorts_df,
            target_labels=['ER Status By IHC', 'TCGA Subtype', 'TMB (nonsynonymous)'],
            max_test_tile_num=512,
            evaluators=[Grouped(auroc), Grouped(F1()), Grouped(count)],
            multi_target_evaluators=[AggregateStats(label='target')],
            train=Load(
                project_dir=project_dir,
                training_project_dir='multi_target_train'),
        ),
        devices={'cuda:0': 4},
        num_concurrent_tasks=0,
    )


if __name__ == '__main__':
    main()
