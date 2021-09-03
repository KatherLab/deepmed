#!/usr/bin/env python3
"""Deploys previously trained models on a new data set.

This example assumes that you have previously trained some models using e.g. the
``multi_target_train`` example.
"""

from deepmed.experiment_imports import *

# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.tar.gz')

test_cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def main():
    project_dir = 'multi_target_deploy'

    do_experiment(
        project_dir=project_dir,
        get=partial(
            get.multi_target,
            get.simple_run,
            test_cohorts_df=test_cohorts_df,
            target_labels=['ER Status By IHC'],
            max_test_tile_num=512,
            evaluators=[Grouped(auroc), Grouped(f1), Grouped(count)],
            multi_target_evaluators=[aggregate_stats],
            train=partial(
                load,
                project_dir=project_dir,
                training_project_dir='multi_target_train'),
        ),
        devices={'cuda:0': 4}
    )
