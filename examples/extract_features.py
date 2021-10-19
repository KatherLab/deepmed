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
        project_dir='extract_features',
        get=get.SimpleRun(
            train_cohorts_df=train_cohorts_df,
            train=extract_features.Extract(),
            target_label='ER Status By IHC',
            feature_save_path='extracted_features',
        ),
        num_concurrent_tasks=0,
    )


if __name__ == '__main__':
    main()
