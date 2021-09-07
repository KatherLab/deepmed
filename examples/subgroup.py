#!/usr/bin/env python3
from deepmed.experiment_imports import *


# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')

cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def subgrouper(x: pd.Series):
    if x['Diagnosis Age'] > 50:
        return 'old'
    elif x['Diagnosis Age'] <= 50:
        return 'young'
    else:
        return None


def main():
    do_experiment(
        project_dir='subgroup',
        get=get.Subgroup(
            get.SimpleRun(),
            train_cohorts_df=cohorts_df,
            test_cohorts_df=cohorts_df,
            target_label='ER Status By IHC',
            subgrouper=subgrouper,
            valid_frac=.2,
            evaluators=[Grouped(auroc), Grouped(count)],
            subgroup_evaluators=[AggregateStats()],
            train=Train(max_epochs=1),
        ),
        devices={'cuda:0': 4}
    )


if __name__ == '__main__':
    main()
