
#!/usr/bin/env python3
from deepmed.experiment_imports import *


# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')

cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def main():
    do_experiment(
        project_dir='continuous',
        get=get.SimpleRun(
            train_cohorts_df=cohorts_df,
            test_cohorts_df=cohorts_df,
            target_label='TMB (nonsynonymous)',
            n_bins=None,
            evaluators=[Grouped(r2), r2],
        ),
    )


if __name__ == '__main__':
    main()
