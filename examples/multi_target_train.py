#!/usr/bin/env python3
from deepmed.experiment_imports import *


# this is a tiny toy data set; do not expect any good results from this
cohort_path = untar_data(
    'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.tar.gz')

train_cohorts_df = cohort(
    tiles_path=cohort_path/'tiles',
    clini_path=cohort_path/'clini.csv',
    slide_path=cohort_path/'slide.csv')


def main():
    do_experiment(
        project_dir=r'multi_target_train',
        get=partial(
            get.multi_target,   # train for multiple targets
            get.simple_run,
            train_cohorts_df=train_cohorts_df,
            target_labels=['ER Status By IHC'],  # target labels to train for
            max_train_tile_num=128,  # maximum number of tiles per patient to train with
            max_valid_tile_num=128,  # maximum number of tiles per patient to validate with
            # amount of data to use as validation set (for early stopping)
            valid_frac=.2,
            balance=True,   # weather to balance the training set
            na_values=['inconclusive'],  # labels to exclude in training
            min_support=10,  # minimal required patient-level class samples for a class to be considered
        ),
        train=partial(
            train,
            batch_size=96,
            # absolute maximum number of epochs to train for (usually preceeded by early stopping)
            max_epochs=32,
            metrics=[BalancedAccuracy()],   # additional metrics
            # epochs to continue training without improvement (will still select best model in the end)
            patience=3,
            monitor='valid_loss',   # metric to monitor for improvement
            # augmentations to apply to data
            tfms=aug_transforms(flip_vert=True, max_rotate=360,
                                max_zoom=1, max_warp=0, size=224),
        ),
        devices={'cuda:0': 4}
    )


if __name__ == '__main__':
    main()
