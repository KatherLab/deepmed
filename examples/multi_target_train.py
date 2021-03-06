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
        project_dir='multi_target_train',
        get=get.MultiTarget(    # train for multiple targets
            get.SimpleRun(),
            train_cohorts_df=train_cohorts_df,
            target_labels=['ER Status By IHC', 'TCGA Subtype',
                           'TMB (nonsynonymous)'],  # target labels to train for
            get_items=get.GetTiles(max_tile_nums={
                get.DatasetType.TRAIN: 128, # maximum number of tiles per patient to train with
                get.DatasetType.VALID: 256, # maximum number of tiles per patient to validate with
                get.DatasetType.TEST: 512   # maximum number of tiles per patient to test on
            }),
            # amount of data to use as validation set (for early stopping)
            valid_frac=.2,
            balance=True,   # weather to balance the training set
            na_values=['inconclusive'],  # labels to exclude in training
            n_bins=3,
            min_support=10,  # minimal required patient-level class samples for a class to be considered
            train=Train(
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
        ),
    )


if __name__ == '__main__':
    main()
