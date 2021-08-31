#!/usr/bin/env python3
from deepmed.experiment_imports import *


if __name__ == '__main__':
    __spec__ = None
    do_experiment(
        project_dir='multi_target_crossval',
        get=partial(
            get.multi_target,
            get.crossval,
            get.simple_run,
            cohorts_df=cohort(
                tiles_path='I:/tcga-brca-testing-tiles/tiles',
                clini_path='I:/tcga-brca-testing-tiles/tcga-brca-test-clini.xlsx',
                slide_path='I:/tcga-brca-testing-tiles/tcga-brca-test-slide.xlsx'),
            target_labels=['ER Status By IHC'],
            max_train_tile_num=128,
            max_valid_tile_num=64,
            max_test_tile_num=256,
            valid_frac=.2,
            multi_target_evaluators=[
                partial(aggregate_stats, group_levels=[0, -1])],
            crossval_evaluators=[aggregate_stats],
            evaluators=[Grouped(auroc), Grouped(count)],
        ),
        train=partial(
            train,
            batch_size=96,
            max_epochs=4),
        devices={'cuda:0': 4})
