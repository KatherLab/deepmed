#!/usr/bin/env python3
from deepmed.experiment_imports import *


if __name__ == '__main__':
    __spec__ = None
    project_dir = 'multi_target_deploy'

    do_experiment(
        project_dir=project_dir,
        get=partial(
            get.multi_target,
            get.simple_run,
            test_cohorts_df=cohort(
                tiles_path='I:/tcga-brca-testing-tiles/tiles',
                clini_path='I:/tcga-brca-testing-tiles/tcga-brca-test-clini.xlsx',
                slide_path='I:/tcga-brca-testing-tiles/tcga-brca-test-slide.xlsx'),
            target_labels=['ER Status By IHC'],
            max_test_tile_num=512,
            evaluators=[Grouped(auroc), Grouped(f1), Grouped(count)],
            multi_target_evaluators=[aggregate_stats]
        ),
        train=partial(
            load,
            project_dir=project_dir,
            training_project_dir='multi_target_train'),
        devices={'cuda:0': 4}
    )
