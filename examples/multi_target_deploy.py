#!/usr/bin/env python3
from deepest_histology.experiment_imports import *


if __name__ == '__main__':
    __spec__ = None
    project_dir = r'K:\Marko\BRCA_DX_TCGA_deploy'
    do_experiment(
        project_dir=project_dir,
        get=partial(
            get.multi_target,
            get.simple_run,
            test_cohorts_df=cohort(
                tiles_path='D:/TCGA-BRCA-DX/BLOCKS_NORM_MACENKO',
                clini_path='D:/BRCA_Docs/TCGA-BRCA-DX_CLINI.xlsx',
                slide_path='D:/BRCA_Docs/TCGA-BRCA-DX_SLIDE_FULLNAMES.csv'),
            target_labels=['PRStatus', 'HER2FinalStatus'],
            max_tile_num=10,
            valid_frac=.2,
            na_values=['Not Available', 'Equivocal', 'Not Performed', 'Performed but Not Available']
        ),
        train=partial(
            load,
            project_dir=project_dir,
            training_project_dir=r'K:\Marko\debug\BRCA_DX_TCGA_full_training_refactor_test'),
        evaluator_groups=[[aggregate_stats], [Grouped(auroc)]],
        num_concurrent_runs=4)
