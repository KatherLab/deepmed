#!/usr/bin/env python3
from deepest_histology.experiment_imports import *


if __name__ == '__main__':
    __spec__ = None
    do_experiment(
        project_dir=r'K:\Marko\BRCA_DX_TCGA_crossval',
        get=partial(
            get.multi_target,
            get.crossval,
            get.simple_run,
            cohorts_df=cohort(
                tiles_path='D:/TCGA-BRCA-DX/BLOCKS_NORM_MACENKO',
                clini_path='D:/BRCA_Docs/TCGA-BRCA-DX_CLINI.xlsx',
                slide_path='D:/BRCA_Docs/TCGA-BRCA-DX_SLIDE_FULLNAMES.csv'),
            target_labels=['ERStatus', 'PRStatus', 'HER2FinalStatus'],
            max_tile_num=8,
            valid_frac=.2,
            na_values=['Not Available', 'Equivocal', 'Not Performed', 'Performed but Not Available'],
            evaluator_groups=[partial(aggregate_stats, group_levels=[-3, -1])],
            crossval_evaluators=[aggregate_stats],
            evaluators=[Grouped(auroc), Grouped(count)],
            ),
        train=partial(
            train,
            batch_size=96,
            max_epochs=1),
        num_concurrent_runs=4)
