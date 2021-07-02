"""Does crossvalidation."""
from deepest_histology.experiment_imports import *

if __name__ == '__main__':  # required on windows
    do_experiment(
        project_dir='/path/to/example_crossval_project_dir',
        mode=crossval,
        cohorts=[
            # Don't forget the `r` in Windows-style paths!
            Cohort(
                tile_path=r'G:\DACHS-CRC-DX\BLOCKS_NORM_MACENKO',
                clini_path=r'G:\path\to\clini.csv', slide_path=r'G:\path\to\slide.csv'),
            Cohort(
                tile_path=r'G:\RAINBOW-CRC-DX\BLOCKS_NORM',
                clini_path=r'G:\path\to\clini.csv', slide_path=r'G:\path\to\slide.csv')],
        target_labels=['isMSIH', 'gender'], # labels to train for
        evaluator_groups=[
            # global aggregation over targets: calculate the means / confidence intervals for each fold
            [partial(aggregate_stats, group_levels=[0, -1])],
            # ROC curves for each fold, collect below statistics in a new file
            [aggregate_stats, Grouped(roc)],
            # fold-wise, patient AUROC, (optimal) F1 score, patient count, as well as tile count
            [Grouped(auroc), Grouped(f1), Grouped(count), count]],
        folds=3,    # number of folds to train
        max_tile_num=250,   # the number of tiles per patient
        batch_size=64,
        max_epochs=16,
        patience=3, # how long do we want to wait before stopping?
        na_values=['N/A', 'unknown'],   # class labels to ignore
        num_concurrent_runs=6,  # train up to six models concurrently
        devices=['cuda:0', 'cuda:1']    # use two gpus
    )