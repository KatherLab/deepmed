"""Does crossvalidation."""
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/path/to/example_crossval_project_dir',
    mode=crossval,
    cohorts=[
        Cohort('/media/markovt/KATHER-P01/DACHS-CRC-DX', tile_dir='BLOCKS_NORM_MACENKO'),
        Cohort('/media/markovt/KATHER-P01/RAINBOW-CRC-DX', tile_dir='BLOCKS_NORM_MACENKO')],
    target_labels=['isMSIH'],
    # metrics to be calculated for each fold; will also calculate average / confidence intervals
    fold_evaluators=[Grouped(f1), Grouped(auroc)],
    target_evaluators=[Grouped(roc)],
    max_tile_num=500,
    batch_size=64,
    max_epochs=4,
)