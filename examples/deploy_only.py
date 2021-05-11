"""Deploy a previously trained model on one or multiple testing cohorts."""
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/path/to/example_deployment_project_dir',
    mode=deploy_only,
    test_cohorts=[  # data to use for testing
        Cohort('/media/markovt/KATHER-P01/DACHS-CRC-DX',
        tile_dir='BLOCKS_NORM_MACENKO')],
    target_labels=['isMSIH'],
    model_path='/path/to/model.pt', # model to use for evaluation
    target_evaluators=[ # evaluations to do
        f1, auroc,  # tile-wise
        Grouped(f1), Grouped(auroc),    # grouped by patient
        Grouped(auroc, by='SLIDE'),     # grouped by slide
        SubGrouped(auroc, by='gender'), # evaluate seperately by subgroup
        # evaluate seperately by subgroup, with each subgroup grouped by patient
        SubGrouped(Grouped(auroc), by='gender')],
    batch_size=64,
    max_tile_num=500)