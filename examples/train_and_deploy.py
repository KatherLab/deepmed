"""Trains and deploys a model."""
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/path/to/example_training_and_deployment_project_dir',
    mode=train_test,
    train_cohorts=[
        Cohort('/media/markovt/KATHER-P01/DACHS-CRC-DX', tile_dir='BLOCKS_NORM_MACENKO')],
    test_cohorts=[
        Cohort('/media/markovt/KATHER-P01/RAINBOW-CRC-DX', tile_dir='BLOCKS_NORM_MACENKO')],
    target_labels=['isMSIH'],
    target_evaluators=[Grouped(f1), Grouped(auroc)],
    max_tile_num=500,   # up to 500 tiles are sampled from each patient
    batch_size=64,
    save_models=False,  # don't save the trained model
    max_epochs=4,       # training will be done for up to four epochs
)