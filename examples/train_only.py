"""Trains a single model on one or multiple cohorts."""
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/path/to/example_training_project_dir',
    mode=train_only,
    train_cohorts=[
        Cohort('/media/markovt/KATHER-P01/DACHS-CRC-DX',    # the cohort folder
               tile_dir='BLOCKS_NORM_MACENKO',  # the subdir the blocks are stored in
               # the name or path of the slide table (optional)
               slide_table='CUSTOM_SLIDE_TABLE.csv',
               # the name of or path the clini table (optional)
               clini_table='/path/to/CUSTOM_CLINI_TABLE.xlsx'),
        # a second cohort
        Cohort('/media/markovt/KATHER-P01/RAINBOW-CRC-DX', tile_dir='BLOCKS')],
    target_labels=['isMSIH', 'gender'],   # the targets to train for
    # all the options below are optional
    valid_frac=.15,     # 15% of patients are used for training validation
    max_tile_num=500,   # up to 500 tiles are sampled from each patient
    batch_size=64,
    max_epochs=4,       # training will be done for up to four epochs
)