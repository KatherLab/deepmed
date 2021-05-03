#!/usr/bin/env python3
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/media/markovt/KATHER_P03/markos_test_project',
    get_runs=crossval.create_experiments,
    #train=basic.train,
    #deploy=basic.deploy,
    #evaluate=crossval.evaluate,
    targets=['isMSIH'],
    fold_evaluators=[auroc, SubGrouped(Grouped(auroc), by='GENDER')],
    target_evaluators=[Grouped(roc)],
    cohorts=[
        Cohort(root_dir='/media/markovt/KATHER-P01/DACHS-CRC-DX',
               tile_dir='BLOCKS_NORM_MACENKO',
               clini_table='CLINI_SMALL.csv')],
    batch_size=64,
    max_epochs=1,
    max_tile_num=10)
