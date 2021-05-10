#!/usr/bin/env python3
from deepest_histology.experiment_imports import *

do_experiment(
    project_dir='/media/markovt/KATHER_P03/BRCA_DX_PROJECT_Marko',
    mode=crossval,
    target_labels=['ERStatus', 'PRStatus'],
    fold_evaluators=[accuracy, auroc, Grouped(auroc)],
    target_evaluators=[Grouped(roc)],
    save_models=True,
    batch_size=64,
    max_epochs=4,
    max_tile_num=500)