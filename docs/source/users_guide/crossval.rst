Cross-Validation
================

::

    from deepest_histology.experiment_imports import *

    train_cohorts_df = pd.concat([
        cohort(tile_path='E:/TCGA-BRCA-DX/BLOCKS_NORM',
               clini_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA-IMMUNO_CLINI.xlsx',
               slide_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA_SLIDE.csv'),
        cohort(tile_path='E:/TCGA-BRCA-DX/BLOCKS_NORM',
               clini_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA-IMMUNO_CLINI.xlsx',
               slide_path='G:/immunoproject/TCGA-IMMUNO_Clini_Slide/TCGA-BRCA_SLIDE.csv')])

    crossval_get = partial(
        get.crossval,
        get.simple_run,
        cohorts_df=cohorts_df,
        target_label='isMSIH',
        max_tile_num=100,
        na_values=['inconclusive'],
        folds=3)