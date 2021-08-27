from tempfile import TemporaryDirectory
import unittest
from deepmed.experiment_imports import *

class SimpleTrainingAndDeployment(unittest.TestCase):
    def test_categorical(self):
        # train a model
        with TemporaryDirectory() as training_dir:
            do_experiment(
                project_dir=training_dir,
                get=partial(
                    get.simple_run,
                    train_cohorts_df=cohort(
                        tiles_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-TILES/BLOCKS_NORM_MACENKO',
                        clini_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-CLINI.xlsx',
                        slide_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-SLIDE.xlsx'),
                    target_label='ER Status By IHC',
                    max_tile_num=4),
                train=partial(train, max_epochs=1),
                devices={'cuda:0': 1},
                logfile=None)

            train_df = pd.read_csv(Path(training_dir)/'training_set.csv.zip')
            counts = train_df['ER Status By IHC'].value_counts()
            self.assertEqual(
                counts['Positive'], counts['Negative'], msg='Training set not balanced!')

            with TemporaryDirectory() as testing_dir:
                # deploy it
                do_experiment(
                    project_dir=testing_dir,
                    get=partial(
                        get.simple_run,
                        test_cohorts_df=cohort(
                            tiles_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-TILES/BLOCKS_NORM_MACENKO',
                            clini_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-CLINI.xlsx',
                            slide_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-SLIDE.xlsx'),
                        target_label='ER Status By IHC',
                        max_tile_num=4),
                    train=partial(
                        load,
                        project_dir=Path(testing_dir),
                        training_project_dir=Path(training_dir)),
                    devices={'cuda:0': 1},
                    logfile=None)

                # add some evaluation
                do_experiment(
                    project_dir=testing_dir,
                    get=partial(
                        get.simple_run,
                        test_cohorts_df=cohort(
                            tiles_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-TILES/BLOCKS_NORM_MACENKO',
                            clini_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-CLINI.xlsx',
                            slide_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-SLIDE.xlsx'),
                        target_label='ER Status By IHC',#,'TCGA Subtype','TMB (nonsynonymous)'],
                        max_tile_num=4,
                        evaluators=[Grouped(auroc), count, Grouped(count)]),
                    devices={'cuda:0': 1},
                    logfile=None)

                stats_df = pd.read_csv(Path(testing_dir)/'stats.csv', index_col=0, header=[0,1])
                self.assertEqual(stats_df[('count', 'PATIENT')]['Positive'], 76)
                self.assertEqual(stats_df[('count', 'PATIENT')]['Negative'], 24)
                self.assertAlmostEqual(stats_df[('count', 'nan')]['Positive'], 76*4)
                self.assertAlmostEqual(stats_df[('count', 'nan')]['Negative'], 24*4)
                auroc_ = stats_df[('auroc', 'PATIENT')]['Positive']
                self.assertTrue(
                    auroc_ >= 0 and auroc_ <= 1,
                    msg='AUROC not in [0,1]')
                self.assertAlmostEqual(
                    stats_df[('auroc', 'PATIENT')]['Positive'],
                    stats_df[('auroc', 'PATIENT')]['Negative'],
                    msg='AUROC for binary target not symmetric!')

    def test_perfect_training(self):
        """Train and deploy on the same data for very good results."""
        with TemporaryDirectory() as training_dir:
            cohorts_df=cohort(
                tiles_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-TILES/BLOCKS_NORM_MACENKO',
                clini_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-CLINI.xlsx',
                slide_path='I:/TCGA-BRCA-BENCHMARK-DEEPMED-SLIDE.xlsx')
            do_experiment(
                project_dir=training_dir,
                get=partial(
                    get.simple_run,
                    train_cohorts_df=cohorts_df,
                    test_cohorts_df=cohorts_df,
                    target_label='ER Status By IHC',
                    evaluators=[auroc, Grouped(auroc)]),
                train=partial(train, max_epochs=1),
                devices={'cuda:0': 1},
                logfile=None)
            stats_df = pd.read_csv(Path(training_dir)/'stats.csv', index_col=0, header=[0,1])
            self.assertTrue(
                (stats_df[('auroc', 'nan')] < stats_df[('auroc', 'PATIENT')]).all(),
                msg='Aggregating over patients yields no improvement!')
            self.assertTrue(
                (stats_df[('auroc', 'PATIENT')] > .9).all(),
                msg='Low AUROC for trivial deployment!')

if __name__ == '__main__':
    unittest.main()