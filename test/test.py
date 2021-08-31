import unittest
from tempfile import TemporaryDirectory

from deepmed.evaluators.types import Evaluator
from deepmed.experiment_imports import *


cohorts_df=cohort(
    tiles_path='I:/tcga-brca-testing-tiles/tiles',
    clini_path='I:/tcga-brca-testing-tiles/tcga-brca-test-clini.xlsx',
    slide_path='I:/tcga-brca-testing-tiles/tcga-brca-test-slide.xlsx'),


class TestCategorical(unittest.TestCase):
    def test_simple(self):
        # train a model
        with TemporaryDirectory() as training_dir:
            do_experiment(
                project_dir=training_dir,
                get=partial(
                    get.simple_run,
                    train_cohorts_df=cohorts_df,
                    target_label='ER Status By IHC',
                    max_train_tile_num=4,
                    max_valid_tile_num=4),
                train=partial(train, max_epochs=1),
                logfile=None)

            train_df = pd.read_csv(Path(training_dir)/'training_set.csv.zip')
            counts = train_df['ER Status By IHC'].value_counts()
            self.assertEqual(
                counts['Positive'], counts['Negative'], msg='Training set not balanced!')

            with TemporaryDirectory() as testing_dir:
                # deploy it
                max_test_tile_num = 2
                
                do_experiment(
                    project_dir=testing_dir,
                    get=partial(
                        get.simple_run,
                        test_cohorts_df=cohorts_df,
                        target_label='ER Status By IHC',
                        max_test_tile_num=max_test_tile_num),
                    train=partial(
                        load,
                        project_dir=Path(testing_dir),
                        training_project_dir=Path(training_dir)),
                    logfile=None)

                # add some evaluation
                do_experiment(
                    project_dir=testing_dir,
                    get=partial(
                        get.simple_run,
                        test_cohorts_df=cohorts_df,
                        target_label='ER Status By IHC',
                        evaluators=[Grouped(auroc), count, Grouped(count)]),
                    logfile=None)

                stats_df = pd.read_csv(Path(testing_dir)/'stats.csv', index_col=0, header=[0,1])
                self.assertEqual(stats_df[('count', 'PATIENT')]['Positive'], 76)
                self.assertEqual(stats_df[('count', 'PATIENT')]['Negative'], 24)
                self.assertEqual(stats_df[('count', 'nan')]['Positive'], max_test_tile_num*76)
                self.assertEqual(stats_df[('count', 'nan')]['Negative'], max_test_tile_num*24)


class TestEvaluators(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training_dir = TemporaryDirectory()

        cls.max_train_tile_num = 99

        # train and deploy
        do_experiment(
            project_dir=cls.training_dir.name,
            get=partial(
                get.simple_run,
                train_cohorts_df=cohorts_df,
                test_cohorts_df=cohorts_df,
                target_label='ER Status By IHC',
                max_train_tile_num=cls.max_train_tile_num),
            train=partial(train, max_epochs=4),
            logfile=None)


    @classmethod
    def tearDownClass(cls) -> None:
        cls.training_dir.cleanup()


    def test_auroc(self):
        """Test AUROC Metric."""
        evaluate(self.training_dir.name, [auroc, Grouped(auroc)])
        stats_df = pd.read_csv(Path(self.training_dir.name)/'stats.csv', index_col=0, header=[0,1])

        auroc_ = stats_df[('auroc', 'PATIENT')]['Positive']
        self.assertTrue(auroc_ >= 0 and auroc_ <= 1, msg='AUROC not in [0,1]')
        self.assertAlmostEqual(
            stats_df[('auroc', 'PATIENT')]['Positive'],
            stats_df[('auroc', 'PATIENT')]['Negative'],
            msg='AUROC for binary target not symmetric!')

        self.assertTrue(
            (stats_df[('auroc', 'nan')] < stats_df[('auroc', 'PATIENT')]).all(),
            msg='Aggregating over patients yields no improvement!')
        self.assertGreater(auroc_, .9, msg='Low AUROC for trivial deployment!')


    def test_f1(self):
        evaluate(self.training_dir.name, [f1, Grouped(f1)])
        stats_df = pd.read_csv(Path(self.training_dir.name)/'stats.csv', index_col=0, header=[0,1])
        print(stats_df)
        self.assertTrue(
            (stats_df[('f1 optimal', 'nan')] < stats_df[('f1 optimal', 'PATIENT')]).all(),
            msg='Aggregating over patients yields no improvement!')


    def test_count(self):
        evaluate(self.training_dir.name, [count, Grouped(count)])
        stats_df = pd.read_csv(Path(self.training_dir.name)/'stats.csv', index_col=0, header=[0,1])
        self.assertTrue(
            (stats_df[('count', 'nan')] ==
             stats_df[('count', 'PATIENT')] * self.max_train_tile_num).all(),
            msg='Did not sample the correct number of tiles')


def evaluate(project_dir: Union[str, Path], evaluators: Iterable[Evaluator]):
    do_experiment(
        project_dir=project_dir,
        get=partial(
            get.simple_run,
            test_cohorts_df=cohorts_df,
            target_label='ER Status By IHC',
            evaluators=evaluators),
        logfile=None)


if __name__ == '__main__':
    unittest.main()