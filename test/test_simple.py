import unittest
from tempfile import TemporaryDirectory
from itertools import product

from deepmed.evaluators.types import Evaluator
from deepmed.experiment_imports import *


class TestSeperateTrainAndDeploy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = untar_data(
            'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')
        cls.cohorts_df = cohort(
            tiles_path=path/'tiles',
            clini_path=path/'clini.csv',
            slide_path=path/'slide.csv')

    def test_simple(self):
        # train a model
        with TemporaryDirectory() as training_dir:
            do_experiment(
                project_dir=training_dir,
                get=get.SimpleRun(
                    train_cohorts_df=self.cohorts_df,
                    target_label='ER Status By IHC',
                    max_train_tile_num=4,
                    max_valid_tile_num=4,
                    train=Train(max_epochs=1)),
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
                    get=get.SimpleRun(
                        test_cohorts_df=self.cohorts_df,
                        target_label='ER Status By IHC',
                        max_test_tile_num=max_test_tile_num,
                        train=Load(
                            project_dir=Path(testing_dir),
                            training_project_dir=Path(training_dir))),
                    logfile=None)

                # add some evaluation
                do_experiment(
                    project_dir=testing_dir,
                    get=get.SimpleRun(
                        test_cohorts_df=self.cohorts_df,
                        target_label='ER Status By IHC',
                        evaluators=[Grouped(auroc), count, Grouped(count)]),
                    logfile=None)

                stats_df = pd.read_pickle(
                    Path(testing_dir)/'stats.pkl')
                self.assertEqual(
                    stats_df[('count', 'PATIENT')]['Positive'], 76)
                self.assertEqual(
                    stats_df[('count', 'PATIENT')]['Negative'], 24)
                self.assertEqual(
                    stats_df[('count', 'nan')]['Positive'], max_test_tile_num*76)
                self.assertEqual(
                    stats_df[('count', 'nan')]['Negative'], max_test_tile_num*24)


class TestDiscretization(unittest.TestCase):
    def test_class(self):
        path = untar_data(
            'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')
        cohorts_df = cohort(
            tiles_path=path/'tiles',
            clini_path=path/'clini.csv',
            slide_path=path/'slide.csv')

        # train a model
        with TemporaryDirectory() as project_dir:
            do_experiment(
                project_dir=project_dir,
                get=get.SimpleRun(
                    train_cohorts_df=cohorts_df,
                    test_cohorts_df=cohorts_df,
                    target_label='TMB (nonsynonymous)',
                    evaluators=[Grouped(count)],
                    n_bins=3,
                    train=Train(max_epochs=1)),
                logfile=None)

            stats_df = pd.read_pickle(Path(project_dir)/'stats.pkl')
            self.assertEqual(stats_df[('count', 'PATIENT')]['[-inf,0.7)'], 31)
            self.assertEqual(stats_df[('count', 'PATIENT')
                                      ]['[0.7,1.43333333333)'], 34)
            self.assertEqual(stats_df[('count', 'PATIENT')
                                      ]['[1.43333333333,inf)'], 34)


class TestEvaluators(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = untar_data(
            'https://katherlab-datasets.s3.eu-central-1.amazonaws.com/tiny-test-data.zip')
        cls.cohorts_df = cohort(
            tiles_path=path/'tiles',
            clini_path=path/'clini.csv',
            slide_path=path/'slide.csv')

        cls.training_dir = TemporaryDirectory()
        cls.max_train_tile_num = 4

        # train and deploy
        do_experiment(
            project_dir=cls.training_dir.name,
            get=get.SimpleRun(
                train_cohorts_df=cls.cohorts_df,
                test_cohorts_df=cls.cohorts_df,
                target_label='ER Status By IHC',
                max_train_tile_num=cls.max_train_tile_num,
                train=Train(max_epochs=4)),
            logfile=None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.training_dir.cleanup()

    def test_auroc(self):
        """Test AUROC Metric."""
        evaluate(self.training_dir.name, self.cohorts_df,
                 [auroc, Grouped(auroc)])
        stats_df = pd.read_pickle(
            Path(self.training_dir.name)/'stats.pkl')

        auroc_ = stats_df[('auroc', 'PATIENT')]['Positive']
        self.assertTrue(auroc_ >= 0 and auroc_ <= 1, msg='AUROC not in [0,1]')
        self.assertAlmostEqual(
            stats_df[('auroc', 'PATIENT')]['Positive'],
            stats_df[('auroc', 'PATIENT')]['Negative'],
            msg='AUROC for binary target not symmetric!')

    def test_f1(self):
        evaluate(self.training_dir.name,
                 self.cohorts_df, [F1(), Grouped(F1())])
        stats_df = pd.read_pickle(
            Path(self.training_dir.name)/'stats.pkl')
        self.assertIn(('f1 optimal', 'nan'), stats_df.columns)
        self.assertIn(('f1 optimal', 'PATIENT'), stats_df.columns)

    def test_count(self):
        evaluate(self.training_dir.name, self.cohorts_df,
                 [count, Grouped(count)])
        stats_df = pd.read_picklel(
            Path(self.training_dir.name)/'stats.pkl')
        self.assertTrue(
            (stats_df[('count', 'nan')] ==
             stats_df[('count', 'PATIENT')] * self.max_train_tile_num).all(),
            msg='Did not sample the correct number of tiles')

    def test_top_tiles(self):
        n_patients, n_tiles = 6, 3
        evaluate(self.training_dir.name, self.cohorts_df, [
                 TopTiles(n_patients=n_patients, n_tiles=n_tiles)])
        for class_ in ['Positive', 'Negative']:
            self.assertTrue(
                (Path(self.training_dir.name) /
                    f'ER Status By IHC_{class_}_best-{n_patients}-patients_best-{n_tiles}-tiles.svg').exists())
            df = pd.read_csv(
                Path(self.training_dir.name) /
                f'ER Status By IHC_{class_}_best-{n_patients}-patients_best-{n_tiles}-tiles.csv')
            self.assertEqual(df.PATIENT.nunique(), n_patients)
            self.assertTrue(
                (df.groupby('PATIENT').tile_path.count() == n_tiles).all())


def evaluate(project_dir: Union[str, Path], cohorts_df: pd.DataFrame, evaluators: Iterable[Evaluator]):
    do_experiment(
        project_dir=project_dir,
        get=get.SimpleRun(
            test_cohorts_df=cohorts_df,
            target_label='ER Status By IHC',
            evaluators=evaluators),
        logfile=None)


if __name__ == '__main__':
    unittest.main()
