import unittest
import subprocess
import sys
import os
from tempfile import TemporaryDirectory
from pathlib import Path


class TestExamples(unittest.TestCase):
    def test_examples(self):
        cwd = Path.cwd()
        env = os.environ.copy()
        env['PYTHONPATH'] = str(cwd)

        examples = [
            'multi_target_train.py',
            'multi_target_deploy.py',
            'crossval.py',
            'subgroup.py',
            'parameterize.py',
            'continuous.py'
        ]
        example_path = Path('examples').absolute()

        with TemporaryDirectory(prefix='deepmed-example-test-') as project_dir:
            for example in examples:
                with self.subTest(example=example):
                    example = example_path/example

                try:
                    os.chdir(project_dir)
                    subprocess.run([sys.executable, example],
                                   env=env, check=True)
                finally:
                    os.chdir(cwd)
