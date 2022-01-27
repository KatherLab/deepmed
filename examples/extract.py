#!/usr/bin/env python3
from deepmed.experiment_imports import *
from deepmed.get._extract_features import Extract
from deepmed.get._extract_features import PretrainedModel


def main():
    do_experiment(
        project_dir='.',
        get=Extract(
            feat_dir='/feature/output/dir',
            tile_dir='/tile/dir',
            arch=PretrainedModel('https://katherlab.s3.eu-central-1.amazonaws.com/dachs-moco-v2-best-XXXXXXXX.pt'),
        ),
    )


if __name__ == '__main__':
    main()
