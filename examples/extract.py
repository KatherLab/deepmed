#!/usr/bin/env python3
from deepmed.experiment_imports import *
from deepmed.get._extract_features import Extract


def main():
    do_experiment(
        project_dir='/feature/output/dir',
        get=Extract(
            tile_dir='/tile/dir',
            arch=resnet18,
        ),
    )


if __name__ == '__main__':
    main()
