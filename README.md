# Welcome to Deepest Histology

## What is this?

This is an open source platform for end-to-end artificial intelligence (AI) in computational pathology. It will enable you to use AI for prediction of any "label" directly from digitized pathology slides. Common use cases which can be reproduced by this pipeline are:
- prediction of microsatellite instability in colorectal cancer (Kather et al., Nat Med 2019)
- prediction of mutations in lung cancer (Coudray et al., Nat Med 2018)
- prediction of subtypes of renal cell carcinoma (Lu et al., Nat Biomed Eng 2021)
- other possible use cases are summarized by Echle et al., Br J Cancer 2021: https://www.nature.com/articles/s41416-020-01122-x

By default, the user of this pipeline can choose between different AI algorithms, while the pre/post-processing is unchanged:

- vanilla deep learning workflow (Coudray et al., Nat Med 2018)
- vanilla Multiple instance learning (Campanella et al., Nat Med 2019)
- CLAM (Lu et al., Nat Biomed Eng 2021)

This pipeline is modular, which means that new methods for pre-/postprocessing or new AI methods can be easily integrated. 

# Prerequisites

We use Deepest Histology on a local workstation server with Ubuntu 20.04 or Windows Server 2019 and a CUDA-enabled NVIDIA GPU. The following packages are required

[MARKO PLEASE ADD REQUIREMENTS]


# How to use

## Global settings

| Option        | Type | Default | Description                   |
|---------------|------|---------|-------------------------------|
| project_dir   | Path |         | Directory to save results to. |
| seed          | int  | 0       | Seed for random processes.    |


## Simple Training and Testing

### Run Generation

| Option        | Type             | Default | Description                     |
|---------------|------------------|---------|---------------------------------|
| target_labels | Iterable[str]    |         | A list of targets to train for. |
| train_cohorts | Iterable[Cohort] | []      | Cohorts to use for training.    |
| test_cohorts  | Iterable[Cohort] | []      | Cohorts to use for testing.     |
| max_tile_num  | int              | 500     | Tiles per patient to use.       |
| valid_frac    | float            | .1      | Amount of the training set to use for validation. |
| n_bins        | int              | 2       | Number of bins to discretize continuous values into. |
| na_values     | Iterable[str]    | []      | Values to exclude in the train / test sets. |

### Training

| Option     | Type                 | Default | Description                |
|------------|----------------------|---------|----------------------------|
| max_epochs | int                  | 10      | Max number of epochs to train for. (May be influenced by early stopping) |
| lr         | float                | 2e-3    | The initial learning rate. |
| batch_size | int                  | 64      | The training batch size. |
| patience   | int                  | 3       | Number of epochs to wait for learning rate improvement. |
| device     | torch.cuda._device_t | None    | The device to train on, or `None` for the default device. |

### Deployment

| Option     | Type                 | Default | Description                |
|------------|----------------------|---------|----------------------------|
| batch_size | int                  | 64      | The deployment batch size. |
| device     | torch.cuda._device_t | None    | The device to deploy on, or `None` for the default device. |

### Evaluation

| Option            | Type                | Default | Description |
|-------------------|---------------------|---------|-------------|
| target_evaluators | Iterable[Evaluator] | []      | Evaluators to be applied to each target. |


## Cross-Validation

### Run Generation

| Option        | Type             | Default | Description               |
|---------------|------------------|---------|---------------------------|
| target_labels | Iterable[str]    |         | A list of targets to train for. |
| cohorts       | Iterable[Cohort] | []      | Cohorts to use for cross-validation. |
| max_tile_num  | int              | 500     | Tiles per patient to use. |
| valid_frac    | float            | .1      | Amount of the training set to use for validation. |
| n_bins        | int              | 2       | Number of bins to discretize continuous values into. |
| na_values     | Iterable[str]    | []      | Values to exclude in the train / test sets. |

### Training

As for simple training.

### Evaluation

| Option            | Type                | Default | Description |
|-------------------|---------------------|---------|-------------|
| fold_evaluators   | Iterable[Evaluator] | []      | Evaluators to be applied to each fold. For numeric evaluators, the means and 95% confidence intervals will also be calculated. |
| target_evaluators | Iterable[Evaluator] | []      | Evaluators to be applied to each target. |

## Types

### Cohorts

```python
Cohort(
    '/path/to/cohort/',
    tile_dir='NAME_OF_TILE_DIR',
    slide_table='/path/to/slide/table', # optional
    clini_table='/path/to/clini/table', # optional
)
```
