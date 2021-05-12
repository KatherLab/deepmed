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
| na_values     | Iterable[str]    | []      | Values to exclude in the train / test sets. |

### Training

| Option     | Type                 | Default | Description                |
|------------|----------------------|---------|----------------------------|
| max_epochs | int                  | 10      | Max number of epochs to train for. (May be influenced by early stopping |
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