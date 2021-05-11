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
| valid_frac    | float            | .1      | Amount of the training set to use for validation |

### Training

| Option     | Type  | Default | Description                |
|------------|-------|---------|----------------------------|
| max_epochs | int   | 10      | Max number of epochs to train for. (May be influenced by early stopping |
| lr         | float | 1e-4    | The initial learning rate. |

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