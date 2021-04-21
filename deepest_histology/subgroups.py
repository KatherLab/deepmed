from pathlib import Path

from .experiment import Evaluator, EvalDF

def subgroup_evaluator(base_evaluator: Evaluator, *, subgroup_label: str) -> Evaluator:
    """Takes an evaluator and applies it to each subgroup."""

    def evaluate(eval_df: EvalDF, result_dir: Path, **kwargs) -> None:
        for group, group_df in eval_df.groupby(subgroup_label):
            base_evaluator(eval_df=group_df, result_dir=result_dir/group, **kwargs)

    return evaluate