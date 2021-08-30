from typing import Iterable, Tuple
from pathlib import Path

import pandas as pd

import pandas as pd
from pathlib import Path
import scipy.stats as st


def aggregate_stats(
        _target_label, _preds_df, result_dir: Path, group_levels: Iterable[int] = [], /
        ) -> pd.DataFrame:
    """Accumulates stats from subdirectories.

    By default, this function simply concatenates the contents of all the
    ``stats.csv`` files in ``result_dir``'s immediate subdirectories.  Each of
    the subdirectories' names will be added as to the index at its top level.

    This function may also aggregate over metrics: if the ``group_levels``
    option is given, the stats will be grouped by the specified index levels.
    """
    # collect all parent stats dfs
    dfs = []
    stats_df_paths = list(result_dir.glob('*/stats.csv'))
    for df_path in stats_df_paths:
        header, index_col = _get_header_and_index_col(df_path)
        dfs.append(pd.read_csv(df_path, header=header, index_col=index_col))
    assert dfs, 'Could not find any stats.csvs to aggregate!'
    stats_df = pd.concat(dfs, keys=[path.parent.name for path in stats_df_paths])

    if group_levels:
        # sum all labels which have 'count' in their topmost column level; calculate means,
        # confidence intervals for the rest
        count_labels = [col for col in stats_df.columns
                        if 'count' in (col[0] if isinstance(col, tuple) else col)]
        metric_labels = [
            col for col in stats_df.columns if col not in count_labels]

        # calculate count sums
        grouped = stats_df[count_labels].groupby(level=group_levels)
        counts = grouped.sum(min_count=1)

        # calculate means, confidence interval bounds
        grouped = stats_df[metric_labels].groupby(level=group_levels)
        means, ns, sems = grouped.mean(), grouped.count(), grouped.sem()
        l, h = st.t.interval(alpha=.95, df=ns-1, loc=means, scale=sems)
        confs = pd.DataFrame(
            (h - l) / 2, index=means.index, columns=means.columns)

        # for some reason concat doesn't like it if one of the dfs is empty and we supply a key
        # nonetheless... so only generate the headers if needed
        keys = (([] if means.empty else ['mean', '95% conf']) +
                ([] if counts.empty else ['total']))
        stats_df = pd.concat([means, confs, counts], keys=keys, axis=1)

        # make mean, conf, total the lowest of the column levels
        stats_df = pd.DataFrame(
            stats_df.values, index=stats_df.index,
            columns=stats_df.columns.reorder_levels([*range(1, stats_df.columns.nlevels), 0]))

        # sort by every but the last (mean, 95%) columns so we get a nice hierarchical order
        stats_df = stats_df[sorted(stats_df.columns,
                                   key=lambda x: x[:stats_df.columns.nlevels-1])]

    return stats_df


def _get_header_and_index_col(csv_path: Path) -> Tuple(Iterable[int], Iterable[int]):
    """Gets the number of header rows and index columns."""
    # FIXME bad, bad evil hack
    # assumes that the first header row contains as many empty fields as there are index columns and
    # that each header row starts with a ','.  For the table
    #
    #     ,,,auroc,f1
    #     ,,,PATIENT,nan
    #     isMSIH,fold_0,MSIH,.7,.4
    #
    # this function would return ([0,1], [0,1,2])
    with open(csv_path) as f:
        index_no = f.readline().split(',').count('')
        header_no = next(i for i, line in enumerate(f) if line[0] != ',') + 1

    return (list(range(header_no)), list(range(index_no)))