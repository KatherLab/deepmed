from typing import Iterable, Tuple, Optional, Union, Sequence
import logging
import re
from pathlib import Path

import pandas as pd

import pandas as pd
from pathlib import Path
import scipy.stats as st

from ..utils import factory


def _aggregate_stats(
        _target_label, _preds_df, path: Path, /, label: Optional[str] = None,
        over: Optional[Iterable[Union[str, int]]] = None, conf: float = .95) -> pd.DataFrame:
    """Accumulates stats from subdirectories.

    Args:
        over:  Index columns to aggregate over.
        conf:  The confindence interval calculated during aggregation.

    Returns:
        An aggregation of the subdirectories' statistics.

    By default, this function simply concatenates the contents of all the
    ``stats.pkl`` files in ``path``'s immediate subdirectories.  Each of
    the subdirectories' names will be added as to the index at its top level.

    The ``over`` argument can be used to aggregate over certain index columns of
    the resulting concatenated dataframe; let's assume the concatenated
    dataframe looks like this:

    #TODO update!!!!! ((
    ======  ======  =======  =======  ===
    Metric                   auroc    f1
    ------  ------  -------  -------  ---
    Group                    Patient  nan
    ------  ------  -------  -------  ---
    target  fold    class
    ======  ======  =======  =======  ===
    isMSIH  fold_0  MSIH     0.7      0.4
    isMSIH  fold_0  nonMSIH  0.6      0.1
    isMSIH  fold_1  MSIH     0.8      0.2
    isMSIH  fold_1  nonMSIH  0.2      0.4
    ======  ======  =======  =======  ===

    Then ``aggregate_stats(over=['fold'])`` would calculate the means and
    confidence intervals for all (target, class) pairs, using the different
    folds as samples.  Alternatively, numerical indices can be given (c.f.
    :func:`pandas.DataFrame.groupby` :obj:`level`).
    """
    # collect all parent stats dfs
    dfs = []
    stats_df_paths = list(path.glob('*/stats.pkl'))
    for df_path in stats_df_paths:
        dfs.append(pd.read_pickle(df_path))

    assert dfs, 'could not find any stats.pkls to aggregate!  ' \
        'Did you accidentally use AggregateStats on the bottommost evaluator level?'
    assert all(df.index.names == dfs[0].index.names for df in dfs[1:]), \
        'index labels differ between stats.pkls to aggregate over!'
    stats_df = pd.concat(
        dfs,
        keys=[path.parent.name for path in stats_df_paths],
        names=[label] + dfs[0].index.names)

    if over is not None:
        level = _get_groupby_levels(stats_df, over)
        # sum all labels which have 'count' in their topmost column level; calculate means,
        # confidence intervals for the rest
        count_labels = [col for col in stats_df.columns
                        if 'count' in (col[0] if isinstance(col, tuple) else col)]
        extreme_labels = [col for col in stats_df.columns
                          if (col[0] if isinstance(col, tuple) else col) == 'p value']  #TODO make configurable
        metric_labels = list(set(stats_df.columns)
                             - set(count_labels))

        # calculate count sums
        try:
            grouped = stats_df.groupby(level=level)
        except IndexError as e:
            logging.getLogger(str(path)).critical(
                'Invalid group levels in aggregate_stats!  '
                'Did you use it in the right evaluator group?'
            )
            raise e
        counts = grouped[count_labels].sum(min_count=1)

        maxs = grouped[extreme_labels].max()
        mins = grouped[extreme_labels].min()

        # calculate means, confidence interval bounds
        grouped = stats_df[metric_labels].groupby(level=level)
        means, ns, sems = grouped.mean(), grouped.count(), grouped.sem()
        l, h = st.t.interval(alpha=conf, df=ns-1, loc=means, scale=sems)
        confs = pd.DataFrame(
            (h - l) / 2, index=means.index, columns=means.columns)

        # for some reason concat doesn't like it if one of the dfs is empty and we supply a key
        # nonetheless... so only generate the headers if needed
        keys = (([] if means.empty else ['mean', '95% conf'])
                + ([] if counts.empty else ['total'])
                + ([] if maxs.empty else ['max'])
                + ([] if mins.empty else ['min']))
        stats_df = pd.concat(
            [means, confs, counts, maxs, mins], keys=keys, axis=1)

        # make mean, conf, total the lowest of the column levels
        stats_df = pd.DataFrame(
            stats_df.values, index=stats_df.index,
            columns=stats_df.columns.reorder_levels([*range(1, stats_df.columns.nlevels), 0]))

        # sort by every but the last (mean, 95%) columns so we get a nice hierarchical order
        stats_df = stats_df[sorted(stats_df.columns,
                                   key=lambda x: x[:stats_df.columns.nlevels-1])]

    return stats_df


def _get_groupby_levels(df: pd.DataFrame, over: Iterable[Union[str, int]]) -> Sequence[int]:
    """Returns numeric levels to give to pd.DataFrame.group.

    If we have the index ``df.index.names=['target', 'subgroup', 'fold',
    'class']`` and ``over=[1, 'fold']``, then this function will return [0, 3],
    i.e. the indices which are *not* index 1 and the index with the name
    ``fold``.
    """
    assert not isinstance(over, str), f'`over` has to be a list of labels.  Try `over=[{over}]`.'

    # check if any of the labels appears zero / more than one time
    assert (label := next((label
                           for label in over
                           if isinstance(label, str) and df.index.names.count(label) != 1),
                          None)) is None, \
        f'{label!r} appears {df.index.names.count(label)} times in stats.pkl! ' \
        'Use index numbers to disambiguate!'

    return [
        i for i, name in enumerate(df.index.names)
        if name not in over and i not in over]


AggregateStats = factory(_aggregate_stats)
