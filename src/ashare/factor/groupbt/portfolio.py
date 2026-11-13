"""分组回测：按行分组、组内等权目标、多空目标，以及带交易限制的持仓推进。

面板形状均为 (T, N)，行是交易日，列是股票；NaN 表示缺失。所有函数都不修改
传入的数组，结果确定（排序用稳定排序，同值按列下标从小到大）。
"""

from __future__ import annotations

import numpy as np


class GroupingError(ValueError):
    """某些行的有效股票数或不同取值数不足以分成指定组数。"""

    def __init__(self, rows):
        self.rows = list(rows)
        super().__init__(
            f"{len(self.rows)} 行无法分组，前 10 个行下标：{self.rows[:10]}"
        )


def assign_groups(alpha, n_groups):
    """按行把有效值从大到小稳定排序，组号 = 名次 * n_groups // 有效个数。

    0 组是值最大的一组（top），n_groups - 1 是最小的一组（bottom）；NaN 为 -1。
    整行 NaN 返回整行 -1 且不算错误；其余行若有效个数或不同取值个数不足
    n_groups，则收集行下标后统一抛出 GroupingError。
    """
    values = np.asarray(alpha, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("alpha 必须是二维数组")
    n_groups = int(n_groups)
    if n_groups < 1:
        raise ValueError("n_groups 必须为正整数")

    n_rows, n_cols = values.shape
    out = np.full((n_rows, n_cols), -1, dtype=np.int64)
    bad_rows = []

    for t in range(n_rows):
        valid = ~np.isnan(values[t])
        count = int(valid.sum())
        if count == 0:
            continue
        if count < n_groups:
            bad_rows.append(t)
            continue
        cols = np.flatnonzero(valid)
        order = np.argsort(-values[t][cols], kind="stable")
        sorted_values = values[t, cols[order]]
        if 1 + np.count_nonzero(sorted_values[1:] != sorted_values[:-1]) < n_groups:
            bad_rows.append(t)
            continue
        ranks = np.empty(count, dtype=np.int64)
        ranks[order] = np.arange(count, dtype=np.int64)
        out[t, cols] = ranks * n_groups // count

    if bad_rows:
        raise GroupingError(bad_rows)
    return out


def group_targets(groups, g):
    """每行中组号等于 g 的股票等权，其余为 0；该组为空的轨道全为 0。"""
    grp = np.asarray(groups)
    out = np.zeros(grp.shape, dtype=np.float64)
    mask = grp == g
    counts = mask.sum(axis=1).astype(np.float64)
    has = counts > 0
    out[has] = mask[has] / counts[has, None]
    return out


def long_short_targets(groups, n_groups):
    """多空目标：一半做多 0 组，一半做空 n_groups - 1 组。"""
    return 0.5 * group_targets(groups, 0) - 0.5 * group_targets(
        groups, int(n_groups) - 1
    )


def simulate(target, fwd_ret, can_buy=None, can_sell=None, *, keep_weights=True):
    """兼容旧内部入口；组合模拟统一由 panelbt 的纯权重模式执行。"""
    from panelbt import run_weights

    result = run_weights(target, fwd_ret, can_buy, can_sell, keep_weights=keep_weights)
    return {"weights": result.weights, "pnl": result.pnl, "turnover": result.turnover}
