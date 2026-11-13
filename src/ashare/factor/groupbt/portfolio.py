"""分组回测：按行分组、组内等权目标、多空目标，以及带交易限制的持仓推进。

面板形状均为 (T, N)，行是交易日，列是股票；NaN 表示缺失。所有函数都不修改
传入的数组，结果确定（排序用稳定排序，同值按列下标从小到大）。
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-12


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
        if count < n_groups or np.unique(values[t][valid]).size < n_groups:
            bad_rows.append(t)
            continue
        cols = np.flatnonzero(valid)
        order = np.argsort(-values[t][cols], kind="stable")
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


def _advance(target, prev, buy, sell):
    """推进一步：受限股票冻结在上一行权重，其余按剩余额度分多空两边缩放。"""
    budget_long = float(target[target > 0].sum())
    budget_short = float(-target[target < 0].sum())
    frozen = np.zeros(target.shape, dtype=bool)

    while True:
        free = ~frozen
        used_long = float(prev[frozen & (prev > 0)].sum())
        used_short = float(-prev[frozen & (prev < 0)].sum())
        free_long = float(target[free & (target > 0)].sum())
        free_short = float(-target[free & (target < 0)].sum())
        scale_long = (
            max(budget_long - used_long, 0.0) / free_long if free_long > 0 else 0.0
        )
        scale_short = (
            max(budget_short - used_short, 0.0) / free_short if free_short > 0 else 0.0
        )

        cand = np.where(frozen, prev, 0.0)
        up = free & (target > 0)
        down = free & (target < 0)
        cand[up] = target[up] * scale_long
        cand[down] = target[down] * scale_short

        violate = free & (
            ((cand > prev + _TOL) & ~buy) | ((cand < prev - _TOL) & ~sell)
        )
        if not violate.any():
            return cand
        frozen |= violate


def simulate(target, fwd_ret, can_buy=None, can_sell=None):
    """逐行推进持仓，返回 weights、pnl 与 turnover。

    target 是目标权重，fwd_ret 是持仓在持有期内的收益，两者中的 NaN 都当 0。
    can_buy/can_sell 是 (T, N) 的布尔面板，None 表示不限；只给一个时另一个
    视为全 True。权重变大需要 can_buy，变小需要 can_sell（负权重按数值大小）。
    """
    tgt = np.array(target, dtype=np.float64, copy=True)
    ret = np.array(fwd_ret, dtype=np.float64, copy=True)
    if tgt.ndim != 2 or ret.shape != tgt.shape:
        raise ValueError("target 与 fwd_ret 必须是同形状的二维数组")
    tgt[np.isnan(tgt)] = 0.0
    ret[np.isnan(ret)] = 0.0

    no_buy = can_buy is None
    no_sell = can_sell is None
    if no_buy:
        buy = np.ones(tgt.shape, dtype=bool)
    else:
        buy = np.array(can_buy, dtype=bool, copy=True)
        if buy.shape != tgt.shape:
            raise ValueError("can_buy 必须与 target 同形状")
    if no_sell:
        sell = np.ones(tgt.shape, dtype=bool)
    else:
        sell = np.array(can_sell, dtype=bool, copy=True)
        if sell.shape != tgt.shape:
            raise ValueError("can_sell 必须与 target 同形状")

    n_rows, n_cols = tgt.shape
    held = np.zeros_like(tgt)
    pnl = np.zeros(n_rows, dtype=np.float64)
    turnover = np.zeros(n_rows, dtype=np.float64)
    prev = np.zeros(n_cols, dtype=np.float64)

    for t in range(n_rows):
        if no_buy and no_sell:
            cur = tgt[t]
        else:
            cur = _advance(tgt[t], prev, buy[t], sell[t])
        held[t] = cur
        pnl[t] = float(np.sum(cur * ret[t]))
        turnover[t] = float(np.sum(np.abs(cur - prev)))
        prev = cur

    return {"weights": held, "pnl": pnl, "turnover": turnover}
