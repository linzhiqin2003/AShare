"""逐日截面指标：Rank IC、ICIR、分组多空收益、换手。全部用 `panelops` 向量化，不逐日循环。

样本向量（按日排序）先按（相对日，列）散到一张小面板上，再交给 `panelops`。
"""

from __future__ import annotations

import math

import numpy as np
import panelops


def _local(values, day, col, n_cols):
    """样本向量 → (天数 × n_cols) 面板，行是 day 里出现过的日子（升序）。"""
    day = np.asarray(day)
    if len(day) == 0:
        return (
            np.full((0, n_cols), np.nan),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
        )
    first = int(day[0])
    span = int(day[-1]) - first + 1
    out = np.full((span, n_cols), np.nan)
    out[day - first, col] = values
    return out, day - first, np.unique(day - first)


def daily_rank_ic(pred, target, day, col, n_cols):
    """每天一个 Spearman 相关（只算样本出现的日子），返回一维数组。"""
    if len(day) == 0:
        return np.empty(0)
    p, rel, present = _local(np.asarray(pred, dtype=np.float64), day, col, n_cols)
    y, _, _ = _local(np.asarray(target, dtype=np.float64), day, col, n_cols)
    ic = panelops.rank_ic(p, y)
    return ic[present]


def mean_rank_ic(pred, target, day, col, n_cols):
    ic = daily_rank_ic(pred, target, day, col, n_cols)
    ic = ic[np.isfinite(ic)]
    return float(ic.mean()) if len(ic) else float("nan")


def rank_within_days(values, day, col, n_cols):
    """按日截面排名（映射到 [−0.5, 0.5]），用于按截面排名平均多个模型。"""
    p, rel, _ = _local(np.asarray(values, dtype=np.float64), day, col, n_cols)
    r = panelops.cs_rank(p, normalize=True) - 0.5
    return r[rel, col]


def ic_stats(ic, horizon=1):
    """均值、标准差、ICIR（均值/标准差，日频不年化）、t 值与双侧 p 值。

    标签跨度 h 天的日 IC 相邻高度相关，t 值按不重叠样本数 n/h 折算（保守的简单做法）。
    """
    ic = np.asarray(ic, dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    n = len(ic)
    if n < 2:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "icir": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "days": n,
        }
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    icir = mean / std if std > 0 else float("nan")
    n_eff = max(n / max(horizon, 1), 1.0)
    t = icir * math.sqrt(n_eff) if std > 0 else float("nan")
    p = math.erfc(abs(t) / math.sqrt(2)) if math.isfinite(t) else float("nan")
    return {"ic_mean": mean, "ic_std": std, "icir": icir, "t": t, "p": p, "days": n}


def panel_summary(pred, raw, horizon, q=10, rows=None):
    """预测面板对原始未来收益的汇总：Rank IC、ICIR、q 组收益与多空、头组换手。

    pred、raw：T×N（无预测的格子为 NaN）；rows：参与统计的行号（默认 pred 有值的行）。
    分组收益是每个信号日各组的等权 h 日收益，`long_short` 为最高组减最低组的均值（h 日），
    `top_turnover` 为最高组按 h 日间隔的平均换手比例。
    """
    pred = np.ascontiguousarray(pred, dtype=np.float64)
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    if rows is None:
        rows = np.flatnonzero(np.isfinite(pred).any(axis=1))
    rows = np.asarray(rows)
    if len(rows) == 0:
        return {"ic": np.empty(0), **ic_stats([], horizon)}
    a, b = int(rows[0]), int(rows[-1]) + 1
    p, r = pred[a:b], raw[a:b]
    ic = panelops.rank_ic(p, r)[rows - a]
    stats = ic_stats(ic, horizon)
    groups = panelops.group_returns(p, r, q)[rows - a]
    turnover = panelops.group_turnover(p, q, horizon)[rows - a]
    with np.errstate(invalid="ignore"):
        spread = groups[:, -1] - groups[:, 0]
    stats.update(
        ic=ic,
        groups=np.nanmean(groups, axis=0) if len(groups) else np.empty(0),
        long_short=float(np.nanmean(spread))
        if np.isfinite(spread).any()
        else float("nan"),
        top_turnover=float(np.nanmean(turnover[:, -1]))
        if np.isfinite(turnover[:, -1]).any()
        else float("nan"),
    )
    return stats
