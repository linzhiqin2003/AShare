"""分组回测的 RankIC 诊断序列：因子对下一期收益的逐日秩相关，只用 numpy。

口径与分组/多空回测一致：因子第 t 行（完整日期轴）用 t 日收盘及以前的数据，
第 t+1 个交易日成交，收益算到再下一个交易日；即第 t 行因子的未来收益是
`price[t+1] / price[t] - 1`，与 `api._forward_return` 的同一口径。

`pool` 是参与横截面的股票池。两条序列：

- 全池 RankIC：池内所有有因子值的股票，因子与未来收益的 Spearman 相关；
- top 1/10 组内 RankIC：池内按因子从高到低排序（并列按面板列顺序）取前 1/10，
  组内再做一次 Spearman 相关，即 10 分组时第 1 组内部的判别力。

停牌或缺价沿用上一个有效复权价（收益记 0），与 `api._adjusted` 的前填一致。
"""

from __future__ import annotations

import math

import numpy as np

MIN_POOL = 20  # 全池 RankIC 至少要有这么多只有效股票
MIN_TOP = 10  # top 1/10 组内 RankIC 至少要有这么多只有效股票
TOP_DIVISOR = 10  # top 组的大小：池内前 1/10


def _spearman(a, b):
    """顺序秩 Spearman 相关（并列按原顺序给不同秩，与分组的并列规则一致）。"""
    n = a.size
    if n < 2:
        return math.nan
    rank_a = np.empty(n, dtype=np.float64)
    rank_a[np.argsort(a, kind="stable")] = np.arange(n, dtype=np.float64)
    rank_b = np.empty(n, dtype=np.float64)
    rank_b[np.argsort(b, kind="stable")] = np.arange(n, dtype=np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = math.sqrt(float((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
    return float((rank_a * rank_b).sum() / denom) if denom > 0 else math.nan


def series(
    factor,
    pool,
    price,
    adjustment,
    first,
    *,
    min_pool=MIN_POOL,
    min_top=MIN_TOP,
):
    """返回 (全池 RankIC, top 1/10 组内 RankIC)，长度都是完整日期轴 T。

    - `factor`：窗口内的因子矩阵，行对应完整日期轴的 `first..T-1`；
    - `pool`：同形状的布尔股票池；
    - `price`：完整历史的价格面板（`panel[trade_at]`，未复权）；
    - `adjustment`：同形状的复权因子；
    - `first`：窗口第一行在完整日期轴上的下标。

    第 t 行 RankIC 用因子第 t-1 行（信号日）与 t 成交行的下一行收益；最后一行没有
    未来收益、窗口第一行之前没有因子行，都是 NaN。返回的数组不修改输入。
    """
    factor = np.asarray(factor, dtype=np.float64)
    pool = np.asarray(pool, dtype=bool)
    price = np.asarray(price, dtype=np.float64)
    adjustment = np.asarray(adjustment, dtype=np.float64)
    if price.shape != adjustment.shape:
        raise ValueError("price 与 adjustment 形状不一致")
    rows, columns = price.shape
    if factor.shape != (rows - first, columns) or pool.shape != factor.shape:
        raise ValueError("factor / pool 必须是窗口内的 (T - first, n) 矩阵")
    out_pool = np.full(rows, np.nan)
    out_top = np.full(rows, np.nan)
    if rows - first < 2:
        return out_pool, out_top
    # 窗口第一行前填需要的历史：先扫一遍 first 之前的最后有效复权价。
    last = np.full(columns, np.nan)
    for row in range(first):
        value = price[row] * adjustment[row]
        ok = np.isfinite(value) & (value > 0.0)
        last[ok] = value[ok]
    value = price[first] * adjustment[first]
    ok = np.isfinite(value) & (value > 0.0)
    last[ok] = value[ok]
    current = last
    for t in range(first, rows - 1):
        value = price[t + 1] * adjustment[t + 1]
        ok = np.isfinite(value) & (value > 0.0)
        following = np.where(ok, value, current)
        with np.errstate(invalid="ignore", divide="ignore"):
            forward = np.where(current > 0.0, following / current - 1.0, np.nan)
        signal_row = t - 1 - first
        if signal_row >= 0:
            x = factor[signal_row]
            base = pool[signal_row] & np.isfinite(x)
            keep = base & np.isfinite(forward)
            if int(keep.sum()) >= min_pool:
                out_pool[t] = _spearman(x[keep], forward[keep])
            candidates = np.flatnonzero(base)
            if candidates.size:
                order = np.argsort(-x[candidates], kind="stable")
                size = max(1, int(math.ceil(candidates.size / TOP_DIVISOR)))
                top = candidates[order[:size]]
                top_keep = np.isfinite(forward[top])
                if int(top_keep.sum()) >= min_top:
                    out_top[t] = _spearman(x[top[top_keep]], forward[top[top_keep]])
        current = following
    return out_pool, out_top


def cumulative(values):
    """累计 RankIC：NaN 当 0，和 `metrics.curves` 的 cumsum 口径一致。"""
    array = np.asarray(values, dtype=np.float64)
    return np.cumsum(np.where(np.isnan(array), 0.0, array))
