"""事件研究：异常收益、平均累计异常收益和考虑日历扎堆的统计检验。

口径：

- 第 0 天是事件的可成交日 ``avail``（收盘后公告即公告日的下一个交易日）；第 −1 天对
  收盘后公告就是公告日本身。第 k 天的收益是第 ``avail+k`` 日相对前一交易日的
  收盘到收盘收益。
- 异常收益 = 个股收益 − 基准。基准由调用方给：长度 T 的市场（指数）收益，或 T×N 的
  预期收益（例如 :func:`group_benchmark` 的同行业同市值组合，或风险模型的因子部分）。
- 停牌日个股收益记 0（复牌日的跳空记在复牌日），异常收益是 −基准；退市后和上市前的
  缺失记 0 并计入 ``coverage``。只用窗口完整落在面板内的事件。
- 检验：朴素 t（事件间独立）、按事件日聚类的标准误、按事件日整块重抽的自助法。
  同一天发生的事件共享当天的市场冲击，朴素 t 会高估显著性；聚类和自助法把同一天的
  事件当作一个整体。窗口互相重叠的不同事件日之间仍有相关，``cluster="month"``
  可以把聚类放宽到同一个自然月。

最小例子::

    import numpy as np, polars as pl
    from ashare.events import event_study

    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, (300, 50)); dates = np.array([str(20200101 + i) for i in range(300)])
    ev = pl.DataFrame({"symbol": [str(i) for i in range(50)], "date": dates[100:150].tolist(),
                       "kind": ["demo"] * 50, "timing": ["pre_open"] * 50})
    res = event_study(ev, ret, ret.mean(axis=1), dates, np.array([str(i) for i in range(50)]),
                      window=(-5, 10))
    res.test(0, 10)      # 一行：均值、朴素/聚类/自助法的标准误与 t
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats as _stats

from .table import AlignedEvents, align


def daily_returns(close, adj_factor, has_quote, is_listed=None):
    """复权收盘到收盘收益（T×N）。

    缺行情日收益为 0（价格沿用上一个有效值，复牌日记跳空）；首个有效价之前和
    ``is_listed`` 为假的格子是 NaN。
    """
    close = np.asarray(close, dtype=np.float64)
    price = close * np.asarray(adj_factor, dtype=np.float64)
    valid = np.asarray(has_quote, dtype=bool) & np.isfinite(price) & (price > 0)
    t, n = price.shape
    idx = np.where(valid, np.arange(t)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = np.where(idx >= 0, price[np.maximum(idx, 0), np.arange(n)], np.nan)
    ret = np.full((t, n), np.nan)
    ret[1:] = filled[1:] / filled[:-1] - 1.0
    if is_listed is not None:
        ret[~np.asarray(is_listed, dtype=bool)] = np.nan
    return ret


def size_industry_groups(industry, market_cap, buckets=5):
    """同行业同市值分组编号（T×N，int64；−1 表示不参与）。

    第 t 天按第 t−1 天的市值把当天有市值的股票分成 ``buckets`` 个分位组，再与行业交叉。
    行业 0（缺失）不参与。
    """
    industry = np.asarray(industry)
    cap = np.asarray(market_cap, dtype=np.float64)
    lag = np.full_like(cap, np.nan)
    lag[1:] = cap[:-1]
    lag[(lag <= 0) | ~np.isfinite(lag)] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        edges = np.nanquantile(lag, np.linspace(0, 1, buckets + 1)[1:-1], axis=1).T
    bucket = np.zeros(cap.shape, dtype=np.int64)
    for k in range(buckets - 1):
        bucket += lag > edges[:, k : k + 1]
    _, dense = np.unique(industry, return_inverse=True)
    dense = dense.reshape(industry.shape).astype(np.int64)
    group = dense * buckets + bucket
    group[(industry == 0) | np.isnan(lag)] = -1
    return group


def group_benchmark(returns, groups, min_size=5):
    """组内留一等权平均收益（T×N）：每只股票的基准是同组其他股票的平均。

    组内有效股票少于 ``min_size`` 时基准为 NaN。
    """
    r = np.asarray(returns, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    t, n = r.shape
    ok = (g >= 0) & np.isfinite(r)
    span = int(g.max()) + 1 if g.size and g.max() >= 0 else 1
    key = np.where(ok, np.arange(t)[:, None] * span + g, 0)
    flat = key.ravel()
    okf = ok.ravel()
    sums = np.bincount(flat[okf], weights=r.ravel()[okf], minlength=t * span)
    cnts = np.bincount(flat[okf], minlength=t * span).astype(np.float64)
    s = sums[key] - np.where(ok, r, 0.0)
    c = cnts[key] - ok
    out = np.where(
        (g >= 0) & (c >= max(min_size - 1, 1)), s / np.where(c > 0, c, 1), np.nan
    )
    return out


def abnormal_returns(returns, benchmark):
    """个股收益减基准；``benchmark`` 为长度 T（逐日广播）或 T×N。"""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    return r - b


def _cluster_se(x, cluster):
    """均值的聚类稳健标准误（CR1 小样本修正），返回 (se, 聚类数)。"""
    n = len(x)
    uniq, inv = np.unique(cluster, return_inverse=True)
    g = len(uniq)
    if g < 2:
        return np.nan, g
    u = np.bincount(inv, weights=x - x.mean(), minlength=g)
    var = (u @ u) / n**2 * g / (g - 1)
    return float(np.sqrt(var)), g


def _bootstrap(x, cluster, reps, rng):
    """按聚类整块重抽的均值分布（向量化：每次重抽是聚类的多项式计数）。"""
    uniq, inv = np.unique(cluster, return_inverse=True)
    g = len(uniq)
    sums = np.bincount(inv, weights=x, minlength=g)
    cnts = np.bincount(inv, minlength=g).astype(np.float64)
    draws = rng.multinomial(g, np.full(g, 1.0 / g), size=reps).astype(np.float64)
    return (draws @ sums) / (draws @ cnts)


@dataclass
class EventStudy:
    """事件研究结果。

    ``ar``：E×W 异常收益（缺失记 0）；``offsets``：窗口内的相对日；``events``：参与的事件
    （带 ``avail``、``cluster`` 等列）；``coverage``：E×W，哪些格子有真实数据。
    """

    ar: np.ndarray
    coverage: np.ndarray
    offsets: np.ndarray
    events: pl.DataFrame
    cluster: np.ndarray
    dropped: dict
    seed: int = 0
    reps: int = 1000
    _cache: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return self.ar.shape[0]

    def _cols(self, a, b):
        if a < self.offsets[0] or b > self.offsets[-1] or a > b:
            raise ValueError(
                f"区间 [{a},{b}] 超出窗口 [{self.offsets[0]},{self.offsets[-1]}]"
            )
        return slice(a - self.offsets[0], b - self.offsets[0] + 1)

    def car(self, a, b):
        """每个事件在 [a, b] 上的累计异常收益（长度 E）。"""
        return self.ar[:, self._cols(a, b)].sum(axis=1)

    def caar(self, mask=None):
        """平均累计异常收益曲线（从窗口起点累计），polars 表：offset、aar、caar、n。"""
        ar = self.ar if mask is None else self.ar[np.asarray(mask, dtype=bool)]
        aar = ar.mean(axis=0) if len(ar) else np.full(len(self.offsets), np.nan)
        return pl.DataFrame(
            {
                "offset": self.offsets,
                "aar": aar,
                "caar": np.cumsum(aar),
                "n": np.full(len(self.offsets), len(ar)),
            }
        )

    def test(self, a, b, mask=None, cluster=None):
        """[a, b] 区间 CAR 的检验：朴素 t、按事件日聚类、按事件日整块自助法，并排给出。"""
        sel = np.ones(len(self), bool) if mask is None else np.asarray(mask, dtype=bool)
        x = self.car(a, b)[sel]
        cl = (self.cluster if cluster is None else np.asarray(cluster))[sel]
        n = len(x)
        row = {
            "window": f"[{a},{b}]",
            "events": n,
            "clusters": int(len(np.unique(cl))) if n else 0,
        }
        if n < 2:
            return pl.DataFrame([row | {"mean": np.nan}])
        mean = float(x.mean())
        se_naive = float(x.std(ddof=1) / np.sqrt(n))
        se_cl, g = _cluster_se(x, cl)
        rng = np.random.default_rng(self.seed)
        boot = _bootstrap(x, cl, self.reps, rng)
        se_bs = float(boot.std(ddof=1))
        p_bs = float(np.mean(np.abs(boot - mean) >= abs(mean)))
        t_naive, t_cl = mean / se_naive, mean / se_cl
        row |= {
            "mean": mean,
            "se_naive": se_naive,
            "t_naive": t_naive,
            "p_naive": float(2 * _stats.t.sf(abs(t_naive), n - 1)),
            "se_cluster": se_cl,
            "t_cluster": t_cl,
            "p_cluster": float(2 * _stats.t.sf(abs(t_cl), g - 1)),
            "se_bootstrap": se_bs,
            "p_bootstrap": p_bs,
            "se_ratio": se_cl / se_naive,
            "positive": float((x > 0).mean()),
        }
        return pl.DataFrame([row])

    def by_year(self):
        """事件数按可成交日年份。"""
        return (
            self.events.group_by(pl.col("avail_date").str.slice(0, 4).alias("year"))
            .agg(pl.len().alias("events"))
            .sort("year")
        )

    def split(self, mask, a, b, labels=("yes", "no")):
        """按布尔掩码把事件分两组，各自做 [a, b] 的检验并上下拼起来。"""
        mask = np.asarray(mask, dtype=bool)
        return pl.concat(
            [
                self.test(a, b, mask).with_columns(pl.lit(labels[0]).alias("group")),
                self.test(a, b, ~mask).with_columns(pl.lit(labels[1]).alias("group")),
            ],
            how="diagonal_relaxed",
        )


def event_study(
    events,
    returns,
    benchmark,
    dates,
    symbols,
    *,
    window=(-20, 60),
    cluster="date",
    can_buy=None,
    entry_lag=0,
    reps=1000,
    seed=0,
) -> EventStudy:
    """计算每个事件的异常收益并汇总；口径见模块说明。

    ``returns``：T×N 日收益（:func:`daily_returns`）；``benchmark``：长度 T 或 T×N；
    ``cluster``：``"date"``（按可成交日）或 ``"month"``（按可成交日所在自然月）；
    ``can_buy``：给出时在事件表里加 ``entry_buyable`` 列（第 ``avail+entry_lag`` 日能否买入），
    用来把开盘涨停、停牌买不进的事件单独统计。
    """
    dates = np.asarray(dates).astype("U8")
    a, b = int(window[0]), int(window[1])
    if a > b:
        raise ValueError("window 起点必须不晚于终点")
    ar_full = abnormal_returns(returns, benchmark)
    t, n = ar_full.shape
    aligned = (
        events if isinstance(events, AlignedEvents) else align(events, dates, symbols)
    )
    inside = (aligned.avail + a >= 0) & (aligned.avail + b < t)
    if can_buy is not None:
        inside &= aligned.avail + entry_lag < t
    frame = aligned.frame.filter(pl.Series(inside))
    avail, col = aligned.avail[inside], aligned.col[inside]
    offsets = np.arange(a, b + 1)
    rows = avail[:, None] + offsets[None, :]
    raw = ar_full[rows, col[:, None]]
    coverage = np.isfinite(raw)
    ar = np.where(coverage, raw, 0.0)
    if cluster == "date":
        cl = avail
    elif cluster == "month":
        cl = np.array([int(d[:6]) for d in dates[avail]], dtype=np.int64)
    else:
        raise ValueError('cluster 只能是 "date" 或 "month"')
    if can_buy is not None:
        buyable = np.asarray(can_buy, dtype=bool)[avail + entry_lag, col]
        frame = frame.with_columns(pl.Series("entry_buyable", buyable))
    dropped = dict(aligned.dropped)
    dropped["window_outside_panel"] = int((~inside).sum())
    return EventStudy(ar, coverage, offsets, frame, cl, dropped, seed=seed, reps=reps)
