"""事件研究：异常收益、平均累计异常收益和考虑日历扎堆的统计检验。

口径：

- 第 0 天是事件的可成交日 ``avail``（收盘后公告即公告日的下一个交易日）；第 −1 天对
  收盘后公告就是公告日本身。第 k 天的收益是第 ``avail+k`` 日相对前一交易日的
  收盘到收盘收益。
- 异常收益 = 个股收益 − 基准。基准由调用方给：长度 T 的市场（指数）收益，或 T×N 的
  预期收益（例如 :func:`group_benchmark` 的同行业同市值组合，或风险模型的因子部分）。
- 停牌日个股收益记 0（复牌日的跳空记在复牌日），异常收益是 −基准。
- 缺失（默认 ``complete=True``）：窗口必须完整落在面板内；窗口起点早于上市（首个有效收益）
  的事件剔除（``listed_too_short``）；窗口内有其他缺失（例如同行业同市值基准组内不足 5 只）
  的事件剔除（``incomplete_window``）；只有退市之后的缺失记 0（相当于退市后持有现金），不剔除，
  免得把退市股票挑掉造成幸存者偏差。剔除数量在 ``dropped``。``complete=False`` 时所有缺失记 0。
- 同一证券、同一可成交日、同一类型的多条事件（例如同一天公告年报和一季报预告）默认只算
  一条（保留事件表里排在前面的一条），合并的条数在 ``dropped["same_day_duplicates"]``。
- 检验（:meth:`EventStudy.test`）：朴素 t、聚类标准误、按聚类整块重抽的自助法并排。默认按
  可成交日所在**自然月**聚类，可选按日、按季度。同一天的事件共享当天冲击，同行业相邻几天
  的事件共享持续的行业冲击，窗口互相重叠的事件共享同一段市场走势——按日聚类只挡得住第一种。
- **首选的显著性口径是日历时间组合检验**（:meth:`EventStudy.calendar_time`）：每个交易日把处在
  事件窗口里的事件等权成一个组合，对组合的日异常收益做带 Newey–West 调整的 t 检验。它不需要
  假设事件之间独立，窗口重叠和扎堆都体现在组合收益的时间序列里。

最小例子::

    import numpy as np, polars as pl
    from ashare.events import event_study

    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, (300, 50)); dates = np.array([str(20200101 + i) for i in range(300)])
    ev = pl.DataFrame({"symbol": [str(i) for i in range(50)], "date": dates[100:150].tolist(),
                       "kind": ["demo"] * 50, "timing": ["pre_open"] * 50})
    res = event_study(ev, ret, ret.mean(axis=1), dates, np.array([str(i) for i in range(50)]),
                      window=(-5, 10))
    res.test(0, 10)           # 一行：均值、朴素/聚类/自助法的标准误与 t
    res.calendar_time(0, 10)  # 日历时间组合检验（Newey–West t）
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


def size_industry_groups(industry, market_cap, buckets=5, listed=None):
    """同行业同市值分组编号（T×N，int64；−1 表示不参与）。

    第 t 天按第 t−1 天及以前最近一个有效市值把股票分成 ``buckets`` 个分位组，再与行业交叉。
    停牌日没有市值，沿用停牌前最后一个（只用过去的数据），所以停牌的股票仍有分组和基准；
    否则「基准缺失 → 窗口不完整 → 剔除」会把窗口里有停牌的事件全部挑掉。``listed``
    （T×N 布尔，通常是面板的 ``is_listed``）给出时，未上市和退市后的格子不参与（也不进入
    分位点的计算）。行业 0（缺失）不参与。
    """
    industry = np.asarray(industry)
    cap = np.asarray(market_cap, dtype=np.float64)
    t, n = cap.shape
    valid = np.isfinite(cap) & (cap > 0)
    idx = np.where(valid, np.arange(t)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = np.where(idx >= 0, cap[np.maximum(idx, 0), np.arange(n)], np.nan)
    lag = np.full_like(cap, np.nan)
    lag[1:] = filled[:-1]
    if listed is not None:
        lag[~np.asarray(listed, dtype=bool)] = np.nan
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


def group_benchmark(returns, groups, min_size=5, member=None):
    """组内留一等权平均收益（T×N）：每只股票的基准是同组其他股票的平均。

    ``member``（T×N 布尔，通常是 ``has_quote``）给出时只有它为真的股票计入组合平均——
    停牌股票当天收益记 0，不该拉低同组的基准；停牌股票自己的基准照常给出。
    组内有效股票少于 ``min_size`` 时基准为 NaN。
    """
    r = np.asarray(returns, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    t, n = r.shape
    ok = (g >= 0) & np.isfinite(r)
    if member is not None:
        ok &= np.asarray(member, dtype=bool)
    span = int(g.max()) + 1 if g.size and g.max() >= 0 else 1
    key = np.where(g >= 0, np.arange(t)[:, None] * span + g, 0)
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


MIN_CLUSTERS = 30  # 少于这么多聚类时聚类标准误和自助法都偏宽松，发警告


def _newey_west_se(r, lags):
    """序列均值的 Newey–West 标准误（Bartlett 权）。"""
    n = len(r)
    d = r - r.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return float(np.sqrt(max(var, 0.0) / n))


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

    ``ar``：E×W 异常收益（缺失记 0）；``coverage``：E×W，哪些格子有真实数据；``offsets``：
    窗口内的相对日；``events``：参与的事件（带 ``avail``、``col`` 等列）；``avail``：可成交日
    下标；``cluster``：默认聚类标签；``dropped``：各类剔除数量；``days``：面板交易日数。
    """

    ar: np.ndarray
    coverage: np.ndarray
    offsets: np.ndarray
    events: pl.DataFrame
    avail: np.ndarray
    cluster: np.ndarray
    clusters: dict
    dropped: dict
    days: int
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

    def _mask(self, mask):
        if mask is None:
            return np.ones(len(self), bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(f"mask 长度应为事件数 {len(self)}，实际 {mask.shape}")
        return mask

    def car(self, a, b):
        """每个事件在 [a, b] 上的累计异常收益（长度 E）。"""
        return self.ar[:, self._cols(a, b)].sum(axis=1)

    def caar(self, mask=None):
        """平均累计异常收益曲线（从窗口起点累计），polars 表：offset、aar、caar、n（该日有真实
        数据的事件数）。"""
        sel = self._mask(mask)
        ar = self.ar[sel]
        aar = ar.mean(axis=0) if len(ar) else np.full(len(self.offsets), np.nan)
        return pl.DataFrame(
            {
                "offset": self.offsets,
                "aar": aar,
                "caar": np.cumsum(aar),
                "n": self.coverage[sel].sum(axis=0),
            }
        )

    def test(self, a, b, mask=None, cluster=None):
        """[a, b] 区间 CAR 的检验：朴素 t、聚类标准误（默认按月）、按聚类整块自助法，并排给出。

        ``cluster``：``"date"``、``"month"``、``"quarter"`` 或与事件等长的标签数组。聚类数少于
        30 时发 ``RuntimeWarning``（这时聚类标准误和自助法都偏宽松）。
        """
        sel = self._mask(mask)
        x = self.car(a, b)[sel]
        if cluster is None:
            labels = self.cluster
        elif isinstance(cluster, str):
            if cluster not in self.clusters:
                raise ValueError(f"cluster 只能是 {list(self.clusters)} 或标签数组")
            labels = self.clusters[cluster]
        else:
            labels = np.asarray(cluster)
            if labels.shape != (len(self),):
                raise ValueError(f"cluster 标签长度应为事件数 {len(self)}")
        cl = labels[sel]
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
        if g < MIN_CLUSTERS:
            warnings.warn(
                f"只有 {g} 个聚类（少于 {MIN_CLUSTERS}），聚类标准误与自助法偏宽松",
                RuntimeWarning,
                stacklevel=2,
            )
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
            "p_cluster": float(2 * _stats.t.sf(abs(t_cl), g - 1)) if g > 1 else np.nan,
            "se_bootstrap": se_bs,
            "p_bootstrap": p_bs,
            "se_ratio": se_cl / se_naive,
            "positive": float((x > 0).mean()),
        }
        return pl.DataFrame([row])

    def calendar_time(self, a, b, mask=None, lags=None):
        """日历时间组合检验（首选的显著性口径）。

        每个交易日 τ，把相对日落在 [a, b] 内、当天有真实数据的事件等权成一个组合，得到组合的
        日异常收益 r_τ；只用组合非空的日子。检验 r_τ 的均值是否为 0，标准误用 Newey–West
        （Bartlett 权，``lags`` 默认 floor(4·(T/100)^(2/9))）。``implied_car`` = 日均 × 区间天数，
        与按事件平均的 CAR 不完全相同（这里每个交易日权重相同，事件多的日子不加权）。
        """
        sel = self._mask(mask)
        sl = self._cols(a, b)
        ar, cov = self.ar[sel][:, sl], self.coverage[sel][:, sl]
        day = self.avail[sel][:, None] + self.offsets[sl][None, :]
        sums = np.bincount(day[cov], weights=ar[cov], minlength=self.days)
        cnt = np.bincount(day[cov], minlength=self.days)
        active = cnt > 0
        r = sums[active] / cnt[active]
        n = len(r)
        row = {"window": f"[{a},{b}]", "days": n}
        if n < 3:
            return pl.DataFrame([row | {"mean_daily": np.nan}])
        if lags is None:
            lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        mean = float(r.mean())
        se = _newey_west_se(r, lags)
        t_nw = mean / se if se > 0 else np.nan
        row |= {
            "mean_daily": mean,
            "implied_car": mean * (b - a + 1),
            "se_nw": se,
            "t_nw": t_nw,
            "p_nw": float(2 * _stats.t.sf(abs(t_nw), n - 1)),
            "lags": lags,
            "events_per_day": float(cnt[active].mean()),
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
        """按布尔掩码把事件分两组，各自做 [a, b] 的检验，另附日历时间组合的 ``t_nw``。"""
        mask = self._mask(mask)
        rows = []
        for m, label in ((mask, labels[0]), (~mask, labels[1])):
            row = self.test(a, b, m).with_columns(pl.lit(label).alias("group"))
            ct = self.calendar_time(a, b, m)
            t_nw = ct["t_nw"][0] if "t_nw" in ct.columns else None
            rows.append(row.with_columns(pl.lit(t_nw, dtype=pl.Float64).alias("t_nw")))
        return pl.concat(rows, how="diagonal_relaxed")


def _check_shapes(returns, benchmark, dates, symbols, can_buy):
    t, n = len(dates), len(symbols)
    if returns.ndim != 2 or returns.shape != (t, n):
        raise ValueError(
            f"returns 形状应为 (交易日数 {t}, 证券数 {n})，实际 {returns.shape}；"
            "dates、symbols 必须与收益面板同一个日历和证券表"
        )
    if benchmark.shape not in ((t,), (t, n)):
        raise ValueError(
            f"benchmark 形状应为 ({t},) 或 ({t}, {n})，实际 {benchmark.shape}"
        )
    if can_buy is not None and np.shape(can_buy) != (t, n):
        raise ValueError(f"can_buy 形状应为 ({t}, {n})，实际 {np.shape(can_buy)}")


def _cluster_labels(dates, avail):
    days = dates[avail]
    month = np.array([int(d[:6]) for d in days], dtype=np.int64)
    quarter = (month // 100) * 10 + ((month % 100) - 1) // 3 + 1
    return {"date": avail.astype(np.int64), "month": month, "quarter": quarter}


def event_study(
    events,
    returns,
    benchmark,
    dates,
    symbols,
    *,
    window=(-20, 60),
    cluster="month",
    complete=True,
    dedupe=True,
    can_buy=None,
    entry_lag=0,
    reps=1000,
    seed=0,
) -> EventStudy:
    """计算每个事件的异常收益并汇总；口径见模块说明。

    ``returns``：T×N 日收益（:func:`daily_returns`）；``benchmark``：长度 T 或 T×N；
    ``dates``、``symbols`` 必须与 ``returns`` 同一个日历和证券表（形状不一致直接报错）；
    ``cluster``：默认聚类，``"month"``（默认）、``"date"``、``"quarter"``；
    ``complete``：剔除窗口不完整和上市不足的事件（默认）；``dedupe``：同股同日同类只算一条；
    ``can_buy``：给出时在事件表里加 ``entry_buyable`` 列（第 ``avail+entry_lag`` 日能否买入），
    用来把开盘涨停、停牌买不进的事件单独统计。
    """
    dates = np.asarray(dates).astype("U8")
    symbols = np.asarray(symbols)
    returns = np.asarray(returns, dtype=np.float64)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    _check_shapes(returns, benchmark, dates, symbols, can_buy)
    a, b = int(window[0]), int(window[1])
    if a > b:
        raise ValueError("window 起点必须不晚于终点")
    clusters_known = ("date", "month", "quarter")
    if cluster not in clusters_known:
        raise ValueError(f"cluster 只能是 {clusters_known}")
    ar_full = abnormal_returns(returns, benchmark)
    t, n = ar_full.shape
    aligned = (
        events if isinstance(events, AlignedEvents) else align(events, dates, symbols)
    )
    dropped = dict(aligned.dropped)
    keep = np.ones(len(aligned), bool)
    if dedupe and len(aligned):
        kinds = aligned.frame["kind"].to_numpy()
        _, codes = np.unique(kinds, return_inverse=True)
        key = (aligned.avail * (n + 1) + aligned.col) * (codes.max() + 1) + codes
        _, first = np.unique(key, return_index=True)
        keep = np.zeros(len(aligned), bool)
        keep[first] = True
    dropped["same_day_duplicates"] = int((~keep).sum())
    inside = keep & (aligned.avail + a >= 0) & (aligned.avail + b < t)
    if can_buy is not None:
        inside &= aligned.avail + entry_lag < t
    dropped["window_outside_panel"] = int((keep & ~inside).sum())
    avail, col = aligned.avail[inside], aligned.col[inside]
    offsets = np.arange(a, b + 1)
    rows = avail[:, None] + offsets[None, :]
    raw = ar_full[rows, col[:, None]]
    coverage = np.isfinite(raw)
    valid = np.isfinite(returns)
    any_valid = valid.any(axis=0)
    first_valid = np.where(any_valid, valid.argmax(axis=0), t)
    last_valid = np.where(any_valid, t - 1 - valid[::-1].argmax(axis=0), -1)
    final = np.ones(len(avail), bool)
    if complete:
        short = avail + a < first_valid[col]
        after_delist = rows > last_valid[col][:, None]
        incomplete = (~coverage & ~after_delist).any(axis=1) & ~short
        final = ~short & ~incomplete
        dropped["listed_too_short"] = int(short.sum())
        dropped["incomplete_window"] = int(incomplete.sum())
    else:
        dropped["listed_too_short"] = dropped["incomplete_window"] = 0
    avail, col, raw, coverage = avail[final], col[final], raw[final], coverage[final]
    idx = np.flatnonzero(inside)[final]
    frame = aligned.frame[idx]
    ar = np.where(coverage, raw, 0.0)
    clusters = _cluster_labels(dates, avail)
    if can_buy is not None:
        buyable = np.asarray(can_buy, dtype=bool)[avail + entry_lag, col]
        frame = frame.with_columns(pl.Series("entry_buyable", buyable))
    return EventStudy(
        ar,
        coverage,
        offsets,
        frame,
        avail,
        clusters[cluster],
        clusters,
        dropped,
        t,
        seed=seed,
        reps=reps,
    )
