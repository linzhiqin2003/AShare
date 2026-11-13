"""单因子评估：Rank IC、分组收益、多头端回测、分年度、分市值段、自相关、与已有因子的相关。

评价以**多头端**为准（做不了空）：最高组（默认前 10%，或前 N 只）等权，每 ``rebalance``
个交易日在信号日收盘后选股，下一交易日开盘成交（`ashare.backtest.execution_targets`），
由 `panelbt.run_batch` 按 A 股规则扣成本回测，相对基准算超额。IC 只是筛选用的中间量。

收益标签直接用 `ashare.ml.labels`（与回测同一成交约定）：信号日 t，t+1 开盘买入、
t+1+h 开盘卖出；买入日不可买（停牌、涨停、一字板）的样本不进 IC；卖出日不可卖按当日价
（`sell_blocked="price"`）。因子值本身在 t 日只用 t 日及以前的数据，由表达式层保证。

```python
from ashare.factor.evaluate import evaluate_factor
report = evaluate_factor(scores, panel)        # scores：T×N 因子面板
print(report.text())                           # 或 report.to_dict()
```

多个因子用 `FactorEvaluator(panel).evaluate_many({名字: 面板})`：标签、股票池、行情、
基准只准备一次，IC 共用收益排序，回测一次 `run_batch`。口径与性能见
`docs/MINING_NOTES.md`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import panelops

TRADING_DAYS = 252
DEFAULT_HORIZONS = (1, 5, 20)


# ---- 小工具 -----------------------------------------------------------------------


def _finite(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else math.nan


def ic_summary(ic, horizon=1):
    """日 IC 序列的均值、标准差、ICIR（日频不年化）、t 值与单侧 p。

    t 值的标准误用 Hansen–Hodrick 均匀核 HAC、滞后 h − 1（持有 h 天的收益相邻重叠，
    零假设下日 IC 是 MA(h − 1)），见 `ashare.factor.stats.hac_t`；p 值按自由度
    n/h − 1 的 Student t 算（HAC 方差在小样本里偏小，正态近似的尾部太薄，纯噪声模拟里
    Holm 的族错误率会到名义水平的两倍，见 MINING_NOTES）。"""
    from .stats import hac_t, p_from_t

    v = np.asarray(ic, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {"mean": math.nan, "std": math.nan, "icir": math.nan, "t": math.nan,
                "p": math.nan, "days": n}  # fmt: skip
    mean, std = float(v.mean()), float(v.std(ddof=1))
    h = max(int(horizon), 1)
    t, _ = hac_t(v, h - 1, "uniform")
    df = max(n / h - 1, 1.0)  # 不重叠的样本数减一
    return {
        "mean": mean,
        "std": std,
        "icir": mean / std if std > 0 else math.nan,
        "t": t,
        "p": p_from_t(t, df=df) if math.isfinite(t) else math.nan,
        "days": n,
    }


def long_summary(r, bench, *, turnover=None, cost=None, periods=TRADING_DAYS):
    """多头端日收益 r 与基准日收益 bench 的汇总（同长度，已截到要统计的区间）。

    年化收益为几何年化；超额 = r − bench 的日序列，年化超额为其算术均值 × periods，
    信息比率 IR = 均值/标准差 × √periods；夏普同 `ashare.report.metrics.sharpe`（无风险
    利率 0）；回撤以区间起点为 1；sr_period、skew、kurt 是超额日序列的每期夏普和矩（给
    折算夏普用）；turnover 为年化单边换手；cost 为年化成本占净值比例。"""
    from ..report import metrics as rm
    from .stats import moments

    r = np.asarray(r, dtype=np.float64)
    bench = np.asarray(bench, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(bench)
    r, bench = r[ok], bench[ok]
    excess = r - bench
    n = len(r)
    sd = float(excess.std(ddof=1)) if n >= 2 else math.nan
    sr, g3, g4 = moments(excess)
    out = {
        "days": n,
        "annual_return": _finite(rm.annual_return(r, periods)) if n else None,
        "benchmark_return": _finite(rm.annual_return(bench, periods)) if n else None,
        "excess_return": float(excess.mean() * periods) if n else None,
        "information_ratio": float(excess.mean() / sd * math.sqrt(periods))
        if n >= 2 and sd > 0
        else None,
        "sharpe": _finite(rm.sharpe(r, periods)),
        "max_drawdown": float(rm.max_drawdown_of_returns(r)) if n else None,
        "sr_period": _finite(sr),
        "skew": _finite(g3),
        "kurt": _finite(g4),
    }
    if turnover is not None:
        t = np.asarray(turnover, dtype=np.float64)[ok]
        out["turnover"] = float(np.nansum(t) / max(n, 1) * periods) if n else None
    if cost is not None:
        c = np.asarray(cost, dtype=np.float64)[ok]
        out["cost"] = float(np.nansum(c) / max(n, 1) * periods) if n else None
    return out


def equal_weight_benchmark(panel, universe):
    """股票池等权基准的日收益（长度 T，首行 NaN）：第 d 日用 d−1 日在股票池内、
    两日复权收盘价都有效的股票的平均收盘到收盘收益。"""
    price = np.asarray(panel["close"], dtype=np.float64) * np.asarray(
        panel["adj_factor"], dtype=np.float64
    )
    price = np.where(np.isfinite(price) & (price > 0), price, np.nan)
    out = np.full(price.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = price[1:] / price[:-1] - 1
    ok = np.asarray(universe, dtype=bool)[:-1] & np.isfinite(ret)
    count = ok.sum(axis=1)
    total = np.where(ok, ret, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


class _RankIC:
    """一份收益面板对多个因子的日 Rank IC；收益的逐日排序只做一次。

    与 `panelops.rank_ic(score, returns)` 逐位相同（同一个内核，见 `rank_ic_many`）。"""

    def __init__(self, returns, mask=None):
        self.returns = np.ascontiguousarray(returns, dtype=np.float64)
        self.mask = None if mask is None else np.ascontiguousarray(mask, dtype=bool)
        self._order = None
        core = getattr(panelops.api, "_core", None)
        self._fast = core is not None and hasattr(core, "rank_ic_ordered")

    def __call__(self, score):
        if not self._fast or panelops.api.BACKENDS.get("rank_ic", "rust") != "rust":
            return panelops.rank_ic(score, self.returns, mask=self.mask)
        core = panelops.api._core
        panelops.api.panel(score, shape=self.returns.shape)
        if self._order is None:
            self._order = core.return_order(
                self.returns,
                panelops.api.optional(self.mask, self.returns, "mask", bool),
                panelops.get_num_threads(),
            )
        return core.rank_ic_ordered(score, self._order, panelops.get_num_threads())


# ---- 结果 -------------------------------------------------------------------------


def _clean(value):
    """to_dict：只剩 dict、list、str、int、float、bool、None；NaN 变 None。"""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite(value)
    return value


@dataclass
class FactorReport:
    """一个因子的评估结果。segments：{段名: 指标}；series 里是日序列（不进 to_dict）。"""

    name: str
    segments: dict
    autocorr: dict
    correlation: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        return _clean(
            {
                "name": self.name,
                "segments": self.segments,
                "autocorr": self.autocorr,
                "correlation": self.correlation,
            }
        )

    def metric(self, segment, path):
        """取一个标量指标：path 如 "long.information_ratio" 或 "ic.5.mean"。"""
        value = self.segments[segment]
        for key in path.split("."):
            value = (
                value[int(key)] if key.isdigit() and int(key) in value else value[key]
            )
        return value

    def text(self):
        """给终端和代理看的文本摘要（每段一块，行宽 ≤ 100 列）。"""
        from ..report.fmt import num, pct

        lines = [f"因子 {self.name}"]
        for seg, m in self.segments.items():
            lines.append(f"[{seg}] {m['start']} – {m['end']}")
            for h, s in m["ic"].items():
                lines.append(
                    f"  Rank IC h={h:<3} 均值 {num(s['mean'], 4)}  ICIR {num(s['icir'], 3)}  "
                    f"t {num(s['t'], 2)}  p {num(s['p'], 4)}  天数 {s['days']}"
                )
            if m.get("groups"):
                g = " ".join(num(v * 1e4, 1) for v in m["groups"]["mean"])
                lines.append(
                    f"  分组 h={m['groups']['horizon']} 平均收益（bp）{g}  单调性 "
                    f"{num(m['groups']['monotonicity'], 2)}"
                )
            lo = m.get("long")
            if lo:
                lines.append(
                    f"  多头端 年化 {pct(lo['annual_return'])}  基准 {pct(lo['benchmark_return'])}  "
                    f"超额 {pct(lo['excess_return'])}  IR {num(lo['information_ratio'], 2)}  "
                    f"夏普 {num(lo['sharpe'], 2)}  回撤 {pct(lo['max_drawdown'])}"
                )
                lines.append(
                    f"         换手 {num(lo.get('turnover'), 1)} 倍/年  成本 {pct(lo.get('cost'), 2)}/年"
                )
            if m.get("years"):
                parts = [
                    f"{y}: IC {num(v['ic'], 3)} 超额 {pct(v.get('excess_return'))}"
                    for y, v in m["years"].items()
                ]
                for k in range(0, len(parts), 3):
                    lines.append("  " + "  ".join(parts[k : k + 3]))
            if m.get("size"):
                lines.append(
                    "  市值段 IC "
                    + "  ".join(f"{k} {num(v['ic'], 3)}" for k, v in m["size"].items())
                )
        lines.append(
            "自相关 "
            + "  ".join(f"滞后 {k}: {num(v, 3)}" for k, v in self.autocorr.items())
        )
        if self.correlation:
            lines.append(
                "与已有因子的截面相关 "
                + "  ".join(f"{k} {num(v, 3)}" for k, v in self.correlation.items())
            )
        return "\n".join(lines)

    def __str__(self):
        return self.text()


# ---- 评估器 -----------------------------------------------------------------------


class FactorEvaluator:
    """在一份面板上评估很多因子：标签、股票池、行情、基准只准备一次。

    panel：`ashare.data` 的面板或任何有 dates、symbols、version、按字段名取 T×N 数组的对象。
    需要的字段见 `required_fields()`。
    - horizons：IC 的持有期；screen_horizon：分组收益与筛选用的那个（默认 rebalance）。
    - groups：分组数；top / top_fraction：多头端取前 top 只或前 top_fraction 比例（默认 10%）。
    - rebalance：多头端每隔几个交易日调仓；lag：信号到成交的行数（默认 1）。
    - universe：T×N 布尔股票池，默认 `ashare.ml.default_universe`。
    - segments：{段名: (起始行, 结束行)}，默认整段 {"all": (0, T)}；IC 只用标签完整落在
      段内的信号日（t + lag + h < 结束行），多头端用段内每天的日收益。
    - market / costs / config：`panelbt` 的行情（或 `ashare.backtest.BacktestMarket`）、
      费用、账户；默认由面板建行情、按 A 股历史印花税和过户费算费用、整手 100 股。
    - benchmark：长度 T 的日收益；默认股票池等权（`equal_weight_benchmark`）。
    - existing：{名字: T×N 面板}，报告与它们的平均日截面秩相关（给去重用）。
    - size_buckets：按总市值把股票池分几段（默认 3：小、中、大）算 IC。
    """

    def __init__(
        self,
        panel,
        *,
        horizons=DEFAULT_HORIZONS,
        screen_horizon=None,
        groups=10,
        top=None,
        top_fraction=0.1,
        rebalance=5,
        lag=1,
        universe=None,
        segments=None,
        market=None,
        costs=None,
        config=None,
        benchmark=None,
        existing=None,
        size_buckets=3,
        threads=0,
    ):
        from ..ml import labels as _labels
        from ..ml.walk import default_universe

        self.panel = panel
        t, n = len(panel.dates), len(panel.symbols)
        self.shape = (t, n)
        self.dates = np.asarray(panel.dates).astype(str)
        self.horizons = tuple(int(h) for h in horizons)
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("horizons: 需要正整数")
        self.rebalance = int(rebalance)
        self.screen_horizon = int(screen_horizon or self.rebalance)
        if self.rebalance < 1:
            raise ValueError("rebalance: 必须是正整数")
        self.lag = int(lag)
        self.groups = int(groups)
        self.top = None if top is None else int(top)
        self.top_fraction = None if top is not None else float(top_fraction)
        self.threads = int(threads)
        self.universe = np.ascontiguousarray(
            default_universe(panel) if universe is None else universe, dtype=bool
        )
        if self.universe.shape != self.shape:
            raise ValueError("universe: 形状必须与面板一致")
        self.segments = dict(segments or {"all": (0, t)})
        for name, (a, b) in self.segments.items():
            if not (0 <= a < b <= t):
                raise ValueError(f"segments[{name!r}]: 行号区间无效")
        # 标签：每个持有期一份，买入日不可买的置缺失（不进 IC）
        self.returns = {}
        horizons_needed = sorted(set(self.horizons) | {self.screen_horizon})
        for h in horizons_needed:
            values = _labels.compute(
                panel, _labels.Label(horizon=h, lag=self.lag), self.universe
            )
            ret = np.where(values.buyable, values.raw, np.nan)
            self.returns[h] = np.ascontiguousarray(ret)
        self._ic = {h: _RankIC(r) for h, r in self.returns.items()}
        self.span = {h: self.lag + h for h in self.returns}
        # 市值段（按信号日截面分位）
        self.size_buckets = int(size_buckets)
        self._size_masks = self._size(panel) if self.size_buckets > 1 else {}
        self._size_ic = {
            k: _RankIC(self.returns[self.screen_horizon], m)
            for k, m in self._size_masks.items()
        }
        self.benchmark = (
            equal_weight_benchmark(panel, self.universe)
            if benchmark is None
            else np.asarray(benchmark, dtype=np.float64)
        )
        if self.benchmark.shape != (t,):
            raise ValueError("benchmark: 需要长度 T 的日收益")
        self._market, self.costs, self.config = self._backtest_inputs(
            market, costs, config
        )
        self.existing = {
            k: np.asarray(v, dtype=np.float64) for k, v in (existing or {}).items()
        }
        self.years = np.array([d[:4] for d in self.dates])

    # -- 准备 --

    @staticmethod
    def required_fields(exec="open"):
        """默认设置下要从面板读的字段。"""
        from ..backtest.market import required_fields as market_fields
        from ..ml.labels import MASKS

        return tuple(
            dict.fromkeys(
                [
                    *market_fields(exec),
                    "adj_factor",
                    "close",
                    "total_market_cap",
                    "is_listed",
                    "is_st",
                    "listing_age",
                    "has_quote",
                    *MASKS,
                ]
            )
        )

    def _size(self, panel):
        try:
            cap = np.asarray(panel["total_market_cap"], dtype=np.float64)
        except (KeyError, FileNotFoundError):
            return {}
        cap = np.where(self.universe & np.isfinite(cap) & (cap > 0), cap, np.nan)
        rank = panelops.cs_rank(np.ascontiguousarray(cap), normalize=True)
        k = self.size_buckets
        names = ["小", "中", "大"] if k == 3 else [f"段{i + 1}" for i in range(k)]
        out = {}
        for i in range(k):
            lo, hi = i / k, (i + 1) / k
            m = np.isfinite(rank) & (rank > lo if i else rank >= 0) & (rank <= hi)
            out[names[i]] = m
        return out

    def _backtest_inputs(self, market, costs, config):
        import panelbt
        from panelbt.markets import ASHARE

        from ..backtest.market import BacktestMarket, iso_dates, market_inputs

        if isinstance(market, BacktestMarket):
            config = config or market.config(initial_cash=1e8)
            costs = costs or market.costs()
            market = market.market
        elif market is None:
            market = panelbt.Market(**market_inputs(self.panel), validate=True)
        if costs is None:
            costs = ASHARE.costs(iso_dates(self.dates))
        if config is None:
            config = panelbt.Config(initial_cash=1e8)
        if market.shape != self.shape:
            raise ValueError("market: 形状必须与面板一致")
        return market, costs, config

    # -- 评估 --

    def _scores(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.shape != self.shape:
            raise ValueError(f"因子面板形状 {x.shape} 与面板 {self.shape} 不一致")
        return np.ascontiguousarray(np.where(self.universe & np.isfinite(x), x, np.nan))

    def rebalance_days(self, scores):
        """调仓信号日：从第一个有因子值的行起每 rebalance 行一次（与面板截断无关）。"""
        rows = np.flatnonzero(np.isfinite(scores).any(axis=1))
        if not len(rows):
            return np.empty(0, dtype=np.int64)
        return np.arange(int(rows[0]), self.shape[0], self.rebalance, dtype=np.int64)

    def targets(self, scores):
        """多头端目标（成交日行号），直接交给 `panelbt`。没有可选股票时返回 None。"""
        from ..backtest import execution_targets

        days = self.rebalance_days(scores)
        if not len(days):
            return None
        kw = (
            {"k": self.top} if self.top is not None else {"fraction": self.top_fraction}
        )
        sparse = panelops.targets(scores, days=days, **kw)
        return execution_targets(sparse, t=self.shape[0], lag=self.lag)

    def screen(self, values):
        """粗筛只要的部分：股票池内的因子面板和筛选持有期的日 IC。返回 (scores, ic)。"""
        scores = self._scores(values)
        return scores, self._ic[self.screen_horizon](scores)

    def screen_metrics(self, ic):
        """筛选持有期日 IC 在各段上的汇总（与 `finish` 给出的同名字段一致，只含基本类型）。"""
        sh = self.screen_horizon
        out = {}
        for seg, (a, b) in self.segments.items():
            s = ic_summary(ic[a : max(a, b - self.span[sh])], sh)
            out[seg] = {
                "start": str(self.dates[a]),
                "end": str(self.dates[b - 1]),
                "ic": s,
                "ic_all": {str(sh): {"mean": s["mean"], "t": s["t"]}},
                "monotonicity": None,
                "long": None,
            }
        return _clean(out)

    def cheap(self, values=None, *, scores=None, screen_ic=None):
        """只算不需要回测的部分：IC、分组、分年度 IC、市值段、自相关、与已有因子相关。

        返回 (scores, partial)；partial 交给 `finish` 补上多头端。已经做过 `screen` 的
        可以把 scores 和 screen_ic 传进来，不再重算。"""
        if scores is None:
            scores = self._scores(values)
        sh = self.screen_horizon
        ic = {
            h: screen_ic if h == sh and screen_ic is not None else self._ic[h](scores)
            for h in self.returns
        }
        groups = panelops.group_returns(scores, self.returns[sh], self.groups)
        size_ic = {k: f(scores) for k, f in self._size_ic.items()}
        autocorr = {
            p: _mean(panelops.factor_autocorr(scores, p))
            for p in sorted({1, self.rebalance})
        }
        corr = {k: _mean(panelops.rank_ic(scores, v)) for k, v in self.existing.items()}
        return scores, {"ic": ic, "groups": groups, "size_ic": size_ic,
                        "autocorr": autocorr, "correlation": corr}  # fmt: skip

    def _segment(self, partial, a, b, long=None):
        sh = self.screen_horizon
        out = {"start": str(self.dates[a]), "end": str(self.dates[b - 1])}
        out["ic"] = {
            h: ic_summary(partial["ic"][h][a : max(a, b - self.span[h])], h)
            for h in self.horizons
        }
        g_rows = slice(a, max(a, b - self.span[sh]))
        g = partial["groups"][g_rows]
        means = np.array([_mean(g[:, k]) for k in range(g.shape[1])])
        ok = np.isfinite(means)
        mono = math.nan
        if ok.sum() >= 3:
            ranks = np.argsort(np.argsort(means[ok])).astype(float)
            idx = np.arange(ok.sum(), dtype=float)
            mono = float(np.corrcoef(idx, ranks)[0, 1])
        out["groups"] = {
            "horizon": sh,
            "mean": means.tolist(),
            "top_minus_bottom": float(means[-1] - means[0])
            if ok[0] and ok[-1]
            else math.nan,
            "monotonicity": mono,
        }
        out["size"] = {
            k: {"ic": _mean(v[g_rows])} for k, v in partial["size_ic"].items()
        }
        years = {}
        ic_sh = partial["ic"][sh]
        for y in dict.fromkeys(self.years[a:b]):
            rows = np.flatnonzero(self.years == y)
            ic_rows = rows[(rows >= a) & (rows < b - self.span[sh])]
            years[str(y)] = {"ic": _mean(ic_sh[ic_rows]) if len(ic_rows) else math.nan}
            if long is not None:
                d = rows[(rows >= a) & (rows < b)]
                excess = long["returns"][d] - self.benchmark[d]
                ok_ = np.isfinite(excess)
                years[str(y)]["excess_return"] = (
                    float(excess[ok_].mean() * TRADING_DAYS) if ok_.any() else math.nan
                )
        out["years"] = years
        if long is not None:
            d = slice(a, b)
            out["long"] = long_summary(
                long["returns"][d],
                self.benchmark[d],
                turnover=long["turnover"][d],
                cost=long["cost"][d],
            )
        return out

    def run_long(self, targets):
        """一批多头端目标一次 `run_batch`；返回每个的 {returns, turnover, cost} 日序列。

        targets 里的 None（没有可选股票）给出全 NaN 的序列。"""
        import panelbt

        live = [k for k, x in enumerate(targets) if x is not None]
        t = self.shape[0]
        empty = {"returns": np.full(t, np.nan), "turnover": np.full(t, np.nan),
                 "cost": np.full(t, np.nan)}  # fmt: skip
        out = [empty] * len(targets)
        if not live:
            return out
        res = panelbt.run_batch(
            self._market, [targets[k] for k in live], self.costs, self.config,
            threads=self.threads,
        )  # fmt: skip
        for i, k in enumerate(live):
            nav = res.nav[i]
            prev = np.r_[1.0, nav[:-1]] * self.config.initial_cash
            spent = (
                res.commission[i] + res.tax[i] + res.other_fee[i] + res.slippage_cost[i]
                + (0 if res.impact_cost is None else res.impact_cost[i])
            )  # fmt: skip
            with np.errstate(invalid="ignore", divide="ignore"):
                cost = spent / prev
            out[k] = {
                "returns": res.returns[i],
                "turnover": res.turnover[i],
                "cost": cost,
            }
        return out

    def finish(self, name, partial, long=None):
        segments = {
            seg: self._segment(partial, a, b, long)
            for seg, (a, b) in self.segments.items()
        }
        series = {"ic": partial["ic"]}
        if long is not None:
            series["long"] = long
            series["excess"] = long["returns"] - self.benchmark
        return FactorReport(
            name, segments, partial["autocorr"], partial["correlation"], series
        )

    def evaluate(self, values, name="factor", *, long=True):
        """评估一个因子面板（T×N）。long=False 时不跑回测（只有 IC 等便宜的部分）。"""
        return self.evaluate_many({name: values}, long=long)[name]

    def evaluate_many(self, factors, *, long=True):
        """评估多个因子。factors：{名字: 面板}、(名字, 面板) 的可迭代对象（例如
        `Evaluator.evaluate_many` 的结果，逐个消费、不同时持有全部面板）。
        回测部分攒成一批 `run_batch`。返回 {名字: FactorReport}，按输入顺序。"""
        items = factors.items() if isinstance(factors, dict) else factors
        partials, targets, names = {}, [], []
        for name, values in items:
            scores, partial = self.cheap(values)
            if name in partials:
                raise ValueError(f"因子重名：{name}")
            partials[name] = partial
            names.append(name)
            if long:
                targets.append(self.targets(scores))
            del scores
        longs = self.run_long(targets) if long else [None] * len(names)
        return {
            name: self.finish(name, partials[name], longs[k])
            for k, name in enumerate(names)
        }


def evaluate_factor(values, panel, *, name="factor", long=True, **options):
    """评估一个因子面板的快捷入口；options 同 `FactorEvaluator`。"""
    return FactorEvaluator(panel, **options).evaluate(values, name, long=long)
