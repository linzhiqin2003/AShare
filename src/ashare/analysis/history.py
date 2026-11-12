"""逐日风险模型的历史：暴露、因子收益，以及（可选）因子协方差和特质方差。

`ashare.risk.model_series` 交出的每日 `RiskModel` 只有 B、F、D，没有因子收益；这里在
遍历时按风险模型自己的口径（同一个截面回归内核、同一估计池规则）补算因子收益：

    f_t = argmin Σ_{i∈估计池_{t-1}} sqrt(cap_{i,t-1}) · (r_{i,t} − B_{i,t-1} f)²，
          行业因子收益按 t-1 日市值加权和为零。

因此第 t 日的因子收益只用 t-1 收盘的暴露和 t 日实现的收益，与风险模型
「B_t 解释 r_{t+1}」的时点一致。全市场的暴露放不下内存，所以只保存 `columns`
指定的那些证券的暴露（通常是组合和基准出现过的证券），截面回归仍用当天的全部证券。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUPS = ("market", "industry", "style")


def factor_groups(factors):
    """因子名 → 类别：country 是市场，industry_* 是行业，其余（含外部暴露）是风格。"""
    out = []
    for name in factors:
        if name in ("country", "market"):
            out.append("market")
        elif name.startswith("industry"):
            out.append("industry")
        else:
            out.append("style")
    return tuple(out)


@dataclass
class FactorHistory:
    """逐日风险模型的历史。T 个日期、n 只保存的证券、K 个因子。

    - `exposures[t]`：第 t 日收盘的暴露 n×K（解释 t+1 日收益）；NaN 行表示当天没有覆盖。
    - `factor_returns[t]`：t-1 收盘到 t 收盘的因子收益（由 `exposures[t-1]` 解释），K。
    - `covariance[t]`、`specific_variance[t]`：第 t 日的 F（K×K）与 D（n），预测 t+1 日。
    """

    dates: np.ndarray
    symbols: tuple[str, ...]
    factors: tuple[str, ...]
    exposures: np.ndarray
    factor_returns: np.ndarray
    covariance: np.ndarray | None = None
    specific_variance: np.ndarray | None = None
    groups: tuple[str, ...] | None = None

    def __post_init__(self):
        self.dates = np.asarray(self.dates).astype(str)
        self.symbols = tuple(map(str, self.symbols))
        self.factors = tuple(map(str, self.factors))
        t, n, k = len(self.dates), len(self.symbols), len(self.factors)
        self.exposures = np.asarray(self.exposures, dtype=float)
        self.factor_returns = np.asarray(self.factor_returns, dtype=float)
        if self.exposures.shape != (t, n, k):
            raise ValueError(
                f"exposures: 需要 {(t, n, k)}，得到 {self.exposures.shape}"
            )
        if self.factor_returns.shape != (t, k):
            raise ValueError("factor_returns: 需要 T×K")
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
            if self.covariance.shape != (t, k, k):
                raise ValueError("covariance: 需要 T×K×K")
        if self.specific_variance is not None:
            self.specific_variance = np.asarray(self.specific_variance, dtype=float)
            if self.specific_variance.shape != (t, n):
                raise ValueError("specific_variance: 需要 T×n")
        if self.groups is None:
            self.groups = factor_groups(self.factors)
        if len(self.groups) != k or not set(self.groups) <= set(GROUPS):
            raise ValueError(f"groups: 每个因子一个，取值 {GROUPS}")

    @property
    def shape(self):
        return self.exposures.shape

    def align(self, dates, symbols):
        """把历史对齐到面板的 (dates, symbols)：返回 (行号, 列号)，缺的为 -1。"""
        row = {d: i for i, d in enumerate(self.dates)}
        col = {s: j for j, s in enumerate(self.symbols)}
        rows = np.array([row.get(str(d), -1) for d in dates], dtype=np.int64)
        cols = np.array([col.get(str(s), -1) for s in symbols], dtype=np.int64)
        return rows, cols

    @classmethod
    def collect(cls, models, source, columns=None, keep_risk=True, config=None):
        """遍历 `model_series` 交出的模型，收集暴露并按风险模型口径补算因子收益。

        models：按日连续的 `RiskModel` 可迭代对象（通常就是 `model_series(source, ...)`）。
        source：构建这些模型用的 `PanelSource`（`risk_source` 的返回值）。
        columns：要保存暴露的证券（代码序列或长度 N 的布尔掩码）；None 为全部。
        """
        from ashare.risk.data import adjusted_returns
        from ashare.risk.engine import cross_section
        from ashare.risk.fundamental import FundamentalConfig

        c = config or FundamentalConfig()
        dates = np.asarray(source.dates).astype(str)
        symbols = tuple(map(str, source.symbols))
        keep = _column_mask(columns, symbols)
        kept = np.flatnonzero(keep)
        position = {d: i for i, d in enumerate(dates)}
        returns = adjusted_returns(source, len(dates) - 1)
        try:
            returns = returns - np.asarray(source["risk_free"], dtype=float)
        except KeyError:
            pass
        suspended = np.asarray(source["is_suspended"], dtype=bool)
        returns[suspended] = np.nan
        listed = np.asarray(source["is_listed"], dtype=bool)
        ages = np.cumsum(listed, axis=0, dtype=np.int32)
        total_mv = np.asarray(source["total_mv"], dtype=float)

        rows, b_out, f_out, cov_out, d_out = [], [], [], [], []
        factors = None
        previous = None  # (day, B 全截面)
        industry = None
        for model in models:
            if tuple(model.symbols) != symbols:
                raise ValueError("模型的证券顺序与 source 不一致")
            day = position.get(str(model.date))
            if day is None:
                raise ValueError(f"模型日期 {model.date} 不在 source 里")
            if rows and day != rows[-1] + 1:
                raise ValueError("模型必须按交易日连续")
            if factors is None:
                factors = tuple(model.factors)
                industry = np.array(
                    [i for i, g in enumerate(factor_groups(factors)) if g == "industry"]
                )
            elif tuple(model.factors) != factors:
                raise ValueError("各日的因子名不一致")
            f = np.full(len(factors), np.nan)
            if previous is not None and previous[0] == day - 1:
                p = day - 1
                cap = total_mv[p]
                good_cap = np.isfinite(cap) & (cap > 0)
                est = listed[p] & (ages[p] >= c.min_listing_days) & ~suspended[p]
                est &= good_cap
                eligible = est & np.isfinite(returns[day])
                if eligible.sum() >= 2:
                    f, _ = cross_section(
                        returns[day],
                        previous[1],
                        np.where(good_cap, cap, 1.0),
                        eligible,
                        industry,
                    )
            rows.append(day)
            b_out.append(np.array(model.B[kept]))
            f_out.append(f)
            if keep_risk:
                cov_out.append(np.array(model.F))
                d_out.append(np.array(model.D[kept]))
            previous = (day, model.B)
        if not rows:
            raise ValueError("models 为空")
        return cls(
            dates[rows],
            tuple(symbols[j] for j in kept),
            factors,
            np.asarray(b_out),
            np.asarray(f_out),
            np.asarray(cov_out) if keep_risk else None,
            np.asarray(d_out) if keep_risk else None,
        )

    @classmethod
    def build(cls, panel, columns=None, config=None, cache=None, keep_risk=True):
        """一行建好：用面板跑 `ashare.risk.model_series`（基本面模型）并收集历史。

        ```python
        panel = an.LazyPanel(ashare.data.open(), start="2014-01-02")
        history = an.FactorHistory.build(panel, columns=held, cache=".cache/risk")
        ```
        面板的起点就是风险模型的起点；暴露的描述子要约一年历史才稳定，起点请比要分析的
        区间早一到两年。
        """
        from ashare.risk import model_series

        from .adapters import risk_source

        source, cfg = risk_source(panel, config)
        models = model_series(source, config=cfg, cache=cache)
        return cls.collect(models, source, columns, keep_risk=keep_risk, config=cfg)


def _column_mask(columns, symbols):
    if columns is None:
        return np.ones(len(symbols), bool)
    c = np.asarray(columns)
    if c.dtype == bool:
        if c.shape != (len(symbols),):
            raise ValueError("columns: 布尔掩码长度须为 N")
        return c
    wanted = set(map(str, c.tolist()))
    unknown = wanted - set(symbols)
    if unknown:
        raise KeyError(f"columns 里有 source 没有的证券：{sorted(unknown)[:5]}")
    return np.array([s in wanted for s in symbols])


def held_columns(*weights, symbols):
    """权重矩阵里出现过非零值的证券代码（给 `FactorHistory.build(columns=...)` 用）。"""
    mask = np.zeros(len(symbols), bool)
    for w in weights:
        if w is not None:
            mask |= np.any(np.nan_to_num(np.asarray(w, float)) != 0, axis=0)
    return tuple(s for s, m in zip(symbols, mask, strict=True) if m)
