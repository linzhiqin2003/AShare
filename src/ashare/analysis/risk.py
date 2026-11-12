"""事前风险归因：把组合波动率（有基准时是跟踪误差）拆到每个因子和特质部分。

第 t 日收盘后的主动权重 a、风险模型 (B_t, F_t, D_t)：

    x = Bᵀa；σ² = xᵀ F x + Σ D_i a_i²
    因子 k 的方差贡献 = x_k (F x)_k；特质的方差贡献 = Σ D_i a_i²   （各项之和 = σ²）
    对波动率的贡献 = 方差贡献 / σ（各项之和 = σ）；占比 = 方差贡献 / σ²（和为 1）
    因子边际贡献 ∂σ/∂x_k = (F x)_k / σ；个股边际贡献 ∂σ/∂a_i = (B F x + D a)_i / σ

这是 Euler 拆分：σ 是 a 的一次齐次函数，所以贡献 = 暴露 × 边际，加起来正好是 σ。
因子之间的协方差平均分到两边（x_k F_kl x_l 记一半在 k、一半在 l），不单列交叉项。

逐日序列的预测 σ_t 预测的是 t+1 日的主动收益；偏差统计量 = std(实现 / 预测)，接近 1
说明预测准确，> 1 低估风险，< 1 高估风险；区间见 `bias_statistics`（默认块自助法，照顾厚尾与波动聚集）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .history import GROUPS, FactorHistory
from .inputs import benchmark_held, result_weights, total_returns
from .table import Table, plain


@dataclass
class RiskDecomposition:
    """某一天的事前风险拆分。`factors` 每行一个因子，末行 specific 与 total。"""

    date: str
    volatility: float
    table: Table
    groups: Table
    stocks: Table | None = None

    def to_dict(self):
        out = {
            "date": self.date,
            "volatility": self.volatility,
            "factors": self.table.to_dict(),
            "groups": self.groups.to_dict(),
        }
        if self.stocks is not None:
            out["stocks"] = self.stocks.to_dict()
        return out


RISK_COLUMNS = ("exposure", "variance", "contribution", "percent", "marginal")


def decompose_risk(weights, model, benchmark=None, factor_groups=None, date=""):
    """单日拆分。model 可以是 `ashare.risk.RiskModel`，或 (B, F, D, factors) 元组。

    ```python
    rd = an.decompose_risk(w, model, benchmark=wb)
    rd.volatility, rd.groups["percent"]    # 日频跟踪误差，及市场/行业/风格/特质占比
    ```
    weights、benchmark：长度 N 的权重（与模型同一证券顺序）。返回日频波动率；年化请乘 sqrt(252)。
    """
    if model is None:
        raise TypeError(
            "model 为 None：需要 ashare.risk.RiskModel 或 (B, F, D, factors)"
        )
    if hasattr(model, "B"):
        b, f, d, factors = model.B, model.F, model.D, model.factors
    elif isinstance(model, tuple) and len(model) == 4:
        b, f, d, factors = model
    else:
        raise TypeError(
            f"model 需要 ashare.risk.RiskModel 或 (B, F, D, factors) 元组，"
            f"收到 {type(model).__name__}"
        )
    b, f, d = (np.asarray(v, dtype=float) for v in (b, f, d))
    a = np.asarray(weights, dtype=float)
    if benchmark is not None:
        wb = np.asarray(benchmark, dtype=float)
        if wb.shape != a.shape:
            raise ValueError(
                f"benchmark 须与 weights 同形状 {a.shape}，收到 {wb.shape}"
            )
        a = a - wb
    if a.shape != d.shape or not np.isfinite(a).all():
        raise ValueError("weights: 需要与模型同长度的有限向量")
    x = b.T @ a
    fx = f @ x
    var_f = x * fx
    var_s = d * a * a
    total = float(var_f.sum() + var_s.sum())
    vol = float(np.sqrt(max(total, 0.0)))
    inv = 1.0 / vol if vol > 0 else 0.0
    names = tuple(factors) + ("specific", "total")
    spec = float(var_s.sum())
    rows = np.column_stack(
        [
            np.r_[x, np.nan, np.nan],
            np.r_[var_f, spec, total],
            np.r_[var_f, spec, total] * inv,
            np.r_[var_f, spec, total] / total if total > 0 else np.zeros(len(names)),
            np.r_[fx * inv, np.nan, np.nan],
        ]
    )
    table = Table(np.array(names), RISK_COLUMNS, rows, "risk", "factor")
    from .history import factor_groups as infer

    g = tuple(factor_groups) if factor_groups is not None else infer(factors)
    gv = np.array([var_f[[x == name for x in g]].sum() for name in GROUPS] + [spec])
    group_rows = np.column_stack(
        [gv, gv * inv, gv / total if total > 0 else np.zeros(len(gv))]
    )
    groups = Table(
        np.array(GROUPS + ("specific",)),
        ("variance", "contribution", "percent"),
        group_rows,
        "risk_groups",
        "group",
    )
    marginal = (b @ fx + d * a) * inv
    stocks = Table(
        np.array(getattr(model, "symbols", None) or [str(i) for i in range(len(a))]),
        ("active_weight", "marginal", "contribution"),
        np.column_stack([a, marginal, a * marginal]),
        "risk_stocks",
        "symbol",
    )
    return RiskDecomposition(
        str(date or getattr(model, "date", "")), vol, table, groups, stocks
    )


@dataclass
class RiskSeries:
    """逐日事前风险及其与事后实现的对比。"""

    predicted: Table  # total、factor、specific 三列的日频波动率，及四类占比
    contributions: Table  # 每个因子与特质对日频波动率的贡献
    realized: Table  # 次日的实现主动收益与 z = 实现 / 预测
    bias: dict
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "predicted": self.predicted.to_dict(),
            "contributions": self.contributions.to_dict(),
            "realized": self.realized.to_dict(),
            "bias": plain({k: v for k, v in self.bias.items() if k != "rolling"}),
        }


def risk_series(
    result, panel, history, benchmark=None, window=252, start=None, end=None
):
    """逐日事前风险拆分与偏差统计。

    ```python
    rs = an.risk_series(result, panel, history, benchmark="000300.SH")
    rs.bias["statistic"]         # 全样本 std(实现/预测)
    ```
    第 t 行用第 t 日收盘后的权重与第 t 日的风险模型，预测第 t+1 日的主动收益；
    实现值是 t+1 日的持仓主动收益（不含交易与成本）。
    """
    if not isinstance(history, FactorHistory):
        raise TypeError(
            f"history 需要 FactorHistory（FactorHistory.build(panel, ...)），"
            f"收到 {type(history).__name__}"
        )
    if history.covariance is None:
        raise ValueError(
            "history 没有保存协方差：请用 FactorHistory.build(..., keep_risk=True)（默认）"
        )
    dates = np.asarray(panel.dates).astype(str)
    symbols = tuple(map(str, panel.symbols))
    returns = total_returns(panel)
    if hasattr(result, "nav"):
        weights, _ = result_weights(result, panel)
    else:
        weights = np.asarray(result, dtype=float)
    held_b = benchmark_held(panel, benchmark, returns)
    # 第 t 日收盘后的基准权重 = 第 t+1 日持有的权重（快照与漂移已在 benchmark_held 里处理）。
    # 这里读第 t+1 行，依赖数据层的约定「指数权重快照从下一交易日起可用」（快照描述的是
    # 此前月末收盘的权重，第 t+1 行出现的新快照在第 t 日收盘时就已经成立）：读它不是前视。
    # 若数据层改成「快照当日可用」，这一行要改为直接用当日快照。
    bench_end = None
    if held_b is not None:
        bench_end = np.zeros_like(held_b)
        bench_end[:-1] = held_b[1:]
        # 最后一天收盘后的基准权重：按当日收益漂移。
        last = held_b[-1] * (1 + np.nan_to_num(returns[-1]))
        bench_end[-1] = last / last.sum() if last.sum() > 0 else 0
    active = weights - (0 if bench_end is None else bench_end)
    rows, cols = history.align(dates, symbols)
    used = np.flatnonzero(cols >= 0)
    lost = np.abs(active).sum(axis=1) - np.abs(active[:, used]).sum(axis=1)
    days = np.flatnonzero(rows >= 0)
    if start is not None:
        days = days[dates[days] >= start]
    if end is not None:
        days = days[dates[days] <= end]
    if not len(days):
        raise ValueError("风险历史与面板没有共同日期")
    hr = rows[days]
    a = active[days][:, used]
    k = len(history.factors)
    x = np.zeros((len(days), k))
    var_s = np.zeros(len(days))
    uncovered = lost[days].copy()
    for s in range(0, len(days), 128):
        e = slice(s, s + 128)
        bb = history.exposures[hr[e]][:, cols[used], :]
        dd = history.specific_variance[hr[e]][:, cols[used]]
        ok = np.isfinite(bb).all(axis=2) & np.isfinite(dd)
        aa = np.where(ok, a[e], 0.0)
        uncovered[e] += np.where(ok, 0.0, np.abs(a[e])).sum(axis=1)
        x[e] = np.einsum("tnk,tn->tk", np.where(ok[..., None], bb, 0.0), aa)
        var_s[e] = (np.where(ok, dd, 0.0) * aa * aa).sum(axis=1)
    fx = np.einsum("tkl,tl->tk", history.covariance[hr], x)
    var_f = x * fx
    total = var_f.sum(axis=1) + var_s
    vol = np.sqrt(np.maximum(total, 0))
    inv = np.divide(1.0, vol, out=np.zeros_like(vol), where=vol > 0)
    groups = history.groups
    gvar = np.column_stack(
        [var_f[:, [g == name for g in groups]].sum(axis=1) for name in GROUPS] + [var_s]
    )
    pct = np.divide(
        gvar, total[:, None], out=np.zeros_like(gvar), where=total[:, None] > 0
    )
    predicted = Table(
        dates[days],
        ("total", "factor", "specific")
        + tuple(f"{n}_percent" for n in GROUPS + ("specific",)),
        np.column_stack(
            [vol, np.sqrt(np.maximum(var_f.sum(axis=1), 0)), np.sqrt(var_s), pct]
        ),
        "predicted_risk",
        "date",
    )
    contributions = Table(
        dates[days],
        history.factors + ("specific",),
        np.column_stack([var_f * inv[:, None], var_s * inv]),
        "risk_contribution",
        "date",
    )
    # 次日实现：持仓主动收益 Σ a_t r_{t+1}。
    nxt = days + 1
    valid = nxt < len(dates)
    realized = np.full(len(days), np.nan)
    rr = np.nan_to_num(returns[nxt[valid]])
    realized[valid] = (active[days[valid]] * rr).sum(axis=1)
    z = np.divide(realized, vol, out=np.full_like(vol, np.nan), where=vol > 0)
    realized_table = Table(
        dates[days],
        ("next_day_active", "z"),
        np.column_stack([realized, z]),
        "realized",
        "date",
    )
    bias = bias_statistics(z, window)
    return RiskSeries(
        predicted,
        contributions,
        realized_table,
        bias,
        {"uncovered_weight": uncovered},
    )


def bias_statistics(z, window=252, block=20, draws=2000, seed=0):
    """偏差统计量 std(z)（ddof=1）与三种 95% 区间。

    区间都是「模型准确（z 的真实标准差为 1）时统计量的 95% 范围」：统计量高于上界说明
    预测风险偏低，低于下界说明偏高。

    - `interval`：块自助法（`block` 日一块的移动块，`draws` 次，固定 `seed`）的 2.5%–97.5%
      分位。z 的峰度明显大于 3、且有波动聚集时，只有它同时照顾到厚尾和序列相关，作为默认；
    - `interval_kurtosis`：按样本峰度 κ 修正的正态近似 1 ± 1.96·sqrt((κ − 1) / (4n))；
    - `interval_normal`：z 服从正态时的 1 ± sqrt(2/n)（κ = 3 时与上一条几乎相同）。
    滚动窗口：`rolling_outside` 是落在峰度修正区间（按窗口长度）之外的窗口比例，
    `rolling_outside_normal` 是按正态区间的比例。
    """
    z = np.asarray(z, dtype=float)
    good = z[np.isfinite(z)]
    n = len(good)
    stat = float(np.std(good, ddof=1)) if n > 1 else float("nan")
    out = {
        "statistic": stat,
        "count": n,
        "kurtosis": np.nan,
        "interval": (np.nan, np.nan),
        "interval_kurtosis": (np.nan, np.nan),
        "interval_normal": (np.nan, np.nan),
        "block": block,
        "window": window,
    }
    if n < 3:
        return out
    centered = good - good.mean()
    kurt = float(np.mean(centered**4) / np.mean(centered**2) ** 2)
    out["kurtosis"] = kurt
    out["interval_normal"] = (1 - np.sqrt(2 / n), 1 + np.sqrt(2 / n))
    half_k = 1.96 * np.sqrt(max(kurt - 1, 0) / (4 * n))
    out["interval_kurtosis"] = (1 - half_k, 1 + half_k)
    size = max(1, min(int(block), n))
    starts = n - size + 1
    count = -(-n // size)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, starts, size=(draws, count))
    idx = (pick[:, :, None] + np.arange(size)[None, None, :]).reshape(draws, -1)[:, :n]
    boot = np.std(good[idx], axis=1, ddof=1)
    # 模型准确（真实 std = 1）时统计量的抽样分布：自助样本 ÷ s 把尺度移到 1（std 的抽样
    # 误差与尺度成比例），厚尾和波动聚集的形状保留下来。
    lo, hi = np.percentile(boot / stat, [2.5, 97.5])
    out["interval"] = (float(lo), float(hi))
    if n >= window > 1:
        roll = np.array(
            [np.std(good[i : i + window], ddof=1) for i in range(0, n - window + 1)]
        )
        half_w = 1.96 * np.sqrt(max(kurt - 1, 0) / (4 * window))
        out["rolling_mean"] = float(roll.mean())
        out["rolling_outside"] = float(np.mean(np.abs(roll - 1) > half_w))
        out["rolling_outside_normal"] = float(
            np.mean(np.abs(roll - 1) > np.sqrt(2 / window))
        )
        out["rolling"] = roll
    return out
