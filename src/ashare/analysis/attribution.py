"""基于风险模型的收益归因：主动收益 = 市场 + 行业 + 风格 + 特质（选股）+ 交易 + 成本。

逐日恒等式（第 t 日，全部是 t-1 收盘 → t 收盘）：

    a_t   = 组合持有权重 − 基准持有权重（两者都是 t-1 收盘后的权重，防前视）
    x_t   = B_{t-1}ᵀ a_t                         主动暴露（t-1 收盘的暴露）
    因子项 = x_{t,k} · f_{t,k}                    f_t 由 B_{t-1} 和 t 日收益回归得到
    特质项 = Σ a_{t,i} r_{t,i} − Σ_k 因子项         持仓主动收益里因子解释不了的部分
    交易项 = Rp_t − Σ h_{t,i} r_{t,i} + 成本率_t     当日成交价与收盘价之差、现金利息、
                                                  公司行为与估值的细小差异
    成本项 = −成本率_t                              佣金、税、过户费、滑点、冲击、融券费

各项之和恰好是 Rp_t − Rb_t（组合净值收益减基准持仓收益）。没有交易的日子里交易项只剩
现金利息等细小差异；直接给权重时 Rp 取持仓收益，交易和成本都为 0。

暴露缺失（风险模型没覆盖的证券）或当日没有因子收益时，那部分全部记进特质项，
并在 `diagnostics` 里给出逐日未覆盖的权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adapters import risk_source
from .history import FactorHistory
from .inputs import (
    benchmark_held,
    held_from_weights,
    result_weights,
    total_returns,
)
from .linking import cumulative_active, link
from .table import Table, plain

EPS = np.finfo(float).eps
ITEMS = ("specific", "trading", "cost")
GROUP_COLUMNS = ("market", "industry", "style", "specific", "trading", "cost")


@dataclass
class Attribution:
    """归因结果。`daily`、`cumulative` 按因子明细；`groups`、`groups_cumulative` 按四类加交易、成本。"""

    daily: Table
    cumulative: Table
    groups: Table
    groups_cumulative: Table
    exposures: Table
    returns: Table
    summary: dict
    diagnostics: dict = field(default_factory=dict)

    def __repr__(self):
        g = self.groups_cumulative.last()
        parts = ", ".join(f"{k}={v:+.4%}" for k, v in g.items())
        return f"Attribution({self.summary['dates'][0]}..{self.summary['dates'][1]}: {parts})"

    def to_dict(self):
        return {
            "summary": plain(self.summary),
            "daily": self.daily.to_dict(),
            "cumulative": self.cumulative.to_dict(),
            "groups": self.groups.to_dict(),
            "groups_cumulative": self.groups_cumulative.to_dict(),
            "exposures": self.exposures.to_dict(),
            "returns": self.returns.to_dict(),
        }


def decompose(
    held,
    benchmark_held,
    returns,
    exposures,
    factor_returns,
    portfolio_returns=None,
    cost_rate=None,
    chunk=128,
):
    """归因的数值核心（纯数组）。

    held、benchmark_held、returns：T×n，第 t 行是 t 日持有的权重与 t 日收益；
    exposures：T×n×K，第 t 行是 t 日收盘的暴露（解释 t+1）；factor_returns：T×K。
    返回 dict：factor（T×K）、specific、trading、cost、portfolio、benchmark、
    active_exposure（T×K）、uncovered（T，未覆盖的主动权重绝对值之和）。
    """
    h = np.asarray(held, dtype=float)
    hb = np.zeros_like(h) if benchmark_held is None else np.asarray(benchmark_held)
    r = np.asarray(returns, dtype=float)
    t_len, n = h.shape
    b = exposures
    k = b.shape[2]
    f = np.asarray(factor_returns, dtype=float)
    if r.shape != h.shape or hb.shape != h.shape or b.shape[:2] != (t_len, n):
        raise ValueError("decompose: 权重、收益、暴露的形状不一致")
    missing = ~np.isfinite(r) & ((h != 0) | (hb != 0))
    r = np.where(np.isfinite(r), r, 0.0)
    active = h - hb
    x = np.zeros((t_len, k))
    xp = np.zeros((t_len, k))  # 组合自身的暴露，只用于把特质项拆成组合侧与基准侧
    uncovered = np.zeros(t_len)
    for start in range(1, t_len, chunk):
        stop = min(t_len, start + chunk)
        prev = np.asarray(b[start - 1 : stop - 1], dtype=float)
        covered = np.isfinite(prev).all(axis=2)
        a = active[start:stop]
        uncovered[start:stop] = np.where(covered, 0.0, np.abs(a)).sum(axis=1)
        clean = np.where(covered[..., None], prev, 0.0)
        x[start:stop] = np.einsum("tnk,tn->tk", clean, a)
        xp[start:stop] = np.einsum("tnk,tn->tk", clean, h[start:stop])
    good_f = np.isfinite(f).all(axis=1)
    factor = np.where(good_f[:, None], x * np.where(good_f[:, None], f, 0.0), 0.0)
    factor[0] = 0.0
    holding_p = (h * r).sum(axis=1)
    benchmark = (hb * r).sum(axis=1)
    specific = (holding_p - benchmark) - factor.sum(axis=1)
    if portfolio_returns is None:
        portfolio = holding_p
        cost = np.zeros(t_len)
        trading = np.zeros(t_len)
    else:
        portfolio = np.asarray(portfolio_returns, dtype=float)
        cost = -np.asarray(cost_rate, dtype=float)
        trading = portfolio - holding_p - cost
    fp = np.where(good_f[:, None], xp * np.where(good_f[:, None], f, 0.0), 0.0)
    fp[0] = 0.0
    specific_portfolio = holding_p - fp.sum(axis=1)
    return {
        "factor": factor,
        "specific": specific,
        # 特质项 = 组合持仓的特质收益 − 基准持仓的特质收益（诊断用）。
        "specific_portfolio": specific_portfolio,
        "specific_benchmark": specific_portfolio - specific,
        "trading": trading,
        "cost": cost,
        "portfolio": portfolio,
        "benchmark": benchmark,
        "active_exposure": x,
        "uncovered": uncovered,
        "missing_return": missing.sum(axis=1),
        "covered_factor_days": good_f,
    }


def attribute(
    result,
    panel,
    risk_model,
    benchmark=None,
    *,
    method="grap",
    start=None,
    end=None,
):
    """一行做收益归因。

    ```python
    import ashare.analysis as an
    report = an.attribute(result, panel, history, benchmark="000300.SH")
    report.groups_cumulative.last()   # 市场、行业、风格、特质、交易、成本的累计贡献
    ```

    result：`panelbt.Result`（须 record_trades=True）或收盘后权重 T×N（`weights[t]` 赚 t+1 日收益）。
    panel：与回测同日期、同证券的面板（`LazyPanel` 或含 close、adj_factor 和基准权重字段的面板）。
    risk_model：`FactorHistory`，或 `ashare.risk.model_series(risk_source(panel)[0], ...)` 交出的模型序列。
    benchmark：指数代码（"000300.SH"）、面板权重字段名、收盘后权重数组，或 None（绝对收益归因）。
    method：多期链接方法，"grap"（默认）或 "carino"，见 `linking`。
    start、end：只分析这一段日期（含两端，"YYYYMMDD"）；默认取风险历史覆盖的区间。
    """
    dates = np.asarray(panel.dates).astype(str)
    symbols = tuple(map(str, panel.symbols))
    returns = total_returns(panel)
    if hasattr(result, "nav"):
        weights, info = result_weights(result, panel)
    else:
        weights, info = np.asarray(result, dtype=float), None
        if weights.shape != returns.shape:
            raise ValueError("权重须与面板同形状 T×N")
    held = held_from_weights(weights)
    held_b = benchmark_held(panel, benchmark, returns)
    if not isinstance(risk_model, FactorHistory):
        needed = np.any(held != 0, axis=0)
        if held_b is not None:
            needed |= np.any(held_b != 0, axis=0)
        source, cfg = risk_source(panel)
        risk_model = FactorHistory.collect(
            risk_model, source, needed, keep_risk=False, config=cfg
        )
    history = risk_model
    rows, cols = history.align(dates, symbols)
    used = np.flatnonzero(cols >= 0)
    outside = np.abs(held).sum(axis=0) + (
        0 if held_b is None else np.abs(held_b).sum(axis=0)
    )
    outside[used] = 0
    # 暴露按面板日期重排；历史里没有的日期为 NaN（该日因子项为 0，全部记特质）。
    k = len(history.factors)
    covered_rows = np.flatnonzero(rows >= 0)
    if not len(covered_rows):
        raise ValueError("风险历史与面板没有共同日期")
    first = int(covered_rows[0]) + 1
    last = int(covered_rows[-1])
    lo = first if start is None else max(first, int(np.searchsorted(dates, start)))
    hi = (
        last
        if end is None
        else min(last, int(np.searchsorted(dates, end, "right")) - 1)
    )
    if hi < lo:
        raise ValueError("分析区间为空：请检查 start、end 与风险历史的日期")
    window = np.arange(lo - 1, hi + 1)  # 多取前一行，供第一天使用 t-1 的暴露

    class _Exposures:
        shape = (len(window), len(used), k)

        def __getitem__(self, s):
            rr = rows[window][s]
            out = np.full((len(rr), len(used), k), np.nan)
            ok = rr >= 0
            out[ok] = history.exposures[rr[ok]][:, cols[used]]
            return out

    f = np.full((len(window), k), np.nan)
    ok = rows[window] >= 0
    f[ok] = history.factor_returns[rows[window][ok]]
    hp = held[window][:, used]
    hb = None if held_b is None else held_b[window][:, used]
    rr = returns[window][:, used]
    # 不在历史列里的证券：权重仍计入组合与基准的持仓收益，收益全部记特质项。
    extra = np.flatnonzero((cols < 0) & (outside > 0))
    if len(extra):
        hp = np.hstack([hp, held[window][:, extra]])
        hb = None if hb is None else np.hstack([hb, held_b[window][:, extra]])
        rr = np.hstack([rr, returns[window][:, extra]])
        base = _Exposures()

        class _Padded:
            shape = (len(window), len(used) + len(extra), k)

            def __getitem__(self, s):
                part = base[s]
                pad = np.full((part.shape[0], len(extra), k), np.nan)
                return np.concatenate([part, pad], axis=1)

        exposures = _Padded()
    else:
        exposures = _Exposures()
    port = None if info is None else info["returns"][window]
    cost = None if info is None else info["cost_rate"][window]
    out = decompose(hp, hb, rr, exposures, f, port, cost)
    # 去掉多取的前一行。
    for key in out:
        out[key] = out[key][1:]
    days = dates[window[1:]]
    return _report(out, days, history, method, len(extra))


def _report(out, days, history, method, n_extra):
    factors, groups = history.factors, history.groups
    rp, rb = out["portfolio"], out["benchmark"]
    detail = np.column_stack(
        [out["factor"], out["specific"], out["trading"], out["cost"]]
    )
    names = factors + ITEMS
    group_values = np.column_stack(
        [
            out["factor"][:, [g == name for g in groups]].sum(axis=1)
            for name in ("market", "industry", "style")
        ]
        + [out["specific"], out["trading"], out["cost"]]
    )
    linked = link(detail, rp, rb, method)
    linked_groups = link(group_values, rp, rb, method)
    target = cumulative_active(rp, rb)
    active = rp - rb
    scale = np.maximum(1.0, np.abs(detail).sum(axis=1))
    daily_error = np.abs(detail.sum(axis=1) - active)
    cum_scale = np.maximum(1.0, np.abs(linked).sum(axis=1))
    cum_error = np.abs(linked.sum(axis=1) - target)
    group_cum_error = np.abs(linked_groups.sum(axis=1) - target)
    summary = {
        "dates": (str(days[0]), str(days[-1])),
        "days": int(len(days)),
        "method": method,
        "portfolio_cumulative": float(np.prod(1 + rp) - 1),
        "benchmark_cumulative": float(np.prod(1 + rb) - 1),
        "active_cumulative": float(target[-1]),
        "groups": {
            c: float(v) for c, v in zip(GROUP_COLUMNS, linked_groups[-1], strict=True)
        },
        "identity": {
            "daily_max_abs": float(daily_error.max()),
            "daily_max_eps": float((daily_error / scale).max() / EPS),
            "cumulative_max_abs": float(cum_error.max()),
            "cumulative_max_eps": float((cum_error / cum_scale).max() / EPS),
            "groups_cumulative_max_abs": float(group_cum_error.max()),
        },
    }
    diagnostics = {
        "uncovered_weight": out["uncovered"],
        "missing_return_count": out["missing_return"],
        "days_without_factor_returns": int((~out["covered_factor_days"]).sum()),
        "columns_outside_history": n_extra,
        # 特质项拆成组合侧与基准侧的逐日值（单利相加，不链接）。
        "specific_portfolio": out["specific_portfolio"],
        "specific_benchmark": out["specific_benchmark"],
    }
    return Attribution(
        daily=Table(days, names, detail, "daily", "date"),
        cumulative=Table(days, names, linked, "cumulative", "date"),
        groups=Table(days, GROUP_COLUMNS, group_values, "groups", "date"),
        groups_cumulative=Table(
            days, GROUP_COLUMNS, linked_groups, "groups_cumulative", "date"
        ),
        exposures=Table(
            days, factors, out["active_exposure"], "active_exposure", "date"
        ),
        returns=Table(
            days,
            ("portfolio", "benchmark", "active"),
            np.column_stack([rp, rb, rp - rb]),
            "returns",
            "date",
        ),
        summary=summary,
        diagnostics=diagnostics,
    )
