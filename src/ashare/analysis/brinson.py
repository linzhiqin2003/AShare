"""Brinson 行业归因：配置、选择、交互三项，多期链接。

单期（行业 s，组合与基准的行业权重 Wp、Wb，行业收益 Rp_s、Rb_s，基准总收益 Rb）：

    Brinson-Fachler（默认）：配置 = (Wp − Wb)(Rb_s − Rb)
    BHB（Brinson-Hood-Beebower）：配置 = (Wp − Wb) Rb_s
    两者相同：选择 = Wb (Rp_s − Rb_s)；交互 = (Wp − Wb)(Rp_s − Rb_s)

各行业三项之和 = 组合持仓收益 − 基准收益。两种口径的总配置相同（因为 Σ(Wp − Wb) = 0），
差别只在配置效应怎么分到各行业：BF 衡量「超配跑赢基准整体的行业」，BHB 衡量「超配涨的行业」。

约定：
- 现金单列为一个行业（收益 0），这样组合与基准的行业权重之和都为 1，恒等式在有现金时仍成立。
  现金收益在两边都确定为 0，所以现金的拖累全部记在配置项（BF）；BHB 下现金的配置项为 0，
  拖累体现在其他行业的配置项之和里。
- 组合在某行业没有持仓时取 Rp_s = Rb_s（选择与交互为 0）；基准没有该行业时取 Rb_s = Rb
  （BF 配置为 0，超额全部记交互）。
- 第 t 日的行业分类与权重都取 t-1 收盘的值。
- 给 `panelbt.Result` 时另有交易与成本两项（含义同 `attribute`），五项之和 = Rp − Rb。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inputs import (
    benchmark_held,
    held_from_weights,
    industry_codes,
    result_weights,
    total_returns,
)
from .linking import cumulative_active, link
from .table import Table, plain

EFFECTS = ("allocation", "selection", "interaction")
EPS = np.finfo(float).eps


def brinson_effects(wp, wb, rp, rb, method="fachler"):
    """单期（或逐期向量化）的行业三项。

    wp、wb、rp、rb：形状 (..., S) 的行业权重与行业收益（rp、rb 在权重为 0 处可为 NaN）。
    返回 (allocation, selection, interaction)，形状同输入。
    """
    wp, wb = np.asarray(wp, dtype=float), np.asarray(wb, dtype=float)
    rp, rb = np.asarray(rp, dtype=float), np.asarray(rb, dtype=float)
    total_b = np.sum(wb * np.where(wb != 0, rb, 0.0), axis=-1, keepdims=True)
    rb_s = np.where(wb != 0, rb, total_b)
    rp_s = np.where(wp != 0, rp, rb_s)
    diff = wp - wb
    if method == "fachler":
        allocation = diff * (rb_s - total_b)
    elif method == "bhb":
        allocation = diff * rb_s
    else:
        raise ValueError("method: 只支持 'fachler' 或 'bhb'")
    selection = wb * (rp_s - rb_s)
    interaction = diff * (rp_s - rb_s)
    return allocation, selection, interaction


@dataclass
class Brinson:
    daily: Table  # 配置、选择、交互、交易、成本
    cumulative: Table
    sectors: Table  # 每个行业的平均权重与累计三项
    summary: dict

    def to_dict(self):
        return {
            "summary": plain(self.summary),
            "daily": self.daily.to_dict(),
            "cumulative": self.cumulative.to_dict(),
            "sectors": self.sectors.to_dict(),
        }


def sector_aggregate(held, returns, sector, n_sectors):
    """逐日按行业汇总：行业权重 W（T×S）与行业贡献 C = Σ w r（T×S）。最后一列是现金。"""
    t_len = held.shape[0]
    s_all = n_sectors + 1
    flat = (np.arange(t_len)[:, None] * s_all + sector).ravel()
    w = np.bincount(flat, held.ravel(), minlength=t_len * s_all).reshape(t_len, s_all)
    c = np.bincount(flat, (held * returns).ravel(), minlength=t_len * s_all).reshape(
        t_len, s_all
    )
    w[:, -1] = 1.0 - w[:, :-1].sum(axis=1)
    c[:, -1] = 0.0
    return w, c


def brinson(
    result,
    panel,
    benchmark,
    *,
    method="fachler",
    link_method="grap",
    sectors=None,
    start=None,
    end=None,
):
    """一行做 Brinson 行业归因。

    ```python
    b = an.brinson(result, panel, benchmark="000300.SH")
    b.cumulative.last()       # 累计配置、选择、交互、交易、成本
    ```
    result：`panelbt.Result`（record_trades=True）或收盘后权重 T×N。
    sectors：(T×N 整数类别, 类别名)；默认用面板的申万一级行业（`industry` 字段）。
    """
    dates = np.asarray(panel.dates).astype(str)
    returns = total_returns(panel)
    if hasattr(result, "nav"):
        weights, info = result_weights(result, panel)
    else:
        weights, info = np.asarray(result, dtype=float), None
    held = held_from_weights(weights)
    held_b = benchmark_held(panel, benchmark, returns)
    if held_b is None:
        raise ValueError("Brinson 归因需要基准")
    codes, names = industry_codes(panel) if sectors is None else sectors
    codes = np.asarray(codes, dtype=np.int64)
    prev = np.zeros_like(codes)
    prev[1:] = codes[:-1]  # 第 t 日用 t-1 收盘的分类
    r = np.nan_to_num(returns)
    lo = 1 if start is None else max(1, int(np.searchsorted(dates, start)))
    hi = len(dates) if end is None else int(np.searchsorted(dates, end, "right"))
    rows = np.arange(lo, hi)
    wp, cp = sector_aggregate(held[rows], r[rows], prev[rows], len(names))
    wb, cb = sector_aggregate(held_b[rows], r[rows], prev[rows], len(names))
    with np.errstate(invalid="ignore", divide="ignore"):
        rp_s = np.where(wp != 0, cp / wp, np.nan)
        rb_s = np.where(wb != 0, cb / wb, np.nan)
    alloc, sel, inter = brinson_effects(wp, wb, rp_s, rb_s, method)
    holding_p, rb = cp.sum(axis=1), cb.sum(axis=1)
    # 现金的收益在两边都确定是 0，不套用「基准没有该行业时取 Rb_s = Rb」的约定：
    # 现金的拖累整体记在配置项里（BF：(Wp − Wb)(0 − Rb)；BHB：0），选择与交互为 0。
    # 这样恒等式不变（现金一列仍是 Wp·0 − Wb·0 − (Wp − Wb)·Rb）。
    cash_diff = wp[:, -1] - wb[:, -1]
    alloc[:, -1] = cash_diff * (0.0 - rb) if method == "fachler" else 0.0
    sel[:, -1] = 0.0
    inter[:, -1] = 0.0
    if info is None:
        rp_total = holding_p
        cost = np.zeros(len(rows))
        trading = np.zeros(len(rows))
    else:
        rp_total = info["returns"][rows]
        cost = -info["cost_rate"][rows]
        trading = rp_total - holding_p - cost
    daily = np.column_stack(
        [alloc.sum(axis=1), sel.sum(axis=1), inter.sum(axis=1), trading, cost]
    )
    cum = link(daily, rp_total, rb, link_method)
    by_sector = [link(e, rp_total, rb, link_method)[-1] for e in (alloc, sel, inter)]
    sector_names = tuple(names) + ("cash",)
    sector_values = np.column_stack(
        [wp.mean(axis=0), wb.mean(axis=0), *by_sector, sum(by_sector)]
    )
    target = cumulative_active(rp_total, rb)
    columns = EFFECTS + ("trading", "cost")
    daily_err = np.abs(daily.sum(axis=1) - (rp_total - rb))
    cum_err = np.abs(cum.sum(axis=1) - target)
    summary = {
        "dates": (str(dates[rows[0]]), str(dates[rows[-1]])),
        "method": method,
        "link": link_method,
        "portfolio_cumulative": float(np.prod(1 + rp_total) - 1),
        "benchmark_cumulative": float(np.prod(1 + rb) - 1),
        "active_cumulative": float(target[-1]),
        "effects": {c: float(v) for c, v in zip(columns, cum[-1], strict=True)},
        "identity": {
            "daily_max_abs": float(daily_err.max()),
            "cumulative_max_abs": float(cum_err.max()),
            "cumulative_max_eps": float(
                (cum_err / np.maximum(1.0, np.abs(cum).sum(axis=1))).max() / EPS
            ),
            "sector_vs_total_max_abs": float(
                np.abs(sector_values[:, 2:5].sum(axis=0) - cum[-1, :3]).max()
            ),
        },
    }
    return Brinson(
        Table(dates[rows], columns, daily, "brinson_daily", "date"),
        Table(dates[rows], columns, cum, "brinson_cumulative", "date"),
        Table(
            np.array(sector_names),
            ("portfolio_weight", "benchmark_weight") + EFFECTS + ("total",),
            sector_values,
            "brinson_sectors",
            "sector",
        ),
        summary,
    )
