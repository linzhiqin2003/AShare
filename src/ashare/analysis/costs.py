"""成本拆解：把 `panelbt.Result` 的佣金、印花税、其他费用（过户费）、滑点、冲击按年列出。

每一项按「当日成本 ÷ 当日收益的分母（E[t-1] + 资金流）」折成收益率再按年求和，
这是该项对当年收益的拖累（单利口径，便于相加；复利差异在成本很小时可以忽略）。
`gross` = 净收益 + 总成本，是「没有这些成本时」的近似收益；`cost_share` = 总成本 / gross。

与 `ashare.report` 的口径：
- `turnover` 与报告相同是**双边**（引擎单边换手 × 2，即成交额 ÷ 交易前权益），按年求和；
  `turnover_one_sided` 是引擎的单边值。
- 报告的年度拆解把全年费用除以上年末权益；这里逐日除以前一日权益再相加。两者在年内
  净值变化不大时几乎相同，大涨大跌的年份这里更准确地反映「每一笔费用占当时资金的比例」。
  报告把滑点和冲击合在 `slippage` 里，这里分开列。
- `all` 行按 252 个交易日年化：成本与换手乘 252 / 天数，净收益按几何年化。
"""

from __future__ import annotations

import numpy as np

from .table import Table

ITEMS = (
    ("commission", "commission"),
    ("stamp_tax", "tax"),
    ("other_fee", "other_fee"),
    ("slippage", "slippage_cost"),
    ("impact", "impact_cost"),
    ("borrow", "borrow_cost"),
)


def cost_breakdown(result, dates, periods_per_year=252, start=None):
    """按年的成本表；最后一行 `all` 是全期按年折算的值。

    ```python
    table = an.cost_breakdown(result, panel.dates)
    table.to_dict()      # 各年的净收益、各项成本、总成本、gross、cost_share、turnover
    ```
    """
    dates = np.asarray(dates).astype(str)
    equity = np.asarray(result.equity, dtype=float)
    returns = np.asarray(result.returns, dtype=float)
    if equity.ndim != 1:
        raise ValueError("只支持单策略 Result")
    if len(dates) != len(equity):
        raise ValueError("dates 与 Result 长度不一致")
    flow = np.asarray(result.cash_flow, dtype=float)
    denom = np.empty_like(equity)
    denom[1:] = equity[:-1] + flow[1:]
    denom[0] = equity[0] / (1 + returns[0]) if returns[0] != -1 else np.nan
    rates = []
    for _, attr in ITEMS:
        v = np.asarray(getattr(result, attr), dtype=float)
        rates.append(np.divide(v, denom, out=np.zeros_like(v), where=denom != 0))
    rates = np.column_stack(rates)
    turnover = np.asarray(result.turnover, dtype=float)
    if start is not None:
        # 只统计 start 及以后（例如去掉策略开始前的空仓期，免得拉低年化）。
        keep = dates >= str(start)
        dates, returns, rates, turnover = (
            dates[keep],
            returns[keep],
            rates[keep],
            turnover[keep],
        )
    years = np.array([d[:4] for d in dates])
    labels = list(dict.fromkeys(years))
    columns = (
        ("net_return",)
        + tuple(name for name, _ in ITEMS)
        + (
            "total_cost",
            "gross",
            "cost_share",
            "turnover",
            "turnover_one_sided",
            "days",
        )
    )
    rows = []

    def row(mask, scale=1.0):
        net = np.prod(1 + returns[mask]) - 1
        items = rates[mask].sum(axis=0) * scale
        total = items.sum()
        net_value = (1 + net) ** scale - 1 if scale != 1.0 else net
        gross = net_value + total
        share = total / gross if gross > 0 else np.nan
        return [
            net_value,
            *items,
            total,
            gross,
            share,
            2 * turnover[mask].sum() * scale,
            turnover[mask].sum() * scale,
            mask.sum(),
        ]

    for y in labels:
        rows.append(row(years == y))
    every = np.ones(len(dates), bool)
    rows.append(row(every, periods_per_year / len(dates)))
    return Table(
        np.array(labels + ["all"]), columns, np.array(rows), "cost_breakdown", "year"
    )
