"""真实数据上的完整例子：沪深 300 内低波动 30 只、月度调仓，跑全部五项分析。

    python -m ashare.analysis.example --start 20130104 --analysis-start 20150105

策略：每月第一个交易日开盘成交；信号用前一交易日收盘及以前的数据——当天在沪深 300
成分里、此前 60 个交易日日收益标准差最小的 30 只，等权 99%（留 1% 现金付费用）。
风险模型从 `--start` 起建（暴露的描述子要一年以上历史），归因从 `--analysis-start` 起算。
结果以 Markdown 表格打印，`--json` 另存为一份只含基本类型的 JSON。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ashare.data

from . import (
    FactorHistory,
    LazyPanel,
    ashare_config,
    attribute,
    brinson,
    capacity,
    cost_breakdown,
    default_costs,
    format_table,
    held_columns,
    market_from_panel,
    risk_series,
    total_returns,
)
from .table import Table


def low_vol_targets(panel, first_day, count=30, window=60, invest=0.99):
    """月度调仓的稠密目标（整行 NaN 不调仓），行号是成交日。

    波动率只用有行情、未停牌的交易日的收益（停牌日价格不动，算进去会把长期停牌的股票
    当成「低波动」选进来）；信号日当天停牌或无行情的股票不选。"""
    dates = np.asarray(panel.dates).astype(str)
    r = total_returns(panel)
    quoted = np.asarray(panel["has_quote"], bool) & ~np.asarray(
        panel["is_suspended"], bool
    )
    r = np.where(quoted, r, np.nan)
    member = np.asarray(panel["member_csi300"], bool)
    t, n = r.shape
    dense = np.full((t, n), np.nan)
    months = np.array([d[:6] for d in dates])
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    for day in starts[starts >= max(first_day, window + 1)]:
        signal = day - 1  # 前一交易日收盘
        hist = r[signal - window + 1 : signal + 1]
        ok = (
            member[signal]
            & quoted[signal]
            & (np.isfinite(hist).sum(axis=0) >= window * 0.8)
        )
        vol = np.full(n, np.inf)
        vol[ok] = np.nanstd(hist[:, ok], axis=0)
        pick = np.argsort(vol, kind="stable")[:count]
        pick = pick[np.isfinite(vol[pick])]
        dense[day] = 0.0
        dense[day, pick] = invest / len(pick)
    return dense


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=None, help="数据根目录（默认 ASHARE_DATA_DIR 或仓库 data/）"
    )
    parser.add_argument("--start", default="20130104")
    parser.add_argument("--analysis-start", default="20150105")
    parser.add_argument("--cache", default=".cache/analysis-risk")
    parser.add_argument("--json", default=None)
    parser.add_argument("--aum", default="1e7,1e8,5e8,1e9,3e9,1e10,3e10")
    args = parser.parse_args(argv)
    clock = {}
    t0 = time.perf_counter()
    import panelbt

    data = ashare.data.open(args.root) if args.root else ashare.data.open()
    panel = LazyPanel(data, start=args.start)
    dates = np.asarray(panel.dates).astype(str)
    first = int(np.searchsorted(dates, args.analysis_start))
    dense = low_vol_targets(panel, first)
    targets = panelbt.Targets.from_dense(dense)
    market = market_from_panel(panel)
    config = ashare_config(panel.symbols, record_trades=True, initial_cash=1e7)
    result = panelbt.run(market, targets, default_costs(dates), config)
    clock["backtest"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    weights = np.nan_to_num(np.asarray(result.positions)) != 0
    bench = np.asarray(panel["weight_csi300"])
    columns = held_columns(weights, bench, symbols=panel.symbols)
    history = FactorHistory.build(panel, columns=columns, cache=args.cache)
    clock["risk_history"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    report = attribute(
        result, panel, history, benchmark="000300.SH", start=args.analysis_start
    )
    clock["attribution"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    risk = risk_series(
        result, panel, history, benchmark="000300.SH", start=args.analysis_start
    )
    clock["risk_series"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    sectors = brinson(result, panel, "000300.SH", start=args.analysis_start)
    clock["brinson"] = time.perf_counter() - t4
    t5 = time.perf_counter()
    aum = [float(x) for x in args.aum.split(",")]
    cap = capacity(
        targets, panel, aum=aum, benchmark="000300.SH", rerun=True, market=market
    )
    clock["capacity"] = time.perf_counter() - t5
    costs = cost_breakdown(result, dates, start=args.analysis_start)

    out = {
        "timing_seconds": clock,
        "attribution": report.to_dict(),
        "risk": risk.to_dict(),
        "brinson": sectors.to_dict(),
        "capacity": cap.to_dict(),
        "costs": costs.to_dict(),
    }
    print("## 收益归因（累计，四类 + 交易 + 成本）")
    print(json.dumps(report.summary, ensure_ascii=False, indent=1, default=float))
    diag = report.diagnostics
    print(
        "特质项拆分（逐日单利相加）：组合侧",
        float(diag["specific_portfolio"].sum()),
        "基准侧",
        float(diag["specific_benchmark"].sum()),
    )
    out["attribution"]["specific_split"] = {
        "portfolio": float(diag["specific_portfolio"].sum()),
        "benchmark": float(diag["specific_benchmark"].sum()),
    }
    yearly = _yearly(report)
    print(format_table(yearly, 4))
    print("\n## 平均主动暴露")
    exp = report.exposures
    mean = exp.values.mean(axis=0)
    order = np.argsort(-np.abs(mean))[:12]
    print(
        format_table(
            Table(np.array(exp.columns)[order], ("mean_active_exposure",), mean[order]),
            3,
        )
    )
    print("\n## 风险归因（日频跟踪误差的平均占比）与偏差统计")
    pred = risk.predicted
    print(
        format_table(
            Table(np.array(["mean"]), pred.columns, pred.values.mean(axis=0)[None, :]),
            4,
        )
    )
    print({k: v for k, v in risk.bias.items() if k != "rolling"})
    print("\n## Brinson（累计）")
    print(
        format_table(
            Table(
                np.array(["cumulative"]),
                sectors.cumulative.columns,
                sectors.cumulative.values[-1:],
            ),
            4,
        )
    )
    print(json.dumps(sectors.summary["identity"], indent=1))
    print("\n## 容量")
    print(format_table(cap.table, 4))
    print(json.dumps(cap.capacity, indent=1, default=float))
    print(format_table(cap.liquidation, 2))
    print(format_table(cap.rerun, 4))
    print("\n## Brinson 行业表")
    print(format_table(sectors.sectors, 4))
    print("\n## 成本拆解")
    print(format_table(costs, 4))
    print("\n计时（秒）", clock)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, allow_nan=False)
    return out


def _yearly(report):
    """按年链接的四类贡献（每年单独从 0 开始链接）。"""
    from .linking import link

    days = report.groups.index
    years = np.array([d[:4] for d in days])
    rows, labels = [], []
    rp, rb = report.returns["portfolio"], report.returns["benchmark"]
    for y in dict.fromkeys(years):
        m = years == y
        linked = link(report.groups.values[m], rp[m], rb[m])
        rows.append(np.r_[linked[-1], np.prod(1 + rp[m]) - np.prod(1 + rb[m])])
        labels.append(y)
    return Table(
        np.array(labels),
        report.groups.columns + ("active",),
        np.array(rows),
        "yearly",
        "year",
    )


if __name__ == "__main__":
    main()
