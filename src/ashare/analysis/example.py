"""真实数据上的完整例子：月度调仓的两个策略，跑全部五项分析。

    python -m ashare.analysis.example --start 20130104 --analysis-start 20150105
    python -m ashare.analysis.example --strategy momentum      # 中证 500 动量，对照偏差统计

策略（`--strategy`）：
- `lowvol`（默认）：沪深 300 成分里此前 60 个交易日日收益标准差最小的 30 只，等权 99%，
  基准沪深 300；
- `momentum`：中证 500 成分里 12-1 月动量（t-252 到 t-21 的后复权涨幅）最高的 50 只，
  等权 92%，基准中证 500。
信号在每月最后一个交易日收盘后算出，用 `ashare.backtest.execution_targets` 移到下一交易日
（每月第一个交易日）开盘成交；行情、账户规则、费用表都用 `ashare.backtest.market_from_panel`。
风险模型从 `--start` 起建（暴露的描述子要一年以上历史），归因从 `--analysis-start` 起算。
结果以 Markdown 表格打印，`--json` 另存为一份只含基本类型的 JSON。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ashare.data
from ashare.backtest import execution_targets, market_from_panel

from . import (
    FactorHistory,
    LazyPanel,
    attribute,
    brinson,
    capacity,
    cost_breakdown,
    format_table,
    held_columns,
    risk_series,
    total_returns,
)
from .table import Table

STRATEGIES = {
    "lowvol": {"benchmark": "000300.SH", "weights": "weight_csi300", "cash": 1e7},
    "momentum": {"benchmark": "000905.SH", "weights": "weight_csi500", "cash": 3e7},
}


def _month_ends(dates, first_day):
    """每月最后一个交易日的行号（信号日），其下一交易日不早于 first_day。"""
    months = np.array([d[:6] for d in dates])
    ends = np.flatnonzero(np.r_[months[1:] != months[:-1], False])
    return ends[ends + 1 >= first_day]


def _quoted(panel):
    return np.asarray(panel["has_quote"], bool) & ~np.asarray(
        panel["is_suspended"], bool
    )


def low_vol_signal(panel, first_day, count=30, window=60, invest=0.99):
    """信号日（月末）上的稠密目标，整行 NaN 表示当日不出信号。

    波动率只用有行情、未停牌的交易日的收益（停牌日价格不动，算进去会把长期停牌的股票
    当成「低波动」选进来）；信号日当天停牌或无行情的股票不选。"""
    dates = np.asarray(panel.dates).astype(str)
    quoted = _quoted(panel)
    r = np.where(quoted, total_returns(panel), np.nan)
    member = np.asarray(panel["member_csi300"], bool)
    t, n = r.shape
    dense = np.full((t, n), np.nan)
    for signal in _month_ends(dates, first_day):
        if signal < window:
            continue
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
        dense[signal] = 0.0
        dense[signal, pick] = invest / len(pick)
    return dense


def momentum_signal(panel, first_day, count=50, invest=0.92):
    """中证 500 成分里 12-1 月动量最高的 count 只（信号日为月末）。"""
    dates = np.asarray(panel.dates).astype(str)
    quoted = _quoted(panel)
    r = np.nan_to_num(total_returns(panel))
    px = np.cumprod(1 + r, axis=0)
    member = np.asarray(panel["member_csi500"], bool)
    t, n = r.shape
    dense = np.full((t, n), np.nan)
    for signal in _month_ends(dates, first_day):
        if signal < 252:
            continue
        with np.errstate(all="ignore"):
            mom = px[signal - 21] / px[signal - 252] - 1
        ok = member[signal] & quoted[signal] & np.isfinite(mom)
        score = np.where(ok, mom, -np.inf)
        pick = np.argsort(-score, kind="stable")[:count]
        pick = pick[np.isfinite(score[pick])]
        dense[signal] = 0.0
        dense[signal, pick] = invest / len(pick)
    return dense


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=None, help="数据根目录（默认 ASHARE_DATA_DIR 或仓库 data/）"
    )
    parser.add_argument("--start", default="20130104")
    parser.add_argument("--analysis-start", default="20150105")
    parser.add_argument("--strategy", default="lowvol", choices=tuple(STRATEGIES))
    parser.add_argument("--cache", default=".cache/analysis-risk")
    parser.add_argument("--json", default=None)
    parser.add_argument("--aum", default="1e7,1e8,5e8,1e9,3e9,1e10,3e10")
    args = parser.parse_args(argv)
    spec = STRATEGIES[args.strategy]
    bench_code = spec["benchmark"]
    clock = {}
    t0 = time.perf_counter()
    import panelbt

    data = ashare.data.open(args.root) if args.root else ashare.data.open()
    panel = LazyPanel(data, start=args.start)
    dates = np.asarray(panel.dates).astype(str)
    first = int(np.searchsorted(dates, args.analysis_start))
    make = low_vol_signal if args.strategy == "lowvol" else momentum_signal
    targets = execution_targets(make(panel, first))
    bm = market_from_panel(data, start=args.start)
    if tuple(map(str, bm.symbols)) != panel.symbols:
        raise RuntimeError("行情与分析面板的证券不一致")
    config = bm.config(record_trades=True, initial_cash=spec["cash"])
    result = panelbt.run(bm.market, targets, bm.costs(), config)
    clock["backtest"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    weights = np.nan_to_num(np.asarray(result.positions)) != 0
    bench = np.asarray(panel[spec["weights"]])
    columns = held_columns(weights, bench, symbols=panel.symbols)
    history = FactorHistory.build(panel, columns=columns, cache=args.cache)
    clock["risk_history"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    report = attribute(
        result, panel, history, benchmark=bench_code, start=args.analysis_start
    )
    clock["attribution"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    risk = risk_series(
        result, panel, history, benchmark=bench_code, start=args.analysis_start
    )
    clock["risk_series"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    sectors = brinson(result, panel, bench_code, start=args.analysis_start)
    clock["brinson"] = time.perf_counter() - t4
    t5 = time.perf_counter()
    aum = [float(x) for x in args.aum.split(",")]
    cap = capacity(
        targets,
        panel,
        aum=aum,
        benchmark=bench_code,
        rerun=True,
        market=bm.market,
        config=bm.config(max_participation=0.1),
        costs=bm.costs(),
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
        # 逐日单利相加：停牌持仓、未覆盖证券在特质项里的部分。
        "suspended": float(diag["suspended_specific"].sum()),
        "uncovered": float(diag["uncovered_specific"].sum()),
        "uncovered_weight_mean": float(diag["uncovered_weight"].mean()),
    }
    print("特质项诊断", out["attribution"]["specific_split"])
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
