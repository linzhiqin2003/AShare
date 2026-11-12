"""真实数据例子：股票、国债、黄金、商品四只 ETF 的四种配置，月度调仓，`panelbt` 带费用回测。

    python bench/allocation/etf_example.py --root /path/to/funds-root [--out 目录]

读 `ashare.data.open(root).funds()` 的基金面板（先 sync-funds、build-funds）。输出每种配置的报告数据
（`<out>/<名字>.json`，可用 `python -m ashare.report show` 查看）和一张汇总表（`summary.md`）。
这不是投资建议，只讨论回测。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ashare.allocation as aa
import ashare.data as ad
from ashare.data.fund_rules import fund_market

ASSETS = {
    "510300.SH": "沪深300ETF（股票）",
    "511010.SH": "国债ETF（国债）",
    "518880.SH": "黄金ETF（黄金）",
    "159985.SZ": "豆粕ETF（商品）",
}
METHODS = {
    "equal": aa.EqualWeight(),
    "fixed_60_40": aa.FixedWeights({"510300.SH": 0.6, "511010.SH": 0.4}),
    "risk_parity": aa.RiskParity(),
    # 风险平价的事前波动只有 3% 左右：目标 6%、杠杆上限 2 倍，融资年利率按 3% 计（保证金账户）。
    "risk_parity_vol6": aa.TargetVolatility(
        aa.RiskParity(), target=0.06, max_leverage=2.0
    ),
}
FINANCING = 0.03  # 年利率，按 252 日折成日利率


def adjusted_returns(panel):
    """后复权收益：close × adj_factor 的环比；缺行情的日子为 NaN（不跨缺口拼接）。"""
    price = np.asarray(panel["close"]) * np.asarray(panel["adj_factor"])
    price = np.where(np.asarray(panel["has_quote"]), price, np.nan)
    out = np.full(price.shape, np.nan)
    out[1:] = price[1:] / price[:-1] - 1
    return out


def window_stats(ts, start, end=None):
    """报告数据里的逐日净值 → 区间统计（年化收益、年化波动、夏普（无风险 0）、最大回撤、年化双边换手）。"""
    daily = ts.data["daily"]
    dates = np.array(daily["date"])
    nav = np.array(daily["nav"], dtype=float)
    turnover = np.array(daily["turnover"], dtype=float)
    keep = (dates >= start) & (dates <= (end or dates[-1]))
    nav, turnover = nav[keep], turnover[keep]
    r = nav[1:] / nav[:-1] - 1
    years = len(r) / 252
    annual = (nav[-1] / nav[0]) ** (1 / years) - 1
    vol = float(np.std(r, ddof=1) * np.sqrt(252))
    peak = np.maximum.accumulate(nav)
    return dict(
        start=str(dates[keep][0]),
        end=str(dates[keep][-1]),
        annual_return=float(annual),
        volatility=vol,
        sharpe=float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)),
        max_drawdown=float((nav / peak - 1).min()),
        turnover=float(turnover[1:].sum() * 2 / years),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", default="2015-01-05")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    funds = ad.open(args.root).funds()
    out = Path(args.out or Path(args.root) / "cn" / "reports" / "allocation")
    out.mkdir(parents=True, exist_ok=True)
    symbols = list(ASSETS)
    fm = fund_market(funds, symbols, args.start)
    panel = funds.panel(
        "fund", ["close", "adj_factor", "has_quote"], start=args.start, symbols=symbols
    )
    returns = adjusted_returns(panel)
    dates = [str(d) for d in panel.dates]
    costs = fm.costs(commission_rate=1e-4, commission_min=0.0, slippage=5e-4)
    config = fm.config(initial_cash=1e6, record_trades=True)
    from dataclasses import replace

    import panelbt as bt

    levered = replace(
        config,
        margin=bt.MarginModel(
            allow_financing=True,
            financing_rate=FINANCING / 252,
            long_initial=0.5,
            long_maintenance=0.25,
            margin_call="record",
        ),
    )
    rows = []
    for name, method in METHODS.items():
        started = time.perf_counter()
        res = aa.allocate(
            returns,
            method,
            dates=dates,
            symbols=symbols,
            rebalance="monthly",
            lookback=252,
            min_history=120,
        )
        used = (
            levered
            if isinstance(method, aa.TargetVolatility) and method.max_leverage > 1
            else config
        )
        ts = aa.tearsheet(
            res,
            fm.market,
            costs=costs,
            config=used,
            title=f"资产配置：{name}",
            names=ASSETS,
        )
        ts.save_json(out / f"{name}.json")
        (out / f"{name}.weights.json").write_text(
            json.dumps(res.to_dict(), ensure_ascii=False)
        )
        rows.append((name, ts, res, time.perf_counter() - started))
        print(name, f"{time.perf_counter() - started:.1f}s")
    lines = [
        "| 配置 | 区间 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 年化双边换手 |",
        "|---|---|---|---|---|---|---|",
    ]
    stats = {}
    for label, begin in (
        ("全样本", args.start.replace("-", "")),
        ("共同区间", "2021-01-04"),
    ):
        for name, ts, _, _ in rows:
            begin_iso = (
                begin if "-" in begin else f"{begin[:4]}-{begin[4:6]}-{begin[6:]}"
            )
            m = window_stats(ts, begin_iso)
            stats[f"{label}/{name}"] = m
            lines.append(
                f"| {name}（{label}） | {m['start']}–{m['end']} | {m['annual_return']:.2%} | {m['volatility']:.2%} | "
                f"{m['sharpe']:.2f} | {m['max_drawdown']:.2%} | {m['turnover']:.0%} |"
            )
    (out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
