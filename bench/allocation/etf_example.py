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
RISK_FREE = "511880.SH"  # 货币 ETF：超额收益夏普的无风险利率
COMMON = "2021-01-04"  # 四只都满足历史要求的共同区间起点


def adjusted_returns(panel):
    """后复权收益：close × adj_factor 的环比；缺行情的日子为 NaN（不跨缺口拼接）。"""
    price = np.asarray(panel["close"]) * np.asarray(panel["adj_factor"])
    price = np.where(np.asarray(panel["has_quote"]), price, np.nan)
    out = np.full(price.shape, np.nan)
    out[1:] = price[1:] / price[:-1] - 1
    return out


def window_stats(ts, start, risk_free, end=None):
    """报告数据里的逐日净值 → 区间统计。

    年化收益、年化波动、夏普（无风险利率 0）、超额夏普（扣货币 ETF 的日收益）、最大回撤、年化双边换手
    （报告里的 turnover 已经是双边，不再乘 2）。
    """
    daily = ts.data["daily"]
    dates = np.array(daily["date"])
    nav = np.array(daily["nav"], dtype=float)
    turnover = np.array(daily["turnover"], dtype=float)
    keep = (dates >= start) & (dates <= (end or dates[-1]))
    nav, turnover, kept = nav[keep], turnover[keep], dates[keep]
    r = nav[1:] / nav[:-1] - 1
    rf = np.array([risk_free.get(d, 0.0) for d in kept[1:]])
    excess = r - rf
    years = len(r) / 252
    annual = (nav[-1] / nav[0]) ** (1 / years) - 1
    vol = float(np.std(r, ddof=1) * np.sqrt(252))
    peak = np.maximum.accumulate(nav)
    return dict(
        start=str(kept[0]),
        end=str(kept[-1]),
        annual_return=float(annual),
        volatility=vol,
        sharpe=float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)),
        excess_sharpe=float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)),
        risk_free_annual=float(np.prod(1 + rf) ** (1 / years) - 1),
        max_drawdown=float((nav / peak - 1).min()),
        turnover=float(turnover[1:].sum() / years),
    )


def margin(config, rate):
    from dataclasses import replace

    import panelbt as bt

    return replace(
        config,
        margin=bt.MarginModel(
            allow_financing=True,
            financing_rate=rate / 252,
            long_initial=0.5,
            long_maintenance=0.25,
            margin_call="record",
        ),
    )


def main(argv=None):
    from dataclasses import replace

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
    iso = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]
    money = adjusted_returns(
        funds.panel(
            "fund",
            ["close", "adj_factor", "has_quote"],
            start=args.start,
            symbols=[RISK_FREE],
        )
    )[:, 0]
    risk_free = {d: float(v) for d, v in zip(iso, money, strict=True) if np.isfinite(v)}
    costs = fm.costs(commission_rate=1e-4, commission_min=0.0, slippage=5e-4)
    config = fm.config(initial_cash=1e6, record_trades=True)

    def allocate(method, cols=symbols):
        idx = [symbols.index(c) for c in cols]
        return aa.allocate(
            returns[:, idx],
            method,
            dates=dates,
            symbols=list(cols),
            rebalance="monthly",
            lookback=252,
            min_history=120,
        )

    def report(name, res, market, used):
        started = time.perf_counter()
        ts = aa.tearsheet(
            res,
            market,
            costs=costs,
            config=used,
            title=f"资产配置：{name}",
            names=ASSETS,
        )
        ts.save_json(out / f"{name}.json")
        (out / f"{name}.weights.json").write_text(
            json.dumps(res.to_dict(), ensure_ascii=False)
        )
        print(name, f"{time.perf_counter() - started:.1f}s")
        return ts

    runs = {}
    allocations = {}
    for name, method in METHODS.items():
        res = allocate(method)
        allocations[name] = res
        levered = isinstance(method, aa.TargetVolatility) and method.max_leverage > 1
        runs[name] = report(
            name, res, fm.market, margin(config, FINANCING) if levered else config
        )
    # 对照：原定的目标波动 8%、不加杠杆（与纯风险平价逐位相同）。
    vol8 = allocate(aa.TargetVolatility(aa.RiskParity(), target=0.08))
    runs["risk_parity_vol8_unlevered"] = report(
        "risk_parity_vol8_unlevered", vol8, fm.market, config
    )
    same = np.array_equal(vol8.weights, allocations["risk_parity"].weights)
    # 分解：国债仓位换成 0 收益现金；只用三类风险资产；融资利率 6%；2 万元小资金。
    rp = allocations["risk_parity"]
    bond = symbols.index("511010.SH")
    no_bond = replace(rp, weights=rp.weights.copy())
    no_bond.weights[:, bond] = 0.0
    decomposition = {
        "risk_parity_bond_as_cash": report(
            "risk_parity_bond_as_cash", no_bond, fm.market, config
        ),
    }
    risky = [s for s in symbols if s != "511010.SH"]
    fm3 = fund_market(funds, risky, args.start)
    decomposition["risk_parity_three_risky"] = report(
        "risk_parity_three_risky",
        allocate(aa.RiskParity(), risky),
        fm3.market,
        fm3.config(initial_cash=1e6, record_trades=True),
    )
    decomposition["risk_parity_vol6_financing6"] = report(
        "risk_parity_vol6_financing6",
        allocations["risk_parity_vol6"],
        fm.market,
        margin(config, 0.06),
    )
    small = replace(config, initial_cash=2e4)
    decomposition["risk_parity_20k"] = report("risk_parity_20k", rp, fm.market, small)

    header = [
        "| 配置 | 区间 | 年化收益 | 年化波动 | 夏普（无风险 0） | 超额夏普（扣货币 ETF） | 最大回撤 | 年化双边换手 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines = list(header)
    stats = {}

    def row(label, name, ts, begin):
        m = window_stats(ts, begin, risk_free)
        stats[f"{label}/{name}"] = m
        lines.append(
            f"| {name}（{label}） | {m['start']}–{m['end']} | {m['annual_return']:.2%} | "
            f"{m['volatility']:.2%} | {m['sharpe']:.2f} | {m['excess_sharpe']:.2f} | "
            f"{m['max_drawdown']:.2%} | {m['turnover']:.0%} |"
        )

    for label, begin in (("全样本", iso[0]), ("共同区间", COMMON)):
        for name, ts in runs.items():
            row(label, name, ts, begin)
    lines += ["", *header]
    for name in ("risk_parity", *decomposition):
        row("共同区间", name, decomposition.get(name) or runs[name], COMMON)
    rf = window_stats(runs["equal"], COMMON, risk_free)["risk_free_annual"]
    weights = {
        s: float(np.mean(rp.weights[np.array(rp.dates) >= COMMON.replace("-", ""), j]))
        for j, s in enumerate(symbols)
    }
    lev = allocations["risk_parity_vol6"]
    extra = dict(
        vol8_identical_to_risk_parity=bool(same),
        risk_free_annual_common=rf,
        risk_parity_mean_weights_common=weights,
        vol6_mean_gross=float(lev.weights.sum(axis=1).mean()),
    )
    stats["extra"] = extra
    (out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(extra, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
