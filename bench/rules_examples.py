"""Two real-data examples of declarative exit rules (read-only on data/).

1. CSI 300 index timing: hold the index while its close is above the 60-day
   moving average, with an 8% trailing stop. The index is not itself
   tradable; it stands in for an index ETF (no stamp tax, unit lots).
2. Monthly stock selection: the 30 CSI 300 members with the strongest
   120-day momentum, equal weight, rebalanced every 21 trading days, with a
   10% stop loss.

Signals are computed at the close and trade at the next row's open (the
shift happens here; panelbt never shifts). Each variant runs with and without
rules; every run is timed 5 times and the median is reported.

Usage: python bench/rules_examples.py [--data DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.metrics import summarize

import ashare.data
from ashare.backtest import market_from_panel

ROOT = Path(__file__).resolve().parents[1]


def default_data():
    """data/ of this checkout, or of the main checkout for a git worktree."""
    if (ROOT / "data/cn/panel/current.json").exists():
        return ROOT / "data"
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.strip()
    return Path(common).parent / "data"


REPEAT = 5


def timed(fn):
    times, out = [], None
    for _ in range(REPEAT):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return out, statistics.median(times) * 1e3


def report(result):
    s = summarize(result)
    reasons = {}
    for t in result.trades:
        reasons[t.reason.name] = reasons.get(t.reason.name, 0) + 1
    rules = {
        k: result.diagnostics[k]
        for k in bt.models.RULE_COUNTERS
        if k in result.diagnostics
    }
    return {
        "annual_return": round(s["annual_return"], 4),
        "max_drawdown": round(s["max_drawdown"], 4),
        "sharpe": round(s["sharpe"], 3),
        "trades": len(result.trades),
        "reasons": reasons,
        "rules": {k: v for k, v in rules.items() if v},
    }


def index_timing(data):
    frame = data.table("index_daily", symbols=["000300.SH"]).sort("date").collect()
    o, h, low, c = (
        frame[k].to_numpy()[:, None].astype(float)
        for k in ("open", "high", "low", "close")
    )
    t = len(c)
    ok = np.isfinite(o)
    market = bt.Market(o, c, ok, ok, open=o, high=h, low=low)
    ma = np.full(t, np.nan)
    cs = np.cumsum(c[:, 0])
    ma[59:] = (cs[59:] - np.concatenate([[0.0], cs[:-60]])) / 60
    signal = np.where(np.isfinite(ma), c[:, 0] > ma, False).astype(float)
    weights = np.full((t, 1), np.nan)
    previous = 0.0
    for d in range(t - 1):  # signal at close d, trade at the open of d + 1
        if signal[d] != previous:
            weights[d + 1, 0] = signal[d]
            previous = signal[d]
    targets = bt.Targets.from_dense(weights)
    costs = bt.Costs(
        commission_rate=0.0001,
        commission_min=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0.0005,
    )
    base = bt.Config(initial_cash=1e6, min_qty=0, qty_step=0, record_trades=True)
    rules = bt.Rules(fill_time="open", trailing_stop=0.08)
    out = {"rows": t, "first": frame["date"][0], "last": frame["date"][-1]}
    for name, config in (
        ("no_rules", base),
        ("trailing_8pct", replace(base, rules=rules)),
    ):
        result, ms = timed(lambda c=config: bt.run(market, targets, costs, c))
        out[name] = {**report(result), "run_ms": round(ms, 3)}
    return out


def stock_selection(data):
    start = time.perf_counter()
    # Open-price execution with A-share masks, lots and fees from ashare.backtest;
    # the rules also need the day's open, high and low.
    bm = market_from_panel(
        data, fields=("open", "high", "low", "adj_factor", "member_csi300")
    )
    panel = bm.panel
    load_s = time.perf_counter() - start
    t, n = bm.shape
    market = replace(
        bm.market,
        open=np.asarray(panel["open"], dtype=float),
        high=np.asarray(panel["high"], dtype=float),
        low=np.asarray(panel["low"], dtype=float),
    )
    adj = np.asarray(panel["close"], dtype=float) * np.asarray(
        panel["adj_factor"], dtype=float
    )
    member = np.asarray(panel["member_csi300"], dtype=bool)
    weights = np.full((t, n), np.nan)
    for d in range(252, t - 1, 21):
        score = adj[d] / adj[d - 120] - 1
        score[~member[d] | ~np.isfinite(score)] = np.nan
        valid = np.flatnonzero(np.isfinite(score))
        top = valid[np.argsort(-score[valid], kind="stable")[:30]]
        row = np.zeros(n)
        row[top] = 1.0 / 30
        weights[d + 1] = row
    targets = bt.Targets.from_dense(weights)
    costs = bm.costs()
    base = bm.config(initial_cash=1e7, record_trades=True)
    rules = bt.Rules(fill_time="open", stop_loss=0.10)
    out = {"rows": t, "securities": n, "panel_load_s": round(load_s, 2)}
    out["first"], out["last"] = str(bm.dates[0]), str(bm.dates[-1])
    for name, config in (
        ("no_rules", base),
        ("stop_10pct", replace(base, rules=rules)),
    ):
        result, ms = timed(lambda c=config: bt.run(market, targets, costs, c))
        out[name] = {**report(result), "run_ms": round(ms, 3)}
    # The same plan stepped row by row gives the same result.
    config = replace(base, rules=rules)
    sim = bt.Simulator(market, costs, config)
    plan = {int(d): k for k, d in enumerate(targets.days)}
    start = time.perf_counter()
    for d in range(t):
        k = plan.get(d)
        if k is None:
            sim.step()
        else:
            a, b = targets.indptr[k], targets.indptr[k + 1]
            sim.step((targets.cols[a:b], targets.weights[a:b]))
    out["stepped_ms"] = round((time.perf_counter() - start) * 1e3, 3)
    stepped = sim.result()
    expected = bt.run(market, targets, costs, config)
    assert stepped.nav.tobytes() == expected.nav.tobytes()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    data = ashare.data.open(root=args.data or default_data())
    out = {
        "load_average": [round(x, 2) for x in os.getloadavg()],
        "index_timing": index_timing(data),
        "stock_selection": stock_selection(data),
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
