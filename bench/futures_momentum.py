"""Multi-product time-series momentum on real domestic futures (read-only data).

    python bench/futures_momentum.py --root /path/to/data [--repeat 5]

All commodity products, the 20 most traded (trailing 60 rows, point in
time) eligible on each rebalance; 60-row momentum on the ratio-adjusted
continuous price (signal only), position = sign × 10% / (annualized 60-row
volatility × sqrt(20)) of equity in notional, rebalanced every 21 rows, held
in the open-interest main contract with real rolls (signal at the close,
fill at the next row's open). Also runs the same weights on continuous prices
directly (the wrong way) and times one run and a 100-plan `run_batch`.
Writes a JSON summary next to this file (bench/results/futures_momentum.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import panelbt as bt
import polars as pl
from panelbt import metrics

import ashare.data as ad
from ashare.backtest import execution_targets
from ashare.backtest.futures import futures_portfolio

# Universe chosen point in time: every commodity product (CFFEX excluded); on
# each rebalance row the 20 products with the largest traded value over the
# previous 60 rows (volume x settlement x multiplier of the held contract,
# known at that close) are eligible. No whole-sample selection.
TOP, LIQ_WINDOW = 20, 60
START, END = "2012-01-04", "2026-11-12"
# Before 2021 (SHFE/INE) or 2011 the source has no margin/fee parameters:
# assumed, conservative constants (see docs/FUTURES_BACKTEST.md).
MISSING = dict(margin_rate=0.15, fee_rate=1e-4, fee_per_contract=0.0)


def liquidity(pf, window=LIQ_WINDOW):
    """Trailing traded value per product (rows x products), NaN when not held."""
    book = pf.book
    t, n_products = pf.held.shape
    rows = np.arange(t)
    f = book.market.futures
    vol = np.asarray(book.market.volume)
    settle = np.asarray(f.settlement)
    mult = np.asarray(f.multiplier)
    value = np.full((t, n_products), np.nan)
    for k in range(n_products):
        h = pf.held[:, k]
        c = np.maximum(h, 0)
        m = mult[c] if mult.ndim == 1 else mult[rows, c]
        v = vol[rows, c] * settle[rows, c] * m
        v[(h < 0) | ~np.isfinite(v)] = 0.0
        value[:, k] = v
    csum = np.cumsum(value, axis=0)
    out = csum - np.vstack([np.zeros((window, n_products)), csum[:-window]])
    out[pf.held < 0] = np.nan
    return out


def momentum_weights(
    prices, lookback=60, vol_window=60, risk=0.10, step=21, liquid=None, top=TOP
):
    p = len(prices[0]) if liquid is None else top
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.log(prices)
        mom = logp - np.roll(logp, lookback, axis=0)
        mom[:lookback] = np.nan
        ret = np.diff(logp, axis=0, prepend=np.nan)
        vol = np.full_like(prices, np.nan)
        for t in range(vol_window, len(prices)):
            vol[t] = np.nanstd(ret[t - vol_window + 1 : t + 1], axis=0, ddof=1)
        vol *= np.sqrt(244)
        w = np.sign(mom) * risk / (vol * np.sqrt(p))
    w = np.clip(w, -1.0, 1.0)
    plan = np.full(prices.shape, np.nan)
    rows = np.arange(max(lookback, vol_window, LIQ_WINDOW), len(prices), step)
    plan[rows] = np.nan_to_num(w[rows], nan=0.0)
    if liquid is not None:
        for t in rows:
            score = np.where(np.isfinite(w[t]), liquid[t], np.nan)
            ok = np.flatnonzero(np.isfinite(score))
            keep = ok[np.argsort(-score[ok], kind="stable")[:top]]
            mask = np.zeros(len(score), bool)
            mask[keep] = True
            plan[t, ~mask] = 0.0
    return plan


def naive_continuous(prices, plan, cost=0.0):
    """Wrong way: daily return = previous weights × continuous price return."""
    held = np.zeros(prices.shape[1])
    out = np.zeros(len(prices))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nan_to_num(prices / np.roll(prices, 1, axis=0) - 1, nan=0.0)
    r[0] = 0
    for t in range(len(prices)):
        out[t] = held @ r[t]
        row = plan[t]
        if not np.all(np.isnan(row)):
            new = np.nan_to_num(row)
            out[t] -= cost * np.abs(new - held).sum()
            held = new
    return out


def summary(returns):
    s = metrics.calculate(returns, periods_per_year=244, allow_bankruptcy=True)
    return dict(
        annual=float(s["annual_return"]),
        max_drawdown=float(s["max_drawdown"]),
        sharpe=float(s["sharpe"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    fut = ad.open(args.root).futures()
    t0 = time.perf_counter()
    products = fut.products().filter(pl.col("exchange") != "CFFEX")["product"].to_list()
    pf = futures_portfolio(
        fut,
        products,
        start=START,
        end=END,
        missing=MISSING,
        config=bt.Config(initial_cash=1e7, record_trades=True),
    )
    build = time.perf_counter() - t0
    liquid = liquidity(pf)
    plan = momentum_weights(pf.signal_prices, liquid=liquid)
    signal = execution_targets(plan)
    book = pf.book
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        r = pf.run(signal, "notional")
        times.append(time.perf_counter() - t0)
    t, n_products = pf.held.shape
    # Fill categories: rebalance rows (signal rows), roll rows of the rolling
    # product (its outgoing and incoming contract), anything else.
    lagged = signal.to_dense(t, n_products)
    rebalance_rows = set(np.flatnonzero(~np.all(np.isnan(lagged), axis=1)).tolist())
    roll_cells = set()
    for day in range(1, t):
        for k in range(n_products):
            a, b = pf.held[day - 1, k], pf.held[day, k]
            if a != b:
                roll_cells.update({(day, a), (day, b)})
    cost = lambda x: x.commission + x.slippage_cost  # noqa: E731
    cats = dict(rebalance=[0, 0.0], roll=[0, 0.0], other=[0, 0.0])
    for x in r.trades:
        key = (
            "rebalance"
            if x.day in rebalance_rows
            else "roll"
            if (x.day, x.col) in roll_cells
            else "other"
        )
        cats[key][0] += 1
        cats[key][1] += cost(x)
    cats["roll"][1] += float(r.other_fee.sum())  # forced closes at the end row
    daily_cost = r.commission + r.slippage_cost + r.other_fee
    previous = np.concatenate([[book.config.initial_cash], r.equity[:-1]])
    years = t / 244
    total = float(daily_cost.sum())
    f = book.market.futures
    positions = np.nan_to_num(r.positions)
    gross = (
        np.abs(positions) * f.multiplier[None, :] * np.nan_to_num(f.settlement)
    ).sum(axis=1) / r.equity
    engine = summary(r.returns)
    # Wrong way: same signal-row weights applied to continuous prices.
    raw_cont = np.column_stack(
        [
            np.asarray(
                pf.book.market.futures.settlement[
                    np.arange(t), np.maximum(pf.held[:, k], 0)
                ]
            )
            for k in range(n_products)
        ]
    )
    raw_cont[pf.held < 0] = np.nan
    wrong_raw = summary(naive_continuous(raw_cont, lagged))
    wrong_adj = summary(naive_continuous(pf.signal_prices, lagged))
    # 100 parameter sets: signals (Python), then the stepped runs in threads.
    plans = []
    t0 = time.perf_counter()
    for lookback in (20, 40, 60, 80, 100, 120, 160, 200, 240, 260):
        for risk in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22):
            plans.append(
                execution_targets(
                    momentum_weights(
                        pf.signal_prices, lookback, 60, risk, liquid=liquid
                    )
                )
            )
    batch_prepare = time.perf_counter() - t0
    t0 = time.perf_counter()
    pf.run_batch(plans, "notional", threads=8)
    batch = time.perf_counter() - t0
    out = dict(
        universe=products,
        top=TOP,
        products_ever_selected=sorted(
            products[k] for k in np.flatnonzero(np.nanmax(np.abs(plan), axis=0) > 0)
        ),
        start=str(pf.dates[0]),
        end=str(pf.dates[-1]),
        rows=t,
        contracts=len(pf.symbols),
        rolls=pf.rolls.height,
        build_seconds=build,
        run_ms_median=1000 * float(np.median(times)),
        batch100_signals_seconds=batch_prepare,
        batch100_runs_seconds_8_threads=batch,
        engine=engine,
        wrong_unadjusted_continuous=wrong_raw,
        wrong_adjusted_continuous=wrong_adj,
        trades=len(r.trades),
        fills_by_kind={k: dict(fills=v[0], cost=v[1]) for k, v in cats.items()},
        cost_per_year_pct_of_equity=100 * float((daily_cost / previous).sum()) / years,
        cost_per_year_pct_of_initial=100 * total / book.config.initial_cash / years,
        roll_cost_share=cats["roll"][1] / total if total else 0.0,
        gross_leverage_mean=float(np.mean(gross)),
        gross_leverage_max=float(np.max(gross)),
        forced_closes=len(r.expiry_events),
        margin_calls=int(r.margin_call.sum()),
        mean_margin_usage=float(np.mean(r.margin_usage)),
    )
    path = Path(__file__).parent / "results" / "futures_momentum.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
