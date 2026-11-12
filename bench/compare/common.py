"""Shared deterministic prices and sparse plans for every comparison adapter."""

import numpy as np
import panelbt as bt

INITIAL = 1e7
FEE = 0.00025


def inputs(t, n, interval=21, variant=0, task=1):
    rng = np.random.default_rng(1937 + t + n)
    p = 25 * np.exp(np.cumsum(rng.normal(0, 0.008, (t, n)), axis=0))
    p = np.ascontiguousarray(p)
    target = plan(t, n, interval, variant, task)
    if task == 3:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from benchmark import market_inputs

        market = bt.Market(**market_inputs(t, n))
        costs = bt.Costs(tax_sell=0.0005)
        config = bt.Config(initial_cash=INITIAL, max_participation=0.1)
    else:
        market = bt.Market(p, p, np.ones((t, n), bool), np.ones((t, n), bool))
        costs = bt.Costs(
            commission_rate=FEE,
            commission_min=0,
            fee_rate_buy=0,
            fee_rate_sell=0,
            slippage=0,
        )
        config = bt.Config(
            initial_cash=INITIAL * (100 if task == 2 else 1),
            min_qty=0 if task == 1 else 100,
            qty_step=0 if task == 1 else 100,
        )
    return market, target, costs, config


def plan(t, n, interval=21, variant=0, task=1):
    h = min(100, n)
    days = np.arange(0, t, interval, dtype=np.int64)
    cols = (
        np.arange(h)[None, :] + np.arange(len(days))[:, None] * 37 + variant * 11
    ) % n
    exposure = (0.85 + 0.1 * (variant % 1000) / 999) if task == 1 else 1.0
    weights = np.full(cols.size, exposure / h)
    return bt.Targets(
        days, np.arange(len(days) + 1, dtype=np.int64) * h, cols.ravel(), weights
    )
