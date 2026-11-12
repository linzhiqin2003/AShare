"""Exit rules in mixed panels: the spot rule corpus plus futures columns."""

from dataclasses import replace

import numpy as np
import panelbt as bt

from .generators_rules import case as spot_case


def _widen(a, nf, fill):
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return np.concatenate([a, np.full(nf, fill, dtype=a.dtype)])
    return np.concatenate([a, np.full((a.shape[0], nf), fill, dtype=a.dtype)], axis=1)


def mixed_case(seed, *, fill_time=None, hedge=True):
    """(market, targets, costs, config): spot columns first, then futures."""
    market, targets, costs, config = spot_case(seed, fill_time=fill_time)
    rng = np.random.default_rng(seed + 7_919)
    t, n = market.shape
    nf = int(rng.integers(1, 3))
    s = rng.uniform(80, 120, nf) * np.exp(rng.normal(0, 0.03, (t, nf)).cumsum(axis=0))
    o = s * np.exp(rng.normal(0, 0.01, (t, nf)))
    h = np.maximum(o, s) * 1.01
    lo = np.minimum(o, s) * 0.99
    ex = o.copy()
    ex[rng.random((t, nf)) < 0.08] = np.nan
    ok = rng.random((t, nf)) > 0.08
    expiry = np.where(rng.random(nf) < 0.4, rng.integers(1, t + 1, nf), t)
    kind = np.concatenate([np.zeros(n, bool), np.ones(nf, bool)])

    def panel(a, b):
        return None if a is None else np.concatenate([a, b], axis=1)

    futures = bt.Futures(
        is_future=kind,
        multiplier=np.concatenate([np.ones(n), rng.integers(5, 30, nf).astype(float)]),
        margin_rate=rng.uniform(0.08, 0.3, (t, n + nf)),
        settlement=panel(np.ones((t, n)), s),
        expiry=np.concatenate([np.full(n, t), expiry]),
        fee_rate=float(rng.choice([0.0, 1e-4])),
        fee_per_contract=float(rng.choice([0.0, 2.0])),
    )
    mixed = bt.Market(
        panel(market.exec_price, ex),
        panel(market.close, s),
        panel(market.can_buy, ok),
        panel(market.can_sell, ok),
        None if market.volume is None else panel(market.volume, np.full((t, nf), 1e6)),
        panel(market.split, np.ones((t, nf))),
        panel(market.cash_div, np.zeros((t, nf))),
        panel(market.delist, np.zeros((t, nf), bool)),
        panel(market.can_short, np.ones((t, nf), bool)),
        futures=futures,
        open=panel(market.open, o),
        high=panel(market.high, h),
        low=panel(market.low, lo),
    )
    dense = targets.to_dense(t, n)
    fw = np.full((t, nf), np.nan)
    rows = ~np.all(np.isnan(dense), axis=1)
    if hedge:
        # Large hedges leave little own cash beside the collateral, so rule
        # reductions get trimmed and accounts get called.
        scale = float(rng.choice([1.0, 3.0, 6.0]))
        fw[rows] = (
            scale
            * rng.uniform(-1.5, 1.0, (int(rows.sum()), nf))
            * (rng.random((int(rows.sum()), nf)) < 0.8)
        )
        if targets.kind == "shares":
            fw[rows] = np.trunc(fw[rows] * 5)
    plan = bt.Targets.from_dense(np.concatenate([dense, fw], axis=1), kind=targets.kind)
    rules = config.rules
    rules = replace(
        rules,
        stop_loss=_widen(rules.stop_loss, nf, np.nan),
        take_profit=_widen(rules.take_profit, nf, np.nan),
        trailing_stop=_widen(rules.trailing_stop, nf, np.nan),
    )
    config = replace(
        config,
        rules=rules,
        min_qty=_widen(config.min_qty, nf, 1.0),
        qty_step=_widen(config.qty_step, nf, 1.0),
    )
    return mixed, plan, costs, config
