"""Standing property: an untraded futures column never changes spot results.

Every spot case is rerun with one extra futures column that is never held or
targeted. All spot outputs must be bitwise identical, for the cash
specialization and for margin accounts, including forced-liquidation rows.
"""

from dataclasses import astuple, fields, replace

import numpy as np
import panelbt as bt
from panelbt.models import METRICS

from .generators import case as cash_case
from .test_extended import extended_case

FUTURE_ONLY = ("futures_pnl", "futures_margin", "delivery_fee", "n_expired_target")


def _pad(a, t, n, fill):
    """Append one column to a T×N panel; leave scalars and length-T vectors."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 2 and a.shape == (t, n):
        return np.hstack([a, np.full((t, 1), fill, dtype=a.dtype)])
    return a


def with_untraded_future(market, targets, costs, config):
    t, n = market.shape
    panels = {
        "exec_price": 100.0,
        "close": 100.0,
        "can_buy": True,
        "can_sell": True,
        "volume": 1e9,
        "split": 1.0,
        "cash_div": 0.0,
        "delist": False,
        "can_short": True,
    }
    kwargs = {k: _pad(getattr(market, k), t, n, v) for k, v in panels.items()}
    settlement = np.ones((t, n + 1))
    settlement[:, n] = 100.0
    futures = bt.Futures(
        np.r_[np.zeros(n, bool), True],
        np.r_[np.ones(n), 300.0],
        0.12,
        settlement,
        np.r_[np.full(n, t), t],
        fee_rate=0.0003,
    )
    wide = bt.Market(**kwargs, futures=futures)
    costs = replace(
        costs,
        **{
            f.name: _pad(getattr(costs, f.name), t, n, 0.0)
            for f in fields(costs)
            if np.ndim(getattr(costs, f.name)) == 2
        },
    )
    margin = replace(
        config.margin,
        **{
            f.name: _pad(getattr(config.margin, f.name), t, n, 0.0)
            for f in fields(config.margin)
            if np.ndim(getattr(config.margin, f.name)) == 2
        },
    )
    per_column = {
        name: np.r_[np.broadcast_to(getattr(config, name), (n,)), 100.0]
        for name in ("min_qty", "qty_step")
    }
    return wide, targets, costs, replace(config, margin=margin, **per_column)


def bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).tobytes(), a.dtype, a.shape


def assert_spot_bitwise_equal(spot, wide, n):
    for name in METRICS:
        if name in FUTURE_ONLY:
            assert not np.any(getattr(wide, name)), name
            continue
        assert bits(getattr(spot, name)) == bits(getattr(wide, name)), name
    assert bits(spot.positions) == bits(wide.positions[:, :n])
    assert not np.any(wide.positions[:, n:])
    assert bits(spot.contribution) == bits(wide.contribution[:n])
    assert wide.contribution[n] == 0
    assert spot.diagnostics == wide.diagnostics
    assert spot.coverage == wide.coverage
    assert spot.bankruptcy_events == wide.bankruptcy_events
    assert wide.expiry_events == []
    for name in ("trades", "delist_events", "liquidation_events"):
        a = [astuple(x) for x in getattr(spot, name)]
        b = [astuple(x) for x in getattr(wide, name)]
        assert np.asarray(a, float).tobytes() == np.asarray(b, float).tobytes(), name


def _check(case, seeds):
    seen = dict.fromkeys(("trades", "liquidation", "margin_call", "cash_scaled"), 0)
    for seed in seeds:
        args = case(seed)
        spot = bt.run(*args)
        wide = bt.run(*with_untraded_future(*args))
        try:
            assert_spot_bitwise_equal(spot, wide, args[0].shape[1])
        except AssertionError as exc:
            raise AssertionError(f"{case.__name__} seed={seed}: {exc}") from exc
        seen["trades"] += spot.diagnostics["trades"]
        seen["cash_scaled"] += spot.diagnostics["cash_scaled"]
        seen["liquidation"] += spot.coverage["liquidation"]
        seen["margin_call"] += spot.coverage["margin_call"]
    return seen


def test_untraded_future_leaves_cash_account_bitwise_unchanged():
    seen = _check(cash_case, range(1500))
    assert seen["trades"] > 0 and seen["cash_scaled"] > 0, seen


def test_untraded_future_leaves_margin_account_bitwise_unchanged():
    seen = _check(extended_case, range(1500))
    # Liquidation rows are where the two account loops used to diverge.
    assert seen["liquidation"] > 100 and seen["margin_call"] > 100, seen
