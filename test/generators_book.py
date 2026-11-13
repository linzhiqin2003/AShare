"""Seeded raw inputs for `FuturesBook` (pure futures) and its naive reference."""

import numpy as np
import panelbt as bt


def book_case(seed, *, small_cash=None):
    """Return (kwargs for FuturesBook, dense plan, kind, config)."""
    rng = np.random.default_rng(seed + 40_417)
    t = int(rng.integers(5, 25))
    n = int(rng.integers(1, 5))
    s = rng.uniform(50, 200, n) * np.exp(rng.normal(0, 0.05, (t, n)).cumsum(axis=0))
    if rng.random() < 0.3:
        # A large gap, so that some accounts go insolvent.
        s[int(rng.integers(1, t)) :] *= rng.choice([0.3, 3.0], n)
    o = s * np.exp(rng.normal(0, 0.02, (t, n)))
    h = np.maximum(o, s) * np.exp(np.abs(rng.normal(0, 0.01, (t, n))))
    lo = np.minimum(o, s) * np.exp(-np.abs(rng.normal(0, 0.01, (t, n))))
    # Some one-price (locked) days.
    locked = rng.random((t, n)) < 0.12
    h = np.where(locked, o, h)
    lo = np.where(locked, o, lo)
    exec_price = o.copy()
    exec_price[rng.random((t, n)) < 0.1] = np.nan
    settle = s.copy()
    settle[rng.random((t, n)) < 0.05] = np.nan
    settle[0] = s[0]
    expiry = rng.integers(2, t + 2, n)
    expiry = np.minimum(expiry, t)
    last_hold = np.where(rng.random(n) < 0.4, rng.integers(0, t, n), t)
    margin = rng.uniform(0.05, 0.3, (t, n))
    if rng.random() < 0.3:
        margin[rng.random((t, n)) < 0.05] = np.nan
    fee = rng.choice([0.0, 1e-4, 5e-4]) * np.ones((t, n))
    fixed = np.full((1, n), rng.choice([0.0, 2.0, 5.0]))
    volume = rng.integers(0, 50, (t, n)).astype(float)
    slip = rng.choice([0.0, 1e-3])
    acct = bt.FuturesAccount(
        margin_add=float(rng.choice([0.0, 0.05])),
        margin_scale=float(rng.choice([1.0, 1.2])),
        fee_scale=float(rng.choice([1.0, 3.0])),
    )
    kw = dict(
        settlement=settle,
        close=s,
        exec_price=exec_price,
        open=o,
        high=h,
        low=lo,
        volume=volume,
        multiplier=rng.integers(1, 20, n).astype(float),
        expiry=expiry,
        margin_rate=margin,
        fee_rate=fee,
        fee_per_contract=fixed,
        tradable=rng.random((t, n)) > 0.08,
        no_buy=rng.random((t, n)) < 0.05,
        no_sell=rng.random((t, n)) < 0.05,
        infer_locked=bool(rng.random() < 0.6),
        last_hold=last_hold,
        cash_settled=rng.random(n) < 0.5,
        delivery_fee_rate=float(rng.choice([0.0, 2e-4])),
        delivery_fee_per_contract=float(rng.choice([0.0, 1.0])),
        account=acct,
        costs=bt.Costs(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
            slippage=slip,
        ),
        missing="previous" if rng.random() < 0.5 else 0.2,
    )
    if rng.random() < 0.3:
        up = s * 1.03
        kw["limit_up"] = np.where(rng.random((t, n)) < 0.5, up, np.nan)
        kw["limit_down"] = s * 0.97
    kind = str(rng.choice(["lots", "notional", "margin"]))
    plan = np.full((t, n), np.nan)
    for d in range(t):
        if rng.random() < 0.35 or d == 0:
            if kind == "lots":
                row = rng.integers(-6, 7, n).astype(float)
            elif kind == "notional":
                row = rng.uniform(-3, 3, n)
            else:
                row = rng.uniform(-0.5, 0.5, n)
            row[rng.random(n) < 0.3] = 0.0
            if kind == "margin":
                # Margin targets need an active margin rate on that row.
                ok = kw["tradable"] & np.isfinite(exec_price)
                for j in range(n):
                    start = int(np.argmax(ok[:, j])) if ok[:, j].any() else t
                    if not (start <= d <= min(expiry[j], last_hold[j] + 1)):
                        row[j] = 0.0
            plan[d] = row
    cash = small_cash if small_cash is not None else float(rng.choice([2e3, 2e4, 2e5]))
    config = bt.Config(
        initial_cash=cash,
        record_trades=True,
        max_participation=float(rng.choice([0.0, 0.0, 0.5])),
        carry_blocked=bool(rng.random() < 0.8),
        cash_buffer=float(rng.choice([0.0, 0.05])),
    )
    kw["config"] = config
    return kw, plan, kind, config
