"""Seeded mixed spot/futures inputs independent of execution implementations."""

import numpy as np
import panelbt as bt


def case(seed):
    rng = np.random.default_rng(seed + 91_003)
    t = int(rng.integers(4, 9))
    n = int(rng.integers(3, 7))
    nf = int(rng.integers(1, min(3, n)))
    kind = np.zeros(n, bool)
    kind[-nf:] = True
    p = rng.uniform(8, 30, (t, n))
    p[:, kind] = rng.uniform(50, 150, (t, nf))
    p *= np.exp(rng.normal(0, 0.1, (t, n)).cumsum(axis=0))
    settlement = p.copy()
    settlement[:, kind] *= np.exp(rng.normal(0, 0.14, (t, nf)))
    close = p.copy()
    close[:, ~kind] *= np.exp(rng.normal(0, 0.04, (t, n - nf)))
    missing = rng.random((t, n)) < 0.12
    p[missing] = np.nan
    close[rng.random((t, n)) < 0.12] = np.nan
    buy = rng.random((t, n)) > 0.12
    sell = rng.random((t, n)) > 0.12
    split = np.ones((t, n))
    dividend = np.zeros((t, n))
    for i in range(n - nf):
        if rng.random() < 0.3:
            split[int(rng.integers(1, t)), i] = float(rng.choice([0.5, 2.0]))
        if rng.random() < 0.3:
            dividend[int(rng.integers(1, t)), i] = float(rng.uniform(0.1, 1))
    delist = np.zeros((t, n), bool)
    if rng.random() < 0.25:
        delist[int(rng.integers(2, t)), int(rng.integers(n - nf))] = True
    expiry = np.full(n, t, dtype=np.int64)
    expiry[-nf:] = rng.integers(1, t + 1, nf)
    fm_rate = rng.uniform(0.05, 0.5, (t, n))
    f = bt.Futures(
        kind,
        np.where(kind, rng.integers(5, 25, n), 1),
        fm_rate,
        settlement,
        expiry,
        fee_rate=rng.choice([0, 0.0003, 0.001]),
        fee_per_contract=rng.choice([0, 0.3, 2]),
        delivery_fee_rate=rng.choice([0, 0.0001]),
        delivery_fee_per_contract=rng.choice([0, 1.5]),
    )
    market = bt.Market(
        p,
        close,
        buy,
        sell,
        volume=rng.integers(1, 2000, (t, n)).astype(float),
        split=split,
        cash_div=dividend,
        delist=delist,
        can_short=rng.random((t, n)) > 0.2,
        futures=f,
    )
    margin = bt.MarginModel(
        allow_short=bool(rng.integers(2)),
        long_initial=float(rng.uniform(0.25, 1)),
        short_initial=float(rng.uniform(0.4, 1.5)),
        long_maintenance=float(rng.uniform(0.1, 0.7)),
        short_maintenance=float(rng.uniform(0.1, 0.8)),
        allow_financing=bool(rng.integers(2)),
        short_proceeds_usable=bool(rng.integers(2)),
        financing_rate=rng.choice([0.0, 0.002]),
        borrow_rate=rng.choice([0.0, 0.001]),
        margin_call=str(rng.choice(["record", "liquidate"])),
    )
    shares = bool(rng.integers(2))
    dense = np.full((t, n), np.nan)
    for d in range(0, t, int(rng.integers(1, 3))):
        if shares:
            target = rng.integers(-10, 11, n).astype(float)
            target[~kind] = rng.integers(0, 120, n - nf)
            if margin.allow_short:
                target[~kind] *= rng.choice([-1, 1], n - nf)
        else:
            target = rng.uniform(-1.5, 1.5, n)
            target[~kind] = rng.uniform(0, 0.35, n - nf)
            if margin.allow_short:
                target[~kind] *= rng.choice([-1, 1], n - nf)
            if not margin.allow_short and not margin.allow_financing:
                total = target[~kind].sum()
                if total > 0.8:
                    target[~kind] *= 0.8 / total
        target[rng.random(n) < 0.25] = 0.0
        dense[d] = target
    targets = bt.Targets.from_dense(dense, kind="shares" if shares else "weights")
    costs = bt.Costs(
        commission_rate=float(rng.choice([0, 0.001])),
        commission_min=float(rng.choice([0, 2])),
        tax_sell=float(rng.choice([0, 0.0005])),
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=float(rng.choice([0, 0.001])),
        impact_k=(rng.uniform(0, 0.001, (t, n)) if rng.random() < 0.25 else None),
    )
    config = bt.Config(
        initial_cash=float(rng.uniform(500, 5000)),
        min_qty=0,
        qty_step=float(rng.choice([0, 1, 10])),
        max_participation=float(rng.choice([0, 0.2, 0.8])),
        cash_buffer=float(rng.choice([0, 0.05])),
        trade_band=float(rng.choice([0, 0.02])),
        dividend_tax_rate=float(rng.choice([0, 0.1])),
        cash_flow=rng.uniform(-100, 100, t),
        cash_rate=float(rng.choice([0, 0.001])),
        record_trades=True,
        margin=margin,
    )
    return market, targets, costs, config
