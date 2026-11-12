"""Seeded mixed spot/futures inputs independent of execution implementations."""

import numpy as np
import panelbt as bt
from panelbt.futures import roll_targets


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


def _rate_shape(rng, panel):
    """Present a T×N rate panel as a scalar, (1, N), (T, 1) or T×N input."""
    t, n = panel.shape
    shape = int(rng.integers(4))
    if shape == 0:
        return float(panel[0, 0])
    if shape == 1:
        return panel[:1].copy()
    if shape == 2:
        return panel[:, :1].copy()
    return panel


def rich_case(seed):
    """Mixed spot/futures account exercising financing, calls and expiry.

    Contracts form one ordered chain on an index; spot names load on the same
    index so that hedges, liquidations and expiry settlement interact.
    """
    rng = np.random.default_rng(seed + 7_700_001)
    t = int(rng.integers(6, 17))
    ns = int(rng.integers(2, 6))
    nf = int(rng.integers(1, 4))
    n = ns + nf
    kind = np.zeros(n, bool)
    kind[ns:] = True
    stress = rng.random() < 0.35
    vol = float(rng.uniform(0.04, 0.14) if stress else rng.uniform(0.01, 0.05))
    index = 100 * np.exp(np.cumsum(rng.normal(0, vol, t)))
    beta = rng.uniform(0.5, 1.5, ns)
    spot = rng.uniform(5, 60, ns) * np.exp(
        np.outer(np.log(index / 100), beta)
        + np.cumsum(rng.normal(0, 0.03, (t, ns)), axis=0)
    )
    # Contract chain: ordered expiries, some beyond the panel.
    expiry = np.sort(rng.choice(np.arange(1, t + 1), nf, replace=False)).astype(
        np.int64
    )
    basis = rng.normal(0, 0.01, (t, nf)).cumsum(axis=0) * 0.3
    settlement_f = index[:, None] * np.exp(basis)
    open_f = np.vstack([settlement_f[:1], settlement_f[:-1]]) * np.exp(
        rng.normal(0, vol * 0.7, (t, nf))
    )
    exec_price = np.empty((t, n))
    exec_price[:, :ns] = spot * np.exp(rng.normal(0, 0.01, (t, ns)))
    exec_price[:, ns:] = open_f
    close = np.empty((t, n))
    close[:, :ns] = spot
    close[:, ns:] = settlement_f * np.exp(rng.normal(0, 0.002, (t, nf)))
    settlement = np.ones((t, n))
    settlement[:, ns:] = settlement_f
    # Suspensions (NaN quotes), missing closes and one-sided limit masks.
    exec_price[rng.random((t, n)) < 0.07] = np.nan
    close[rng.random((t, n)) < 0.08] = np.nan
    move = np.vstack(
        [np.zeros(n), np.diff(np.log(np.nan_to_num(exec_price, nan=1.0)), axis=0)]
    )
    can_buy = rng.random((t, n)) > 0.06
    can_sell = rng.random((t, n)) > 0.06
    can_buy &= ~((move > 0.05) & (rng.random((t, n)) < 0.5))
    can_sell &= ~((move < -0.05) & (rng.random((t, n)) < 0.5))
    split = np.ones((t, n))
    dividend = np.zeros((t, n))
    for i in range(ns):
        if rng.random() < 0.35:
            # Often on a contract expiry row so delivery and actions coincide.
            row = (
                int(expiry[rng.integers(nf)])
                if rng.random() < 0.6 and expiry.min() < t
                else int(rng.integers(1, t))
            )
            row = min(max(row, 1), t - 1)
            if rng.random() < 0.5:
                split[row, i] = float(rng.choice([0.5, 1.5, 2.0]))
            else:
                dividend[row, i] = float(rng.uniform(0.05, 1.5))
    delist = np.zeros((t, n), bool)
    if rng.random() < 0.15:
        delist[int(rng.integers(2, t)), int(rng.integers(ns))] = True
    volume = rng.choice(
        [0.0, 5.0, 40.0, 300.0, 1e5, np.nan],
        (t, n),
        p=[0.03, 0.12, 0.15, 0.2, 0.47, 0.03],
    )
    multiplier = np.ones(n)
    multiplier[ns:] = float(rng.choice([1, 5, 10, 20, 50]))
    base = float(rng.uniform(0.06, 0.25))
    margin_panel = np.full((t, n), base)
    for k in range(nf):
        # Exchanges raise margin ahead of delivery.
        lo = max(int(expiry[k]) - int(rng.integers(1, 4)), 0)
        margin_panel[lo : int(expiry[k]) + 1, ns + k] *= float(rng.uniform(1.2, 2.0))
    if rng.random() < 0.3:
        margin_panel[int(rng.integers(t)) :] *= float(rng.uniform(1.1, 1.6))
    fee_panel = rng.choice([0.0, 2.3e-5, 1e-4, 1e-3], (t, n))
    per_contract = rng.choice([0.0, 1.0, 5.0, 60.0], (t, n))
    f = bt.Futures(
        kind,
        multiplier,
        _rate_shape(rng, margin_panel),
        settlement,
        np.concatenate([np.full(ns, t, dtype=np.int64), expiry]),
        fee_rate=_rate_shape(rng, fee_panel),
        fee_per_contract=_rate_shape(rng, per_contract),
        delivery_fee_rate=_rate_shape(rng, rng.choice([0.0, 1e-4], (t, n))),
        delivery_fee_per_contract=_rate_shape(rng, rng.choice([0.0, 2.0], (t, n))),
    )
    impact = rng.random() < 0.3
    market = bt.Market(
        exec_price,
        close,
        can_buy,
        can_sell,
        volume=volume,
        split=split,
        cash_div=dividend,
        delist=delist,
        can_short=rng.random((t, n)) > 0.2,
        futures=f,
    )
    cash_account = rng.random() < 0.35
    if cash_account:
        margin = bt.MarginModel(
            financing_rate=float(rng.choice([0.0, 0.002])),
            margin_call=str(rng.choice(["liquidate", "liquidate", "record"])),
        )
    else:
        financing = rng.random() < 0.6
        li = rng.uniform(0.2, 1.0)
        margin = bt.MarginModel(
            allow_short=bool(rng.random() < 0.5),
            long_initial=(float(li) if rng.random() < 0.6 else rng.uniform(li, 1.0, t)),
            short_initial=float(rng.uniform(0.3, 1.5)),
            long_maintenance=float(rng.uniform(0.05, li)),
            short_maintenance=float(rng.uniform(0.1, 0.9)),
            allow_financing=financing,
            short_proceeds_usable=bool(rng.random() < 0.4),
            financing_rate=float(rng.choice([0.0, 0.001, 0.004])),
            borrow_rate=float(rng.choice([0.0, 0.001])),
            margin_call=str(rng.choice(["liquidate"] * 4 + ["record"])),
        )
    shares = rng.random() < 0.35
    initial_cash = float(10 ** rng.uniform(3.2, 5))
    notional = float(np.nanmean(settlement_f)) * multiplier[-1]
    # Futures gross notional relative to equity; larger in stress groups.
    lever = float(rng.uniform(1.0, 6.0) if stress else rng.uniform(0.2, 1.6))
    sign = -1.0 if rng.random() < 0.7 else 1.0
    roll = rng.random() < 0.5
    roll_plan = None
    if roll:
        roll_plan = roll_targets(
            expiry, t, int(rng.integers(0, 3)), sign * lever
        ).to_dense(t, nf)
    dense = np.full((t, n), np.nan)
    spot_w = np.zeros(ns)
    every = int(rng.integers(1, 4))
    for d in range(t):
        scheduled = d % every == 0 or (
            roll_plan is not None and not np.isnan(roll_plan[d, 0])
        )
        if not scheduled:
            continue
        if d % every == 0 or d == 0:
            gross = float(rng.uniform(0.3, 1.0))
            if not cash_account and margin.allow_financing:
                gross *= float(rng.uniform(1.0, 2.5))
            spot_w = rng.dirichlet(np.ones(ns)) * gross
            spot_w[rng.random(ns) < 0.2] = 0.0
            if margin.allow_short:
                spot_w *= rng.choice([-1, 1], ns, p=[0.3, 0.7])
        row = np.zeros(n)
        row[:ns] = spot_w
        if roll_plan is not None and not np.isnan(roll_plan[d, 0]):
            row[ns:] = roll_plan[d]
        elif roll_plan is not None:
            # Keep holding the contract the roll plan currently points to.
            last = roll_plan[: d + 1][~np.isnan(roll_plan[: d + 1, 0])][-1]
            row[ns:] = last
        else:
            live = np.flatnonzero(expiry >= d)
            k = int(live[0]) if len(live) else nf - 1
            row[ns + k] = sign * lever * float(rng.uniform(0.5, 1.5))
        if rng.random() < 0.15:
            # A stale target on an earlier (possibly expired) contract.
            row[ns + int(rng.integers(nf))] = float(rng.uniform(-1, 1))
        if rng.random() < 0.1:
            # Tiny hedge that rounds to zero contracts.
            row[ns + int(rng.integers(nf))] = (
                float(rng.uniform(-0.2, 0.2)) * notional / initial_cash
            )
        if shares:
            px = np.nan_to_num(exec_price[d], nan=np.nanmean(exec_price[:, :ns]))
            row[:ns] = np.round(row[:ns] * initial_cash / px[:ns], 2)
            row[ns:] = np.trunc(row[ns:] * initial_cash / notional)
        dense[d] = row
    targets = bt.Targets.from_dense(dense, kind="shares" if shares else "weights")
    costs = bt.Costs(
        commission_rate=float(rng.choice([0, 2.5e-4, 0.001])),
        commission_min=float(rng.choice([0, 1, 5])),
        tax_sell=float(rng.choice([0, 0.0005])),
        fee_rate_buy=float(rng.choice([0, 1e-5])),
        fee_rate_sell=float(rng.choice([0, 1e-5])),
        slippage=(
            float(rng.choice([0, 0.0005, 0.002]))
            if rng.random() < 0.7
            else rng.uniform(0, 0.003, (t, n))
        ),
        impact_k=rng.uniform(0, 0.01, (t, n)) if impact else None,
    )
    lot = float(rng.choice([0, 1, 10, 100]))
    config = bt.Config(
        initial_cash=initial_cash,
        min_qty=lot,
        qty_step=lot,
        max_participation=float(rng.choice([0, 0, 0.1, 0.5])),
        carry_blocked=bool(rng.random() < 0.8),
        delist_recovery=float(rng.choice([0, 0.5, 1])),
        cash_buffer=float(rng.choice([0, 0.02])),
        trade_band=float(rng.choice([0, 0, 0.03])),
        dividend_tax_rate=float(rng.choice([0, 0.1])),
        cash_flow=rng.choice(
            [0.0, 0.0, 0.0, initial_cash * 0.05, -initial_cash * 0.08], t
        ),
        cash_rate=float(rng.choice([0, 0.0005])),
        record_trades=True,
        margin=margin,
    )
    return market, targets, costs, config
