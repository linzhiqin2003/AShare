"""带开高低收、退出规则和限价目标的随机市场（与实现无关）。"""

from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.rules.ashare import tradability

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def percent(rng, t, n, low, high):
    kind = rng.integers(0, 4)
    if kind == 0:
        return None
    if kind == 1:
        return float(rng.uniform(low, high))
    if kind == 2:
        a = rng.uniform(low, high, n)
        a[rng.random(n) < 0.2] = np.nan
        return a
    a = rng.uniform(low, high, (t, n))
    a[rng.random((t, n)) < 0.2] = np.nan
    return a


def case(seed: int, *, margin=None, fill_time=None):
    rng = np.random.default_rng(seed + 55_000)
    t, n = int(rng.integers(6, 30)), int(rng.integers(2, 9))
    fill_time = fill_time or str(rng.choice(["open", "close", "average"]))
    close = np.empty((t, n))
    open_ = np.empty((t, n))
    level = rng.uniform(3, 50, n)
    vol = rng.choice([0.01, 0.03, 0.06])
    for d in range(t):
        open_[d] = level * np.exp(
            rng.normal(0, vol, n)
            + rng.choice([0, 0.06, -0.06], n, p=[0.85, 0.075, 0.075])
        )
        close[d] = open_[d] * np.exp(rng.normal(0, vol, n))
        level = close[d]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol, (t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol, (t, n))))
    if fill_time == "open":
        p = open_.copy()
    elif fill_time == "close":
        p = close.copy()
    else:
        p = low + (high - low) * rng.uniform(0.2, 0.8, (t, n))
    prev = np.vstack([open_[:1], close[:-1]])
    upper, lower = prev * 1.1, prev * 0.9
    flags = rng.integers(0, 14, (t, n))
    suspended = flags == 4
    buy, sell = tradability(p, high, low, upper, lower, suspended)
    sell &= rng.random((t, n)) > 0.08  # limit-down style blocks
    for a in (p, open_, high, low, close):
        a[suspended] = np.nan
    close[rng.random((t, n)) < 0.05] = np.nan
    volume = rng.choice(
        [0.0, 100.0, 300.0, 1e4, 1e8, np.nan],
        (t, n),
        p=[0.03, 0.1, 0.1, 0.17, 0.55, 0.05],
    )
    split = rng.choice([1.0, 1.5, 2.0], (t, n), p=[0.95, 0.03, 0.02])
    div = np.where(rng.random((t, n)) < 0.08, rng.uniform(0.01, 0.5, (t, n)), 0.0)
    delist = np.zeros((t, n), bool)
    if rng.random() < 0.3:
        delist[rng.integers(2, t), rng.integers(0, n)] = True
    shorting = margin if margin is not None else seed % 3 == 0
    market = bt.Market(
        p,
        close,
        buy,
        sell,
        volume if seed % 5 else None,
        split,
        div,
        delist,
        rng.random((t, n)) > 0.2 if shorting else None,
        open=open_,
        high=high,
        low=low,
    )
    w = np.full((t, n), np.nan)
    limits = np.full((t, n), np.nan)
    daily = rng.random() < 0.3
    for d in range(t):
        if d == 0 or daily or rng.random() < 0.3:
            active = rng.choice(n, int(rng.integers(1, n + 1)), replace=False)
            row = np.zeros(n)
            row[active] = rng.dirichlet(np.ones(len(active))) * rng.choice([0.5, 0.9])
            if shorting:
                row *= rng.choice([-1, 1], n)
            w[d] = row
            if rng.random() < 0.4:
                ref = (
                    np.where(np.isfinite(close[d - 1]), close[d - 1], level)
                    if d
                    else open_[0]
                )
                pick = rng.random(n) < 0.4
                limits[d, pick] = ref[pick] * rng.uniform(0.94, 1.06, pick.sum())
    limits[~np.isfinite(limits)] = np.nan
    use_limits = rng.random() < 0.5
    kind = "shares" if seed % 7 == 0 else "weights"
    if kind == "shares":
        w = np.round(w * 1e5 / 100) * 100
    targets = bt.Targets.from_dense(w, kind=kind, limits=limits if use_limits else None)
    rules = bt.Rules(
        fill_time=fill_time,
        stop_loss=percent(rng, t, n, 0.02, 0.12),
        take_profit=percent(rng, t, n, 0.02, 0.15),
        trailing_stop=percent(rng, t, n, 0.02, 0.10),
        max_hold=int(rng.integers(1, 6)) if rng.random() < 0.4 else None,
        both_hit=str(rng.choice(["stop", "target"])),
        cooldown=None if rng.random() < 0.5 else int(rng.integers(0, 4)),
        dividend_adjusts_cost=bool(rng.random() < 0.3),
        sell_lock=bool(rng.random() < 0.7),
        limit_valid=str(rng.choice(["day", "until_next"])),
    )
    kinds = rng.integers(0, 3, n)
    costs = bt.Costs(
        commission_rate=float(rng.choice([0.0, 0.00025])),
        commission_min=float(rng.choice([0.0, 5.0])),
        tax_sell=0.0005,
        slippage=float(rng.choice([0.0, 0.001])),
    )
    account = (
        bt.MarginModel(
            allow_short=True,
            short_initial=1.0,
            long_maintenance=0.2,
            short_maintenance=0.25,
            borrow_rate=0.0001,
        )
        if shorting
        else bt.MarginModel()
    )
    config = bt.Config(
        initial_cash=float(10 ** rng.uniform(4, 6)),
        min_qty=np.array([100.0, 1.0, 0.0])[kinds],
        qty_step=np.array([100.0, 1.0, 0.0])[kinds],
        max_participation=float(rng.choice([0.0, 0.2, 1.0])),
        carry_blocked=bool(rng.random() < 0.8),
        cash_buffer=float(rng.choice([0.0, 0.02])),
        record_trades=True,
        margin=account,
        rules=rules,
    )
    return market, targets, costs, config
