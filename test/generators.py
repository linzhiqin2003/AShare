"""包含缺失、限制、公司行为和不同申报单位的随机市场。"""

from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.rules.ashare import tradability

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def case(seed: int):
    rng = np.random.default_rng(seed)
    t, n = int(rng.integers(5, 30)), int(rng.integers(3, 17))
    p = rng.uniform(2, 60, n) * np.exp(np.cumsum(rng.normal(0, 0.08, (t, n)), axis=0))
    close = p * np.exp(rng.normal(0, 0.03, (t, n)))
    upper, lower = p * 1.1, p * 0.9
    flags = rng.integers(0, 12, (t, n))
    upper[flags == 0] = p[flags == 0]
    lower[flags == 1] = p[flags == 1]
    high, low = p * 1.02, p * 0.98
    high[flags == 2] = upper[flags == 2]
    low[flags == 2] = upper[flags == 2]
    high[flags == 3] = lower[flags == 3]
    low[flags == 3] = lower[flags == 3]
    p[rng.random((t, n)) < 0.08] = np.nan
    close[rng.random((t, n)) < 0.08] = np.nan
    buy, sell = tradability(p, high, low, upper, lower, flags == 4)
    volume = rng.choice(
        [0.0, 50.0, 100.0, 500.0, 1e4, 1e8, np.nan],
        (t, n),
        p=[0.05, 0.1, 0.1, 0.1, 0.15, 0.45, 0.05],
    )
    split = rng.choice(
        [1.0, 1.3, 1.5, 2.0, 0.5], (t, n), p=[0.93, 0.02, 0.02, 0.02, 0.01]
    )
    div = np.where(rng.random((t, n)) < 0.08, rng.uniform(0.01, 1, (t, n)), 0.0)
    delist = np.zeros((t, n), bool)
    for i in range(n):
        if rng.random() < 0.3:
            delist[rng.integers(1, t), i] = True
    market = bt.Market(
        p, close, buy, sell, volume if seed % 4 else None, split, div, delist
    )
    w = np.full((t, n), np.nan)
    for day in range(t):
        if day == 0 or rng.random() < 0.35:
            active = rng.choice(n, int(rng.integers(1, n + 1)), replace=False)
            row = np.zeros(n)
            row[active] = rng.dirichlet(np.ones(len(active))) * rng.choice(
                [0.4, 0.8, 1.0]
            )
            if rng.random() < 0.08:
                row[:] = 0
            w[day] = row
    kinds = rng.integers(0, 4, n)
    minimum = np.array([100.0, 200.0, 100.0, 100.0])[kinds]
    steps = np.array([100.0, 1.0, 1.0, 0.0])[kinds]
    costs = bt.Costs(
        commission_rate=float(rng.choice([0.0, 0.00025, 0.005])),
        commission_min=float(rng.choice([0.0, 5.0, 25.0])),
        tax_sell=rng.choice([0.0005, 0.001], t),
        tax_buy=rng.choice([0.0, 0.001], t),
        fee_rate_buy=0.00001,
        fee_rate_sell=0.00001,
        slippage=float(rng.choice([0.0, 0.0005, 0.005])),
    )
    config = bt.Config(
        initial_cash=float(10 ** rng.uniform(3, 6)),
        min_qty=minimum,
        qty_step=steps,
        max_participation=float(rng.choice([0.0, 0.1, 0.3, 1.0])),
        carry_blocked=seed % 5 != 0,
        delist_recovery=float(rng.choice([0.0, 0.3, 1.0])),
        cash_buffer=float(rng.choice([0.0, 0.01, 0.05])),
        record_trades=True,
    )
    return market, bt.Targets.from_dense(w), costs, config
