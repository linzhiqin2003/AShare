"""Run with an isolated installed-wheel interpreter (-I), never the workspace."""

import importlib.metadata
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import panelbt as bt

assert "site-packages" in str(Path(bt.__file__))
installed = sorted(
    x.metadata["Name"].lower() for x in importlib.metadata.distributions()
)
assert installed == ["numpy", "panelbt"], installed
prices = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
market = bt.Market(prices, prices, prices > 0, prices > 0)
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
costs = bt.Costs(
    commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
)
config = bt.Config(initial_cash=1000, min_qty=0, qty_step=0)
result = bt.run(market, targets, costs, config)
np.testing.assert_allclose(result.nav, [1.0, 1.025, 1.125])
rng = np.random.default_rng(19)
t, n = 4000, 5500
p = 20 * np.exp(np.cumsum(rng.normal(0, 0.005, (t, n)), axis=0))
market = bt.Market(
    p,
    p,
    np.ones((t, n), bool),
    np.ones((t, n), bool),
    split=np.ones((t, n)),
    cash_div=np.zeros((t, n)),
    delist=np.zeros((t, n), bool),
)
days = np.arange(0, t, 21, dtype=np.int64)
cols = ((np.arange(len(days))[:, None] * 37 + np.arange(100)) % n).ravel()
targets = bt.Targets(
    days,
    np.arange(len(days) + 1, dtype=np.int64) * 100,
    cols,
    np.full(len(cols), 0.0095),
)
config = bt.Config()
for _ in range(4):
    bt.run(market, targets, costs, config)
samples = []
for _ in range(21):
    start = perf_counter()
    bt.run(market, targets, costs, config)
    samples.append(perf_counter() - start)
print(
    json.dumps(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "median_s": median(samples),
            "samples_s": samples,
            "minimal_example": "passed",
            "runtime_dependencies": importlib.metadata.requires("panelbt"),
            "installed_distributions": installed,
        }
    )
)
