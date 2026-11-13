"""Fixed per-step cost of `Simulator` against a per-bar Backtrader callback.

Both sides run a daily callback that reads the account state and, on one bar
in 21, sends a new target for one security; otherwise nothing trades. The
panelbt cost is reported per step for N=1 and N=5500 (it must not grow with
N), together with `bt.run` on the same plan. Backtrader (optional, from the
`compare` dependency group) runs the same one-security plan with preload,
runonce and no observers. Prints and writes `.cache/panelbt/step_overhead.json`.
"""

import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import panelbt as bt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.build_guard import assert_current_build  # noqa: E402

T = 4000


def market(n, seed=7):
    rng = np.random.default_rng(seed)
    p = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0))
    ones = np.ones((T, n), bool)
    return bt.Market(p, p, ones, ones, volume=np.full((T, n), 1e8))


def panelbt_steps(m, repeats):
    config = bt.Config(initial_cash=1e7)
    samples = []
    for _ in range(repeats):
        sim = bt.Simulator(m, bt.Costs(), config)
        start = perf_counter()
        for day in range(T):
            state = sim.state
            target = None
            if day % 21 == 0:
                w = 0.9 if (day // 21) % 2 == 0 else 0.0
                target = ([0], [w]) if state.cash > 0 else None
            sim.step(target)
        samples.append((perf_counter() - start) / T)
    return samples


def panelbt_bare_steps(m, repeats):
    """`step(None)` alone: the binding and kernel cost of one row."""
    samples = []
    for _ in range(repeats):
        sim = bt.Simulator(m, bt.Costs(), bt.Config(initial_cash=1e7))
        start = perf_counter()
        for _day in range(T):
            sim.step()
        samples.append((perf_counter() - start) / T)
    return samples


def panelbt_run(m, repeats):
    days = np.arange(0, T, 21)
    w = np.where((np.arange(len(days)) % 2) == 0, 0.9, 0.0)
    keep = w != 0
    indptr = np.concatenate(([0], np.cumsum(keep)))
    target = bt.Targets(days, indptr, np.zeros(keep.sum(), np.int64), w[keep])
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        bt.run(m, target, bt.Costs(), bt.Config(initial_cash=1e7))
        samples.append((perf_counter() - start) / T)
    return samples


def backtrader_bars(repeats):
    try:
        import backtrader as btr
        import pandas as pd
    except ImportError:
        return None
    rng = np.random.default_rng(7)
    p = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    frame = pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "volume": 1e8},
        index=pd.bdate_range("2000-01-03", periods=T),
    )

    class Strategy(btr.Strategy):
        def next(self):
            day = len(self.data) - 1
            self.broker.getcash()
            _ = self.getposition(self.data).size
            if day % 21 == 0:
                w = 0.9 if (day // 21) % 2 == 0 else 0.0
                self.order_target_percent(self.data, target=w)

    samples = []
    for _ in range(repeats):
        cerebro = btr.Cerebro(stdstats=False, preload=True, runonce=True)
        cerebro.adddata(btr.feeds.PandasData(dataname=frame))
        cerebro.addstrategy(Strategy)
        cerebro.broker.setcash(1e7)
        start = perf_counter()
        cerebro.run()
        samples.append((perf_counter() - start) / T)
    return samples


def main():
    assert_current_build()
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "load_start": os.getloadavg(),
        "T": T,
    }
    small, large = market(1), market(5500)
    for name, fn in [
        ("panelbt_step_n1", lambda: panelbt_steps(small, 5)),
        ("panelbt_step_n5500", lambda: panelbt_steps(large, 5)),
        ("panelbt_bare_step_n1", lambda: panelbt_bare_steps(small, 5)),
        ("panelbt_bare_step_n5500", lambda: panelbt_bare_steps(large, 5)),
        ("panelbt_run_n1", lambda: panelbt_run(small, 5)),
        ("backtrader_n1", lambda: backtrader_bars(3)),
    ]:
        samples = fn()
        if samples is None:
            continue
        report[name] = {
            "median_us": median(samples) * 1e6,
            "samples_us": [x * 1e6 for x in samples],
        }
        print(f"{name}: {median(samples) * 1e6:.2f} us per row", flush=True)
    report["load_end"] = os.getloadavg()
    out = ROOT / ".cache/panelbt/step_overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
