"""逐日步进接口：与 bt.run 逐位相同、防前视、存档恢复、多线程。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.models import METRICS

from . import generators, generators_futures
from .test_extended import extended_case

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def assert_bitwise(actual, expected):
    for name in (*METRICS, "positions", "contribution"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b, name
            continue
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name
    assert actual.diagnostics == expected.diagnostics
    assert actual.coverage == expected.coverage
    assert actual.bankruptcy_events == expected.bankruptcy_events
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b, name
            continue
        bits = [np.asarray(astuple(x), dtype=float).tobytes() for x in a]
        assert bits == [np.asarray(astuple(x), dtype=float).tobytes() for x in b], name


def rows(target):
    """The plan as {day: (cols, values)}."""
    out = {}
    for k, day in enumerate(target.days.tolist()):
        a, b = target.indptr[k], target.indptr[k + 1]
        out[day] = (target.cols[a:b], target.weights[a:b])
    return out


def step_all(market, target, costs, config, *, dense=False, stop=None):
    sim = bt.Simulator(market, costs, config, kind=target.kind)
    plan = rows(target)
    n = market.shape[1]
    for day in range(market.shape[0] if stop is None else stop):
        row = plan.get(day)
        if dense and row is not None:
            full = np.zeros(n)
            full[row[0]] = row[1]
            # An all-zero dense row is still a target row (it closes positions).
            row = full if full.any() else row
        sim.step(row)
    return sim


CORPORA = {
    "cash": generators.case,
    "margin": extended_case,
    "futures": generators_futures.case,
    "futures_rich": generators_futures.rich_case,
}


@pytest.mark.parametrize("corpus", sorted(CORPORA))
def test_stepping_equals_run_bitwise(corpus):
    make = CORPORA[corpus]
    touched = {"trades": 0, "blocked": 0, "liquidation": 0, "expiry": 0}
    for seed in range(300):
        market, target, costs, config = make(seed)
        expected = bt.run(market, target, costs, config)
        sim = step_all(market, target, costs, config, dense=seed % 3 == 0)
        assert_bitwise(sim.result(), expected)
        touched["trades"] += expected.diagnostics["trades"]
        touched["blocked"] += expected.diagnostics["blocked_buy"]
        touched["liquidation"] += expected.coverage["liquidation"]
        touched["expiry"] += expected.coverage["futures_expiry"]
    # The corpora must reach the branches they are meant to cover.
    assert touched["trades"] > 0 and touched["blocked"] > 0, touched
    if corpus == "margin":
        assert touched["liquidation"] > 0, touched
    if corpus.startswith("futures"):
        assert touched["expiry"] > 0, touched


def test_cash_account_record_trades_off_and_batch_equivalence():
    for seed in range(60):
        market, target, costs, config = generators.case(seed)
        config = replace(config, record_trades=False)
        expected = bt.run(market, target, costs, config)
        assert_bitwise(step_all(market, target, costs, config).result(), expected)


def state_copy(sim):
    s = sim.state
    return (
        s.day,
        s.cash,
        s.equity,
        s.margin_call,
        s.bankrupt,
        s.shares.copy(),
        s.cost.copy(),
        s.entry_day.copy(),
        s.last.copy(),
        s.futures_basis.copy(),
        s.pending_cols.copy(),
        s.pending_targets.copy(),
    )


def states_equal(a, b):
    for x, y in zip(a, b, strict=True):
        if isinstance(x, np.ndarray):
            assert x.tobytes() == y.tobytes()
        else:
            assert x == y or (x != x and y != y)


def perturb(market, t, rng):
    """Change every price/mask/volume on rows >= t and actions on rows > t."""
    shape = market.shape
    future = np.arange(shape[0])[:, None] >= t
    spot = np.ones(shape[1], bool)
    if market.futures is not None:
        spot = ~np.asarray(market.futures.is_future)
    # Futures columns cannot carry corporate actions or delisting.
    later = (np.arange(shape[0])[:, None] > t) & spot[None, :]
    noise = np.exp(rng.normal(0, 0.2, shape))
    kw = {}
    for name in ("exec_price", "close", "volume"):
        a = getattr(market, name)
        if a is not None:
            kw[name] = np.where(future, a * noise, a)
    for name in ("can_buy", "can_sell", "can_short"):
        a = getattr(market, name)
        if a is not None:
            kw[name] = np.where(future, rng.random(shape) < 0.5, a)
    if market.split is not None:
        kw["split"] = np.where(later, 1.0, market.split)
    if market.cash_div is not None:
        kw["cash_div"] = np.where(later, 0.3, market.cash_div)
    if market.delist is not None:
        kw["delist"] = np.where(later, rng.random(shape) < 0.3, market.delist)
    if market.futures is not None:
        f = market.futures
        settle = np.where(future, f.settlement * noise, f.settlement)
        kw["futures"] = replace(f, settlement=settle)
    return replace(market, **kw)


@pytest.mark.parametrize("corpus", sorted(CORPORA))
def test_state_before_step_t_ignores_rows_from_t(corpus):
    make = CORPORA[corpus]
    for seed in range(80):
        market, target, costs, config = make(seed)
        rng = np.random.default_rng(seed)
        t = int(rng.integers(0, market.shape[0]))
        plan = rows(target)
        first = bt.Simulator(market, costs, config, kind=target.kind)
        before = []
        for day in range(t):
            before.append(state_copy(first))
            first.step(plan.get(day))
        before.append(state_copy(first))
        changed = perturb(market, t, rng)
        second = bt.Simulator(changed, costs, config, kind=target.kind)
        for day in range(t):
            states_equal(state_copy(second), before[day])
            second.step(plan.get(day))
        states_equal(state_copy(second), before[t])


def test_perturbation_is_not_vacuous():
    """The perturbed market really changes the result after row t."""
    changed = 0
    for seed in range(40):
        market, target, costs, config = generators.case(seed)
        t = market.shape[0] // 2
        other = perturb(market, t, np.random.default_rng(seed))
        a = bt.run(market, target, costs, config)
        b = bt.run(other, target, costs, config)
        assert a.nav[:t].tobytes() == b.nav[:t].tobytes()
        changed += a.nav.tobytes() != b.nav.tobytes()
    assert changed > 30


@pytest.mark.parametrize("corpus", sorted(CORPORA))
def test_snapshot_restore_continues_bitwise(corpus):
    make = CORPORA[corpus]
    for seed in range(60):
        market, target, costs, config = make(seed)
        stop = int(np.random.default_rng(seed).integers(0, market.shape[0] + 1))
        plan = rows(target)
        sim = step_all(market, target, costs, config, stop=stop)
        blob = sim.snapshot()
        resumed = bt.Simulator.restore(blob, market, costs, config, kind=target.kind)
        states_equal(state_copy(resumed), state_copy(sim))
        for day in range(stop, market.shape[0]):
            resumed.step(plan.get(day))
            sim.step(plan.get(day))
        expected = bt.run(market, target, costs, config)
        assert_bitwise(resumed.result(), expected)
        assert_bitwise(sim.result(), expected)


def test_snapshot_rejects_other_settings_and_garbage():
    market, target, costs, config = generators.case(3)
    sim = step_all(market, target, costs, config, stop=2)
    blob = sim.snapshot()
    with pytest.raises(ValueError, match="differ"):
        bt.Simulator.restore(blob, market, costs, replace(config, cash_buffer=0.5))
    with pytest.raises(ValueError, match="differ"):
        bt.Simulator.restore(blob, market, costs, config, kind="shares")
    with pytest.raises(ValueError, match="not a panelbt"):
        bt.Simulator.restore(b"garbage", market, costs, config)
    with pytest.raises(ValueError, match="truncated|inconsistent"):
        bt.Simulator.restore(blob[:-9], market, costs, config)
    # A failed restore leaves no half-loaded simulator behind.
    good = bt.Simulator.restore(blob, market, costs, config)
    assert good.day == 2


def test_parallel_threads_match_sequential():
    cases = [generators.case(seed) for seed in range(24)]
    expected = [bt.run(*c) for c in cases]

    def work(c):
        return step_all(*c).result()

    with ThreadPoolExecutor(6) as pool:
        for actual, exp in zip(pool.map(work, cases), expected, strict=True):
            assert_bitwise(actual, exp)


def test_errors_and_lifecycle():
    market, target, costs, config = generators.case(5)
    sim = bt.Simulator(market, costs, config)
    with pytest.raises(ValueError, match="unfinished"):
        sim.result()
    with pytest.raises(ValueError, match="allow_short"):
        sim.step(([0], [-0.5]))
    with pytest.raises(ValueError, match="row sum"):
        sim.step(([0, 1], [0.8, 0.8]))
    with pytest.raises(ValueError, match="index < N"):
        sim.step(([market.shape[1]], [0.1]))
    with pytest.raises(ValueError, match="duplicate"):
        sim.step(([1, 1], [0.1, 0.1]))
    with pytest.raises(ValueError, match="length N"):
        sim.step(np.zeros(market.shape[1] + 1))
    assert sim.day == 0  # rejected targets do not advance the simulation
    while sim.day < sim.n_days:
        sim.step()
    with pytest.raises(ValueError, match="finished"):
        sim.step()
    assert sim.result().nav.shape == (sim.n_days,)


def test_state_fields_cost_entry_split_and_views():
    t, n = 6, 2
    price = np.array([[10.0, 5], [12, 5], [12, 5], [8, 5], [8, 5], [9, 5]])
    split = np.ones((t, n))
    split[3, 0] = 1.5
    ones = np.ones((t, n), bool)
    market = bt.Market(price, price, ones, ones, split=split)
    costs = bt.Costs(
        commission_rate=0,
        commission_min=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0.01,
    )
    config = bt.Config(initial_cash=1e4, min_qty=0, qty_step=0)
    sim = bt.Simulator(market, costs, config, kind="shares")
    s = sim.state
    assert (s.day, s.cash, s.equity) == (0, 1e4, 1e4)
    assert not s.shares.flags.writeable
    with pytest.raises(ValueError):
        s.shares[0] = 1
    sim.step(([0], [100.0]))
    assert s.shares[0] == 100  # live view
    assert sim.state.entry_day[0] == 0
    assert sim.state.cost[0] == 10  # raw price, slippage excluded
    sim.step(([0], [300.0]))
    assert sim.state.cost[0] == pytest.approx((100 * 10 + 200 * 12) / 300)
    sim.step(([0], [200.0]))  # a reduction keeps the cost and entry
    # Row 3's 1.5 split is already applied: it is known before row 3 trades.
    assert sim.state.day == 3
    assert sim.state.shares[0] == 300
    assert sim.state.cost[0] == pytest.approx((100 * 10 + 200 * 12) / 300 / 1.5)
    assert sim.state.entry_day[0] == 0
    sim.step(([], []))  # an empty row closes everything
    assert sim.state.shares[0] == 0 and sim.state.entry_day[0] == -1
    assert sim.state.cost[0] == 0


def test_run_helper_and_day_record():
    market, target, costs, config = generators.case(9)
    plan = rows(target)
    seen = []

    def strategy(day, state):
        seen.append((day, state.day))
        return plan.get(day)

    sim = bt.Simulator(market, costs, config)
    result = sim.run(strategy)
    assert all(a == b for a, b in seen) and len(seen) == market.shape[0]
    assert_bitwise(result, bt.run(market, target, costs, config))
    again = bt.Simulator(market, costs, config)
    record = again.step(plan.get(0))
    assert record.day == 0 and record.nav == result.nav[0]
    assert record.equity == result.equity[0]


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_stepping_example_runs(name):
    import re

    text = (Path(bt.__file__).resolve().parents[2] / name).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    stepping = [b for b in blocks if "bt.Simulator(" in b]
    assert stepping
    ns = {}
    exec(blocks[0], ns)  # the minimal example defines market, costs, config
    for block in stepping:
        exec(block, ns)
    expected = bt.run(
        ns["market"],
        bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]]),
        ns["costs"],
        ns["config"],
    )
    assert_bitwise(ns["result"], expected)
    assert (
        bt.Simulator.restore(ns["blob"], ns["market"], ns["costs"], ns["config"]).day
        == 3
    )
