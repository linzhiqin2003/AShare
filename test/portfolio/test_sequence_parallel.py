from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ashare
from ashare.portfolio import (
    Constraints,
    PortfolioProblem,
    independent_periods,
    objective,
    optimize,
    optimize_sequence,
    violations,
)
from ashare.risk import RiskModel

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def period(seed, n=30, **kwargs):
    rng = np.random.default_rng(seed)
    k = 5
    model = RiskModel(
        tuple(f"s{i}" for i in range(n)),
        rng.normal(size=(n, k)),
        tuple("abcde"),
        np.diag(rng.uniform(1e-4, 4e-4, k)),
        rng.uniform(1e-4, 6e-4, n),
    )
    bench = rng.uniform(0.5, 1.5, n)
    return PortfolioProblem(
        rng.normal(0, 1e-3, n),
        model,
        bench / bench.sum(),
        np.zeros(n),
        constraints=Constraints(max_weight=0.1),
        **kwargs,
    )


def periods(count=6, **kwargs):
    return [period(seed, **kwargs) for seed in range(count)]


def test_independent_periods_is_conservative():
    assert independent_periods(periods())
    base = periods(3)
    for change in (
        dict(buy_cost=1e-3),
        dict(sell_cost=np.r_[1e-3, np.zeros(29)]),
        dict(impact=1e-4),
        dict(constraints=Constraints(max_weight=0.1, turnover=0.5)),
    ):
        mixed = [base[0], replace(base[1], **change), base[2]]
        assert not independent_periods(mixed), change
    frozen = period(1)
    frozen.can_sell[3] = False
    assert not independent_periods([base[0], frozen])
    blocked = period(1)
    blocked.can_buy[0] = False
    assert not independent_periods([blocked])


def test_parallel_matches_serial_within_tolerance():
    plist = periods(8)
    initial = np.full(30, 1 / 30)
    serial, st = optimize_sequence(np.arange(8), plist, initial=initial)
    par, pt = optimize_sequence(
        np.arange(8), plist, initial=initial, parallel=True, threads=4
    )
    assert all(r.success for r in par)
    for p, a, b in zip(plist, serial, par, strict=True):
        # 独立核验：并行解满足原问题全部约束，目标值与串行在容差内相同。
        assert max(violations(p, b.weights).values()) <= 1e-7
        assert b.objective == pytest.approx(objective(p, b.weights), abs=1e-15)
        assert abs(a.objective - b.objective) <= 1e-9
        np.testing.assert_allclose(a.weights, b.weights, atol=1e-5)
    np.testing.assert_array_equal(st.days, pt.days)
    # 第一期的当前持仓都是 initial，两条路径逐位相同；不用 initial 则不同。
    np.testing.assert_array_equal(serial[0].weights, par[0].weights)
    assert not np.array_equal(optimize(plist[0]).weights, par[0].weights)


def test_parallel_is_deterministic_across_thread_counts():
    plist = periods(7)
    runs = [
        optimize_sequence(np.arange(7), plist, parallel=True, threads=t)[0]
        for t in (1, 3, None)
    ]
    for other in runs[1:]:
        for a, b in zip(runs[0], other, strict=True):
            np.testing.assert_array_equal(a.weights, b.weights)
    # 每期等于对该问题单独调用 optimize（当前持仓取问题自带的）。
    for p, r in zip(plist[1:], runs[0][1:], strict=True):
        np.testing.assert_array_equal(optimize(p).weights, r.weights)


@pytest.mark.parametrize(
    "change",
    [
        dict(buy_cost=5e-4),
        dict(impact=1e-3),
        dict(constraints=Constraints(max_weight=0.1, turnover=0.3)),
    ],
)
def test_dependent_sequence_stays_serial_bitwise(change):
    plist = [replace(p, **change) for p in periods(4)]
    serial, _ = optimize_sequence(np.arange(4), plist)
    par, _ = optimize_sequence(np.arange(4), plist, parallel=True, threads=4)
    for a, b in zip(serial, par, strict=True):
        np.testing.assert_array_equal(a.weights, b.weights)


def test_frozen_period_stays_serial_bitwise():
    plist = periods(4)
    plist[2].can_sell[0] = False
    serial, _ = optimize_sequence(np.arange(4), plist)
    par, _ = optimize_sequence(np.arange(4), plist, parallel=True)
    for a, b in zip(serial, par, strict=True):
        np.testing.assert_array_equal(a.weights, b.weights)


def test_failure_resumes_serially_with_previous_holdings():
    plist = periods(6)
    plist[3] = replace(plist[3], constraints=Constraints(max_weight=0.01))
    serial, _ = optimize_sequence(np.arange(6), plist)
    par, _ = optimize_sequence(np.arange(6), plist, parallel=True, threads=3)
    assert [r.success for r in par] == [True, True, True, False, True, True]
    assert par[3].status == "fallback_previous"
    np.testing.assert_array_equal(par[3].weights, par[2].weights)
    # 失败期之后与串行一样以前一期结果为持仓递推；与串行只差在前几期的容差。
    tail, _ = optimize_sequence(np.arange(3), plist[3:], initial=par[2].weights)
    for a, b in zip(tail, par[3:], strict=True):
        np.testing.assert_array_equal(a.weights, b.weights)
    assert serial[3].status == "fallback_previous"


def test_parallel_first_period_failure_and_initial():
    plist = periods(3)
    plist[0] = replace(plist[0], constraints=Constraints(max_weight=0.01))
    initial = np.full(30, 1 / 30)
    serial, _ = optimize_sequence(np.arange(3), plist, initial=initial)
    par, _ = optimize_sequence(np.arange(3), plist, initial=initial, parallel=True)
    for a, b in zip(serial, par, strict=True):
        np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(par[0].weights, initial)


def test_parallel_arguments_and_single_period():
    plist = periods(2)
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="threads"):
            optimize_sequence(np.arange(2), plist, parallel=True, threads=bad)
    one, _ = optimize_sequence([0], plist[:1], parallel=True)
    ref, _ = optimize_sequence([0], plist[:1])
    np.testing.assert_array_equal(one[0].weights, ref[0].weights)
    other = period(9, n=29)
    with pytest.raises(ValueError, match="主证券表"):
        optimize_sequence(np.arange(2), [plist[0], other], parallel=True)
