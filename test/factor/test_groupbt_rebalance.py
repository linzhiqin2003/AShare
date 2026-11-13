"""调仓日历、固定权重冻结及稀疏实际持仓的独立 NumPy 对照。"""

import numpy as np
import pytest

from ashare.data.reader import Panel
from ashare.factor import groupbt
from bench.platform.synthetic import panel


def sample():
    original = panel(n=20)
    arrays = {k: v[:23].copy() for k, v in original._arrays.items()}
    dates = np.char.replace(
        np.arange("2025-01-01", "2025-02-03", dtype="datetime64[D]")[
            np.is_busday(np.arange("2025-01-01", "2025-02-03", dtype="datetime64[D]"))
        ].astype(str),
        "-",
        "",
    )
    rng = np.random.default_rng(39)
    factor = rng.normal(size=arrays["close"].shape)
    arrays["can_sell_open"][6, :8] = False
    arrays["can_buy_open"][11, 8:] = False
    return Panel(dates, original.symbols, arrays, "synthetic"), factor


def reference(factor, source, n, phase=0):
    rows, cols = factor.shape
    held = np.zeros(cols)
    pnl, turn = np.zeros(rows), np.zeros(rows)
    snapshots = []
    for day in range(1, rows):
        if (day - 1 - phase) % n == 0:
            order = np.argsort(-factor[day - 1], kind="stable")
            target = np.zeros(cols)
            target[order[: cols // 5]] = 0.5 / (cols // 5)
            target[order[-(cols // 5) :]] = -0.5 / (cols // 5)
            frozen = np.zeros(cols, bool)
            while True:
                current = np.zeros(cols)
                for positive in (True, False):
                    old_side = held > 0 if positive else held < 0
                    new_side = target > 0 if positive else target < 0
                    used = abs(held[frozen & old_side].sum())
                    free = abs(target[~frozen & new_side].sum())
                    if free:
                        current[~frozen & new_side] = (
                            target[~frozen & new_side] * max(0, 0.5 - used) / free
                        )
                current[frozen] = held[frozen]
                blocked = ((current > held + 1e-12) & ~source["can_buy_open"][day]) | (
                    (current < held - 1e-12) & ~source["can_sell_open"][day]
                )
                if not np.any(blocked & ~frozen):
                    break
                frozen |= blocked
            turn[day] = np.abs(current - held).sum()
            held = current
        snapshots.append(held.copy())
        if day + 1 < rows:
            returns = source["open"][day + 1] / source["open"][day] - 1
            pnl[day] = np.dot(held, returns)
    pnl[[0, -1]] = np.nan
    turn[[0, -1]] = np.nan
    return pnl, turn, snapshots


@pytest.mark.parametrize("n", [1, 5, 10, 20])
def test_numpy_reference_and_calendar_mutation(n):
    source, factor = sample()
    result = groupbt.run(
        factor,
        source,
        "ALL",
        n_groups=5,
        sub_universes=False,
        window=groupbt.VALIDATION,
        rebalance=n,
        record_holdings=True,
        weights_portfolio=True,
        decay=3,
    )
    pnl, turn, held = reference(factor, source, n)
    # NumPy 全宽点积与内核紧凑点积的归约顺序不同，允许 1e-15 的绝对舍入差。
    np.testing.assert_allclose(result.pnl["long_short"], pnl, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.turnover["long_short"], turn, rtol=0, atol=1e-15)
    for i, day in enumerate(result.holdings["days"]):
        a, b = result.holdings["offsets"][i : i + 2]
        dense = np.zeros(len(source.symbols))
        dense[result.holdings["columns"][a:b]] = result.holdings["weights"][a:b]
        np.testing.assert_allclose(dense, held[day - 1], rtol=0, atol=1e-15)
    if n > 1:
        nonscheduled = (np.arange(len(source.dates)) - 1) % n != 0
        nonscheduled[[0, -1]] = False
        np.testing.assert_array_equal(result.turnover["long_short"][nonscheduled], 0)
        wrong, _, _ = reference(factor, source, n, phase=1)
        assert not np.allclose(pnl, wrong, equal_nan=True, rtol=0, atol=1e-15)


def test_default_and_recording_preserve_every_output():
    from bench.groupbt.common import compare, snapshot, synthetic

    source, factor = synthetic()
    options = dict(
        universe="ALL", weights_portfolio=True, group_neutralize="cap+industry"
    )
    compare(
        snapshot(groupbt.run(factor, source, **options)),
        snapshot(
            groupbt.run(factor, source, **options, rebalance=1, record_holdings=True)
        ),
    )


def test_window_starts_its_own_schedule():
    source, factor = sample()
    for start in (source.dates[0], source.dates[3]):
        result = groupbt.run(
            factor,
            source,
            "ALL",
            sub_universes=False,
            window=(start, "20260101"),
            rebalance=5,
            record_holdings=True,
        )
        first = np.searchsorted(source.dates, start)
        np.testing.assert_array_equal(
            result.holdings["days"], np.arange(first + 1, len(source.dates), 5)
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "5"])
def test_invalid_period(value):
    source, factor = sample()
    with pytest.raises(ValueError):
        groupbt.run(factor, source, "ALL", rebalance=value)
