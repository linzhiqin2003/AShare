"""Hand calculations for the remaining 1b review gaps."""

from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest

from .test_extended import ZERO
from .test_review import engine as engine
from .test_review import simulate


@pytest.mark.parametrize("seed", [23425, 38050, 38190])
def test_scaled_dust_seed_regressions(seed):
    from panelbt import reference

    from .comparison import assert_results_equal
    from .generators import case

    args = case(seed)
    assert_results_equal(bt.run(*args), reference.run(*args))


@pytest.mark.parametrize("maintenance", [0.0, 0.3])
def test_scaled_opening_dust_does_not_fill(engine, maintenance):
    first = np.nextafter(1000.0, 0.0) / 10.0
    r = simulate(
        engine,
        [[10, 10], [10, 10]],
        [[first, 0], [first, 100]],
        margin=bt.MarginModel(long_maintenance=maintenance),
    )
    assert len(r.trades) == 1
    np.testing.assert_array_equal(r.positions[:, 1], [0, 0])
    assert 0 <= r.cash[-1] < 1e-9


def test_explicit_maintenance_selects_margin_account(engine):
    cash = simulate(engine, [[10], [11]], [[100], [np.nan]], close=[[np.nan], [11]])
    margin = simulate(
        engine,
        [[10], [11]],
        [[100], [np.nan]],
        close=[[np.nan], [11]],
        margin=bt.MarginModel(long_maintenance=0.3),
    )
    np.testing.assert_array_equal(cash.nav, [0, 1.1])
    np.testing.assert_array_equal(margin.nav, [1, 1.1])
    for maintenance in [0.0, 0.3]:
        r = simulate(
            engine,
            [[10]],
            [[100]],
            costs=replace(ZERO, commission_rate=0.01),
            margin=bt.MarginModel(long_maintenance=maintenance),
        )
        assert r.positions[0, 0] == pytest.approx(100 / 1.01)
        assert r.diagnostics["cash_scaled"] == int(maintenance == 0)
        assert r.coverage["margin_scaled"] == int(maintenance != 0)


def test_invalid_opening_quote_is_counted(engine):
    r = simulate(
        engine,
        [[10]],
        [[-10]],
        market={"volume": [[1]]},
        costs=replace(ZERO, impact_k=np.array([[2.0]])),
        margin=bt.MarginModel(allow_short=True),
    )
    assert r.trades == []
    np.testing.assert_array_equal(r.n_invalid_quote, [1])
    assert r.coverage["invalid_quote"] == 1
    assert r.n_blocked_sell[0] == 1


@pytest.mark.parametrize(
    "kind,target,remaining", [("shares", 10, 10), ("weights", 0.1, 5)]
)
def test_forced_day_reduces_long_from_new_row(engine, kind, target, remaining):
    r = simulate(
        engine,
        [[10], [5], [5]],
        [[150 if kind == "shares" else 1.5], [np.nan], [target]],
        kind=kind,
        margin=bt.MarginModel(
            allow_financing=True, long_initial=0.5, long_maintenance=0.4
        ),
    )
    np.testing.assert_array_equal(r.margin_call, [False, True, False])
    assert r.positions[-1, 0] == remaining
    assert [(x.day, x.qty) for x in r.liquidation_events] == [(2, remaining - 150)]
    np.testing.assert_array_equal(r.equity, [1000, 250, 250])


@pytest.mark.parametrize("dividend", [0.1, 1.0])
def test_simultaneous_dividend_and_split_order(engine, dividend):
    r = simulate(
        engine,
        [[10], [np.nan], [(10 - dividend) / 2]],
        [[100], [np.nan], [np.nan]],
        market={"split": [[1], [2], [1]], "cash_div": [[0], [dividend], [0]]},
    )
    np.testing.assert_allclose(r.equity, [1000, 1000, 1000], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(r.cash, [0, 100 * dividend, 100 * dividend])
    np.testing.assert_array_equal(r.positions[:, 0], [100, 200, 200])


def test_reentry_reconstructs_unheld_action_marks(engine):
    r = simulate(
        engine,
        [[10], [10], [np.nan], [4.5]],
        [[100], [0], [np.nan], [100]],
        close=[[10], [10], [np.nan], [np.nan]],
        market={"split": [[1], [1], [2], [1]], "cash_div": [[0], [0], [1], [0]]},
    )
    np.testing.assert_array_equal(r.equity, [1000] * 4)
    assert r.long_value[-1] == 450
    assert r.dividend_gross.sum() == 0


def test_maintenance_comparison_has_relative_tolerance(engine):
    for extra, called in [(5e-11, False), (2e-10, True)]:
        r = simulate(
            engine,
            [[10]],
            [[100]],
            margin=bt.MarginModel(long_maintenance=1 + extra),
        )
        assert bool(r.margin_call[0]) is called


def test_unvalued_margin_position_defers_bankruptcy(engine):
    # An oversized distribution floors the carried mark at zero; no current
    # quote exists to resolve the value. A later valid mark resolves insolvency.
    r = simulate(
        engine,
        [[10], [np.nan], [1]],
        [[-100], [np.nan], [np.nan]],
        market={"cash_div": [[0], [30], [0]]},
        margin=bt.MarginModel(allow_short=True, margin_call="record"),
    )
    np.testing.assert_array_equal(r.equity, [1000, -1000, -1100])
    assert r.bankruptcy_events == [2]


def test_bankruptcy_materiality_counter_boundary(engine):
    r = simulate(
        engine,
        [[1e-12]],
        [[2]],
        close=[[4e-13]],
        initial_cash=1e-12,
        margin=bt.MarginModel(allow_financing=True, long_initial=0.5),
    )
    assert r.equity[0] < 0 and r.bankruptcy_events == [0]
    assert r.coverage["bankruptcy"] == 1
    assert r.coverage["genuine_bankruptcy"] == 0


def test_interest_coverage_uses_material_threshold(engine):
    for rate, count in [(5e-11, 0), (2e-10, 1)]:
        r = simulate(engine, [[10]], [[0]], initial_cash=1, cash_rate=rate)
        assert r.cash_interest[0] == rate
        assert r.coverage["cash_interest"] == count
        r = simulate(
            engine,
            [[1]],
            [[2]],
            initial_cash=1,
            margin=bt.MarginModel(
                allow_financing=True, long_initial=0.5, financing_rate=rate
            ),
        )
        assert r.financing_interest[0] == rate
        assert r.coverage["financing_interest"] == count
