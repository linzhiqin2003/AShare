"""Independent hand-calculated regression cases from the 1b review."""

from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest
from panelbt import reference
from panelbt.metrics import summarize

from .test_extended import ZERO


@pytest.fixture(params=[bt.run, reference.run], ids=["rust", "reference"])
def engine(request):
    return request.param


def simulate(
    engine,
    prices,
    targets,
    *,
    close=None,
    margin=None,
    costs=ZERO,
    kind="shares",
    market=None,
    **config,
):
    p = np.asarray(prices, float)
    options = dict(market or {})
    buy = options.pop("can_buy", np.ones_like(p, bool))
    sell = options.pop("can_sell", np.ones_like(p, bool))
    m = bt.Market(p, p if close is None else close, buy, sell, **options)
    g = dict(initial_cash=1000, min_qty=0, qty_step=0, record_trades=True)
    g.update(config)
    return engine(
        m,
        bt.Targets.from_dense(targets, kind=kind),
        costs,
        bt.Config(margin=margin or bt.MarginModel(), **g),
    )


def test_cash_account_missing_mark_is_not_bankrupt(engine):
    r = simulate(
        engine,
        [[10], [10], [11], [12]],
        [[100], [np.nan], [100], [100]],
        close=[[np.nan], [10], [11], [12]],
    )
    np.testing.assert_array_equal(r.nav, [0, 1, 1.1, 1.2])
    np.testing.assert_array_equal(r.positions[:, 0], [100] * 4)
    assert not r.bankruptcy_events and not r.liquidation_events


def test_margin_missing_mark_uses_execution_price(engine):
    r = simulate(
        engine,
        [[10], [11]],
        [[150], [np.nan]],
        close=[[np.nan], [11]],
        margin=bt.MarginModel(
            allow_financing=True, long_initial=0.5, long_maintenance=0.3
        ),
    )
    np.testing.assert_array_equal(r.equity, [1000, 1150])
    assert not r.bankruptcy_events


@pytest.mark.parametrize(
    "action,values,last",
    [("split", [[1], [2], [1]], 5), ("cash_div", [[0], [1], [0]], 9)],
)
def test_suspended_action_preserves_wealth(engine, action, values, last):
    r = simulate(
        engine,
        [[10], [np.nan], [last]],
        [[100], [np.nan], [np.nan]],
        market={action: values},
    )
    np.testing.assert_array_equal(r.nav, [1, 1, 1])


def test_short_action_does_not_force_other_long(engine):
    r = simulate(
        engine,
        [[10, 10], [np.nan, 10], [np.nan, 10], [5, 10]],
        [[-50, 40], [np.nan] * 2, [np.nan] * 2, [np.nan] * 2],
        market={"split": [[1, 1], [2, 1], [1, 1], [1, 1]]},
        margin=bt.MarginModel(allow_short=True, short_maintenance=0.8),
    )
    np.testing.assert_array_equal(r.short_value, [500] * 4)
    assert not r.margin_call.any() and not r.liquidation_events
    np.testing.assert_array_equal(r.positions[:, 1], [40] * 4)


def test_empty_account_can_resume_after_deposit(engine):
    for margin in [bt.MarginModel(), bt.MarginModel(allow_short=True)]:
        r = simulate(
            engine,
            [[10]] * 4,
            [[np.nan], [np.nan], [100], [100]],
            cash_flow=[0, -1000, 1000, 0],
            margin=margin,
        )
        np.testing.assert_array_equal(r.positions[:, 0], [0, 0, 100, 100])
        assert not r.bankruptcy_events


def test_restricted_proceeds_interest_bases(engine):
    r = simulate(
        engine,
        [[10], [10]],
        [[-50], [np.nan]],
        margin=bt.MarginModel(allow_short=True, financing_rate=0.01),
        cash_rate=0.01,
    )
    np.testing.assert_allclose(r.cash_interest, [10, 10.1], rtol=1e-14)
    np.testing.assert_array_equal(r.financing_interest, [0, 0])
    # 1200 long + 600 short needs 900 margin; C=400, own=-200.
    r = simulate(
        engine,
        [[10, 10]],
        [[120, -60]],
        margin=bt.MarginModel(
            allow_short=True,
            long_initial=0.5,
            short_initial=0.5,
            allow_financing=True,
            financing_rate=0.01,
        ),
        cash_rate=0.02,
    )
    np.testing.assert_array_equal(r.cash_interest, [0])
    np.testing.assert_array_equal(r.financing_interest, [2])
    np.testing.assert_array_equal(r.cash, [398])
    # Review probe D2: 1500 long + 600 short initially needs 1050 margin.
    # The 20/21 scale leaves C=1000/7, S=4000/7, hence interest=30/7.
    r = simulate(
        engine,
        [[10, 10]],
        [[150, -60]],
        margin=bt.MarginModel(
            allow_short=True,
            long_initial=0.5,
            short_initial=0.5,
            allow_financing=True,
            financing_rate=0.01,
        ),
        cash_rate=0.02,
    )
    np.testing.assert_allclose(r.positions[0], [1000 / 7, -400 / 7], rtol=1e-14)
    np.testing.assert_allclose(r.financing_interest, [30 / 7], rtol=1e-14)
    np.testing.assert_allclose(r.cash, [970 / 7], rtol=1e-14)
    np.testing.assert_array_equal(r.cash_interest, [0])


def test_reversal_shares_daily_volume(engine):
    r = simulate(
        engine,
        [[10], [10]],
        [[20], [-20]],
        market={"volume": [[100], [100]]},
        max_participation=0.3,
        margin=bt.MarginModel(allow_short=True),
    )
    np.testing.assert_array_equal(r.positions[:, 0], [20, -10])
    assert [(x.day, x.qty) for x in r.trades] == [(0, 20), (1, -20), (1, -10)]


def test_opening_and_reduction_order(engine):
    mm = bt.MarginModel(allow_short=True)
    r = simulate(engine, [[10, 10], [10, 10]], [[-40, 40], [0, 0]], margin=mm)
    assert [(x.day, x.col, x.qty) for x in r.trades] == [
        (0, 0, -40),
        (0, 1, 40),
        (1, 1, -40),
        (1, 0, 40),
    ]


def test_liquidation_stops_at_filled_plan(engine):
    r = simulate(
        engine,
        [[10], [16], [16], [16.5], [17], [17.5]],
        [[-80], [np.nan], [np.nan], [np.nan], [-5], [np.nan]],
        close=[[10], [16], [16.5], [17], [17.5], [17.5]],
        margin=bt.MarginModel(allow_short=True, short_maintenance=0.8),
    )
    np.testing.assert_array_equal(r.margin_call, [0, 1, 0, 0, 0, 0])
    assert len(r.liquidation_events) == 1
    assert r.positions[4, 0] == -5


def test_unfilled_liquidation_continues_without_maintenance_breach(engine):
    # Maintenance drops after the call. Day 2 cannot cover, so the plan remains due.
    r = simulate(
        engine,
        [[10], [16], [16], [16]],
        [[-80], [np.nan], [np.nan], [np.nan]],
        market={"can_buy": [[1], [1], [0], [1]]},
        margin=bt.MarginModel(allow_short=True, short_maintenance=[0.8, 0.8, 0, 0]),
    )
    np.testing.assert_array_equal(r.margin_call, [0, 1, 1, 0])
    assert [(e.day, e.qty) for e in r.liquidation_events] == [(3, 47.5)]


def test_liquidation_day_honors_reducing_target(engine):
    r = simulate(
        engine,
        [[10], [16], [16]],
        [[-80], [np.nan], [0]],
        margin=bt.MarginModel(allow_short=True, short_maintenance=0.8),
    )
    assert r.positions[-1, 0] == 0
    assert r.liquidation_events[-1].qty == 80


def test_selling_minimum_and_nonzero_small_target(engine):
    for minimum in [100, 200]:
        r = simulate(
            engine,
            [[10]] * 4,
            [[0.5], [0.45], [0.05], [0]],
            kind="weights",
            initial_cash=10000,
            min_qty=minimum,
            qty_step=1,
        )
        np.testing.assert_array_equal(r.positions[:, 0], [500, 500, 50, 0])
        assert [x.qty for x in r.trades] == [500, -450, -50]


@pytest.mark.parametrize(
    "kind,targets,capital",
    [
        ("shares", [[700], [770], [770]], 100000),
        ("weights", [[0.5], [0.5], [0.5]], 14010),
    ],
)
def test_split_dust_has_no_commission(engine, kind, targets, capital):
    # Weight path: 700 shares at 10, cash 7005; split ex-price is 10/1.1.
    r = simulate(
        engine,
        [[10], [10 / 1.1], [10 / 1.1]],
        targets,
        kind=kind,
        costs=replace(ZERO, commission_min=5),
        initial_cash=capital,
        market={"split": [[1], [1.1], [1]]},
    )
    if kind == "shares":
        assert [x.qty for x in r.trades] == [700]
        np.testing.assert_array_equal(r.commission, [5, 0, 0])
    else:
        # Avoid a genuine first-fill commission-induced weight drift.
        r = simulate(
            engine,
            [[10], [10 / 1.1]],
            [[0.5], [0.5]],
            kind=kind,
            initial_cash=14000,
            market={"split": [[1], [1.1]]},
        )
        assert len(r.trades) == 1


def test_reduction_clipped_and_invalid_quote_counted(engine):
    r = simulate(
        engine,
        [[10, 10]] * 3,
        [[98.5, -50], [98.5, 0], [np.nan, np.nan]],
        costs=replace(ZERO, commission_rate=0.01),
        margin=bt.MarginModel(allow_short=True, long_initial=0.5, short_initial=0.5),
    )
    np.testing.assert_array_equal(r.n_reduction_clipped, [0, 1, 0])
    assert r.coverage["reduction_clipped"] == 1
    r = simulate(
        engine,
        [[10]] * 3,
        [[100], [0], [np.nan]],
        initial_cash=100000,
        market={"volume": [[1e9], [1], [1e9]]},
        costs=replace(ZERO, impact_k=np.full((3, 1), 2.0)),
    )
    np.testing.assert_array_equal(r.n_invalid_quote, [0, 1, 0])
    assert r.coverage["invalid_quote"] == 1


def test_metrics_on_negative_equity(engine):
    r = simulate(
        engine,
        [[10], [30], [30]],
        [[-80], [np.nan], [np.nan]],
        margin=bt.MarginModel(allow_short=True, margin_call="record"),
    )
    summary = summarize(r)
    assert summary["bankrupt"] and not summary["geometric_metrics_valid"]
    assert np.isnan(summary["annual_return"])
    assert summary["cumulative_return"] == pytest.approx(-1.6)


def test_reference_does_not_call_prepare(monkeypatch):
    import panelbt.api

    def fail(*a, **kw):
        raise AssertionError("shared preparation")

    monkeypatch.setattr(panelbt.api, "_prepare", fail)
    r = simulate(reference.run, [[10]], [[10]])
    assert r.cash[0] == 900


def test_zero_mark_reduction_does_not_count_blocked_buy(engine):
    r = simulate(
        engine,
        [[10, 10], [np.nan, 10], [10, 10]],
        [[0.4, 0.4], [0.4, 0.4], [np.nan, np.nan]],
        close=[[np.nan, 10], [np.nan, 10], [10, 10]],
        kind="weights",
        initial_cash=10000,
        min_qty=100,
        qty_step=100,
    )
    np.testing.assert_array_equal(r.n_blocked_buy, [0, 0, 0])
    np.testing.assert_array_equal(r.n_blocked_sell, [0, 1, 0])


@pytest.mark.parametrize("seed", [1048, 8257])
def test_coverage_is_stable_at_tiny_cash_residuals(seed):
    from .test_extended import extended_case

    inputs = extended_case(seed)
    a, b = bt.run(*inputs), reference.run(*inputs)
    assert a.coverage == b.coverage
    np.testing.assert_allclose(a.equity, b.equity, rtol=1e-9, atol=1e-9)


def test_volume_cap_cannot_create_subminimum_partial_sale(engine):
    r = simulate(
        engine,
        [[10], [10]],
        [[0.5], [0]],
        kind="weights",
        initial_cash=10000,
        min_qty=200,
        qty_step=1,
        market={"volume": [[1000], [50]]},
        max_participation=1,
    )
    np.testing.assert_array_equal(r.positions[:, 0], [500, 500])
    np.testing.assert_array_equal(r.n_blocked_sell, [0, 1])


def test_prefix_check_when_previous_fees_created_cash_deficit(engine):
    r = simulate(
        engine,
        [[10, 10, 10], [10, 10, 10]],
        [[150, -50, 0], [150, -50.1, -1]],
        margin=bt.MarginModel(
            allow_short=True,
            long_initial=0.2,
            short_initial=0.2,
            short_proceeds_usable=True,
            borrow_rate=0.01,
        ),
    )
    assert r.cash[0] == -5
    # A 0.1-share short cannot repair a five-unit cash deficit. The next
    # one-share short can, and leaves 5 units before the closing borrow charge.
    assert [(x.col, x.qty) for x in r.trades if x.day == 1] == [(2, -1)]
    np.testing.assert_allclose(r.cash, [-5, -0.1], rtol=1e-12, atol=1e-12)
