from dataclasses import replace
from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.models import METRICS

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])

ZERO = bt.Costs(
    commission_rate=0,
    commission_min=0,
    tax_sell=0,
    tax_buy=0,
    fee_rate_buy=0,
    slippage=0,
    fee_rate_sell=0,
)


def simple(prices, weights, *, costs=ZERO, config=None, **overrides):
    p = np.asarray(prices, dtype=float)
    market = bt.Market(
        p,
        overrides.pop("close", p),
        overrides.pop("can_buy", np.ones_like(p, bool)),
        overrides.pop("can_sell", np.ones_like(p, bool)),
        **overrides,
    )
    cfg = config or bt.Config(
        initial_cash=1000, min_qty=0, qty_step=0, record_trades=True
    )
    return bt.run(market, bt.Targets.from_dense(weights), costs, cfg)


def test_sell_before_buy():
    r = simple([[10, 10], [10, 10]], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(r.positions, [[100, 0], [0, 100]])
    assert [x.qty for x in r.trades] == [100, -100, 100]
    assert r.cash[-1] == 0


def test_fees_and_tax_on_slipped_amount():
    cfg = bt.Config(initial_cash=10000, record_trades=True)
    costs = bt.Costs(
        commission_rate=0.001,
        commission_min=5,
        tax_sell=0.001,
        tax_buy=0.0003,
        fee_rate_buy=0.00002,
        slippage=0.0005,
        fee_rate_sell=0.00002,
    )
    r = simple([[10], [10]], [[0.5], [0]], costs=costs, config=cfg)
    buy, sell = r.trades
    assert buy.qty == 500 and sell.qty == -500
    assert buy.price == 10 * (1 + 0.0005)
    assert sell.price == 10 * (1 - 0.0005)
    assert buy.commission == 500 * buy.price * 0.001
    assert sell.commission == 5
    np.testing.assert_allclose(
        r.tax, [500 * buy.price * 0.0003, 500 * sell.price * 0.001], rtol=1e-14
    )
    np.testing.assert_allclose(
        r.other_fee,
        [500 * buy.price * 0.00002, 500 * sell.price * 0.00002],
        rtol=1e-14,
    )
    np.testing.assert_array_equal(r.slippage_cost, [2.5, 2.5])
    assert (
        abs(
            r.cash[-1]
            - (10000 - sum(r.commission + r.tax + r.other_fee + r.slippage_cost))
        )
        < 1e-9
    )


def test_min_commission_and_cash_scaling():
    cfg = bt.Config(initial_cash=10, min_qty=100, qty_step=0, record_trades=True)
    r = simple(
        [[1]],
        [[1]],
        config=cfg,
        costs=bt.Costs(
            commission_rate=0,
            commission_min=5,
            tax_sell=0,
            tax_buy=0,
            fee_rate_buy=0,
            slippage=0,
            fee_rate_sell=0,
        ),
    )
    assert r.positions[0, 0] == 5
    assert r.cash[0] == 0
    assert r.commission[0] == 5
    assert r.diagnostics["cash_scaled"] == 1
    assert r.n_blocked_buy[0] == 0


def test_cash_shortfall_is_not_carried():
    cfg = bt.Config(initial_cash=2000, min_qty=200, qty_step=1, record_trades=True)
    r = simple(
        [[10], [1]],
        [[1], [np.nan]],
        config=cfg,
        costs=bt.Costs(
            commission_rate=0,
            commission_min=5,
            tax_sell=0,
            tax_buy=0,
            fee_rate_buy=0,
            slippage=0,
            fee_rate_sell=0,
        ),
    )
    np.testing.assert_array_equal(r.positions, [[0], [0]])
    assert r.diagnostics["cash_scaled"] == 1
    assert r.n_blocked_buy.sum() == 0


def test_can_sell_and_carry():
    r = simple([[10], [10], [10]], [[1], [0], [np.nan]], can_sell=[[1], [0], [1]])
    np.testing.assert_array_equal(r.positions, [[100], [100], [0]])
    np.testing.assert_array_equal(r.n_blocked_sell, [0, 1, 0])
    assert r.diagnostics["carry_trade"] == 1


def test_carry_only_blocked_security():
    r = simple(
        [[10, 10], [10, 20], [10, 20]],
        [[0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]],
        can_buy=[[0, 1], [1, 1], [1, 1]],
    )
    np.testing.assert_array_equal(r.positions, [[0, 50], [50, 50], [50, 50]])
    assert [(x.day, x.col) for x in r.trades] == [(0, 1), (1, 0)]
    assert r.diagnostics["carry_trade"] == 1


def test_new_target_cancels_carry():
    r = simple([[10, 10], [10, 10]], [[1, 0], [0, 1]], can_buy=[[0, 1], [1, 1]])
    np.testing.assert_array_equal(r.positions, [[0, 0], [0, 100]])


def test_carry_switch():
    cfg = bt.Config(
        initial_cash=1000,
        min_qty=0,
        qty_step=0,
        carry_blocked=False,
        record_trades=True,
    )
    r = simple([[10], [10]], [[1], [np.nan]], can_buy=[[0], [1]], config=cfg)
    assert not r.positions.any()
    assert r.n_blocked_buy[0] == 1


def test_volume_and_odd_lot_liquidation():
    cfg = bt.Config(initial_cash=2000, max_participation=0.1, record_trades=True)
    r = simple(
        [[10], [10], [10], [10]],
        [[0.5], [0], [np.nan], [np.nan]],
        volume=[[1e8], [1000], [np.nan], [1000]],
        split=[[1], [1.5], [1], [1]],
        config=cfg,
    )
    np.testing.assert_array_equal(r.positions, [[100], [50], [50], [0]])
    np.testing.assert_array_equal(r.n_blocked_sell, [0, 1, 1, 0])
    assert r.trades[-1].qty == -50


def test_dividend_before_split_and_odd_buy():
    cfg = bt.Config(initial_cash=2000, record_trades=True)
    r = simple(
        [[10], [5], [5]],
        [[0.5], [0.5], [0]],
        cash_div=[[0], [1], [0]],
        split=[[1], [1.5], [1]],
        config=cfg,
    )
    assert r.cash[1] == 1100
    assert r.positions[1, 0] == 150
    assert r.trades[-1].qty == -150


def test_delist_prior_close_and_no_reentry():
    cfg = bt.Config(
        initial_cash=1000,
        min_qty=0,
        qty_step=0,
        delist_recovery=0.3,
        record_trades=True,
    )
    r = simple(
        [[10], [999], [999]],
        [[1], [1], [1]],
        close=[[12], [999], [999]],
        delist=[[0], [1], [0]],
        config=cfg,
    )
    np.testing.assert_array_equal(r.positions, [[100], [0], [0]])
    assert len(r.delist_events) == 1
    event = r.delist_events[0]
    assert event.price == 12 * 0.3
    assert event.proceeds == 100 * (12 * 0.3)
    np.testing.assert_allclose(r.cash, [0, 360, 360], atol=1e-12)


def test_missing_price_uses_latest_close_and_carries():
    r = simple(
        [[10], [np.nan], [12]], [[1], [0], [np.nan]], close=[[10], [np.nan], [12]]
    )
    np.testing.assert_allclose(r.nav, [1, 1, 1.2], rtol=1e-14)
    np.testing.assert_array_equal(r.positions, [[100], [100], [0]])
    assert r.n_blocked_sell[1] == 1
    r = simple([[np.nan], [10]], [[1], [np.nan]], close=[[np.nan], [10]])
    np.testing.assert_array_equal(r.positions, [[0], [100]])
    assert r.n_blocked_buy[0] == 1


def test_unheld_missing_close_history():
    r = simple(
        [[10], [10], [np.nan], [10]],
        [[0], [0], [1], [np.nan]],
        close=[[9], [np.nan], [np.nan], [10]],
    )
    assert r.n_blocked_buy[2] == 1
    assert r.positions[-1, 0] == 100


def test_minimum_sell_fee_cannot_overdraw():
    cfg = bt.Config(initial_cash=10, min_qty=0, qty_step=0, record_trades=True)
    r = simple(
        [[1], [0.001]],
        [[1], [0]],
        config=cfg,
        costs=bt.Costs(
            commission_rate=0,
            commission_min=5,
            tax_sell=0,
            tax_buy=0,
            fee_rate_buy=0,
            slippage=0,
            fee_rate_sell=0,
        ),
    )
    assert r.cash[0] == 0
    assert r.cash[1] == 0
    assert r.positions[-1, 0] == 5


def test_batch_bitwise_and_threads():
    from .generators import case

    for seed in range(12):
        m, target, cost, cfg = case(seed)
        plans = [
            target,
            bt.Targets.from_dense(np.zeros(m.shape)),
            bt.Targets.from_dense(np.full(m.shape, np.nan)),
        ]
        one = bt.run_batch(m, plans, cost, cfg, threads=1)
        all_cores = bt.run_batch(m, plans, cost, cfg, threads=0)
        for i, plan in enumerate(plans):
            single = bt.run(m, plan, cost, cfg)
            for key in (*METRICS, "positions"):
                np.testing.assert_array_equal(
                    getattr(one, key)[i], getattr(single, key)
                )
                np.testing.assert_array_equal(
                    getattr(all_cores, key)[i], getattr(single, key)
                )
            assert one.trades[i] == all_cores.trades[i] == single.trades
            assert (
                one.delist_events[i]
                == all_cores.delist_events[i]
                == single.delist_events
            )
            assert {k: v[i] for k, v in one.diagnostics.items()} == single.diagnostics
            assert {
                k: v[i] for k, v in all_cores.diagnostics.items()
            } == single.diagnostics
        empty = bt.run_batch(m, [], cost, cfg)
        assert empty.nav.shape == (0, m.shape[0])
        assert empty.positions.shape == (0, *m.shape)
        lean = bt.run(m, target, cost, replace(cfg, record_trades=False))
        assert lean.positions is None and lean.trades is None
        np.testing.assert_array_equal(lean.nav, one.nav[0])


def test_proportional_cash_scaling_before_column_order():
    cfg = bt.Config(initial_cash=10000, record_trades=True)
    r = simple(
        [[10, 10]],
        [[0.5, 0.5]],
        config=cfg,
        costs=bt.Costs(
            commission_rate=0.1,
            commission_min=0,
            tax_sell=0,
            tax_buy=0,
            fee_rate_buy=0,
            slippage=0,
            fee_rate_sell=0,
        ),
    )
    np.testing.assert_array_equal(r.positions, [[400, 400]])
    assert r.cash[0] == 1200
