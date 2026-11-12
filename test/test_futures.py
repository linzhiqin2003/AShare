"""Hand-computed futures settlement, collateral and mixed-account cases."""

import json

import numpy as np
import panelbt as bt
import pytest
from panelbt import reference
from panelbt.futures import hedge_contracts, roll_targets
from panelbt.markets import cffex_index_futures

from .comparison import assert_results_equal
from .generators_futures import case
from .ledger import audit

ZERO = bt.Costs(
    commission_rate=0,
    commission_min=0,
    tax_buy=0,
    tax_sell=0,
    fee_rate_buy=0,
    fee_rate_sell=0,
    slippage=0,
)


def sample(
    prices,
    settlement,
    kind,
    multiplier,
    expiry,
    targets,
    *,
    margin_rate=0.1,
    fee_rate=0,
    fee_per_contract=0,
    delivery_fee_rate=0,
    delivery_fee_per_contract=0,
    initial_cash=10000,
    config=None,
    costs=None,
    market_kwargs=None,
):
    p = np.asarray(prices, dtype=float)
    f = bt.Futures(
        is_future=kind,
        multiplier=multiplier,
        margin_rate=margin_rate,
        settlement=settlement,
        expiry=expiry,
        fee_rate=fee_rate,
        fee_per_contract=fee_per_contract,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
    )
    market = bt.Market(
        p,
        p.copy(),
        np.isfinite(p),
        np.isfinite(p),
        futures=f,
        **(market_kwargs or {}),
    )
    cfg = config or bt.Config(
        initial_cash=initial_cash,
        min_qty=0,
        qty_step=0,
        record_trades=True,
    )
    plan = bt.Targets.from_dense(targets, kind="shares")
    got = bt.run(market, plan, costs or ZERO, cfg)
    expected = reference.run(market, plan, costs or ZERO, cfg)
    assert_results_equal(got, expected)
    return got


def test_contract_settlement_through_expiry():
    result = sample(
        [[100], [100], [100]],
        [[101], [103], [102]],
        [True],
        [10],
        [2],
        [[2], [np.nan], [np.nan]],
        fee_rate=0.001,
        fee_per_contract=1,
        delivery_fee_per_contract=3,
    )
    np.testing.assert_array_equal(result.cash, [10016, 10056, 10030])
    np.testing.assert_array_equal(result.futures_pnl, [20, 40, -20])
    np.testing.assert_array_equal(result.positions[:, 0], [2, 2, 0])
    assert result.expiry_events == [bt.ExpiryEvent(2, 0, 2, 102, 6)]
    assert result.delivery_fee[-1] == 6
    assert result.contribution[0] == 30


def test_opening_day_uses_effective_execution_price():
    r = sample(
        [[100], [100]],
        [[101], [102]],
        [True],
        [10],
        [2],
        [[2], [np.nan]],
        costs=bt.Costs(
            commission_rate=0,
            commission_min=0,
            fee_rate_buy=0,
            fee_rate_sell=0,
            slippage=0.01,
        ),
    )
    np.testing.assert_array_equal(r.futures_pnl, [0, 20])
    np.testing.assert_array_equal(r.cash, [10000, 10020])


def test_margin_scales_integer_contracts():
    r = sample([[100]], [[100]], [True], [10], [1], [[10]], initial_cash=250)
    assert r.positions[0, 0] == 2
    assert r.futures_margin[0] == 200
    assert r.coverage["futures_margin_scaled"] == 1


def test_expiry_disables_later_targets():
    r = sample(
        [[100], [100], [100]],
        [[100], [100], [100]],
        [True],
        [10],
        [1],
        [[1], [np.nan], [1]],
    )
    np.testing.assert_array_equal(r.positions[:, 0], [1, 0, 0])
    np.testing.assert_array_equal(r.n_expired_target, [0, 0, 1])


def test_futures_result_npz_roundtrip(tmp_path):
    r = sample(
        [[100], [100]],
        [[100], [110]],
        [True],
        [10],
        [1],
        [[1], [np.nan]],
        delivery_fee_per_contract=2,
    )
    path = tmp_path / "futures.npz"
    r.to_npz(path)
    loaded = bt.Result.from_npz(path)
    np.testing.assert_array_equal(loaded.cash, r.cash)
    np.testing.assert_array_equal(loaded.futures_pnl, r.futures_pnl)
    np.testing.assert_array_equal(loaded.futures_margin, r.futures_margin)
    assert loaded.expiry_events == r.expiry_events


def test_pre_futures_version_one_archive_still_loads(tmp_path):
    r = sample([[100]], [[100]], [True], [10], [1], [[0]])
    current = tmp_path / "current.npz"
    previous = tmp_path / "previous.npz"
    r.to_npz(current)
    with np.load(current, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in archive.files
            if key
            not in ("futures_pnl", "futures_margin", "delivery_fee", "n_expired_target")
        }
    metadata = json.loads(arrays["metadata"].item())
    metadata.pop("expiry_events")
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(previous, **arrays)
    loaded = bt.Result.from_npz(previous)
    np.testing.assert_array_equal(loaded.futures_pnl, [0])
    np.testing.assert_array_equal(loaded.n_expired_target, [0])
    assert loaded.expiry_events == []


def test_spot_long_future_short_hedge_offsets_price_moves():
    r = sample(
        [[10, 100], [11, 100], [9, 100]],
        [[1, 100], [1, 110], [1, 90]],
        [False, True],
        [1, 10],
        [3, 3],
        [[100, -1], [np.nan, np.nan], [np.nan, np.nan]],
        initial_cash=2000,
    )
    np.testing.assert_array_equal(r.equity, [2000, 2000, 2000])
    np.testing.assert_array_equal(r.long_value, [1000, 1100, 900])
    np.testing.assert_array_equal(r.short_value, [0, 0, 0])
    np.testing.assert_array_equal(r.futures_pnl, [0, -100, 200])


def test_dated_margin_raise_calls_and_reduces_next_day():
    r = sample(
        [[100], [100], [100]],
        [[100], [100], [100]],
        [True],
        [10],
        [3],
        [[4], [np.nan], [np.nan]],
        margin_rate=[[0.1], [0.3], [0.3]],
        initial_cash=1000,
    )
    np.testing.assert_array_equal(r.margin_call, [False, True, False])
    np.testing.assert_array_equal(r.positions[:, 0], [4, 4, 3])
    assert [(x.day, x.qty) for x in r.liquidation_events] == [(2, -1)]


def test_mixed_liquidation_releases_futures_collateral_first():
    r = sample(
        [[10, 100]] * 3,
        [[1, 100]] * 3,
        [False, True],
        [1, 10],
        [3, 3],
        [[100, -4], [np.nan, np.nan], [np.nan, np.nan]],
        margin_rate=[[0.1, 0.1], [0.1, 0.3], [0.1, 0.3]],
        config=bt.Config(
            initial_cash=1000,
            min_qty=0,
            qty_step=0,
            record_trades=True,
            margin=bt.MarginModel(
                allow_financing=True, long_initial=0.5, long_maintenance=0.5
            ),
        ),
    )
    np.testing.assert_array_equal(r.positions, [[100, -4], [100, -4], [50, -2]])
    assert [x.col for x in r.liquidation_events] == [1, 0]
    assert [x.qty for x in r.liquidation_events] == [2, -50]


def test_dated_margin_is_checked_at_close_and_before_next_trade():
    r = sample(
        [[100], [100], [100]],
        [[100], [100], [100]],
        [True],
        [10],
        [3],
        [[4], [np.nan], [np.nan]],
        margin_rate=[[0.1], [0.3], [0.3]],
        initial_cash=1000,
    )
    np.testing.assert_array_equal(r.futures_margin, [400, 1200, 900])
    np.testing.assert_array_equal(r.margin_call, [False, True, False])
    assert r.liquidation_events[0].day == 2


def test_invalid_future_spec_fails_at_market_entry():
    p = np.ones((2, 1))
    with pytest.raises(ValueError, match="settlement"):
        bt.Market(
            p,
            p,
            p > 0,
            p > 0,
            futures=bt.Futures([True], [10], 0.1, [[100], [np.nan]], [2]),
        )
    with pytest.raises(ValueError, match="integer"):
        m = bt.Market(
            p,
            p,
            p > 0,
            p > 0,
            futures=bt.Futures([True], [10], 0.1, [[100], [100]], [2]),
        )
        bt.run(m, bt.Targets.from_dense([[0.5], [np.nan]], kind="shares"), ZERO)


def test_random_mixed_reference_two_thousand():
    counts = dict.fromkeys(
        ("trades", "open", "expired", "call", "forced", "bankrupt"), 0
    )
    for seed in range(2000):
        args = case(seed)
        try:
            result = bt.run(*args)
            assert_results_equal(result, reference.run(*args))
            audit(args, result)
            counts["trades"] += result.diagnostics["trades"]
            counts["open"] += result.coverage["futures_open"]
            counts["expired"] += result.coverage["expired_target"]
            counts["call"] += result.coverage["margin_call"]
            counts["forced"] += result.coverage["futures_liquidation"]
            counts["bankrupt"] += result.coverage["bankruptcy"]
        except AssertionError as exc:
            raise AssertionError(f"mixed seed={seed}: {exc}") from exc
    assert all(v > 0 for v in counts.values()), counts


def test_roll_changes_on_boundary_and_hedge_contracts():
    plan = roll_targets([4, 8, 12], periods=10, roll_days=2, weight=-0.5)
    np.testing.assert_array_equal(plan.days, [0, 2, 6])
    np.testing.assert_array_equal(plan.cols, [0, 1, 2])
    np.testing.assert_array_equal(plan.weights, [-0.5] * 3)
    assert hedge_contracts(1_000_000, 4000, 300, beta=1.2) == -1
    assert hedge_contracts(1_500_000, 100, 100, beta=1) == -150
    np.testing.assert_array_equal(
        hedge_contracts([400, 1000], [100, 100], [10, 10]), [0, -1]
    )


def test_cffex_multipliers_and_explicit_terms():
    spec = cffex_index_futures(
        [None, "IF", "IH", "IC", "IM"],
        settlement=np.ones((2, 5)) * 100,
        expiry=[2, 2, 2, 2, 2],
        margin_rate=0.13,
        fee_rate=0.00023,
        fee_per_contract=0,
        delivery_fee_rate=0.0001,
        delivery_fee_per_contract=0,
    )
    np.testing.assert_array_equal(spec.multiplier, [1, 300, 300, 200, 200])
    p = np.ones((2, 5)) * 100
    m = bt.Market(p, p, p > 0, p > 0, futures=spec)
    np.testing.assert_array_equal(m.futures.is_future, [False, True, True, True, True])


def test_expiry_cash_settlement_ignores_trade_mask():
    r = sample(
        [[100], [np.nan]],
        [[100], [110]],
        [True],
        [10],
        [1],
        [[1], [np.nan]],
        market_kwargs={"volume": [[100], [0]]},
    )
    np.testing.assert_array_equal(r.positions[:, 0], [1, 0])
    np.testing.assert_array_equal(r.futures_pnl, [0, 100])
    assert len(r.expiry_events) == 1 and r.expiry_events[0].day == 1


def test_weight_target_uses_futures_notional_and_negative_without_spot_short():
    p = np.full((2, 2), 100.0)
    f = bt.Futures([False, True], [1, 10], 0.1, p, [2, 2])
    m = bt.Market(p, p, p > 0, p > 0, futures=f)
    target = bt.Targets.from_dense([[0.5, -1.0], [np.nan, np.nan]])
    cfg = bt.Config(
        initial_cash=10000,
        min_qty=0,
        qty_step=0,
        record_trades=True,
    )
    a, b = bt.run(m, target, ZERO, cfg), reference.run(m, target, ZERO, cfg)
    assert_results_equal(a, b)
    np.testing.assert_array_equal(a.positions[0], [50, -10])
    assert a.futures_margin[0] == 1000
