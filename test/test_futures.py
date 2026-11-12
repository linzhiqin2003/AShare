"""Hand-computed futures settlement, collateral and mixed-account cases."""

import json
import warnings

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
    with pytest.warns(bt.HedgeRoundingWarning):
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


# ---- Review 1c fixes: each case is hand computed and also checked against the
# structurally independent reference through `sample`.

TAXED = bt.Costs(
    commission_rate=0,
    commission_min=0,
    fee_rate_buy=0,
    fee_rate_sell=0,
    slippage=0,
    tax_sell=0.0005,
)
LEVERED = bt.MarginModel(long_initial=0.5, long_maintenance=0.25, allow_financing=True)


def _spot_and_mixed(prices, dense, expiry):
    p = np.asarray(prices, dtype=float)
    o = np.ones(p.shape, bool)
    cfg = bt.Config(initial_cash=1e6, record_trades=True, margin=LEVERED)
    spot = bt.run(
        bt.Market(p[:, :1], p[:, :1], o[:, :1], o[:, :1]),
        bt.Targets.from_dense(np.asarray(dense)[:, :1]),
        TAXED,
        cfg,
    )
    f = bt.Futures([False, True], [1, 300], 0.12, p, [expiry, expiry])
    m = bt.Market(p, p, o, o, futures=f)
    plan = bt.Targets.from_dense(dense)
    mixed = bt.run(m, plan, TAXED, cfg)
    assert_results_equal(mixed, reference.run(m, plan, TAXED, cfg))
    return spot, mixed


def test_financed_spot_long_can_be_reduced_with_futures_attached():
    # P0: 1.8x financed long cut to 1x. Sale 80,000 × 10 pays 400 tax.
    spot, mixed = _spot_and_mixed(
        [[10.0, 4000.0]] * 3, [[1.8, 0.0], [1.0, 0.0], [np.nan, np.nan]], 3
    )
    for r in (spot, mixed):
        np.testing.assert_array_equal(r.positions[:, 0], [180000, 100000, 100000])
        np.testing.assert_array_equal(r.cash, [-800000, -400, -400])
        np.testing.assert_array_equal(r.n_blocked_sell, [0, 0, 0])


def test_financed_spot_margin_call_liquidates_with_futures_attached():
    # P0: 180,000 × 5.5 = 990,000 long, equity 190,000 < 0.25 × 990,000.
    # Liquidation keeps 68,900 shares, exactly as without the futures column.
    spot, mixed = _spot_and_mixed(
        [[10.0, 4000.0]] + [[5.5, 4000.0]] * 3,
        [[1.8, 0.0]] + [[np.nan, np.nan]] * 3,
        4,
    )
    for r in (spot, mixed):
        np.testing.assert_array_equal(r.positions[:, 0], [180000, 180000, 68900, 68900])
        np.testing.assert_array_equal(r.margin_call, [False, True, False, False])
        assert [(x.day, x.col, x.qty) for x in r.liquidation_events] == [
            (2, 0, -111100)
        ]


@pytest.mark.parametrize("financing", [False, True])
def test_futures_close_is_never_trimmed_by_own_cash(financing):
    # Six long contracts gap from 4000 to 3400: realized loss 6×300×600=1,080,000.
    # Both fills pay 5 per contract; the close leaves -80,060 as account debt.
    # financing_rate=0.001 then charges 80.06 on that negative own cash.
    r = sample(
        [[4000.0], [3400.0], [3400.0]],
        [[4000.0], [3400.0], [3400.0]],
        [True],
        [300],
        [3],
        [[6], [0], [np.nan]],
        margin_rate=0.12,
        fee_per_contract=5,
        config=bt.Config(
            initial_cash=1e6,
            min_qty=0,
            qty_step=0,
            record_trades=True,
            margin=bt.MarginModel(allow_financing=financing, financing_rate=0.001),
        ),
    )
    np.testing.assert_array_equal(r.positions[:, 0], [6, 0, 0])
    assert r.coverage["reduction_clipped"] == 0
    assert [x.qty for x in r.trades] == [6, -6]
    assert r.cash[0] == 1e6 - 30
    assert r.cash[1] == pytest.approx(-80060 - 80.06, abs=1e-9)
    assert r.financing_interest[1] == pytest.approx(80.06, abs=1e-12)
    assert r.bankruptcy_events == [1]


def test_expired_target_on_liquidation_row_is_counted():
    # B's collateral rises from 400 to 1200 > 1000 equity: call on row 1.
    # Row 2 liquidates B to 3 contracts; its plan also targets A, which expired
    # on row 0. That target is reported, not silently dropped.
    r = sample(
        [[100, 100]] * 3,
        [[100, 100]] * 3,
        [True, True],
        [10, 10],
        [0, 3],
        [[0, 4], [np.nan, np.nan], [1, 4]],
        margin_rate=[[0.1, 0.1], [0.3, 0.3], [0.3, 0.3]],
        initial_cash=1000,
    )
    np.testing.assert_array_equal(r.margin_call, [False, True, False])
    np.testing.assert_array_equal(r.positions[:, 1], [4, 4, 3])
    np.testing.assert_array_equal(r.n_expired_target, [0, 0, 1])
    assert r.coverage["expired_target"] == 1


def test_weight_to_contracts_rounds_toward_zero():
    # contracts = w × 10,000 / (100 × 10) = 10w: 2.75→2, -2.95→-2, 3.3→3, 0.5→0.
    p = np.full((1, 4), 100.0)
    f = bt.Futures([True] * 4, [10] * 4, 0.01, p, [1] * 4)
    m = bt.Market(p, p, p > 0, p > 0, futures=f)
    plan = bt.Targets.from_dense([[0.275, -0.295, 0.33, 0.05]])
    cfg = bt.Config(initial_cash=10000, min_qty=0, qty_step=0, record_trades=True)
    r = bt.run(m, plan, ZERO, cfg)
    assert_results_equal(r, reference.run(m, plan, ZERO, cfg))
    np.testing.assert_array_equal(r.positions[0], [2, -2, 3, 0])
    assert r.coverage["futures_rounded_to_zero"] == 1


def _dust_market(margin):
    # Row 0 opens one contract whose collateral 10 × 100 × 1.0 = 1000 leaves
    # 1e-9 of own cash. Row 1 asks for 100 spot shares at 10: the basket scale
    # gives 1e-10 shares, below the 1e-7 quantity tolerance, so no fill.
    p = np.array([[10.0, 100.0], [10.0, 100.0]])
    f = bt.Futures([False, True], [1, 10], 1.0, p, [2, 2])
    m = bt.Market(p, p, p > 0, p > 0, futures=f)
    plan = bt.Targets.from_dense([[0, 1], [100, 1]], kind="shares")
    cfg = bt.Config(
        initial_cash=1000 + 1e-9,
        min_qty=0,
        qty_step=0,
        record_trades=True,
        margin=margin,
    )
    r = bt.run(m, plan, ZERO, cfg)
    assert_results_equal(r, reference.run(m, plan, ZERO, cfg))
    return r


@pytest.mark.parametrize(
    "margin",
    [bt.MarginModel(), bt.MarginModel(long_maintenance=0.01)],
    ids=["cash", "margin"],
)
def test_scaled_spot_dust_beside_futures_collateral_is_not_filled(margin):
    r = _dust_market(margin)
    assert [(x.day, x.col) for x in r.trades] == [(0, 1)]
    np.testing.assert_array_equal(r.positions[:, 0], [0, 0])


def test_expiry_row_trades_then_settles_and_later_rows_do_not():
    # Opening on the expiry row is allowed: 2 contracts at 100 settle at 105.
    r = sample(
        [[100], [100], [100]],
        [[100], [105], [100]],
        [True],
        [10],
        [1],
        [[np.nan], [2], [2]],
    )
    assert [(x.day, x.qty) for x in r.trades] == [(1, 2)]
    assert r.expiry_events == [bt.ExpiryEvent(1, 0, 2, 105, 0)]
    np.testing.assert_array_equal(r.futures_pnl, [0, 100, 0])
    np.testing.assert_array_equal(r.positions[:, 0], [0, 0, 0])
    np.testing.assert_array_equal(r.n_expired_target, [0, 0, 1])


def test_unheld_contract_without_quote_is_sized_at_previous_settlement():
    # Row 1 has no execution price. Sizing uses row 0's settlement 100 (not
    # row 1's 120): 0.3 × 10,000 / (100 × 10) = 3 contracts, blocked by the
    # missing quote and carried; row 2 fills them at 110.
    r = sample(
        [[100], [np.nan], [110]],
        [[100], [120], [110]],
        [True],
        [10],
        [3],
        [[np.nan], [3], [np.nan]],
    )
    np.testing.assert_array_equal(r.n_blocked_buy, [0, 1, 0])
    assert [(x.day, x.qty, x.price) for x in r.trades] == [(2, 3, 110)]


def test_hedge_contracts_round_toward_zero_and_warn_on_zero():
    # 2,700,000 / (3000 × 300) = 3.0; × 0.9 = 2.7 → -2 (not -3).
    assert hedge_contracts(2_700_000, 3000, 300, hedge_ratio=0.9) == -2
    assert hedge_contracts(2_700_000, 3000, 300) == -3
    with pytest.warns(bt.HedgeRoundingWarning):
        assert hedge_contracts(800_000, 3000, 300) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hedge_contracts(0, 3000, 300) == 0


def test_roll_never_moves_back_when_contracts_are_closer_than_roll_days():
    plan = roll_targets([5, 6], periods=8, roll_days=3, weight=-1)
    np.testing.assert_array_equal(plan.days, [0, 2, 7])
    np.testing.assert_array_equal(plan.cols, [0, 1])
    np.testing.assert_array_equal(np.diff(plan.indptr), [1, 1, 0])


def test_future_rates_require_an_explicit_axis_when_t_equals_n():
    p = np.full((2, 2), 100.0)
    with pytest.raises(ValueError, match="1-D rate is ambiguous"):
        bt.Futures([True, True], [10, 10], [0.1, 0.3], p, [2, 2]).validated(
            (2, 2), bt.Market(p, p, p > 0, p > 0)
        )
    by_day = bt.Market(
        p,
        p,
        p > 0,
        p > 0,
        futures=bt.Futures([True, True], [10, 10], [[0.1], [0.3]], p, [2, 2]),
    )
    by_col = bt.Market(
        p,
        p,
        p > 0,
        p > 0,
        futures=bt.Futures([True, True], [10, 10], [[0.1, 0.3]], p, [2, 2]),
    )
    plan = bt.Targets.from_dense([[1, 1], [np.nan, np.nan]], kind="shares")
    cfg = bt.Config(initial_cash=10000, min_qty=0, qty_step=0)
    np.testing.assert_allclose(
        bt.run(by_day, plan, ZERO, cfg).futures_margin, [200, 600]
    )
    np.testing.assert_allclose(
        bt.run(by_col, plan, ZERO, cfg).futures_margin, [400, 400]
    )


# Cash account with futures debt: a spot sale must not be blocked merely
# because its proceeds do not repay the whole debt in one fill.
DEBT_COSTS = bt.Costs(
    commission_rate=0.0003,
    commission_min=5,
    fee_rate_buy=0,
    fee_rate_sell=0,
    slippage=0,
    tax_sell=0.0005,
)


def _debt_case(futures_path, dense):
    t = len(futures_path)
    p = np.column_stack([np.full(t, 10.0), np.full(t, 10.0), futures_path])
    o = np.ones(p.shape, bool)
    f = bt.Futures([False, False, True], [1, 1, 100], 0.12, p, [t, t, t])
    m = bt.Market(p, p, o, o, futures=f)
    plan = bt.Targets.from_dense(np.array(dense))
    cfg = bt.Config(initial_cash=1e6, record_trades=True)
    r = bt.run(m, plan, DEBT_COSTS, cfg)
    assert_results_equal(r, reference.run(m, plan, DEBT_COSTS, cfg))
    return r


def test_cash_account_sells_spot_while_futures_loss_leaves_debt():
    # Row 0: A 5000 and B 50000 shares (fees 15 + 150), 2 contracts at 4000:
    # cash 449,835. Row 1 settles at 1500: -500,000 → cash -50,165.
    # Row 2 sells A first (50,000 - 15 - 25 = 49,960: cash -205, still debt)
    # and B down to 24,900 (251,000 - 75.3 - 125.5 = 250,799.2).
    nan = np.nan
    r = _debt_case(
        [4000, 1500, 1500, 1500],
        [[0.05, 0.5, 1.0], [nan] * 3, [0.0, 0.5, 0.0], [nan] * 3],
    )
    np.testing.assert_array_equal(r.positions[:, 0], [5000, 5000, 0, 0])
    np.testing.assert_array_equal(r.positions[:, 1], [50000, 50000, 24900, 24900])
    assert r.cash[1] == pytest.approx(-50165, abs=1e-6)
    assert r.cash[2] == pytest.approx(-50165 + 49960 + 250799.2, abs=1e-6)
    np.testing.assert_array_equal(r.n_blocked_sell, [0, 0, 0, 0])


def test_cash_account_liquidates_spot_after_futures_loss():
    # Five contracts settle from 4000 to 2100: -950,000 → cash -500,165,
    # equity 49,835 < collateral 5×100×2100×0.12 = 126,000: call on row 1.
    # Row 2 sells 4 contracts, 4700 A (46,962.4 net) and 47,900 B
    # (478,616.8 net): cash 25,414.2 ≥ collateral 25,200; the call clears.
    nan = np.nan
    r = _debt_case(
        [4000, 2100, 2100, 2100, 2100],
        [[0.05, 0.5, 2.0]] + [[nan] * 3] * 4,
    )
    np.testing.assert_array_equal(r.positions[2], [300, 2100, 1])
    np.testing.assert_array_equal(r.margin_call, [False, True, False, False, False])
    assert r.cash[2] == pytest.approx(-500165 + 46962.4 + 478616.8, abs=1e-6)
    assert r.futures_margin[2] == pytest.approx(25200)
    assert sorted((x.col, x.qty) for x in r.liquidation_events) == [
        (0, -4700),
        (1, -47900),
        (2, -4),
    ]
