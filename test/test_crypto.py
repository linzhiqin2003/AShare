"""C1 deterministic counterexamples, written before the implementation.

Sources: DESIGN 4.17 v3.1; review R1 (order, lambda, pending targets),
R2 N1/N2/N6/N7/N8 and R3 V1/V2. All prices below are synthetic.
"""

from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest

ZERO = bt.Costs(commission_rate=0, commission_min=0, tax_sell=0, slippage=0)


def scenario(
    prices,
    targets,
    *,
    cash=100,
    marks=None,
    high=None,
    low=None,
    settlement=None,
    funding=None,
    maint=0.01,
    initial=0.1,
    clearance=0,
    cap=1e30,
    expiry=None,
    config=None,
    costs=ZERO,
    buy=None,
    sell=None,
    volume=None,
    **future,
):
    p = np.array(prices, dtype=float)
    t, n = p.shape
    o = p.copy() if marks is None else np.array(marks, dtype=float)
    h = o.copy() if high is None else np.array(high, dtype=float)
    lows = o.copy() if low is None else np.array(low, dtype=float)
    f = bt.Futures(
        np.ones(n, bool),
        np.ones(n),
        initial,
        o.copy() if settlement is None else np.array(settlement, dtype=float),
        np.full(n, t) if expiry is None else expiry,
        funding=funding,
        maint_rate=maint,
        liquidation=bt.Liquidation(o, h, lows, clearance, cap),
        exec_at_open=True,
        **future,
    )
    m = bt.Market(
        p,
        p.copy(),
        np.isfinite(p) if buy is None else buy,
        np.isfinite(p) if sell is None else sell,
        futures=f,
        volume=volume,
    )
    c = config or bt.Config(initial_cash=cash, record_trades=True)
    return m, bt.Targets.from_dense(targets, kind="shares"), costs, c


def test_opening_includes_execution_mark_loss():
    # R3 V1: E_after=10+(90-100)=0 < IM=9. Integer contract rejected.
    r = bt.run(*scenario([[100]], [[1]], cash=10, marks=[[90]]))
    assert r.positions[0, 0] == 0
    assert r.cash[0] == 10


def test_post_trade_reduction_liquidates_at_equality():
    # R3 V1: opening D=3-2=1; reduce at 98 -> C=1, MM=1, D=0.
    r = bt.run(
        *scenario([[100], [98]], [[2], [1]], cash=3, marks=[[100], [100]], initial=0.01)
    )
    np.testing.assert_array_equal(r.cash, [3, 1])
    np.testing.assert_array_equal(r.positions[:, 0], [2, 0])
    assert r.coverage["liquidation_post_trade"] == 1


def test_tier_cap_uses_high_for_longs():
    # R3 V2: the adverse low 99 is not the highest notional, 200.
    r = bt.run(*scenario([[100]], [[1]], high=[[200]], low=[[99]], cap=150))
    assert r.coverage["tier_cap_exceeded"] == 1
    assert r.positions[0, 0] == 1


def test_open_gap_cannot_be_rescued_by_target_or_later_rally():
    r = bt.run(
        *scenario(
            [[100], [80], [100]],
            [[1], [0], [1]],
            cash=10,
            high=[[100], [120], [100]],
            settlement=[[100], [120], [100]],
        )
    )
    np.testing.assert_array_equal(r.cash, [10, 0, 0])
    np.testing.assert_array_equal(r.positions[:, 0], [1, 0, 0])
    assert r.bankruptcy_events == [1]
    assert r.bankruptcy_deficit[1] == 10
    assert r.coverage["liquidation_open"] == 1


def test_funding_boundary_signed_and_no_charge_after_intraday_liquidation():
    r = bt.run(
        *scenario([[100], [100], [100]], [[1], [-1], [0]], funding=[[2], [3], [7]])
    )
    np.testing.assert_array_equal(r.funding, [2, -3, 0])
    np.testing.assert_array_equal(r.cash, [98, 101, 101])
    r = bt.run(*scenario([[100]], [[1]], cash=10, low=[[80]], funding=[[9]]))
    assert r.funding[0] == 0
    assert r.trades[-1].price == pytest.approx(90 / 0.99)


def test_funding_then_close_risk_uses_settlement_not_next_open():
    # R2 N2, accepted v3.1 approximation: 1.1 - .2 < MM(100)=1.
    r = bt.run(
        *scenario(
            [[100], [110]],
            [[1], [np.nan]],
            cash=1.1,
            initial=0.01,
            funding=[[0.2], [0]],
        )
    )
    assert r.cash[0] == pytest.approx(0.9)
    assert r.positions[0, 0] == 0
    assert r.coverage["liquidation_close"] == 1


def test_delivery_uses_delivery_price_once_and_expires_before_funding():
    # R2 N6: quoted close 101, delivery 99; total variation -1, not +1.
    r = bt.run(
        *scenario(
            [[100]],
            [[1]],
            high=[[101]],
            low=[[99]],
            settlement=[[99]],
            expiry=[0],
            funding=[[8]],
            delivery_fee_rate=0.001,
        )
    )
    assert r.futures_pnl[0] == -1
    assert r.delivery_fee[0] == pytest.approx(0.099)
    assert r.cash[0] == pytest.approx(98.901)
    assert r.funding[0] == 0
    assert r.expiry_events[0].settlement == 99


def test_clearance_capped_after_ordinary_commission_and_no_fee_on_deficit():
    r = bt.run(*scenario([[100]], [[1]], cash=10, low=[[80]], clearance=0.1))
    assert r.clearance_fee[0] == pytest.approx(90 / 99)
    assert r.cash[0] == 0
    assert r.bankruptcy_events == []
    assert r.bankruptcy_deficit[0] == 0
    r = bt.run(
        *scenario(
            [[100], [80]],
            [[1], [np.nan]],
            cash=10,
            clearance=0.1,
            fee_rate=0.001,
            initial=0.09,
        )
    )
    assert r.clearance_fee[1] == 0
    assert r.bankruptcy_deficit[1] == pytest.approx(10.18)


def test_lambda_short_and_multi_column_formula():
    r = bt.run(*scenario([[100]], [[-1]], cash=10, high=[[120]], maint=0.02))
    assert r.trades[-1].price == pytest.approx(110 / 1.02)
    r = bt.run(
        *scenario(
            [[100, 100]],
            [[1, -1]],
            cash=20,
            high=[[100, 120]],
            low=[[80, 100]],
            maint=0.02,
        )
    )
    # D0=16, D1=-40 => lambda=.4; prices=92,108; residual equity=4.
    assert [x.price for x in r.trades[-2:]] == pytest.approx([92, 108])
    assert r.cash[0] == pytest.approx(4)


def test_lambda_one_and_nondecreasing():
    r = bt.run(*scenario([[100]], [[1]], cash=10, maint=0.1, initial=0.1))
    assert r.coverage["liquidation_post_trade"] == 1
    r = bt.run(*scenario([[100]], [[1]], cash=10, low=[[90 / 0.99]]))
    assert r.coverage["liquidation_intraday"] == 1
    r = bt.run(*scenario([[100]], [[1]], cash=20))
    assert r.positions[0, 0] == 1
    assert r.coverage["liquidation_nondecreasing"] == 1


def test_liquidation_clears_other_columns_pending_and_allows_later_target():
    r = bt.run(
        *scenario(
            [[100, 100], [100, 100], [50, 100]],
            [[1, 1], [np.nan, np.nan], [1, 0]],
            cash=10,
            low=[[80, 100], [100, 100], [50, 100]],
            buy=[[True, False], [True, True], [True, True]],
            initial=0.001,
        )
    )
    np.testing.assert_array_equal(r.positions, [[0, 0], [0, 0], [1, 0]])
    assert r.diagnostics["carry_trade"] == 0


def test_non_trade_hour_never_retries_with_carry_disabled():
    cfg = bt.Config(initial_cash=100, carry_blocked=False, record_trades=True)
    r = bt.run(
        *scenario([[100], [100]], [[1], [np.nan]], buy=[[False], [True]], config=cfg)
    )
    assert not r.positions.any()


def test_scaled_order_rules_reversal_and_full_close_dust():
    r = bt.run(
        *scenario(
            [[100]],
            [[10]],
            cash=1000,
            qty_step=[[3]],
            min_qty=[[3]],
            min_notional=[[400]],
            volume=[[5]],
            config=bt.Config(
                initial_cash=1000, max_participation=1, record_trades=True
            ),
        )
    )
    assert r.positions[0, 0] == 0  # cap 5 -> step 3 -> notional 300 < 400
    r = bt.run(
        *scenario(
            [[100], [100]],
            [[5], [-1]],
            cash=1000,
            qty_step=[[1], [3]],
            min_qty=[[1], [3]],
            min_notional=[[0], [1000]],
        )
    )
    np.testing.assert_array_equal(r.positions[:, 0], [5, 0])
    r = bt.run(
        *scenario([[100], [100]], [[5], [3]], cash=1000, min_notional=[[0], [1000]])
    )
    np.testing.assert_array_equal(r.positions[:, 0], [5, 3])


def test_lifecycle_nan_marks_only_error_when_held():
    args = scenario(
        [[np.nan], [100], [np.nan]],
        [[np.nan], [1], [np.nan]],
        marks=[[np.nan], [100], [np.nan]],
        settlement=[[100], [100], [100]],
        expiry=[1],
    )
    r = bt.run(*args)
    np.testing.assert_array_equal(r.cash, [100, 100, 100])
    m, plan, costs, cfg = args
    m = replace(m, futures=replace(m.futures, expiry=[3]))
    with pytest.raises(ValueError, match="mark"):
        bt.run(m, plan, costs, cfg)


def test_independent_reference_branch_corpus():
    from collections import Counter

    from panelbt.reference_crypto import run as oracle

    from .comparison import assert_results_equal
    from .generators_crypto import BRANCHES, case, directed_cases

    totals = Counter()
    for index, args in enumerate([*directed_cases(), *(case(s) for s in range(500))]):
        actual, expected = bt.run(*args), oracle(*args)
        try:
            assert_results_equal(actual, expected)
            for field in ("funding", "clearance_fee", "bankruptcy_deficit"):
                np.testing.assert_allclose(
                    getattr(actual, field),
                    getattr(expected, field),
                    rtol=1e-12,
                    atol=1e-9,
                    err_msg=field,
                )
        except AssertionError as exc:
            raise AssertionError(f"crypto corpus case {index}: {exc}") from exc
        totals.update(actual.coverage)
        # Single-collateral conservation: deficit is model support, not cash flow.
        np.testing.assert_allclose(
            actual.cash[-1],
            args[3].initial_cash
            + sum(actual.cash_flow)
            + sum(actual.futures_pnl)
            - sum(actual.commission)
            - sum(actual.delivery_fee)
            - sum(actual.clearance_fee)
            - sum(actual.funding)
            + sum(actual.bankruptcy_deficit)
            + sum(actual.cash_interest)
            - sum(actual.financing_interest),
            rtol=1e-12,
            atol=1e-9,
        )
    assert all(totals[k] > 0 for k in BRANCHES), totals


def test_crypto_run_batch_step_and_snapshot_are_bitwise_identical():
    from .generators_crypto import case, directed_cases
    from .test_step import assert_bitwise

    for args in [*directed_cases(), *(case(s) for s in range(40))]:
        m, plan, costs, cfg = args
        expected = bt.run(*args)
        batched = bt.run_batch(m, [plan, plan], costs, cfg, threads=2)
        for name in (
            "cash",
            "funding",
            "clearance_fee",
            "bankruptcy_deficit",
            "positions",
        ):
            np.testing.assert_array_equal(
                getattr(batched, name)[0], getattr(expected, name)
            )
        sim = bt.Simulator(m, costs, cfg, kind=plan.kind, fingerprint="full")
        dense = plan.to_dense(*m.shape)
        for d, target in enumerate(dense):
            sim.step(target)
            if d in (0, m.shape[0] // 2):
                sim = bt.Simulator.restore(
                    sim.snapshot(), m, costs, cfg, kind=plan.kind, fingerprint="full"
                )
        actual = sim.result()
        assert_bitwise(actual, expected)
        for field in ("funding", "clearance_fee", "bankruptcy_deficit"):
            assert (
                getattr(actual, field).tobytes() == getattr(expected, field).tobytes()
            )


def test_rule_minimum_is_checked_after_step_rounding():
    # Development regression: tradecore's spot minimum-offset lattice is not
    # the exchange's zero-origin step lattice; minimum 3, step 2 means 4.
    r = bt.run(*scenario([[100]], [[5]], qty_step=[[2]], min_qty=[[3]]))
    assert r.positions[0, 0] == 4
    r = bt.run(
        *scenario([[100], [100]], [[7], [4]], qty_step=[[1], [2]], min_qty=[[1], [3]])
    )
    np.testing.assert_array_equal(r.positions[:, 0], [7, 7])


def test_optional_funding_maintenance_and_rules_without_liquidation():
    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal

    args = scenario(
        [[100], [100], [100]],
        [[5], [3], [0]],
        funding=[[1], [-2], [9]],
        qty_step=[[2], [1], [3]],
    )
    m, plan, costs, cfg = args
    m = replace(m, futures=replace(m.futures, liquidation=None, exec_at_open=False))
    got = bt.run(m, plan, costs, cfg)
    assert_results_equal(got, oracle(m, plan, costs, cfg))
    np.testing.assert_array_equal(got.positions[:, 0], [4, 3, 0])
    np.testing.assert_array_equal(got.cash, [96, 102, 102])


@pytest.mark.parametrize(
    "field,value",
    [
        ("qty_step", [[0]]),
        ("qty_step", [[1.5]]),
        ("min_qty", [[-1]]),
        ("min_notional", [[np.inf]]),
        ("funding", [[np.inf]]),
    ],
)
def test_crypto_input_values_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        scenario([[100]], [[1]], **{field: value})


@pytest.mark.parametrize("mode", ["initial", "volume", "cash", "reduction"])
def test_standalone_quantity_filter_reports_zero(mode):
    # Development regression: new quantity filters must count zero orders
    # even when the exchange liquidation model is not selected.
    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal

    kw = dict(qty_step=[[3]], min_qty=[[3]])
    cfg = bt.Config(initial_cash=100, record_trades=True)
    target, prices = [[2]], [[100]]
    if mode == "volume":
        target = [[6]]
        kw["volume"] = [[2]]
        cfg = replace(cfg, max_participation=1)
    elif mode == "cash":
        target = [[6]]
        cfg = replace(cfg, initial_cash=20)
    elif mode == "reduction":
        target, prices = [[6], [5]], [[100], [100]]
        kw = dict(qty_step=[[3], [3]], min_qty=[[3], [3]])
    m, plan, costs, cfg = scenario(prices, target, config=cfg, **kw)
    m = replace(m, futures=replace(m.futures, liquidation=None))
    actual = bt.run(m, plan, costs, cfg)
    assert actual.coverage["futures_rounded_to_zero"] == 1
    assert_results_equal(actual, oracle(m, plan, costs, cfg))


def test_initial_margin_and_maintenance_are_separate_without_liquidation():
    # R1 #6: a marked loss may exhaust initial collateral without breaching MM.
    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal

    m, plan, costs, cfg = scenario(
        [[100], [85]],
        [[1], [np.nan]],
        cash=20,
        settlement=[[100], [85]],
        maint=0.01,
        initial=0.1,
    )
    m = replace(m, futures=replace(m.futures, liquidation=None))
    r = bt.run(m, plan, costs, cfg)
    assert_results_equal(r, oracle(m, plan, costs, cfg))
    assert r.cash[-1] == 5
    assert r.futures_margin[-1] == 8.5
    assert not r.margin_call.any()


def test_clearance_aggregates_columns_before_cash_cap():
    # R3 P2: cap the account's sum, not each column independently.
    r = bt.run(
        *scenario(
            [[100, 100]],
            [[1, -1]],
            cash=20,
            maint=0.02,
            low=[[80, 100]],
            high=[[100, 120]],
            clearance=[[0.1, 0.2]],
        )
    )
    assert r.clearance_fee[0] == pytest.approx(4)
    assert r.cash[0] == 0
    assert sum(r.contribution) == pytest.approx(-20)
    assert not r.bankruptcy_events


def test_rule_expiry_lifecycle_two_contract_fixture():
    # C1/C2/C3 common fixture: perpetual lists on row 1; quarterly expires on
    # row 2. Only lifecycle-exterior marks are NaN; settlements stay positive.
    m, plan, costs, cfg = scenario(
        [[np.nan, 100], [101, 102], [102, 103], [103, np.nan]],
        [[0, 1], [1, 1], [1, 1], [1, 0]],
        marks=[[np.nan, 100], [101, 102], [102, 103], [103, np.nan]],
        settlement=[[101, 100], [101, 102], [102, 99], [103, 99]],
        low=[[np.nan, 100], [101, 102], [102, 99], [103, np.nan]],
        expiry=[4, 2],
        funding=[[0, 0], [0.1, 0], [-0.2, 0], [0, 0]],
    )
    r = bt.run(m, plan, costs, cfg)
    np.testing.assert_array_equal(r.positions, [[0, 1], [1, 1], [1, 0], [1, 0]])
    np.testing.assert_allclose(r.cash, [100, 101.9, 100.1, 101.1])
    assert r.expiry_events[0].settlement == 99


def test_funding_missing_when_held_errors_but_absent_event_is_zero():
    with pytest.raises(ValueError, match="funding"):
        bt.run(*scenario([[100]], [[1]], funding=[[np.nan]]))
    r = bt.run(*scenario([[100]], [[0]], funding=[[np.nan]]))
    assert r.cash[0] == 100


def test_liquidation_forced_execution_ignores_mask_and_participation():
    cfg = bt.Config(initial_cash=10, record_trades=True, max_participation=0.1)
    r = bt.run(
        *scenario(
            [[100], [80]],
            [[1], [np.nan]],
            volume=[[10], [0]],
            sell=[[True], [False]],
            config=cfg,
        )
    )
    assert r.coverage["liquidation_open"] == 1
    assert r.positions[1, 0] == 0


def test_liquidation_requires_standalone_and_open_declaration():
    m, *_ = scenario([[100]], [[1]])
    with pytest.raises(ValueError, match="exec_at_open"):
        replace(m, futures=replace(m.futures, exec_at_open=False))
    with pytest.raises(ValueError, match="mark_open"):
        replace(m, futures=replace(m.futures, liquidation=None, min_notional=[[1]]))


def test_snapshot_fingerprints_new_input_cells():
    m, plan, costs, cfg = scenario([[100], [100]], [[1], [np.nan]], funding=[[0], [0]])
    sim = bt.Simulator(m, costs, cfg, kind=plan.kind, fingerprint="full")
    sim.step([1])
    blob = sim.snapshot()
    changed = replace(m, futures=replace(m.futures, funding=[[0], [1]]))
    with pytest.raises(ValueError, match="differ"):
        bt.Simulator.restore(
            blob, changed, costs, cfg, kind=plan.kind, fingerprint="full"
        )


def test_terminal_delivery_debt_latches_bankruptcy_without_fictitious_credit():
    # Development regression / design gap: an expiry fee can leave debt with
    # no held contracts. Keep that debt, and do not revive trading on a deposit.
    cfg = bt.Config(initial_cash=100, cash_flow=[0, 500], record_trades=True)
    r = bt.run(
        *scenario(
            [[100, 100], [100, 100]],
            [[1, 0], [0, 1]],
            expiry=[0, 2],
            delivery_fee_per_contract=200,
            config=cfg,
        )
    )
    assert r.bankruptcy_events == [0]
    np.testing.assert_array_equal(r.cash, [-100, 400])
    assert r.bankruptcy_deficit.sum() == 0
    assert not r.positions.any()


def test_funding_is_per_internal_contract_and_multiplier_is_applied_once():
    m, plan, costs, cfg = scenario(
        [[100], [101]], [[1000], [np.nan]], funding=[[0.002], [0]], fee_rate=0.001
    )
    m = replace(m, futures=replace(m.futures, multiplier=[0.001]))
    r = bt.run(m, plan, costs, cfg)
    np.testing.assert_allclose(r.cash, [97.9, 98.9])
    np.testing.assert_array_equal(r.funding, [2, 0])
    np.testing.assert_allclose(r.commission, [0.1, 0])
    np.testing.assert_allclose(r.futures_margin, [10, 10.1])


def test_min_notional_uses_mark_not_execution_price():
    r = bt.run(*scenario([[110]], [[1]], marks=[[90]], min_notional=[[100]]))
    assert not r.positions.any()
    assert r.coverage["futures_min_notional"] == 1


def test_closed_target_at_boundary_paid_prior_rows_funding():
    r = bt.run(*scenario([[100], [100]], [[1], [0]], funding=[[3], [7]]))
    np.testing.assert_array_equal(r.funding, [3, 0])
    assert r.cash[-1] == 97


def test_step_exposes_current_funding_only_after_step():
    m, plan, costs, cfg = scenario([[100], [100]], [[1], [np.nan]], funding=[[3], [7]])
    sim = bt.Simulator(m, costs, cfg, kind=plan.kind)
    assert sim.state.cash == 100
    first = sim.step([1])
    assert first.funding == 3 and sim.state.cash == 97
    second = sim.step(None)
    assert second.funding == 7 and sim.state.cash == 90


def test_no_record_batch_keeps_crypto_metrics():
    m, plan, costs, cfg = scenario([[100]], [[1]], funding=[[3]])
    cfg = replace(cfg, record_trades=False)
    r = bt.run_batch(m, [plan, plan], costs, cfg, threads=2)
    np.testing.assert_array_equal(r.funding, [[3], [3]])
    assert r.positions is None and r.trades is None


def test_mixed_optional_inputs_preserve_affordability_cash_flow_counter():
    # Development regression, rich_case(2) with C1 inputs: the independent
    # spot bisection spent one extra ULP versus the legacy analytical cap,
    # removing the residual withdrawn on row 12 and changing cash_flow's count.
    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal
    from .generators_futures import rich_case

    m, plan, costs, cfg = rich_case(2)
    rng = np.random.default_rng(2)
    m = replace(
        m,
        futures=replace(
            m.futures,
            funding=rng.uniform(-2, 2, m.shape),
            maint_rate=0.01,
            qty_step=rng.integers(1, 4, m.shape),
            min_qty=rng.integers(1, 5, m.shape),
        ),
    )
    actual, expected = bt.run(m, plan, costs, cfg), oracle(m, plan, costs, cfg)
    assert actual.cash[11] == 2**-45
    assert actual.cash_flow[12] == -(2**-45)
    assert actual.coverage["cash_flow"] == 7
    assert_results_equal(actual, expected)


def test_invalid_quote_is_not_quantity_rounding_without_liquidation():
    # Development regression from the augmented mixed corpus, seed 76:
    # rejecting an impact quote must not be counted as a quantity filter.
    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal

    m, plan, costs, cfg = scenario(
        [[100]],
        [[1]],
        qty_step=[[1]],
        volume=[[0]],
        costs=replace(ZERO, impact_k=[[0.01]]),
    )
    m = replace(m, futures=replace(m.futures, liquidation=None))
    actual = bt.run(m, plan, costs, cfg)
    assert actual.coverage["invalid_quote"] == 1
    assert actual.coverage["futures_rounded_to_zero"] == 0
    assert_results_equal(actual, oracle(m, plan, costs, cfg))


def test_mixed_optional_input_reference_corpus():
    from collections import Counter

    from panelbt.reference import run as oracle

    from .comparison import assert_results_equal
    from .generators_crypto import mixed_case

    counts = Counter()
    for seed in range(200):
        inputs = mixed_case(seed)
        actual, expected = bt.run(*inputs), oracle(*inputs)
        assert_results_equal(actual, expected)
        np.testing.assert_allclose(
            actual.funding, expected.funding, rtol=1e-12, atol=1e-9
        )
        counts.update(actual.coverage)
    for key in (
        "futures_open",
        "funding_paid",
        "funding_received",
        "futures_rounded_to_zero",
    ):
        assert counts[key] > 0, key
