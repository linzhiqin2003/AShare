"""Review C1 R1: SPEC-derived expectations, independent of either engine.

Every original defect is exercised against both implementations. Sources and
hand calculations are kept beside the assertions, not inferred by comparison.
"""

import io
import json
from dataclasses import replace
from functools import partial

import numpy as np
import panelbt as bt
import pytest
from panelbt.reference_crypto import run as reference_run

from .test_crypto import ZERO, scenario


def test_review_corpus_requires_same_row_combinations():
    from collections import Counter

    from .generators_crypto import COMBINATIONS, combination_counts, directed_cases

    counts = Counter()
    for inputs in directed_cases():
        counts.update(combination_counts(inputs, bt.run(*inputs)))
    assert all(counts[k] >= 2 for k in COMBINATIONS), counts


def step_run(m, plan, costs, cfg, *, restore=False):
    from .test_step import state_copy, states_equal

    sim = bt.Simulator(m, costs, cfg, kind=plan.kind, fingerprint="full")
    for row in plan.to_dense(*m.shape):
        sim.step(row)
        if restore:
            before = state_copy(sim)
            sim = bt.Simulator.restore(
                sim.snapshot(), m, costs, cfg, kind=plan.kind, fingerprint="full"
            )
            states_equal(before, state_copy(sim))
    return sim.result()


@pytest.fixture(
    params=[bt.run, reference_run, step_run, partial(step_run, restore=True)],
    ids=["native", "reference", "step", "snapshot"],
)
def engine(request):
    return request.param


@pytest.mark.parametrize("target,remaining,cash", [(4, 5, 900), (1, 2, 960)])
def test_review_quantity_filtered_reduction_is_not_pending(
    engine, target, remaining, cash
):
    # P0-1: a step-rounded reduction is a completed decision, not a market block.
    r = engine(
        *scenario(
            [[100], [100], [90], [80]],
            [[5], [target], [np.nan], [np.nan]],
            cash=1000,
            qty_step=[[1], [3], [1], [1]],
        )
    )
    np.testing.assert_array_equal(
        r.positions[:, 0], [5, remaining, remaining, remaining]
    )
    assert r.cash[-1] == cash
    assert r.diagnostics["carry_trade"] == 0
    assert r.diagnostics["blocked_sell"] == 0


def test_review_minimum_filtered_reduction_is_not_pending(engine):
    r = engine(
        *scenario(
            [[100], [100], [90]],
            [[5], [4], [np.nan]],
            cash=1000,
            min_qty=[[1], [2], [1]],
        )
    )
    np.testing.assert_array_equal(r.positions[:, 0], [5, 5, 5])
    assert r.cash[-1] == 950
    assert r.diagnostics["blocked_sell"] == 0


def test_review_volume_block_survives_quantity_filter(engine):
    # A real cap of two prevents the requested four-contract reduction. The
    # step of three also filters the attempt, but the market cause still carries.
    r = engine(
        *scenario(
            [[100], [100], [100]],
            [[5], [1], [np.nan]],
            qty_step=[[1], [3], [1]],
            volume=[[100], [2], [100]],
            config=bt.Config(
                initial_cash=1000, max_participation=1, record_trades=True
            ),
        )
    )
    np.testing.assert_array_equal(r.positions[:, 0], [5, 5, 1])
    assert r.diagnostics["blocked_sell"] == 1
    assert r.diagnostics["carry_trade"] == 1


def test_review_zero_equity_cannot_use_candidate_profit_but_deposit_revives(engine):
    # P0-2: clearance leaves exactly zero without bankruptcy. Buying at 80 with
    # mark 100 must not bootstrap collateral. A later external 10 enables it.
    r = engine(
        *scenario(
            [[100], [80], [80]],
            [[1], [1], [1]],
            marks=[[100], [100], [100]],
            low=[[80], [100], [100]],
            clearance=1,
            config=bt.Config(initial_cash=10, cash_flow=[0, 0, 10], record_trades=True),
        )
    )
    np.testing.assert_array_equal(r.cash, [0, 0, 30])
    np.testing.assert_array_equal(r.positions[:, 0], [0, 0, 1])
    assert r.bankruptcy_events == []
    assert r.coverage["futures_reopen"] == 1


def test_review_reduction_to_zero_equity_cannot_finance_another_opening(engine):
    # P0-2: the account is solvent at the open, then the reduction realizes -10.
    r = engine(
        *scenario(
            [[100, 100], [90, 80], [100, 100]],
            [[1, 0], [0, 1], [np.nan, np.nan]],
            cash=10,
            marks=[[100, 100]] * 3,
        )
    )
    np.testing.assert_array_equal(r.cash, [10, 0, 0])
    np.testing.assert_array_equal(r.positions, [[1, 0], [0, 0], [0, 0]])
    assert r.diagnostics["carry_trade"] == 0


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("carry", [True, False])
def test_review_invalid_ordinary_quote_keeps_only_carryable_target(
    engine, side, reduction, carry
):
    # P0-3: volume zero makes impact infinite, independently of participation.
    if reduction:
        prices, targets, volume = (
            [[100]] * 3,
            [[side], [0], [np.nan]],
            [[10000], [0], [100]],
        )
        expected_pos = [side, side, 0 if carry else side]
        expected_cash = 99.9899 if carry else 99.9999
    else:
        prices, targets, volume = [[100]] * 2, [[side], [np.nan]], [[0], [100]]
        expected_pos = [0, side if carry else 0]
        expected_cash = 99.99 if carry else 100
    r = engine(
        *scenario(
            prices,
            targets,
            volume=volume,
            costs=replace(
                ZERO, impact_k=np.full((len(prices), 1), 0.01), impact_exponent=1
            ),
            config=bt.Config(initial_cash=100, carry_blocked=carry, record_trades=True),
        )
    )
    np.testing.assert_array_equal(r.positions[:, 0], expected_pos)
    assert r.cash[-1] == pytest.approx(expected_cash)
    assert r.coverage["invalid_quote"] == 1
    direction = "sell" if (side > 0) == reduction else "buy"
    assert r.diagnostics["blocked_" + direction] == 1
    assert r.diagnostics["carry_trade"] == int(carry)


def test_review_invalid_forced_quote_still_errors(engine):
    with pytest.raises(ValueError, match="liquidation.*quote|quote.*liquidation"):
        engine(
            *scenario(
                [[100], [80]],
                [[1], [np.nan]],
                cash=10,
                initial=0.09,
                volume=[[100], [0]],
                costs=replace(ZERO, impact_k=[[0.01], [0.01]], impact_exponent=1),
            )
        )


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("kind", ["weights", "shares"])
def test_review_executable_intent_requires_marks_before_weight_sizing(
    engine, side, kind
):
    # P1: a missing denominator is data loss, never a zero-sized weight order.
    m, _, costs, cfg = scenario([[100]], [[side]], marks=[[np.nan]], settlement=[[100]])
    plan = bt.Targets.from_dense([[side]], kind=kind)
    with pytest.raises(ValueError, match="mark"):
        engine(m, plan, costs, cfg)


@pytest.mark.parametrize("mode", ["masked", "no_quote", "expired", "zero_target"])
def test_review_unheld_inactive_marks_may_be_nan(engine, mode):
    rows = 2 if mode == "expired" else 1
    m, _, costs, cfg = scenario(
        [[np.nan if mode == "no_quote" else 100]] * rows,
        [[1]] * rows,
        marks=[[np.nan]] * rows,
        settlement=[[100]] * rows,
        buy=[[mode != "masked"]] * rows,
        expiry=[0 if mode == "expired" else 1],
    )
    plan = bt.Targets.from_dense(
        [[0], [1]] if mode == "expired" else [[0 if mode == "zero_target" else 1]]
    )
    r = engine(m, plan, costs, cfg)
    np.testing.assert_array_equal(r.cash, [100] * rows)
    assert not r.positions.any()


@pytest.mark.parametrize("pre_futures", [False, True])
def test_review_old_npz_keeps_crypto_disabled_and_frame_columns(pre_futures):
    # P2: recreate the version-1 layout by removing fields absent in that era.
    m, plan, costs, cfg = scenario([[100]], [[1]])
    legacy = replace(m.futures, liquidation=None, maint_rate=None, exec_at_open=False)
    expected = bt.run(replace(m, futures=legacy), plan, costs, cfg)
    original = io.BytesIO()
    expected.to_npz(original)
    original.seek(0)
    missing = {"funding", "clearance_fee", "bankruptcy_deficit"}
    if pre_futures:
        missing |= {"futures_pnl", "futures_margin", "delivery_fee", "n_expired_target"}
    with np.load(original, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files if k not in missing}
    metadata = json.loads(values["metadata"].item())
    for name in missing:
        metadata.pop(name, None)
    values["metadata"] = np.array(json.dumps(metadata))
    old = io.BytesIO()
    np.savez_compressed(old, **values)
    old.seek(0)
    restored = bt.Result.from_npz(old)
    assert restored.funding is None
    assert restored.clearance_fee is None
    assert restored.bankruptcy_deficit is None
    assert list(restored.to_frame().columns) == list(expected.to_frame().columns)
    if pre_futures:
        for field in (
            "futures_pnl",
            "futures_margin",
            "delivery_fee",
            "n_expired_target",
        ):
            np.testing.assert_array_equal(getattr(restored, field), [0])


@pytest.mark.parametrize("enabled", [False, True])
def test_review_current_npz_preserves_enabled_zero_arrays(enabled):
    m, plan, costs, cfg = scenario([[100]], [[0]])
    if not enabled:
        m = replace(
            m,
            futures=replace(
                m.futures, liquidation=None, maint_rate=None, exec_at_open=False
            ),
        )
    expected = bt.run(m, plan, costs, cfg)
    buffer = io.BytesIO()
    expected.to_npz(buffer)
    buffer.seek(0)
    restored = bt.Result.from_npz(buffer)
    assert list(restored.to_frame().columns) == list(expected.to_frame().columns)
    for name in ("funding", "clearance_fee", "bankruptcy_deficit"):
        if enabled:
            np.testing.assert_array_equal(getattr(restored, name), [0])
        else:
            assert getattr(restored, name) is None


@pytest.mark.parametrize("phase", ["post_trade", "close"])
@pytest.mark.parametrize("side", [1, -1])
def test_review_combined_liquidation_hand_values(engine, phase, side):
    from .generators_crypto import combination_case

    r = engine(*combination_case(phase, side))
    assert r.coverage["liquidation_" + phase] == 1
    for other in {"open", "post_trade", "intraday", "close"} - {phase}:
        assert r.coverage["liquidation_" + other] == 0
    if phase == "post_trade":
        # 6 - .4 opening fees - 3 realized reduction loss - (.1-.003*s)
        # reduction fee - .3 forced fees - .3 clearance. Expiry/funding cancel.
        balance = 1.9 + 0.003 * side
        assert r.expiry_events == []
        np.testing.assert_array_equal(r.funding, np.zeros(5))
        assert r.delivery_fee[1] == 0
        assert len(r.liquidation_events) == 3
        assert r.commission[1] == pytest.approx(0.4 - 0.003 * side)
    else:
        # 6 - .4 openings - .1 delivery - (2*1.5 + 1*.5) funding
        # - .3 forced fees - .3 clearance = 1.4. Expired col's 99 is absent.
        balance = 1.4
        assert len(r.expiry_events) == 1
        e = r.expiry_events[0]
        assert (e.day, e.col, e.qty, e.settlement, e.fee) == (1, 2, side, 100, 0.1)
        assert r.funding[1] == 3.5
        assert r.delivery_fee[1] == 0.1
        assert len(r.liquidation_events) == 2
        assert r.commission[1] == pytest.approx(0.3)
    np.testing.assert_allclose(
        r.cash,
        [5.6, balance, balance, balance - 0.1, balance - 0.1],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        r.positions,
        np.array(
            [
                [2, -1, 1, 0],
                [0] * 4,
                [0] * 4,
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        * side,
    )
    assert r.clearance_fee[1] == pytest.approx(0.3)
    assert r.coverage["futures_reopen"] == 1
    assert r.diagnostics["carry_trade"] == 0
    assert r.bankruptcy_events == []
    assert sum(r.bankruptcy_deficit) == 0


@pytest.mark.parametrize("phase", ["post_trade", "close"])
@pytest.mark.parametrize("side", [1, -1])
def test_review_combined_liquidation_state_survives_every_snapshot(phase, side):
    from .generators_crypto import combination_case
    from .test_step import assert_bitwise, state_copy, states_equal

    inputs = combination_case(phase, side)
    m, plan, costs, cfg = inputs
    expected = bt.run(*inputs)
    direct = bt.Simulator(m, costs, cfg, kind=plan.kind, fingerprint="full")
    restored = bt.Simulator(m, costs, cfg, kind=plan.kind, fingerprint="full")
    for day, row in enumerate(plan.to_dense(*m.shape)):
        direct.step(row)
        restored.step(row)
        restored = bt.Simulator.restore(
            restored.snapshot(), m, costs, cfg, kind=plan.kind, fingerprint="full"
        )
        states_equal(state_copy(direct), state_copy(restored))
        assert direct.state.pending_cols.tolist() == ([3] if day == 0 else [])
        assert direct.state.pending_targets.tolist() == ([side] if day == 0 else [])
        np.testing.assert_array_equal(direct.state.shares, expected.positions[day])
        assert not direct.state.bankrupt
    for result in (direct.result(), restored.result()):
        assert_bitwise(result, expected)
        for field in ("funding", "clearance_fee", "bankruptcy_deficit"):
            assert (
                getattr(result, field).tobytes() == getattr(expected, field).tobytes()
            )
