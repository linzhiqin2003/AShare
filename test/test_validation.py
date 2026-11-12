"""Kernel content checks: same errors as the NumPy checks, and a working switch."""

import json

import numpy as np
import panelbt as bt
import pytest
from panelbt import _core

from .bad_inputs import (
    CHUNK,
    FIXTURE,
    FUTURE_COLS,
    N,
    T,
    base_futures,
    base_market_arrays,
    base_targets,
    cases,
    outcome,
)

RECORDED = json.loads(FIXTURE.read_text())
CASES = cases()


def test_matrix_matches_recording_and_chunk_size():
    assert _core.VALIDATION_CHUNK == CHUNK
    assert sorted(CASES) == sorted(RECORDED)
    assert T * N > 2 * CHUNK
    # Every recorded case failed before the move; nothing became valid.
    assert all(kind != "ok" for kind, _ in RECORDED.values())


@pytest.mark.parametrize("name", sorted(CASES))
def test_bad_input_reports_the_recorded_error(name):
    assert outcome(CASES[name]) == RECORDED[name]


@pytest.mark.parametrize("field", ["can_buy", "can_sell", "delist", "can_short"])
@pytest.mark.parametrize("position", [0, CHUNK - 1, CHUNK, T * N - 1])
def test_mask_bytes_other_than_zero_or_one_are_rejected(field, position):
    arrays = base_market_arrays()
    raw = arrays[field].view(np.uint8)
    raw.reshape(-1)[position] = 2
    with pytest.raises(ValueError, match=f"^{field}: expected boolean values 0 or 1$"):
        bt.Market(**arrays)
    # Unchecked, a nonzero byte is simply true: no undefined behaviour.
    assert not bt.Market(**arrays, validate=False).validated


def _market(validate=True, futures=False, **changes):
    arrays = base_market_arrays()
    arrays.update(changes)
    spec = bt.Futures(**base_futures()) if futures else None
    return bt.Market(**arrays, futures=spec, validate=validate)


def test_validate_false_skips_only_content_checks():
    bad = base_market_arrays()["exec_price"].copy()
    bad[3, 4] = -1.0
    with pytest.raises(ValueError, match="exec_price"):
        _market(exec_price=bad)
    unchecked = _market(validate=False, exec_price=bad)
    assert not unchecked.validated
    # Shape, dtype and layout are still enforced.
    with pytest.raises(ValueError, match="expected shape"):
        _market(validate=False, close=np.ones((T, N + 1)))
    with pytest.raises(ValueError, match="dimensions"):
        _market(validate=False, volume=np.ones(T))
    settlement = base_futures()["settlement"].copy()
    settlement[0, FUTURE_COLS[0]] = np.nan
    spec = bt.Futures(**{**base_futures(), "settlement": settlement})
    with pytest.raises(ValueError, match="settlement"):
        bt.Market(**base_market_arrays(), futures=spec)
    assert not bt.Market(**base_market_arrays(), futures=spec, validate=False).validated
    with pytest.raises(ValueError, match="T×N matching"):
        bt.Market(
            **base_market_arrays(),
            futures=bt.Futures(**{**base_futures(), "settlement": np.ones((2, 2))}),
            validate=False,
        )


def test_unchecked_plans_skip_content_but_not_bounds():
    plan = base_targets()
    plan["weights"] = plan["weights"].copy()
    plan["weights"][7] = np.nan
    with pytest.raises(ValueError, match="weights: expected finite"):
        bt.Targets(**plan)
    unchecked = bt.Targets(**plan, validate=False)
    assert not unchecked.validated
    for field, value, message in (
        ("indptr", np.r_[1, plan["indptr"][1:]], "indptr"),
        ("cols", np.r_[-1, plan["cols"][1:]], "cols/weights"),
    ):
        broken = {**base_targets(), field: value}
        with pytest.raises(ValueError, match=message):
            bt.Targets(**broken, validate=False)
    market = _market()
    beyond = {**base_targets(), "cols": np.r_[base_targets()["cols"][:-1], N]}
    with pytest.raises(ValueError, match=f"cols: expected index < N={N}"):
        bt.run(market, bt.Targets(**beyond, validate=False))
    late = {**base_targets(), "days": np.r_[base_targets()["days"][:-1], T]}
    with pytest.raises(ValueError, match=f"days: expected index < T={T}"):
        bt.run(market, bt.Targets(**late, validate=False))


def test_unchecked_plan_skips_context_checks_and_result_says_so():
    market = _market()
    plan = base_targets()
    plan["weights"] = plan["weights"] * 2  # row sums 1.8 in a cash account
    with pytest.raises(ValueError, match="row sum"):
        bt.run(market, bt.Targets(**plan))
    result = bt.run(market, bt.Targets(**plan, validate=False))
    assert result.inputs_validated is False
    assert bt.run(market, bt.Targets(**base_targets())).inputs_validated is True
    unchecked_market = _market(validate=False)
    r = bt.run(unchecked_market, bt.Targets(**base_targets()))
    assert r.inputs_validated is False


def test_batch_flag_requires_every_input_checked():
    market = _market()
    good = bt.Targets(**base_targets())
    unchecked = bt.Targets(**base_targets(), validate=False)
    assert bt.run_batch(market, [good, good]).inputs_validated is True
    assert bt.run_batch(market, [good, unchecked]).inputs_validated is False
    assert bt.run_batch(_market(validate=False), [good, good]).inputs_validated is False
    traced = bt.run_batch(
        market, [good, unchecked], config=bt.Config(record_trades=True)
    )
    assert traced.inputs_validated is False


def test_batch_reports_the_first_bad_plan_in_input_order():
    market = _market()
    short = {**base_targets()}
    short["weights"] = short["weights"].copy()
    short["weights"][-1] = -0.001
    heavy = {**base_targets(), "weights": base_targets()["weights"] * 2}
    plans = [bt.Targets(**base_targets())] * 40
    plans[30] = bt.Targets(**heavy)
    plans[35] = bt.Targets(**short)
    with pytest.raises(ValueError, match="row sum"):
        bt.run_batch(market, plans, threads=4)
    plans[30] = bt.Targets(**base_targets())
    with pytest.raises(ValueError, match="allow_short"):
        bt.run_batch(market, plans, threads=4)


def test_validated_and_unchecked_good_inputs_give_identical_results():
    from .generators import case

    for seed in range(200):
        m, t, c, g = case(seed)
        fields = {
            k: getattr(m, k)
            for k in (
                "exec_price",
                "close",
                "can_buy",
                "can_sell",
                "volume",
                "split",
                "cash_div",
                "delist",
                "can_short",
            )
        }
        fast_m = bt.Market(**fields, validate=False)
        fast_t = bt.Targets(t.days, t.indptr, t.cols, t.weights, t.kind, validate=False)
        np.testing.assert_array_equal(fast_m.first_delist, m.first_delist)
        np.testing.assert_array_equal(fast_m.action_indptr, m.action_indptr)
        np.testing.assert_array_equal(fast_m.action_cols, m.action_cols)
        a, b = bt.run(m, t, c, g), bt.run(fast_m, fast_t, c, g)
        assert a.inputs_validated and not b.inputs_validated
        np.testing.assert_array_equal(a.nav, b.nav)
        np.testing.assert_array_equal(a.positions, b.positions)


def test_unsorted_plan_is_reordered_and_duplicates_still_caught():
    plan = bt.Targets([0], [0, 3], [2, 0, 1], [0.3, 0.1, 0.2])
    np.testing.assert_array_equal(plan.cols, [0, 1, 2])
    np.testing.assert_array_equal(plan.weights, [0.1, 0.2, 0.3])
    assert plan.max_row_sum == pytest.approx(0.6) and not plan.has_negative
    with pytest.raises(ValueError, match="duplicate"):
        bt.Targets([0], [0, 3], [2, 0, 2], [0.3, 0.1, 0.2])
    with pytest.raises(ValueError, match="duplicate"):
        bt.Targets([0], [0, 2], [1, 1], [0.3, 0.1])
