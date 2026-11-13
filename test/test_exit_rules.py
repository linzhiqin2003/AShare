"""声明式退出规则与限价目标：参考实现对拍、步进等价、性质与边界。"""

import json
from collections import Counter
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.models import METRICS, RULE_COUNTERS
from panelbt.reference import run as reference_run
from panelbt.reference_rules import RulesEngine

from .digest import corpora, digest
from .generators_rules import case
from .test_step import assert_bitwise

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])

R = bt.TradeReason
RULE_REASONS = {R.STOP_LOSS, R.TAKE_PROFIT, R.TRAILING_STOP, R.MAX_HOLD}
ZERO = bt.Costs(
    commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
)


def assert_matches_reference(actual, expected, seed):
    for key in (*METRICS, "positions", "contribution"):
        np.testing.assert_allclose(
            getattr(actual, key),
            getattr(expected, key),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"seed={seed}, field={key}",
        )
    assert actual.diagnostics == expected.diagnostics, seed
    assert actual.coverage == expected.coverage, seed
    assert len(actual.trades) == len(expected.trades), seed
    for a, b in zip(actual.trades, expected.trades, strict=True):
        assert (a.day, a.col, a.reason) == (b.day, b.col, b.reason), (seed, a, b)
        np.testing.assert_allclose(astuple(a), astuple(b), rtol=1e-9, atol=1e-9)
    assert len(actual.delist_events) == len(expected.delist_events), seed
    for a, b in zip(actual.delist_events, expected.delist_events, strict=True):
        np.testing.assert_allclose(astuple(a), astuple(b), rtol=1e-9, atol=1e-9)


def test_random_rules_match_reference():
    counts = Counter()
    branches = Counter()
    reasons = Counter()
    for seed in range(1500):
        market, target, costs, config = case(seed)
        actual = bt.run(market, target, costs, config)
        engine = RulesEngine(market, target, costs, config)
        expected = engine.run()
        assert_matches_reference(actual, expected, seed)
        counts.update({k: actual.diagnostics[k] for k in RULE_COUNTERS})
        counts["liquidation"] += actual.coverage["liquidation"]
        counts["blocked_sell"] += actual.diagnostics["blocked_sell"]
        branches.update(engine.branches)
        reasons.update(t.reason for t in actual.trades)
    # Every rule, every restriction and every order type was really exercised.
    assert all(counts[k] > 0 for k in (*RULE_COUNTERS, "liquidation")), counts
    assert all(reasons[r] > 0 for r in R), reasons
    for branch in (
        "gap_stop",
        "gap_target",
        "gap_stop_short",
        "gap_target_short",
        "both_hit_stop",
        "both_hit_target",
        "trailing_above_stop",
        "hold_at_open",
        "due_retry",
        "reduce_mask_blocked",
        "reduce_sell_lock",
        "reduce_capped",
        "reentry_cooldown",
        "reentry_until_row",
        "dividend_adjusted",
        "limit_at_open",
        "limit_after_exit",
        "limit_carried",
        "limit_touch_only_filled",
        "limit_touch_only_skipped",
        "limit_cash_scaled",
        "limit_margin_scaled",
        "limit_cut_kept",
        "limit_cut_complete",
    ):
        assert branches[branch] > 0, (branch, branches)


@pytest.mark.parametrize("fill_time", ["open", "close", "average"])
def test_stepping_with_rules_is_bitwise_run(fill_time):
    from .test_step import rows

    for seed in range(120):
        market, target, costs, config = case(seed, fill_time=fill_time)
        expected = bt.run(market, target, costs, config)
        plan = rows(target)
        limits = {}
        if target.limits is not None:
            for k, day in enumerate(target.days.tolist()):
                limits[day] = target.limits[target.indptr[k] : target.indptr[k + 1]]
        sim = bt.Simulator(market, costs, config, kind=target.kind)
        stop = market.shape[0] // 2
        for day in range(market.shape[0]):
            if day == stop:
                sim = bt.Simulator.restore(
                    sim.snapshot(), market, costs, config, kind=target.kind
                )
            row = plan.get(day)
            sim.step(None if row is None else (*row, limits.get(day)))
        assert_bitwise(sim.result(), expected)
        for a, b in zip(sim.result().trades, expected.trades, strict=True):
            assert a.reason == b.reason


def test_batch_with_rules_equals_single_runs():
    market, target, costs, config = case(4)
    others = [case(s)[1] for s in (4, 4)]
    batch = bt.run_batch(market, others, costs, config, threads=2)
    single = bt.run(market, target, costs, config)
    for k in RULE_COUNTERS:
        assert batch.diagnostics[k][0] == single.diagnostics[k]
    np.testing.assert_array_equal(batch.nav[0], single.nav)
    np.testing.assert_array_equal(batch.nav[1], single.nav)


def test_runs_without_rules_match_frozen_pre_rules_build():
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/no_rules_digest.json").read_text()
    )
    seen = 0
    for name, args in corpora(150):
        assert digest(bt.run(*args)) == frozen[name], name
        seen += 1
    assert seen == len(frozen)


# ----------------------------------------------------------------- properties
def long_only(seed, fill_time, **rules):
    market, target, costs, config = case(seed, margin=False, fill_time=fill_time)
    rules = bt.Rules(fill_time=fill_time, **rules)
    return market, replace(target, limits=None), costs, replace(config, rules=rules)


@pytest.mark.parametrize("fill_time", ["open", "close", "average"])
def test_stop_at_100_percent_is_no_rule(fill_time):
    for seed in range(80):
        market, target, costs, config = long_only(
            seed, fill_time, stop_loss=1.0, trailing_stop=1.0
        )
        with_rule = bt.run(market, target, costs, config)
        without = bt.run(market, target, costs, replace(config, rules=None))
        for key in (*METRICS, "positions", "contribution"):
            a, b = getattr(with_rule, key), getattr(without, key)
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), (seed, key)
        assert with_rule.diagnostics["stop_loss"] == 0
        assert with_rule.trades == without.trades


def test_rules_only_reduce_and_fill_inside_the_day_range():
    checked = Counter()
    for seed in range(800):
        market, target, costs, config = case(seed)
        r = bt.run(market, target, costs, config)
        positions = r.positions
        lo, hi, op = market.low, market.high, market.open
        limits = {}
        if target.limits is not None:
            for k, day in enumerate(target.days.tolist()):
                a, b = target.indptr[k], target.indptr[k + 1]
                for c, x in zip(target.cols[a:b], target.limits[a:b], strict=True):
                    limits[(day, int(c))] = x
        # Holdings before a row's first fill: that row's close minus its fills
        # (splits and delistings happen before trading).
        before = Counter()
        for t in r.trades:
            before[(t.day, t.col)] -= t.qty
        before = {k: positions[k] + v for k, v in before.items()}
        for t in r.trades:
            key = (t.day, t.col)
            held = before[key]
            raw = t.price / (1 + np.sign(t.qty) * costs.slippage)
            if t.reason in RULE_REASONS:
                # A rule never adds exposure and never crosses zero.
                assert held != 0 and np.sign(t.qty) == -np.sign(held), (seed, t)
                assert abs(t.qty) <= abs(held) + 1e-9, (seed, t)
                if t.reason != R.MAX_HOLD:
                    tol = 1e-9 * hi[t.day, t.col]
                    assert lo[t.day, t.col] - tol <= raw <= hi[t.day, t.col] + tol
                    checked["inside"] += 1
            if t.reason == R.LIMIT:
                limit = limits.get(key)
                if limit is None:  # an until_next order from an earlier row
                    limit = next(
                        v
                        for (d, c), v in sorted(limits.items(), reverse=True)
                        if c == t.col and d < t.day
                    )
                cross = config.rules.limit_fill == "cross"
                if t.qty > 0:
                    assert raw <= limit * (1 + 1e-12), (seed, t, limit)
                    assert raw == pytest.approx(min(op[t.day, t.col], limit))
                    # The default needs the low strictly below the limit.
                    assert not cross or lo[t.day, t.col] < limit, (seed, t, limit)
                else:
                    assert raw >= limit * (1 - 1e-12), (seed, t, limit)
                    assert raw == pytest.approx(max(op[t.day, t.col], limit))
                    assert not cross or hi[t.day, t.col] > limit, (seed, t, limit)
                checked["limit"] += 1
                checked["limit_" + config.rules.limit_fill] += 1
            before[key] = held + t.qty
    assert checked["inside"] > 500 and checked["limit"] > 100, checked
    assert checked["limit_cross"] > 50 and checked["limit_touch"] > 50, checked


# ------------------------------------------------------- deterministic cases
def bars(open_, high, low, close, exec_price=None, **kw):
    open_, high, low, close = (np.asarray(x, float) for x in (open_, high, low, close))
    shape = open_.shape
    ones = np.ones(shape, bool)
    return bt.Market(
        exec_price if exec_price is not None else open_,
        close,
        kw.pop("can_buy", ones),
        kw.pop("can_sell", ones),
        open=open_,
        high=high,
        low=low,
        **kw,
    )


def single(market, rows, rules, *, kind="shares", costs=ZERO, limits=None, **cfg):
    t, n = market.shape
    w = np.full((t, n), np.nan)
    for day, value in rows.items():
        w[day] = value
    targets = bt.Targets.from_dense(w, kind=kind, limits=limits)
    terms = dict(
        initial_cash=1e5, min_qty=0, qty_step=0, record_trades=True, rules=rules
    )
    config = bt.Config(**(terms | cfg))
    result = bt.run(market, targets, costs, config)
    assert_matches_reference(result, reference_run(market, targets, costs, config), 0)
    return result


def col(a):
    return np.asarray(a, float)[:, None]


def test_gap_below_stop_fills_at_open_and_today_high_does_not_lift_trail():
    # Bought at 10 on row 0 (open). Row 1: high 13 then low 11.5. The trail
    # line from yesterday's peak (10.5) is 9.45, not 13 * 0.9 = 11.7, so no exit.
    m = bars(
        col([10, 10.5, 11.0, 9.0]),
        col([10.5, 13.0, 11.2, 9.5]),
        col([9.9, 10.4, 10.9, 8.5]),
        col([10.4, 11.6, 11.1, 9.2]),
    )
    rules = bt.Rules(fill_time="open", trailing_stop=0.1, stop_loss=0.05)
    r = single(m, {0: [100]}, rules)
    exit_ = [t for t in r.trades if t.reason != R.REBALANCE]
    # Row 2's trail line uses the peak through row 1 (13): 11.7 > open 11.0 -> gap.
    assert [(t.day, t.reason, t.price) for t in exit_] == [(2, R.TRAILING_STOP, 11.0)]


def test_both_hit_is_pessimistic_by_default():
    m = bars(col([10, 10]), col([10, 11.5]), col([10, 8.5]), col([10, 10]))
    stop = bt.Rules(fill_time="open", stop_loss=0.1, take_profit=0.1)
    target = replace(stop, both_hit="target")
    a = single(m, {0: [100]}, stop)
    b = single(m, {0: [100]}, target)
    assert [(t.reason, t.price) for t in a.trades[1:]] == [(R.STOP_LOSS, 9.0)]
    assert [(t.reason, t.price) for t in b.trades[1:]] == [(R.TAKE_PROFIT, 11.0)]
    assert a.diagnostics["stop_loss"] == 1 and b.diagnostics["take_profit"] == 1


def test_short_mirror_and_gap_up_through_stop():
    m = bars(
        col([10, 12, 12]), col([10, 12.5, 12]), col([10, 11.8, 12]), col([10, 12, 12])
    )
    rules = bt.Rules(fill_time="open", stop_loss=0.1)
    cfg = {"margin": bt.MarginModel(allow_short=True)}
    r = single(m, {0: [-100]}, rules, **cfg)
    assert [(t.day, t.reason, t.qty, t.price) for t in r.trades[1:]] == [
        (1, R.STOP_LOSS, 100.0, 12.0)
    ]


def test_sell_lock_defers_same_day_exit_to_next_exec_price():
    # Bought at the open of row 0; the low breaks the stop the same row.
    m = bars(
        col([10, 9.6, 9.0]), col([10, 9.7, 9.2]), col([8.5, 9.4, 8.9]), col([9, 9.5, 9])
    )
    locked = single(m, {0: [100]}, bt.Rules(fill_time="open", stop_loss=0.05))
    free = single(
        m, {0: [100]}, bt.Rules(fill_time="open", stop_loss=0.05, sell_lock=False)
    )
    assert [(t.day, t.price) for t in free.trades[1:]] == [(0, 9.5)]
    assert [(t.day, t.price) for t in locked.trades[1:]] == [(1, 9.6)]
    assert locked.diagnostics["rule_deferred"] == 1


def test_blocked_exit_retries_at_exec_price_and_blocks_reentry():
    sell = np.array([[True], [False], [True], [True]])
    m = bars(
        col([10, 9.0, 8.0, 8.5]),
        col([10, 9.2, 8.3, 8.6]),
        col([10, 8.8, 7.9, 8.4]),
        col([10, 9, 8, 8.5]),
        can_sell=sell,
    )
    rules = bt.Rules(fill_time="open", stop_loss=0.05)
    # Row 2 repeats the target (100 shares): the due exit overrides that
    # column on its own row (not a re-entry block); row 3 is a later target row.
    r = single(m, {0: [100], 2: [100], 3: [100]}, rules)
    trades = [(t.day, t.reason, t.qty, t.price) for t in r.trades]
    assert trades == [
        (0, R.REBALANCE, 100.0, 10.0),
        (2, R.STOP_LOSS, -100.0, 8.0),
        (3, R.REBALANCE, 100.0, 8.5),
    ]
    assert r.diagnostics["rule_deferred"] == 1
    assert r.diagnostics["reentry_blocked"] == 0


def test_cooldown_blocks_new_rows_inside_the_window():
    m = bars(
        col([10, 9, 9, 9, 9]),
        col([10, 9, 9, 9, 9]),
        col([10, 8, 9, 9, 9]),
        col([10] * 5),
    )
    rules = bt.Rules(fill_time="open", stop_loss=0.05, cooldown=2)
    r = single(m, {0: [100], 2: [100], 3: [100], 4: [100]}, rules)
    assert [(t.day, t.reason) for t in r.trades] == [
        (0, R.REBALANCE),
        (1, R.STOP_LOSS),
        (4, R.REBALANCE),
    ]
    assert r.diagnostics["reentry_blocked"] == 2


def test_max_hold_and_close_timing_order():
    m = bars(
        col([10, 10, 10, 10]),
        col([10, 10, 10, 13]),
        col([10, 10, 10, 9]),
        col([10, 10, 10, 10]),
        exec_price=col([10, 10, 10, 10]),
    )
    hold = single(m, {0: [100]}, bt.Rules(fill_time="close", max_hold=2))
    assert [(t.day, t.reason) for t in hold.trades] == [
        (0, R.REBALANCE),
        (2, R.MAX_HOLD),
    ]
    # With fill_time="close" the intraday stop on row 3 comes before max_hold.
    both = single(
        m, {0: [100]}, bt.Rules(fill_time="close", max_hold=3, take_profit=0.2)
    )
    assert [(t.day, t.reason) for t in both.trades] == [
        (0, R.REBALANCE),
        (3, R.TAKE_PROFIT),
    ]


def test_average_fill_time_does_not_evaluate_new_positions_the_same_row():
    m = bars(
        col([10, 10]),
        col([12, 10]),
        col([9, 10]),
        col([10, 10]),
        exec_price=col([10, 10]),
    )
    avg = single(
        m, {0: [100]}, bt.Rules(fill_time="average", take_profit=0.1, sell_lock=False)
    )
    opn = single(
        m, {0: [100]}, bt.Rules(fill_time="open", take_profit=0.1, sell_lock=False)
    )
    assert [t.reason for t in avg.trades] == [R.REBALANCE]
    assert [(t.day, t.reason) for t in opn.trades] == [
        (0, R.REBALANCE),
        (0, R.TAKE_PROFIT),
    ]


def test_limit_orders_touch_gap_and_validity():
    m = bars(
        col([10, 10, 9.0, 9.5]),
        col([10, 10.2, 9.4, 9.6]),
        col([10, 9.7, 8.8, 9.4]),
        col([10] * 4),
    )
    lim = np.full((4, 1), np.nan)
    lim[1] = 9.5  # not touched on row 1 (low 9.7), gap fill at 9.0 on row 2
    day = single(m, {1: [100]}, bt.Rules(fill_time="open"), limits=lim)
    gtc = single(
        m, {1: [100]}, bt.Rules(fill_time="open", limit_valid="until_next"), limits=lim
    )
    assert day.trades == [] and day.diagnostics["limit_unfilled"] == 1
    assert [(t.day, t.reason, t.price) for t in gtc.trades] == [(2, R.LIMIT, 9.0)]
    assert (
        gtc.diagnostics["limit_filled"] == 1 and gtc.diagnostics["limit_unfilled"] == 1
    )


def test_margin_cover_is_trimmed_to_own_cash_like_reference():
    # A gap cover whose commission exceeds the remaining own cash.
    m = bars(
        col([10, 30, 30]),
        col([10, 31, 30]),
        col([10, 29, 30]),
        col([10, 30, 30]),
        can_short=np.ones((3, 1), bool),
    )
    costs = bt.Costs(
        commission_rate=0.15,
        commission_min=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    rules = bt.Rules(fill_time="open", stop_loss=0.1)
    cfg = {"margin": bt.MarginModel(allow_short=True, short_initial=0.2)}
    w = {0: [-2.5]}
    t, n = m.shape
    dense = np.full((t, n), np.nan)
    dense[0] = w[0]
    targets = bt.Targets.from_dense(dense)
    config = bt.Config(
        initial_cash=1e4, min_qty=0, qty_step=0, record_trades=True, rules=rules, **cfg
    )
    engine = RulesEngine(m, targets, costs, config)
    expected = engine.run()
    actual = bt.run(m, targets, costs, config)
    assert_matches_reference(actual, expected, 0)
    assert engine.branches["reduce_cash_trimmed"] > 0
    assert 0 < actual.trades[1].qty < 2500
    assert actual.diagnostics["rule_deferred"] > 0


def test_cash_account_sale_that_cannot_pay_its_fee_is_deferred():
    p = [10, 0.001, 0.001]
    m = bars(col(p), col([10, 0.01, 0.01]), col(p), col(p))
    costs = bt.Costs(
        commission_rate=0,
        commission_min=5.0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    rules = bt.Rules(fill_time="open", stop_loss=0.5)
    t, n = m.shape
    dense = np.full((t, n), np.nan)
    dense[0] = [1000]
    targets = bt.Targets.from_dense(dense, kind="shares")
    config = bt.Config(
        initial_cash=1e4 + 5, min_qty=0, qty_step=0, record_trades=True, rules=rules
    )
    engine = RulesEngine(m, targets, costs, config)
    expected = engine.run()
    actual = bt.run(m, targets, costs, config)
    assert_matches_reference(actual, expected, 0)
    assert engine.branches["reduce_fee_blocked"] > 0


def test_dividend_option_and_split_adjust_the_cost():
    m = bars(
        col([10, 9.5, 4.0]),
        col([10, 9.6, 4.1]),
        col([10, 9.4, 3.9]),
        col([10, 9.5, 4.0]),
        cash_div=col([0, 0.5, 0]),
        split=col([1, 1, 2]),
    )
    sim = bt.Simulator(
        m,
        ZERO,
        bt.Config(
            initial_cash=1e5,
            min_qty=0,
            qty_step=0,
            rules=bt.Rules(fill_time="open", dividend_adjusts_cost=True),
        ),
        kind="shares",
    )
    sim.step(([0], [100.0]))
    assert sim.state.cost[0] == 9.5  # after row 1's dividend
    sim.step()
    assert sim.state.cost[0] == 4.75 and sim.state.shares[0] == 200


def test_input_errors():
    m = bars(col([10, 10]), col([10, 10]), col([10, 10]), col([10, 10]))
    plain = bt.Market(
        np.full((2, 1), 10.0),
        np.full((2, 1), 10.0),
        np.ones((2, 1), bool),
        np.ones((2, 1), bool),
    )
    target = bt.Targets.from_dense([[1.0], [np.nan]])
    with pytest.raises(ValueError, match="open, high and low"):
        bt.run(plain, target, config=bt.Config(rules=bt.Rules(fill_time="open")))
    with pytest.raises(ValueError, match="fill_time"):
        bt.run(m, target, config=bt.Config(rules=bt.Rules(fill_time="noon")))
    with pytest.raises(ValueError, match="stop_loss"):
        bt.run(
            m,
            target,
            config=bt.Config(rules=bt.Rules(fill_time="open", stop_loss=-0.1)),
        )
    with pytest.raises(ValueError, match="require Config"):
        bt.run(m, bt.Targets.from_dense([[1.0], [np.nan]], limits=[[9.0], [np.nan]]))
    with pytest.raises(ValueError, match="low <= open <= high"):
        bars(col([10, 10]), col([10, 9]), col([10, 9.5]), col([10, 10]))
    with pytest.raises(ValueError, match="max_hold"):
        bt.run(
            m, target, config=bt.Config(rules=bt.Rules(fill_time="open", max_hold=0))
        )
    futures = bt.Futures(
        is_future=[True],
        multiplier=[1.0],
        margin_rate=0.1,
        settlement=np.full((2, 1), 10.0),
        expiry=[2],
    )
    fm = replace(m, futures=futures, split=None, cash_div=None)
    # Exit rules are accepted with futures columns (they act on spot columns,
    # see test_exit_rules_futures); limit targets and a rule percentage on a
    # futures column are still rejected.
    with pytest.raises(ValueError, match="futures"):
        bt.run(
            fm,
            bt.Targets.from_dense([[1.0], [np.nan]], limits=[[9.0], [np.nan]]),
            config=bt.Config(rules=bt.Rules(fill_time="open")),
        )
    with pytest.raises(ValueError, match="spot columns only"):
        bt.Simulator(
            fm, config=bt.Config(rules=bt.Rules(fill_time="open", stop_loss=[0.1]))
        )


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_rules_example_runs(name):
    import re

    text = (Path(bt.__file__).resolve().parents[2] / name).read_text()
    blocks = [
        b for b in re.findall(r"```python\n(.*?)```", text, re.S) if "bt.Rules(" in b
    ]
    assert len(blocks) == 1
    ns = {}
    exec(blocks[0], ns)
    slip = bt.Costs().slippage  # the example uses the default costs
    trades = [(t.day, t.reason, t.price) for t in ns["exit_result"].trades]
    assert trades[0][:2] == (0, R.REBALANCE)
    # A gap below the trailing line (peak 10.9 through row 1) fills at the open.
    assert trades[1] == (2, R.TRAILING_STOP, pytest.approx(9.0 * (1 - slip)))
    limit = [(t.day, t.reason, t.price) for t in ns["limit_result"].trades]
    assert limit == [(2, R.LIMIT, pytest.approx(9.0 * (1 + slip)))]


def test_limit_fill_cross_is_default_and_touch_is_optional():
    # Row 1: the low equals the buy limit (9.5) without going below it.
    m = bars(
        col([10, 9.8, 9.6]), col([10, 9.9, 9.7]), col([10, 9.5, 9.4]), col([10] * 3)
    )
    lim = np.full((3, 1), np.nan)
    lim[1] = 9.5
    cross = single(m, {1: [100]}, bt.Rules(fill_time="open"), limits=lim)
    touch = single(
        m, {1: [100]}, bt.Rules(fill_time="open", limit_fill="touch"), limits=lim
    )
    assert cross.trades == [] and cross.diagnostics["limit_unfilled"] == 1
    assert [(t.day, t.reason, t.price) for t in touch.trades] == [(1, R.LIMIT, 9.5)]
    # A sell limit at the high mirrors it.
    m = bars(
        col([10, 10.2, 10]), col([10, 10.5, 10]), col([10, 10.1, 10]), col([10] * 3)
    )
    lim = np.full((3, 1), np.nan)
    lim[1] = 10.5
    rows = {0: [100], 1: [0]}
    cross = single(m, rows, bt.Rules(fill_time="open"), limits=lim)
    touch = single(m, rows, bt.Rules(fill_time="open", limit_fill="touch"), limits=lim)
    assert [t.reason for t in cross.trades] == [R.REBALANCE]
    assert [(t.day, t.reason, t.price) for t in touch.trades][1:] == [
        (1, R.LIMIT, 10.5)
    ]
    with pytest.raises(ValueError, match="limit_fill"):
        single(m, rows, bt.Rules(fill_time="open", limit_fill="near"), limits=lim)


def test_full_weight_cash_limit_buy_is_cash_scaled_not_blocked():
    m = bars(col([10, 10, 10]), col([10, 10, 10]), col([10, 9.4, 9.5]), col([10] * 3))
    lim = np.full((3, 1), np.nan)
    lim[1] = 9.6
    costs = bt.Costs(
        commission_rate=0.001, commission_min=0, fee_rate_buy=0, slippage=0
    )
    r = single(
        m,
        {1: [1.0]},
        bt.Rules(fill_time="open", limit_valid="until_next"),
        kind="weights",
        costs=costs,
        limits=lim,
    )
    (t,) = r.trades
    assert t.qty * t.price + t.commission == pytest.approx(1e5, rel=1e-12)
    assert r.diagnostics["limit_blocked"] == 0 and r.diagnostics["cash_scaled"] == 1
    assert r.diagnostics["limit_filled"] == 1  # complete: no remainder carried


# --------------------------------------------------------------- snapshots
def rules_sim(**rules):
    market, target, costs, config = case(5, fill_time="open")
    config = replace(config, rules=replace(config.rules, **rules))
    sim = bt.Simulator(market, costs, config, kind=target.kind)
    for _ in range(market.shape[0] // 2):
        sim.step()
    return sim, market, costs, config, target.kind


def test_snapshot_fingerprints_every_rule_parameter():
    t, n = case(5, fill_time="open")[0].shape
    panel = np.full((t, n), 0.05)
    sim, market, costs, config, kind = rules_sim(stop_loss=panel)
    blob = sim.snapshot()
    bt.Simulator.restore(blob, market, costs, config, kind=kind)  # same: accepted
    other = panel.copy()
    other[t - 1, n - 1] = 0.06  # one cell of a T×N panel
    changes = [
        {"stop_loss": other},
        {"stop_loss": 0.05},
        {"take_profit": 0.3},
        {"trailing_stop": 0.1},
        {"max_hold": 7},
        {"limit_fill": "touch" if config.rules.limit_fill == "cross" else "cross"},
        {"sell_lock": not config.rules.sell_lock},
    ]
    for change in changes:
        changed = replace(config, rules=replace(config.rules, **change))
        with pytest.raises(ValueError, match="differ"):
            bt.Simulator.restore(blob, market, costs, changed, kind=kind)


def test_snapshot_rejects_every_bit_flip_and_truncation():
    sim, market, costs, config, kind = rules_sim()
    blob = sim.snapshot()
    rng = np.random.default_rng(0)
    for k in range(len(blob)):
        bad = bytearray(blob)
        bad[k] ^= 1 << int(rng.integers(0, 8))
        with pytest.raises(ValueError, match="snapshot"):
            bt.Simulator.restore(bytes(bad), market, costs, config, kind=kind)
    for k in range(len(blob)):
        with pytest.raises(ValueError, match="snapshot"):
            bt.Simulator.restore(blob[:k], market, costs, config, kind=kind)
    with pytest.raises(ValueError, match="snapshot"):
        bt.Simulator.restore(blob + b"\0", market, costs, config, kind=kind)


@pytest.mark.parametrize("fill_time", ["open", "close"])
def test_cash_scaled_counts_a_row_once(fill_time):
    # Row 1: a limit buy (column 0) and an ordinary buy (column 1), both cut by
    # cash 300. Either order of execution counts the row once.
    m = bars(
        np.array([[10, 10], [10, 10], [10, 10]]),
        np.array([[10, 10], [10.1, 10.1], [10, 10]]),
        np.array([[10, 10], [9.5, 9.5], [10, 10]]),
        np.array([[10, 10], [10, 10], [10, 10]]),
    )
    lim = np.full((3, 2), np.nan)
    lim[1, 0] = 9.8
    r = single(
        m,
        {1: [1000, 1000]},
        bt.Rules(fill_time=fill_time),
        limits=lim,
        costs=bt.Costs(commission_min=None),
        initial_cash=300.0,
    )
    # Whichever runs first spends the cash; the limit order is cut either way
    # (to a part or to nothing) and, short by far more than a lot, is blocked.
    first = 0 if fill_time == "close" else 1
    assert [t.col for t in r.trades] == [first]
    assert r.diagnostics["limit_blocked"] == 1
    assert r.diagnostics["cash_scaled"] == 1
