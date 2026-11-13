"""Exit rules in mixed spot + futures panels: reference, stepping, invariance, cases."""

from collections import Counter
from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest
from panelbt.models import RULE_COUNTERS
from panelbt.reference_rules_futures import MixedRulesEngine

from .generators_rules import case as spot_case
from .generators_rules_futures import mixed_case
from .test_exit_rules import assert_matches_reference
from .test_step import assert_bitwise, rows

R = bt.TradeReason


def test_random_mixed_rules_match_reference():
    counts, reasons = Counter(), Counter()
    for seed in range(500):
        market, target, costs, config = mixed_case(seed)
        actual = bt.run(market, target, costs, config)
        expected = MixedRulesEngine(market, target, costs, config).run()
        assert_matches_reference(actual, expected, seed)
        assert [(e.day, e.col, e.qty) for e in actual.expiry_events] == [
            (e.day, e.col, e.qty) for e in expected.expiry_events
        ]
        counts.update({k: actual.diagnostics[k] for k in RULE_COUNTERS[:6]})
        counts["futures_trades"] += actual.coverage["futures_open"]
        counts["liquidation"] += actual.coverage["futures_liquidation"]
        counts["margin_account"] += int(config.margin.allow_short)
        f = market.futures.is_future
        rule_trades = [t for t in actual.trades if t.reason != R.REBALANCE]
        # Rules never touch a futures column.
        assert not any(f[t.col] for t in rule_trades), seed
        reasons.update(t.reason for t in rule_trades)
    for key in (*RULE_COUNTERS[:6], "futures_trades", "liquidation", "margin_account"):
        assert counts[key] > 0, (key, counts)
    assert set(reasons) == {R.STOP_LOSS, R.TAKE_PROFIT, R.TRAILING_STOP, R.MAX_HOLD}


@pytest.mark.parametrize("fill_time", ["open", "close", "average"])
def test_stepping_mixed_rules_is_bitwise_run(fill_time):
    for seed in range(60):
        market, target, costs, config = mixed_case(seed, fill_time=fill_time)
        expected = bt.run(market, target, costs, config)
        plan = rows(target)
        sim = bt.Simulator(market, costs, config, kind=target.kind)
        stop = market.shape[0] // 2
        for day in range(market.shape[0]):
            if day == stop:
                sim = bt.Simulator.restore(
                    sim.snapshot(), market, costs, config, kind=target.kind
                )
            sim.step(plan.get(day))
        assert_bitwise(sim.result(), expected)


def test_untraded_futures_column_leaves_spot_rules_bitwise_unchanged():
    checked = 0
    for seed in range(200):
        spot, target, costs, config = spot_case(seed)
        if target.limits is not None:
            continue
        market, mixed_target, _, mixed_config = mixed_case(seed, hedge=False)
        a = bt.run(spot, target, costs, config)
        b = bt.run(market, mixed_target, costs, mixed_config)
        n = spot.shape[1]
        for key in ("equity", "cash", "nav", "commission", "tax", "margin_usage"):
            assert np.array_equal(getattr(a, key), getattr(b, key)), (seed, key)
        assert np.array_equal(a.positions, b.positions[:, :n])
        assert a.trades == b.trades
        assert a.diagnostics == b.diagnostics
        checked += 1
    assert checked > 50


def _hand_case(collateral_rate):
    """Short 100 spot at 10 and long 10 contracts; row 1 gaps up through the stop."""
    t = 2
    spot_open = np.array([10.0, 13.0])
    fut = np.array([100.0, 100.0])
    open_ = np.column_stack([spot_open, fut])
    high = np.column_stack([[10.0, 13.5], fut])
    low = np.column_stack([[10.0, 12.5], fut])
    close = np.column_stack([[10.0, 13.0], fut])
    futures = bt.Futures(
        is_future=[False, True],
        multiplier=[1.0, 10.0],
        margin_rate=np.array([[0.0, 0.05], [0.0, collateral_rate]]),
        settlement=np.column_stack([np.ones(t), fut]),
        expiry=[t, t],
    )
    market = bt.Market(
        close,
        close,
        np.ones((t, 2), bool),
        np.ones((t, 2), bool),
        can_short=np.ones((t, 2), bool),
        futures=futures,
        open=open_,
        high=high,
        low=low,
    )
    costs = bt.Costs(
        commission_rate=0.15,
        commission_min=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    config = bt.Config(
        initial_cash=1700.0,
        min_qty=1.0,
        qty_step=1.0,
        record_trades=True,
        margin=bt.MarginModel(allow_short=True),
        rules=bt.Rules(fill_time="close", stop_loss=0.1),
    )
    w = np.array([[-100.0, 10.0], [np.nan, np.nan]])
    return market, bt.Targets.from_dense(w, kind="shares"), costs, config


def test_rule_cover_is_trimmed_by_futures_collateral_by_hand():
    # Row 0: short 100 at 10 (commission 150) -> cash 2550; 10 contracts.
    # Row 1: margin 14.5% -> collateral 10 * 10 * 100 * 0.145 = 1450.
    # Own cash = cash - short value (last mark 10) - collateral = 100. Covering
    # q at 13 with 15% commission removes 13 q of short value at 13:
    #   (2550 - 14.95 q) - (1000 - 13 q) - 1450 = 100 - 1.95 q >= 0
    # -> q = 100 / 1.95 (share targets have no lot rounding).
    market, target, costs, config = _hand_case(0.145)
    r = bt.run(market, target, costs, config)
    exits = [x for x in r.trades if x.reason == R.STOP_LOSS]
    assert [(x.day, x.col, x.price) for x in exits] == [(1, 0, 13.0)]
    assert exits[0].qty == pytest.approx(100 / 1.95, rel=1e-9)
    assert r.diagnostics["rule_deferred"] == 1
    expected = MixedRulesEngine(market, target, costs, config).run()
    assert_matches_reference(r, expected, 0)
    # Without collateral (margin rate 0 on row 1) the whole short is covered.
    free = bt.run(*_hand_case(0.0))
    assert [x.qty for x in free.trades if x.reason == R.STOP_LOSS] == [100.0]


def test_scalar_rules_never_close_futures():
    t = 4
    s = np.array([100.0, 90.0, 80.0, 70.0])
    one = np.ones((t, 1))
    futures = bt.Futures(
        is_future=[True],
        multiplier=[10.0],
        margin_rate=0.1,
        settlement=s[:, None],
        expiry=[t],
    )
    m = bt.Market(
        s[:, None],
        s[:, None],
        one > 0,
        one > 0,
        futures=futures,
        open=s[:, None],
        high=s[:, None] * 1.01,
        low=s[:, None] * 0.99,
    )
    config = bt.Config(
        initial_cash=1e5,
        record_trades=True,
        rules=bt.Rules(fill_time="open", stop_loss=0.01, max_hold=1),
    )
    w = np.full((t, 1), np.nan)
    w[0] = 2
    r = bt.run(m, bt.Targets.from_dense(w, kind="shares"), config=config)
    assert r.positions[:, 0].tolist() == [2, 2, 2, 2]
    assert all(x.reason == R.REBALANCE for x in r.trades)
    assert r.diagnostics["stop_loss"] == 0 and r.diagnostics["max_hold"] == 0


def test_mixed_rule_input_errors():
    market, target, costs, config = _hand_case(0.1)
    bad = replace(config, rules=replace(config.rules, stop_loss=np.array([0.1, 0.1])))
    with pytest.raises(ValueError, match="spot columns only"):
        bt.run(market, target, costs, bad)
    with pytest.raises(ValueError, match="spot columns only"):
        bt.Simulator(market, costs, bad, kind="shares")
    per_day = np.full((2, 2), np.nan)
    per_day[:, 0] = 0.1
    ok = replace(config, rules=replace(config.rules, stop_loss=per_day))
    bt.run(market, target, costs, ok)
    per_day[1, 1] = 0.2
    with pytest.raises(ValueError, match="spot columns only"):
        bt.run(
            market,
            target,
            costs,
            replace(config, rules=replace(config.rules, stop_loss=per_day)),
        )
    limited = bt.Targets.from_dense(
        [[-100.0, 10.0], [np.nan, np.nan]],
        kind="shares",
        limits=[[9.0, np.nan], [np.nan, np.nan]],
    )
    with pytest.raises(ValueError, match="futures columns"):
        bt.run(market, limited, costs, config)
    sim = bt.Simulator(market, costs, config, kind="shares")
    with pytest.raises(ValueError, match="futures columns"):
        sim.step(([0], [-100.0], [9.0]))


def test_open_rule_phase_uses_collateral_after_the_rebalance():
    # fill_time="open": the row-1 rebalance adds 5 contracts (10 -> 15) before
    # the rules run, so the trim uses 15 * 10 * 100 * 5.6% = 840 of collateral.
    # Row 0: short 100 at 10 with 40% commission -> cash 2300. Own cash after
    # covering q at 13: (2300 - 18.2 q) - (1000 - 13 q) - 840 = 460 - 5.2 q.
    market, _, _, config = _hand_case(0.056)
    market = replace(
        market,
        futures=replace(
            market.futures, margin_rate=np.array([[0.0, 0.05], [0.0, 0.056]])
        ),
    )
    costs = bt.Costs(
        commission_rate=0.4,
        commission_min=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    config = replace(
        config,
        margin=bt.MarginModel(allow_short=True, short_initial=0.1),
        rules=bt.Rules(fill_time="open", stop_loss=0.1),
    )
    target = bt.Targets.from_dense([[-100.0, 10.0], [-100.0, 15.0]], kind="shares")
    r = bt.run(market, target, costs, config)
    assert r.positions[1, 1] == 15
    exits = [x for x in r.trades if x.reason == R.STOP_LOSS]
    assert [(x.day, x.col) for x in exits] == [(1, 0)]
    assert exits[0].qty == pytest.approx(460 / 5.2, rel=1e-9)
    expected = MixedRulesEngine(market, target, costs, config).run()
    assert_matches_reference(r, expected, 0)
