"""手算的规则例子：内核、参考实现、手算期望值三方逐项核对。

参考实现与内核的规则部分结构相近，两边对拍不能排除两边一起错；这些例子的期望值
是照 SPEC 手算的（独立复核时写成），作为兜底。每个例子 3–5 根 K 线，现金账户、
分数股；空头用 `allow_short`。期望值是 (行, 滑点前后的成交价, 原因[, 数量])。
"""

from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt import reference

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])

R = bt.TradeReason
FEE = bt.Costs().commission_rate  # default commission; minimum disabled below


def market(bars, ex="close", **kw):
    a = np.array(bars, float)  # rows: open, high, low, close
    o, h, lo, c = (a[:, k][:, None] for k in range(4))
    t = len(a)
    exec_price = {"open": o, "close": c}[ex]
    can_buy = np.array(kw.pop("can_buy", [True] * t))[:, None]
    can_sell = np.array(kw.pop("can_sell", [True] * t))[:, None]
    return bt.Market(
        exec_price=exec_price,
        close=c,
        can_buy=can_buy,
        can_sell=can_sell,
        open=o,
        high=h,
        low=lo,
    )


def plan(t, rows, limits=None):
    w = np.full((t, 1), np.nan)
    for d, x in rows.items():
        w[d, 0] = x
    lim = None
    if limits:
        lim = np.full((t, 1), np.nan)
        for d, x in limits.items():
            lim[d, 0] = x
    return bt.Targets.from_dense(w, limits=lim)


FLAT = [10, 10, 10, 10]
CASES = [
    # (name, bars, exec, rows, limits, rules, expected fills, counters, short, slippage)
    (
        "gap stop long + slippage",
        [FLAT, [9.0, 9.6, 8.8, 9.2], [9] * 4],
        "close",
        {0: 1.0},
        None,
        dict(stop_loss=0.05),
        [(1, 9.0 * (1 - 0.001), R.STOP_LOSS)],
        {},
        False,
        0.001,
    ),
    (
        "line stop long",
        [FLAT, [9.8, 10.1, 9.3, 9.6], [9] * 4],
        "close",
        {0: 1.0},
        None,
        dict(stop_loss=0.05),
        [(1, 9.5, R.STOP_LOSS)],
        {},
        False,
        0.0,
    ),
    (
        "both hit -> stop",
        [FLAT, [10, 10.6, 9.4, 10], FLAT],
        "close",
        {0: 1.0},
        None,
        dict(stop_loss=0.05, take_profit=0.05),
        [(1, 9.5, R.STOP_LOSS)],
        {},
        False,
        0.0,
    ),
    (
        "both hit -> target",
        [FLAT, [10, 10.6, 9.4, 10], FLAT],
        "close",
        {0: 1.0},
        None,
        dict(stop_loss=0.05, take_profit=0.05, both_hit="target"),
        [(1, 10.5, R.TAKE_PROFIT)],
        {},
        False,
        0.0,
    ),
    (
        "gap target long",
        [FLAT, [11, 11.2, 10.9, 11], FLAT],
        "close",
        {0: 1.0},
        None,
        dict(take_profit=0.05),
        [(1, 11, R.TAKE_PROFIT)],
        {},
        False,
        0.0,
    ),
    (
        "trailing uses previous peak",
        [FLAT, [10.5, 12, 10.4, 11.5], [11.5, 11.6, 10.7, 11], [11] * 4],
        "close",
        {0: 1.0},
        None,
        dict(trailing_stop=0.10),
        [(2, 10.8, R.TRAILING_STOP)],
        {},
        False,
        0.0,
    ),
    (
        "entry row open no lock",
        [FLAT, [10, 10.2, 9.4, 9.8], [9.7, 9.9, 9.6, 9.8]],
        "open",
        {1: 1.0},
        None,
        dict(trailing_stop=0.05, sell_lock=False),
        [(1, 9.5, R.TRAILING_STOP)],
        {},
        False,
        0.0,
    ),
    (
        "entry row open with lock -> next open",
        [FLAT, [10, 10.2, 9.4, 9.8], [9.7, 9.9, 9.6, 9.8]],
        "open",
        {1: 1.0},
        None,
        dict(trailing_stop=0.05, sell_lock=True),
        [(2, 9.7, R.TRAILING_STOP)],
        {"rule_deferred": 1, "trailing_stop": 1},
        False,
        0.0,
    ),
    (
        "entry row close fresh",
        [FLAT, [10, 10.2, 9.0, 9.8], [9.8, 9.9, 9.6, 9.8]],
        "close",
        {1: 1.0},
        None,
        dict(stop_loss=0.05),
        [],
        {"stop_loss": 0},
        False,
        0.0,
    ),
    (
        "short line stop",
        [FLAT, [10.2, 10.8, 10.0, 10.4], FLAT],
        "close",
        {0: -1.0},
        None,
        dict(stop_loss=0.05),
        [(1, 10.5, R.STOP_LOSS)],
        {},
        True,
        0.0,
    ),
    (
        "short gap stop",
        [FLAT, [11, 11.3, 10.9, 11], FLAT],
        "close",
        {0: -1.0},
        None,
        dict(stop_loss=0.05),
        [(1, 11, R.STOP_LOSS)],
        {},
        True,
        0.0,
    ),
    (
        "short take",
        [FLAT, [9.5, 9.8, 8.7, 9.2], FLAT],
        "close",
        {0: -1.0},
        None,
        dict(take_profit=0.10),
        [(1, 9.0, R.TAKE_PROFIT)],
        {},
        True,
        0.0,
    ),
    (
        "short both -> stop",
        [FLAT, [9.5, 10.6, 8.9, 9.2], FLAT],
        "close",
        {0: -1.0},
        None,
        dict(stop_loss=0.05, take_profit=0.10),
        [(1, 10.5, R.STOP_LOSS)],
        {},
        True,
        0.0,
    ),
    (
        "short trailing",
        [FLAT, [9.5, 9.7, 9.3, 9.5], [9.0, 10.5, 8.8, 9.0], [9] * 4],
        "close",
        {0: -1.0},
        None,
        dict(trailing_stop=0.10),
        [(2, 9.3 * 1.1, R.TRAILING_STOP)],
        {},
        True,
        0.0,
    ),
    (
        "cooldown None blocks until next row",
        [FLAT, [9.8, 10, 9.3, 9.6]] + [[9.6, 9.7, 9.5, 9.6]] * 3,
        "close",
        {0: 1.0, 3: 1.0},
        None,
        dict(stop_loss=0.05),
        [(1, 9.5, R.STOP_LOSS), (3, 9.6, R.REBALANCE)],
        {},
        False,
        0.0,
    ),
    (
        "cooldown 1 with daily rows",
        [FLAT, [9.8, 10, 9.3, 9.6]] + [[9.6, 9.7, 9.5, 9.6]] * 3,
        "close",
        {d: 1.0 for d in range(5)},
        None,
        dict(stop_loss=0.05, cooldown=1),
        [(1, 9.5, R.STOP_LOSS), (3, 9.6, R.REBALANCE)],
        {"reentry_blocked": 1},
        False,
        0.0,
    ),
    # The stop fires on row 1 but the stock cannot be sold; row 2 at its close.
    (
        "deferred at next exec",
        [FLAT, [9.8, 10, 9.0, 9.1], [9.0, 9.2, 8.5, 8.7], [8] * 4],
        "close",
        {0: 1.0},
        None,
        dict(stop_loss=0.05),
        [(2, 8.7, R.STOP_LOSS)],
        {"rule_deferred": 1, "stop_loss": 1},
        False,
        0.0,
        {"can_sell": [True, False, True, True]},
    ),
    # Full-weight cash limit buy: cut to what cash affords with the commission.
    (
        "limit buy crossing the limit",
        [FLAT, [10, 10.1, 9.5, 9.8], [9.8, 9.9, 9.7, 9.8]],
        "close",
        {1: 1.0},
        {1: 9.6},
        {},
        [(1, 9.6, R.LIMIT, 1000 / (9.6 * (1 + FEE)))],
        {"limit_filled": 1, "limit_blocked": 0, "cash_scaled": 1},
        False,
        0.0,
    ),
    (
        "limit buy gap at open",
        [FLAT, [9.4, 9.9, 9.3, 9.8], [9.8, 9.9, 9.7, 9.8]],
        "close",
        {1: 1.0},
        {1: 9.6},
        {},
        [(1, 9.4, R.LIMIT)],
        {"limit_filled": 1},
        False,
        0.0,
    ),
    (
        "limit buy untouched day order",
        [FLAT, [10, 10.1, 9.7, 9.8], [9.8, 9.9, 9.5, 9.8]],
        "close",
        {1: 1.0},
        {1: 9.6},
        {},
        [],
        {"limit_unfilled": 1},
        False,
        0.0,
    ),
    (
        "limit buy until_next",
        [FLAT, [10, 10.1, 9.7, 9.8], [9.8, 9.9, 9.5, 9.8]],
        "close",
        {1: 1.0},
        {1: 9.6},
        dict(limit_valid="until_next"),
        [(2, 9.6, R.LIMIT)],
        {"limit_filled": 1, "limit_unfilled": 1},
        False,
        0.0,
    ),
    (
        "limit buy only touching: no fill",
        [FLAT, [10, 10.1, 9.6, 9.8], [9.8, 9.9, 9.7, 9.8]],
        "close",
        {1: 1.0},
        {1: 9.6},
        {},
        [],
        {"limit_unfilled": 1},
        False,
        0.0,
    ),
    (
        "limit buy only touching, touch option",
        [FLAT, [10, 10.1, 9.6, 9.8], [9.8] * 4],
        "close",
        {1: 1.0},
        {1: 9.6},
        dict(limit_fill="touch"),
        [(1, 9.6, R.LIMIT)],
        {"limit_filled": 1},
        False,
        0.0,
    ),
    (
        "max_hold 2 close",
        [[10, 10, 10, 10], [10, 10, 10, 10.1], [10, 10, 10, 10.2], [10, 10, 10, 10.3]],
        "close",
        {0: 1.0},
        None,
        dict(max_hold=2),
        [(2, 10.2, R.MAX_HOLD)],
        {},
        False,
        0.0,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_hand_computed(case):
    name, bars, ex, rows, limits, rules, expected, counters, short, slip, *masks = case
    m = market(bars, ex, **(masks[0] if masks else {}))
    targets = plan(len(bars), rows, limits)
    margin = (
        bt.MarginModel(allow_short=True, short_initial=1.5)
        if short
        else bt.MarginModel()
    )
    config = bt.Config(
        initial_cash=1000.0,
        min_qty=0.0,
        qty_step=0.0,
        record_trades=True,
        rules=bt.Rules(fill_time=ex, **rules),
        margin=margin,
    )
    costs = bt.Costs(
        commission_min=None, fee_rate_buy=0, fee_rate_sell=0, slippage=slip
    )
    kernel = bt.run(m, targets, costs, config)
    ref = reference.run(m, targets, costs, config)
    assert len(kernel.trades) == len(ref.trades), name
    for a, b in zip(kernel.trades, ref.trades, strict=True):
        assert (a.day, a.reason) == (b.day, b.reason), name
        assert (a.qty, a.price) == pytest.approx((b.qty, b.price), rel=1e-12), name
    fills = [(t.day, t.qty, t.price, t.reason) for t in kernel.trades]

    def matches(want):
        return any(
            (day, reason) == (want[0], want[2])
            and price == pytest.approx(want[1], rel=1e-12)
            and (len(want) < 4 or qty == pytest.approx(want[3], rel=1e-12))
            for day, qty, price, reason in fills
        )

    assert all(matches(want) for want in expected), (name, fills)
    # No rule or limit fill beyond the expected ones.
    rule = [f for f in fills if f[3] != R.REBALANCE]
    assert len(rule) == sum(w[2] != R.REBALANCE for w in expected), (name, fills)
    for key, value in counters.items():
        assert kernel.diagnostics[key] == value == ref.diagnostics[key], (name, key)


# ------------------------------------------------- limit orders cut by cash
CUT_BARS = [FLAT, [10, 10.1, 9.5, 9.8], [9.8, 9.9, 9.5, 9.8], [9.8, 9.9, 9.5, 9.8]]


def cut_run(
    flow, *, valid="until_next", target=1000.0, cash=300.0, lot=0.0, kind="shares"
):
    m = market(CUT_BARS, "close")
    w = np.full((4, 1), np.nan)
    w[1, 0] = target
    lim = np.full((4, 1), np.nan)
    lim[1, 0] = 9.6
    targets = bt.Targets.from_dense(w, kind=kind, limits=lim)
    config = bt.Config(
        initial_cash=cash,
        min_qty=lot,
        qty_step=lot,
        record_trades=True,
        cash_flow=np.asarray(flow, float),
        rules=bt.Rules(fill_time="close", limit_valid=valid),
    )
    costs = bt.Costs(commission_min=None)
    kernel = bt.run(m, targets, costs, config)
    ref = reference.run(m, targets, costs, config)
    assert len(kernel.trades) == len(ref.trades)
    for a, b in zip(kernel.trades, ref.trades, strict=True):
        assert (a.day, a.reason) == (b.day, b.reason)
        assert (a.qty, a.price) == pytest.approx((b.qty, b.price), rel=1e-12)
    keys = ("cash_scaled", "limit_filled", "limit_blocked", "limit_unfilled")
    assert {k: kernel.diagnostics[k] for k in keys} == {
        k: ref.diagnostics[k] for k in keys
    }
    return kernel


PAID = 9.6 * (1 + bt.Costs().slippage)  # the limit plus default slippage
# Cash per share bought, including the commission and transfer-fee rates.
FEE_UNIT = PAID * (1 + FEE + bt.Costs().fee_rate_buy)


def test_until_next_keeps_the_remainder_of_a_cash_cut():
    # Cash 300 buys 31.2 of 1000 shares on row 1; 10000 arrives before row 2
    # trades, and the order (still crossed on row 2) buys the rest.
    r = cut_run([0, 0, 10000, 0])
    first = 300 / FEE_UNIT
    assert [(t.day, t.reason) for t in r.trades] == [(1, R.LIMIT), (2, R.LIMIT)]
    assert r.trades[0].qty == pytest.approx(first, rel=1e-12)
    assert r.trades[1].qty == pytest.approx(1000 - first, rel=1e-12)
    assert r.trades[1].price == pytest.approx(PAID, rel=1e-12)
    d = r.diagnostics
    assert (d["cash_scaled"], d["limit_filled"], d["limit_blocked"]) == (1, 2, 1)


def test_zero_fill_is_not_a_completed_order():
    # No cash on row 1: nothing fills, the order stays and fills on row 2.
    r = cut_run([-300, 0, 10000, 0])
    assert [(t.day, t.qty) for t in r.trades] == [(2, 1000.0)]
    assert r.diagnostics["limit_blocked"] == 1


def test_day_order_cut_by_cash_lapses_but_shows_blocked():
    r = cut_run([0, 0, 10000, 0], valid="day")
    assert [t.day for t in r.trades] == [1]
    assert r.diagnostics["limit_blocked"] == 1 and r.diagnostics["cash_scaled"] == 1


def test_cut_by_fees_within_one_lot_completes():
    # Lots of 100, full weight, cash 9.6 * 10400 + 1: the target is 10400
    # shares at the limit; with slippage and fees cash affords 10300, one lot
    # short, so the order is complete and nothing is carried to row 2.
    r = cut_run(
        [0, 0, 10000, 0], target=1.0, cash=9.6 * 10400 + 1, lot=100, kind="weights"
    )
    assert [(t.day, t.qty) for t in r.trades] == [(1, 10300.0)]
    assert r.diagnostics["limit_blocked"] == 0 and r.diagnostics["cash_scaled"] == 1
