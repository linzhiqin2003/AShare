"""逐行推进（``ashare.backtest.futures_step``）：换月、挂起、顺延与强平之间的边界。

全部是手写的合约层面小组合，不读真实行情。
"""

import numpy as np
import panelbt as bt
import polars as pl
import pytest

from ashare.backtest import futures as bf
from ashare.backtest import futures_step


def portfolio(settle, mult, expiry, held, products, *, price=None, cash=1e6, **cfg):
    settle = np.asarray(settle, dtype=float)
    t, n = settle.shape
    book = bt.FuturesBook(
        settlement=settle,
        close=settle,
        exec_price=settle if price is None else price,
        multiplier=mult,
        expiry=expiry,
        margin_rate=0.1,
        fee_rate=1e-4,
        config=bt.Config(initial_cash=cash, record_trades=True, **cfg),
    )
    return bf.FuturesPortfolio(
        book=book,
        dates=np.array([f"2024{i + 1:04d}" for i in range(t)]),
        symbols=np.array([f"C{j}" for j in range(n)]),
        products=products,
        held=np.asarray(held, dtype=np.int64),
        signal_prices=settle[:, : len(products)],
        contracts=pl.DataFrame(),
        splits=[],
        rolls=pl.DataFrame(),
        last_hold={},
    )


def fills(r):
    return sorted((x.day, x.col, x.qty) for x in r.trades)


def _carried_then_liquidated():
    # 列 0、1 是品种 A 的新旧合约（第 7 行换月），列 2 是品种 B（乘数 100）。
    # B 第 0 行还没有合约；第 1 行落地，但第 1–2 行没有成交价，靠引擎顺延到第 3 行开出 20 手；
    # 第 4 行跌价记追保，第 5 行强平到 15 手；第 6 行起价格回升，资金够再买。
    t = 10
    a = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    b = np.array([100, 100, 100, 100, 95, 95, 110, 110, 110, 110], dtype=float)
    s = np.column_stack([a, b])
    price = s.copy()
    price[1:3, 2] = np.nan
    held = np.array([[0, -1]] + [[0, 2]] * 6 + [[1, 2]] * 3)
    pf = portfolio(
        s, [10.0, 10.0, 100.0], [8, t, t], held, ["A", "B"], price=price, cash=2.5e4
    )
    w = np.full((t, 2), np.nan)
    w[0] = [1.0, 20.0]
    return pf, w


def test_carried_landing_is_not_resent_by_another_roll():
    pf, w = _carried_then_liquidated()
    r = pf.run(w, "lots")
    assert r.positions[3, 2] == 20 and r.margin_call[4]
    q = r.positions[5, 2]
    assert 0 < q < 20  # 第 5 行被强平减仓
    # 第 7 行 A 换月：B 不成交，强平后的仓位不会被加回去。
    assert [f for f in fills(r) if f[0] == 7] == [(7, 0, -1.0), (7, 1, 1.0)]
    assert (r.positions[5:, 2] == q).all()


def test_carried_order_of_another_product_survives_a_roll_row():
    # B 的调仓目标在第 1 行给出，第 1–2 行没有成交价、由引擎顺延；A 在第 2 行换月（这一行要写目标）：
    # 写出的这一行要把 B 的顺延单原样带上，B 在第 3 行开出。
    t = 6
    s = np.column_stack([100 + np.arange(t), 102 + np.arange(t), np.full(t, 50.0)])
    price = s.copy()
    price[1:3, 2] = np.nan
    held = np.array([[0, 2]] * 2 + [[1, 2]] * 4)
    pf = portfolio(s, [10.0, 10.0, 10.0], [3, t, t], held, ["A", "B"], price=price)
    w = np.full((t, 2), np.nan)
    w[0] = [2.0, 0.0]
    w[1] = [2.0, 4.0]
    r = pf.run(w, "lots")
    assert fills(r) == [(0, 0, 2.0), (2, 0, -2.0), (2, 1, 2.0), (3, 2, 4.0)]


def test_target_on_a_contract_end_row_trades_that_row():
    # 第 2 行是旧合约的结束行，目标在这一行照常成交，收盘被了结后按手数接到下一个合约。
    t = 5
    s = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    held = np.array([[0]] * 3 + [[1]] * 2)
    pf = portfolio(s, [10.0, 10.0], [2, t], held, ["A"])
    w = np.full((t, 1), np.nan)
    w[2] = 3.0
    r = pf.run(w, "lots")
    assert fills(r)[0] == (2, 0, 3.0)
    assert [(e.day, e.col, e.qty) for e in r.expiry_events] == [(2, 0, 3.0)]
    assert r.positions[3, 1] == 3


def test_roll_and_carry_round_toward_zero_and_report_the_rest():
    # 换月 3 手 × 10 → 乘数 4：7.5 手向零取整为 7，零头 2 个单位计入诊断。
    t = 6
    s = np.full((t, 2), 100.0)
    held = np.array([[0]] * 3 + [[1]] * 3)
    pf = portfolio(s, [10.0, 4.0], [5, t], held, ["A"])
    w = np.full((t, 1), np.nan)
    w[0] = 3.0
    r = pf.run(w, "lots")
    assert r.positions[3, 1] == 7
    d = r.diagnostics["futures_portfolio"]
    assert d["roll_rounding_events"] == 1 and d["roll_rounding_notional"] == 200.0
    # 结束行了结后接续（交易单位分界）同样向零取整：3 手 × 10 → 乘数 4 是 7 手。
    pf = portfolio(s, [10.0, 4.0], [2, t], held, ["A"])
    r = pf.run(w, "lots")
    assert r.positions[2, 0] == 0 and r.positions[3, 1] == 7


def test_roll_with_old_contract_untradable_holds_both_for_a_row():
    # 换月行旧合约没有成交价：新合约照开、旧合约顺延到下一行，持仓加倍一天并计入诊断。
    t = 7
    s = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    price = s.copy()
    price[3, 0] = np.nan
    held = np.array([[0]] * 3 + [[1]] * 4)
    pf = portfolio(s, [10.0, 10.0], [t, t], held, ["A"], price=price)
    w = np.full((t, 1), np.nan)
    w[0] = 5.0
    r = pf.run(w, "lots")
    assert r.positions[3].tolist() == [5.0, 5.0]
    assert r.positions[4].tolist() == [0.0, 5.0]
    assert r.diagnostics["futures_portfolio"]["roll_overlap_events"] == 1


def test_uncarried_residual_is_closed_on_the_next_written_row():
    # 不顺延（carry_blocked=False）时，换月没平掉的旧合约留着；下一次写目标的行（B 换月）把它平掉。
    t = 8
    s = np.column_stack(
        [100 + np.arange(t), 101 + np.arange(t), 50 + np.arange(t), 51 + np.arange(t)]
    ).astype(float)
    price = s.copy()
    price[3, 0] = np.nan
    held = np.array([[0, 2]] * 3 + [[1, 2]] * 2 + [[1, 3]] * 3)
    pf = portfolio(
        s, [10.0] * 4, [t] * 4, held, ["A", "B"], price=price, carry_blocked=False
    )
    w = np.full((t, 2), np.nan)
    w[0] = [2.0, 3.0]
    r = pf.run(w, "lots")
    assert r.positions[4, 0] == 2  # 残余还在
    assert r.positions[5, 0] == 0 and r.positions[5, 3] == 3


@pytest.mark.parametrize("kind", ["lots", "notional", "margin"])
def test_target_before_the_contract_trades_waits_in_every_kind(kind):
    # 持有的合约前两行还没有成交价（不在交易窗口里）：三种写法都挂起，第 2 行落地。
    t = 5
    s = np.full((t, 1), 100.0)
    price = s.copy()
    price[:2] = np.nan
    pf = portfolio(s, [10.0], [t], [[0]] * t, ["A"], price=price)
    w = np.full((t, 1), np.nan)
    w[0] = {"lots": 4.0, "notional": 0.0041, "margin": 0.00041}[kind]  # 都是 4 手
    r = pf.run(w, kind)
    assert fills(r) == [(2, 0, 4.0)]
    assert r.diagnostics["futures_portfolio"]["landings"] == 1


def _random_case(seed):
    rng = np.random.default_rng(seed)
    t = int(rng.integers(10, 40))
    p = int(rng.integers(1, 4))
    kind = str(rng.choice(["lots", "notional", "margin"]))
    held = np.full((t, p), -1)
    expiry, mult = [], []
    c = 0
    for k in range(p):
        cuts = sorted(rng.choice(np.arange(2, t - 1), 2, replace=False))
        start = int(rng.integers(0, 3))
        for i, (a, b) in enumerate(
            [(start, cuts[0]), (cuts[0], cuts[1]), (cuts[1], t)]
        ):
            held[a:b, k] = c + i
            expiry.append(min(b + int(rng.integers(0, 3)), t) if i < 2 else t)
            mult.append(float(rng.choice([10.0, 10.0, 4.0, 5.0])))
        c += 3
    s = rng.uniform(50, 500, c) * np.exp(np.cumsum(rng.normal(0, 0.04, (t, c)), axis=0))
    price = s * np.exp(rng.normal(0, 0.005, (t, c)))
    price[rng.random((t, c)) < 0.08] = np.nan
    pf = portfolio(
        s,
        mult,
        expiry,
        held,
        [f"P{k}" for k in range(p)],
        price=price,
        cash=float(rng.choice([3e4, 1e5, 1e6])),
        carry_blocked=bool(rng.random() < 0.8),
    )
    scale = {"lots": 10.0, "notional": 0.8, "margin": 0.08}[kind]
    w = np.full((t, p), np.nan)
    for d in range(t):
        if d == 0 or rng.random() < 0.15:
            w[d] = np.round(rng.normal(0, 1, p), 2) * scale
    if kind == "lots":
        w = np.trunc(w)
    return pf, w, kind


def test_random_fills_off_target_rows_come_from_the_engine_alone():
    # I3：不是调仓行、也不是这个品种的换月或落地行，成交只能来自引擎自己——行首顺延中的单子、
    # 追保强平或结束行了结。成交原因字段不区分顺延和强平，所以用步进记录的行首顺延集合
    # 和强平、到期事件判断。
    seen = dict(calls=0, carried=0, rolls=0)
    for seed in range(200):
        pf, w, kind = _random_case(seed)
        trace = []
        r = futures_step.simulate(pf, w, kind, trace=trace)
        owner = {
            int(c): k for k in range(w.shape[1]) for c in set(pf.held[:, k]) if c >= 0
        }
        rebalance = set(np.flatnonzero(~np.all(np.isnan(w), axis=1)).tolist())
        liq = {(e.day, e.col) for e in r.liquidation_events}
        end = {(e.day, e.col) for e in r.expiry_events}
        for x in r.trades:
            day, ordered, carried = trace[x.day]
            assert day == x.day
            if x.day in rebalance or owner[x.col] in ordered:
                seen["rolls"] += x.day not in rebalance
                continue
            if x.col in carried:
                seen["carried"] += 1
                continue
            assert (x.day, x.col) in liq or (x.day, x.col) in end, (seed, kind, x)
            seen["calls"] += 1
        # 现金恒等：上日权益 + 期货盈亏 − 手续费 − 强平费 = 当日权益。
        prev = np.r_[pf.book.config.initial_cash, r.equity[:-1]]
        gap = prev + r.futures_pnl - r.commission - r.other_fee - r.equity
        assert np.max(np.abs(gap)) < 1e-6 * max(1.0, np.max(np.abs(r.equity)))
    assert seen["calls"] > 0 and seen["carried"] > 0 and seen["rolls"] > 0, seen


def test_waiting_target_lands_and_closes_a_leftover_old_contract():
    # 第 1 行给 3 手，但当行持有的新合约第 3 行才有成交价（挂起）；旧合约这一行没有成交价、
    # 不顺延，留着 5 手。第 3 行落地：按调用方的 3 手开新合约，同时平掉旧合约（不是按旧手数换月）。
    t = 6
    s = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    price = s.copy()
    price[1, 0] = np.nan
    price[:3, 1] = np.nan
    held = np.array([[0]] + [[1]] * 5)
    pf = portfolio(
        s, [10.0, 10.0], [t, t], held, ["A"], price=price, carry_blocked=False
    )
    w = np.full((t, 1), np.nan)
    w[0] = 5.0
    w[1] = 3.0
    r = pf.run(w, "lots")
    assert fills(r) == [(0, 0, 5.0), (3, 0, -5.0), (3, 1, 3.0)]


@pytest.mark.parametrize("cash", [1.6e4, 2e4])
def test_nothing_is_reopened_after_a_liquidation(cash):
    # 换月行旧合约没有成交价、不顺延：新合约 5 手 + 旧合约残余 5 手。新合约下一行大跌、追保；
    # 第 4 行强平，同一行也是旧合约的结束行。强平掉的仓位之后不能被「接续」或加回去
    # （1.6 万：全部平光；2 万：各剩 2 手，旧合约的 2 手在结束行了结）。
    t = 8
    s = np.column_stack([np.full(t, 100.0), np.r_[100.0, 100, 100, 70, 70, 70, 70, 70]])
    price = s.copy()
    price[2, 0] = np.nan
    held = np.array([[0]] * 2 + [[1]] * 6)
    pf = portfolio(
        s,
        [100.0, 100.0],
        [4, t],
        held,
        ["A"],
        price=price,
        cash=cash,
        carry_blocked=False,
    )
    w = np.full((t, 1), np.nan)
    w[0] = 5.0
    r = pf.run(w, "lots")
    assert r.positions[2].tolist() == [5.0, 5.0] and r.margin_call[3]
    q = r.positions[4, 1]
    assert q < 5 and r.positions[4, 0] == 0
    assert (r.positions[4:, 1] == q).all()
    assert not [f for f in fills(r) if f[0] > 4]
    assert r.diagnostics["futures_portfolio"]["liquidation_rows"] >= 1


@pytest.mark.parametrize("cash, left", [(4.2e4, 14.0), (4.5e4, 17.0)])
def test_liquidation_on_an_end_row_carries_what_is_left(cash, left):
    # 20 手，第 3 行大跌记追保；第 4 行既是强平行、又是旧合约的结束行：强平掉一部分，剩下的
    # 在收盘按结算价了结，第 5 行接到新合约（和追保早一天发生时一样）。
    t = 8
    s = np.column_stack([np.r_[100.0, 100, 100, 85, 85, 85, 85, 85], np.full(t, 80.0)])
    held = np.array([[0]] * 5 + [[1]] * 3)
    pf = portfolio(s, [100.0, 100.0], [4, t], held, ["A"], cash=cash)
    w = np.full((t, 1), np.nan)
    w[0] = 20.0
    r = pf.run(w, "lots")
    assert [(e.day, e.qty) for e in r.liquidation_events] == [(4, left - 20)]
    assert [(e.day, e.qty) for e in r.expiry_events] == [(4, left)]
    assert r.positions[5:, 1].tolist() == [left] * 3


def test_liquidation_drops_what_was_waiting_before_it():
    # B 第 0 行给了目标，但它的合约第 5 行才有成交价（挂起）；A 第 3 行大跌、第 4 行强平：
    # 强平之前就在等的 B 不再开仓，计入诊断。
    t = 8
    s = np.column_stack([np.r_[100.0, 100, 100, 85, 85, 85, 85, 85], np.full(t, 50.0)])
    price = s.copy()
    price[:5, 1] = np.nan
    held = np.array([[0, 1]] * t)
    pf = portfolio(s, [100.0, 10.0], [t, t], held, ["A", "B"], price=price, cash=4.2e4)
    w = np.full((t, 2), np.nan)
    w[0] = [20.0, 3.0]
    r = pf.run(w, "lots")
    assert r.margin_call[3] and not r.positions[:, 1].any()
    d = r.diagnostics["futures_portfolio"]
    assert d["liquidation_dropped_waits"] == 1 and d["liquidation_rows"] == 1


def test_rebalance_on_an_end_row_carries_with_the_multiplier_ratio():
    # 调仓行同时是旧合约（乘数 25）的结束行：当行成交 2 手、收盘了结，下一合约（乘数 5）按
    # 「2 × 25 / 5」接上 10 手。
    t = 5
    s = np.full((t, 2), 100.0)
    held = np.array([[0]] * 3 + [[1]] * 2)
    pf = portfolio(s, [25.0, 5.0], [2, t], held, ["A"])
    w = np.full((t, 1), np.nan)
    w[2] = 2.0
    r = pf.run(w, "lots")
    assert fills(r)[0] == (2, 0, 2.0)
    assert r.positions[3:, 1].tolist() == [10.0, 10.0]


def test_landing_waits_for_the_products_order_in_flight():
    # 第 1 行改目标为新合约 3 手（新合约第 2 行才有价，挂起）；旧合约 5 手第 1–2 行没价，平仓由引擎
    # 顺延。落地要等这个品种在途的平仓做完（第 3 行），第 4 行再开新合约，不同时挂两张单。
    t = 6
    s = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    price = s.copy()
    price[1:3, 0] = np.nan
    price[:2, 1] = np.nan
    held = np.array([[0]] + [[1]] * 5)
    pf = portfolio(s, [10.0, 10.0], [t, t], held, ["A"], price=price)
    w = np.full((t, 1), np.nan)
    w[0] = 5.0
    w[1] = 3.0
    r = pf.run(w, "lots")
    assert fills(r) == [(0, 0, 5.0), (3, 0, -5.0), (4, 1, 3.0)]


def test_roll_respects_the_callers_order_in_flight_on_the_new_contract():
    # 第 2 行既换月又调仓：调用方要新合约 4 手，新合约第 2–3 行没价、由引擎顺延；旧合约 5 手第 2 行
    # 没价、平仓也顺延。第 3 行品种还拿着旧合约：不能按旧手数（5 手）「换月」盖掉调用方的 4 手。
    t = 7
    s = np.column_stack([100 + np.arange(t), 101 + np.arange(t)]).astype(float)
    price = s.copy()
    price[2, 0] = np.nan
    price[2:4, 1] = np.nan
    held = np.array([[0]] * 2 + [[1]] * 5)
    pf = portfolio(s, [10.0, 10.0], [t, t], held, ["A"], price=price)
    w = np.full((t, 1), np.nan)
    w[0] = 5.0
    w[2] = 4.0
    r = pf.run(w, "lots")
    assert fills(r) == [(0, 0, 5.0), (3, 0, -5.0), (4, 1, 4.0)]
