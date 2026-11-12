import numpy as np
import panelbt
import polars as pl
import pytest

from ashare.events import align, to_targets
from test.events.reference import reference_targets, to_dense

# 十个交易日（跳过周末）；三只股票。
DATES = np.array(
    ["20240102", "20240103", "20240104", "20240105", "20240108",
     "20240109", "20240110", "20240111", "20240112", "20240115"]
)  # fmt: skip
SYMS = np.array(["A", "B", "C"])


def hand_events():
    return pl.DataFrame(
        {
            # 1 A 周二盘后 -> 第 1 天；2 B 周六 -> 周一第 4 天；3 A 周五盘前 -> 第 3 天（重叠）
            # 4 C 周二盘后 -> 第 6 天（当天涨停买不进）；5 B 周四盘后 -> 第 8 天
            "symbol": ["A", "B", "A", "C", "B"],
            "date": ["20240102", "20240106", "20240105", "20240109", "20240111"],
            "kind": ["demo"] * 5,
            "timing": [
                "after_close",
                "after_close",
                "pre_open",
                "after_close",
                "after_close",
            ],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def hand_can_buy():
    can_buy = np.ones((10, 3), dtype=bool)
    can_buy[6, 2] = False
    return can_buy


def test_hand_example_equal_weights():
    out = to_targets(hand_events(), DATES, SYMS, hand_can_buy(), holding=3, defer=1)
    t = out.targets
    assert t.days.tolist() == [1, 4, 6, 7, 8]
    rows = [
        dict(
            zip(
                SYMS[t.cols[t.indptr[i] : t.indptr[i + 1]]],
                t.weights[t.indptr[i] : t.indptr[i + 1]],
                strict=True,
            )
        )
        for i in range(len(t.days))
    ]
    assert rows == [
        {"A": 1.0},
        {"A": 0.5, "B": 0.5},
        {"B": 1.0},
        {"C": 1.0},
        {"B": 0.5, "C": 0.5},
    ]
    pos = out.positions.sort("entry")
    assert pos.select("symbol", "entry", "exit", "events").rows() == [
        ("A", 1, 6, 2),  # 第 3 天的新事件把卖出日从 4 延到 3+3=6
        ("B", 4, 7, 1),
        ("C", 7, 10, 1),  # 第 6 天涨停，顺延一天买入；持有到面板结束
        ("B", 8, 11, 1),
    ]
    status = dict(
        zip(out.events["score"].to_list(), out.events["status"].to_list(), strict=True)
    )
    assert status == {
        1.0: "opened",
        2.0: "opened",
        3.0: "extended",
        4.0: "deferred",
        5.0: "opened",
    }


def test_hand_example_capacity_and_skip():
    out = to_targets(
        hand_events(), DATES, SYMS, hand_can_buy(), holding=3, defer=1, max_positions=1
    )
    t = out.targets
    assert t.days.tolist() == [1, 6, 7]
    assert t.indptr.tolist() == [0, 1, 1, 2]  # 第 6 天 A 卖出、C 还没买到：空行即清仓
    assert SYMS[t.cols].tolist() == ["A", "C"]
    status = dict(
        zip(out.events["score"].to_list(), out.events["status"].to_list(), strict=True)
    )
    assert status == {
        1.0: "opened",
        2.0: "capacity",
        3.0: "extended",
        4.0: "deferred",
        5.0: "capacity",
    }
    # 不顺延：C 放弃
    out = to_targets(hand_events(), DATES, SYMS, hand_can_buy(), holding=3, defer=0)
    assert "not_buyable" in out.counts and out.counts["not_buyable"] == 1
    assert 2 not in set(out.positions["col"].to_list())


def test_hand_example_value_fixed_and_cap():
    out = to_targets(
        hand_events(),
        DATES,
        SYMS,
        hand_can_buy(),
        holding=3,
        defer=1,
        weight="value",
        value="score",
    )
    dense = to_dense(out.targets, 10, 3)
    # 第 4 天：A 的数值已更新为 3（第 3 天的延长事件），B 为 2
    np.testing.assert_allclose(dense[4], [3 / 5, 2 / 5, 0])
    np.testing.assert_allclose(dense[8], [0, 5 / 9, 4 / 9])
    out = to_targets(
        hand_events(),
        DATES,
        SYMS,
        hand_can_buy(),
        holding=3,
        defer=1,
        weight="fixed",
        fixed_weight=0.7,
    )
    dense = to_dense(out.targets, 10, 3)
    np.testing.assert_allclose(dense[1], [0.7, 0, 0])
    np.testing.assert_allclose(dense[4], [0.5, 0.5, 0])  # 1.4 缩到 1
    out = to_targets(
        hand_events(), DATES, SYMS, hand_can_buy(), holding=3, defer=1, max_weight=0.4
    )
    dense = to_dense(out.targets, 10, 3)
    np.testing.assert_allclose(dense[1], [0.4, 0, 0])
    np.testing.assert_allclose(dense[4], [0.4, 0.4, 0])


def test_stack_and_ignore():
    ev = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": ["20240102", "20240103"],
            "kind": ["k", "k"],
            "timing": ["after_close"] * 2,
        }
    )
    stack = to_targets(
        ev, DATES, SYMS, holding=3, overlap="stack", weight="fixed", fixed_weight=0.3
    )
    dense = to_dense(stack.targets, 10, 3)
    np.testing.assert_allclose(dense[[1, 2, 4, 5], 0], [0.3, 0.6, 0.3, 0.0])
    ignore = to_targets(ev, DATES, SYMS, holding=3, overlap="ignore")
    assert ignore.targets.days.tolist() == [1, 4]
    assert ignore.counts["ignored"] == 1


def test_entry_lag_and_tail():
    ev = pl.DataFrame(
        {
            "symbol": ["B", "C"],
            "date": ["20240103", "20240112"],
            "kind": ["k", "k"],
            "timing": ["after_close"] * 2,
        }
    )
    out = to_targets(ev, DATES, SYMS, entry_lag=2, holding=2)
    assert out.targets.days.tolist() == [4, 6]
    assert out.counts["after_panel_end"] == 1  # C：可成交日第 9 天 + 2 超出面板


def test_targets_feed_panelbt():
    prices = np.full((10, 3), 10.0)
    market = panelbt.Market(prices, prices, prices > 0, prices > 0)
    out = to_targets(hand_events(), DATES, SYMS, hand_can_buy(), holding=3, defer=1)
    costs = panelbt.Costs(
        commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
    )
    result = panelbt.run(
        market,
        out.targets,
        costs,
        panelbt.Config(initial_cash=1e6, min_qty=0, qty_step=0),
    )
    np.testing.assert_allclose(result.nav, 1.0)


def test_input_checks():
    ev = hand_events()
    with pytest.raises(ValueError):
        to_targets(ev, DATES, SYMS, weight="value")
    with pytest.raises(ValueError):
        to_targets(ev, DATES, SYMS, holding=0)
    with pytest.raises(ValueError):
        to_targets(
            ev.with_columns(pl.lit(-1.0).alias("score")),
            DATES,
            SYMS,
            weight="value",
            value="score",
        )
    with pytest.raises(ValueError):
        to_targets(ev, DATES, SYMS, np.ones((3, 3), bool))
    with pytest.raises(ValueError):
        to_targets(ev, DATES, SYMS, weight="fixed")
    with pytest.raises(ValueError, match="can_sell 形状"):
        to_targets(ev, DATES, SYMS, can_sell=np.ones((10, 2), bool))
    with pytest.raises(ValueError, match="can_buy 形状"):
        to_targets(ev, DATES, SYMS, np.ones((10, 2), bool))


def test_capacity_ties_do_not_depend_on_code_or_row_order():
    """同一天、同一 priority 抢一个名额：按 tie_key 哈希定先后，与代码大小和表的行序无关。"""
    from ashare.events import tie_key

    syms = np.array([f"{i:06d}.SZ" for i in range(8)])
    ev = pl.DataFrame(
        {
            "symbol": syms.tolist(),
            "date": ["20240103"] * 8,
            "kind": ["k"] * 8,
            "timing": ["after_close"] * 8,
        }
    )
    picks = {}
    for seed in range(4):
        out = to_targets(ev, DATES, syms, holding=3, max_positions=1, seed=seed)
        chosen = out.positions["symbol"].to_list()
        expect = syms[np.argmin(tie_key(syms, ["20240103"] * 8, seed))]
        assert chosen == [expect]
        shuffled = ev.sample(fraction=1.0, shuffle=True, seed=seed)
        again = to_targets(shuffled, DATES, syms, holding=3, max_positions=1, seed=seed)
        assert again.positions["symbol"].to_list() == chosen
        picks[seed] = chosen[0]
    assert len(set(picks.values())) > 1  # 换种子换人：不是总挑代码最小的


def test_capacity_released_after_actual_sell():
    """卖出日跌停卖不出：给了 can_sell 时名额等到实际卖出那天才释放。"""
    ev = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": ["20240102", "20240105"],
            "kind": ["k", "k"],
            "timing": ["after_close", "after_close"],
        }
    )
    can_sell = np.ones((10, 3), dtype=bool)
    can_sell[4:6, 0] = False  # A 计划第 4 天卖，第 4、5 天卖不出，第 6 天卖出
    plain = to_targets(ev, DATES, SYMS, holding=3, max_positions=1)
    assert plain.events["status"].to_list() == ["opened", "opened"]  # B 第 4 天拿到名额
    blocked = to_targets(ev, DATES, SYMS, holding=3, max_positions=1, can_sell=can_sell)
    assert blocked.events["status"].to_list() == ["opened", "capacity"]


def random_case(seed, t=40, n=6, m=30):
    rng = np.random.default_rng(seed)
    days = np.array([str(20200101 + 2 * i) for i in range(t)])  # 隔一天一个交易日
    syms = np.array([f"S{i}" for i in range(n)])
    raw_days = rng.integers(20200101 - 3, 20200101 + 2 * t + 3, m)
    ev = pl.DataFrame(
        {
            "symbol": rng.choice(np.append(syms, "X"), m).tolist(),
            "date": [str(d) for d in raw_days],
            "kind": ["k"] * m,
            "timing": rng.choice(["after_close", "pre_open"], m).tolist(),
            "value": rng.uniform(0.5, 3, m).round(2),
        }
    )
    can_buy = rng.random((t, n)) > 0.3
    return days, syms, ev, can_buy


@pytest.mark.parametrize("seed", range(60))
def test_matches_reference(seed):
    days, syms, ev, can_buy = random_case(seed)
    rng = np.random.default_rng(1000 + seed)
    kw = {
        "entry_lag": int(rng.integers(0, 3)),
        "holding": int(rng.integers(1, 8)),
        "weight": ["equal", "value", "fixed"][seed % 3],
        "fixed_weight": 0.15,
        "max_positions": [None, 2, 4][seed % 4 % 3],
        "max_weight": [None, 0.3][seed % 2],
        "overlap": ["extend", "ignore", "stack"][(seed // 3) % 3],
        "defer": int(rng.integers(0, 3)),
        "seed": seed % 5,
    }
    can_sell = rng.random(can_buy.shape) > 0.4 if seed % 2 else None
    prio = (
        "value" if seed % 4 else "flat"
    )  # 四分之一的用例 priority 全相同，全靠哈希定先后
    ev = ev.with_columns(pl.lit(1.0).alias("flat"))
    out = to_targets(
        ev, days, syms, can_buy, can_sell=can_sell, value="value", priority=prio, **kw
    )
    ordered = align(ev, days, syms).frame  # 与正式实现相同的事件顺序（按日期、证券）
    rows = [
        {
            "symbol": r["symbol"],
            "date": r["date"],
            "timing": r["timing"],
            "value": r["value"],
            "priority": r[prio],
        }
        for r in ev.with_row_index("_o")
        .sort(["date", "symbol", "kind", "_o"])
        .iter_rows(named=True)
    ]
    assert len(ordered) <= len(rows)
    ref = reference_targets(rows, days, syms, can_buy, can_sell=can_sell, **kw)
    got = to_dense(out.targets, len(days), len(syms))
    np.testing.assert_array_equal(np.isnan(got[:, 0]), np.isnan(ref[:, 0]))
    np.testing.assert_allclose(
        np.nan_to_num(got, nan=-1), np.nan_to_num(ref, nan=-1), atol=1e-12
    )


def test_reference_cases_hit_every_branch():
    """对拍用例确实走到了各个分支（顺延、名额、延长、忽略、叠加、买不进、超出面板）。"""
    seen = set()
    for seed in range(60):
        days, syms, ev, can_buy = random_case(seed)
        rng = np.random.default_rng(1000 + seed)
        out = to_targets(
            ev, days, syms, can_buy, value="value",
            entry_lag=int(rng.integers(0, 3)), holding=int(rng.integers(1, 8)),
            max_positions=[None, 2, 4][seed % 4 % 3], overlap=["extend", "ignore", "stack"][(seed // 3) % 3],
            defer=int(rng.integers(0, 3)),
        )  # fmt: skip
        seen |= set(out.counts)
    assert {
        "opened",
        "deferred",
        "extended",
        "ignored",
        "capacity",
        "not_buyable",
        "after_panel_end",
    } <= seen


# ---------- 防前视 ----------


def lookahead_case(seed):
    rng = np.random.default_rng(seed)
    t, n, m = 60, 8, 50
    days = np.array([str(20200101 + i) for i in range(t)])
    syms = np.array([f"S{i}" for i in range(n)])
    ev = pl.DataFrame(
        {
            "symbol": rng.choice(syms, m).tolist(),
            "date": [str(20200101 + int(d)) for d in rng.integers(0, t, m)],
            "kind": ["k"] * m,
            "timing": rng.choice(["after_close", "pre_open"], m).tolist(),
            "value": rng.uniform(0.5, 3, m),
        }
    )
    return days, syms, ev, rng.random((t, n)) > 0.2


KW = {
    "holding": 5,
    "defer": 2,
    "max_positions": 4,
    "weight": "value",
    "value": "value",
    "overlap": "extend",
}


@pytest.mark.parametrize("seed", range(20))
def test_no_lookahead_drop_future_events(seed):
    days, syms, ev, can_buy = lookahead_case(seed)
    full = to_dense(
        to_targets(ev, days, syms, can_buy, **KW).targets, len(days), len(syms)
    )
    for cut in range(5, 55, 7):
        part = ev.filter(pl.col("date") <= days[cut])
        got = to_dense(
            to_targets(part, days, syms, can_buy, **KW).targets, len(days), len(syms)
        )
        np.testing.assert_array_equal(
            np.nan_to_num(got[: cut + 1], nan=-1),
            np.nan_to_num(full[: cut + 1], nan=-1),
        )


@pytest.mark.parametrize("seed", range(20))
def test_no_lookahead_future_market_data(seed):
    """把第 t 天之后的可买掩码随机改掉，t 天及以前的目标不变。"""
    days, syms, ev, can_buy = lookahead_case(seed)
    can_sell = np.random.default_rng(seed + 7).random(can_buy.shape) > 0.3
    full = to_dense(
        to_targets(ev, days, syms, can_buy, can_sell=can_sell, **KW).targets,
        len(days),
        len(syms),
    )
    rng = np.random.default_rng(seed + 99)
    for cut in range(5, 55, 7):
        changed = can_buy.copy()
        changed[cut + 1 :] = rng.random(changed[cut + 1 :].shape) > 0.5
        sell = can_sell.copy()
        sell[cut + 1 :] = rng.random(sell[cut + 1 :].shape) > 0.5
        got = to_dense(
            to_targets(ev, days, syms, changed, can_sell=sell, **KW).targets,
            len(days),
            len(syms),
        )
        np.testing.assert_array_equal(
            np.nan_to_num(got[: cut + 1], nan=-1),
            np.nan_to_num(full[: cut + 1], nan=-1),
        )


def test_after_close_event_never_trades_on_announcement_day():
    """收盘后公告不得在公告日当天成交；把『下一交易日』改成『当日』的变异由此抓到。"""
    days = np.array(["20240102", "20240103", "20240104"])
    ev = pl.DataFrame(
        {
            "symbol": ["A"],
            "date": ["20240103"],
            "kind": ["k"],
            "timing": ["after_close"],
        }
    )
    assert to_targets(ev, days, np.array(["A"]), holding=1).targets.days.tolist() == [2]
    ev = ev.with_columns(pl.lit("pre_open").alias("timing"))
    assert to_targets(ev, days, np.array(["A"]), holding=1).targets.days.tolist() == [
        1,
        2,
    ]
