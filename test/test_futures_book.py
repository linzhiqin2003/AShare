"""Stand-alone futures portfolios (`FuturesBook`): reference, properties, cases."""

import math

import numpy as np
import panelbt as bt
import pytest
from panelbt import reference_book as ref

from .generators_book import book_case

SEEDS = range(260)


def _pair(seed):
    kw, plan, kind, config = book_case(seed)
    try:
        book = bt.FuturesBook(**kw)
    except ValueError:
        # Missing margin that cannot be filled: the reference must agree.
        with pytest.raises(ValueError):
            ref.run(kw, plan, kind, config)
        return None
    return (
        kw,
        plan,
        kind,
        config,
        book,
        book.run(plan, kind),
        ref.run(kw, plan, kind, config),
    )


def _close(a, b):
    return np.allclose(a, b, rtol=1e-9, atol=1e-7)


def test_random_agreement_with_naive_reference():
    seen = dict(
        call=0,
        liquidation=0,
        bankrupt=0,
        personal=0,
        physical_end=0,
        delivered=0,
        locked=0,
        blocked=0,
        expired=0,
        rounded=0,
        scaled=0,
        kinds=set(),
        missing_raise=0,
    )
    for seed in SEEDS:
        got = _pair(seed)
        if got is None:
            seen["missing_raise"] += 1
            continue
        kw, plan, kind, config, book, r, x = got
        assert np.array_equal(r.positions, x["positions"]), seed
        for key in ("equity", "cash", "commission", "other_fee"):
            assert _close(getattr(r, key), x[key]), (seed, key)
        assert np.array_equal(r.margin_call, x["margin_call"]), seed
        assert np.allclose(r.margin_usage, x["margin_usage"], rtol=1e-9, atol=1e-12), (
            seed
        )
        assert np.array_equal(r.n_expired_target, x["n_expired_target"]), seed
        assert np.array_equal(r.n_blocked_buy, x["n_blocked_buy"]), seed
        assert np.array_equal(r.n_blocked_sell, x["n_blocked_sell"]), seed
        assert [(e.day, e.col, e.qty) for e in r.expiry_events] == [
            (d, j, q) for d, j, q, _, _ in x["expiries"]
        ], seed
        assert _close([e.fee for e in r.expiry_events], [e[4] for e in x["expiries"]])
        assert [(e.day, e.col, e.qty) for e in r.liquidation_events] == x[
            "liquidations"
        ]
        assert list(r.bankruptcy_events) == x["bankruptcies"], seed
        assert r.coverage["futures_rounded_to_zero"] == x["rounded_to_zero"], seed
        # Branch coverage of the corpus (not of a single case).
        seen["kinds"].add(kind)
        seen["call"] += bool(r.margin_call.any())
        seen["liquidation"] += bool(r.liquidation_events)
        seen["bankrupt"] += bool(len(r.bankruptcy_events))
        seen["blocked"] += bool(r.n_blocked_buy.sum() + r.n_blocked_sell.sum())
        seen["expired"] += bool(r.n_expired_target.sum())
        seen["rounded"] += bool(r.coverage["futures_rounded_to_zero"])
        seen["scaled"] += bool(r.coverage["futures_margin_scaled"])
        seen["locked"] += bool(book.locked_up.any() or book.locked_down.any())
        for e in r.expiry_events:
            if book.personal[e.col]:
                seen["personal"] += 1
            elif book.forced_close[e.col]:
                seen["physical_end"] += 1
            else:
                seen["delivered"] += 1
    assert seen["kinds"] == {"lots", "notional", "margin"}
    for key, value in seen.items():
        if key != "kinds":
            assert value >= 3, (key, value)


def test_simulator_matches_run_bit_for_bit():
    for seed in range(40):
        got = _pair(seed)
        if got is None:
            continue
        kw, plan, kind, config, book, r, _ = got
        sim_kind = "lots" if kind == "lots" else "notional"
        targets = book.targets(plan, kind)
        sim = book.simulator(sim_kind)
        rows = dict(zip(targets.days.tolist(), range(len(targets.days)), strict=True))
        for day in range(book.shape[0]):
            k = rows.get(day)
            if k is None:
                sim.step(None)
            else:
                a, b = targets.indptr[k], targets.indptr[k + 1]
                sim.step((targets.cols[a:b], targets.weights[a:b]))
        s = sim.result()
        for key in ("equity", "cash", "commission", "other_fee", "margin_usage"):
            assert np.array_equal(getattr(r, key), getattr(s, key)), (seed, key)
        assert np.array_equal(r.positions, s.positions)


def test_margin_to_notional_matches_margin_targets():
    for seed in range(60):
        kw, plan, kind, config = book_case(seed)
        if kind != "margin":
            continue
        try:
            book = bt.FuturesBook(**kw)
        except ValueError:
            continue
        t = book.targets(plan, "margin")
        for k, day in enumerate(t.days):
            a, b = t.indptr[k], t.indptr[k + 1]
            raw = plan[day, t.cols[a:b]]
            _, converted = book.margin_to_notional(day, t.cols[a:b], raw)
            assert np.array_equal(converted, t.weights[a:b])


def test_daily_cash_identity_and_flat_book():
    for seed in SEEDS:
        got = _pair(seed)
        if got is None:
            continue
        *_, r, _ = got
        change = np.diff(
            np.concatenate(
                (
                    [
                        r.equity[0]
                        - (r.futures_pnl[0] - r.commission[0] - r.other_fee[0])
                    ],
                    r.equity,
                )
            )
        )
        # equity[t] - equity[t-1] = variation and realized P&L - fees.
        assert np.allclose(
            change, r.futures_pnl - r.commission - r.other_fee, rtol=0, atol=1e-6
        )
    # Never trading: equity is constant whatever the prices do.
    kw, plan, kind, config = book_case(3)
    book = bt.FuturesBook(**kw)
    r = book.run(np.full(book.shape, np.nan), kind)
    assert np.all(r.equity == config.initial_cash)


def test_raising_margin_never_allows_more_contracts():
    rng = np.random.default_rng(5)
    for _ in range(80):
        t, n = 6, 3
        s = rng.uniform(80, 120, (t, n))
        base = rng.uniform(0.05, 0.2, (1, n))
        plan = np.full((t, n), np.nan)
        plan[0] = rng.uniform(-8, 8, n)
        opened = []
        for bump in (1.0, 1.5, 3.0):
            book = bt.FuturesBook(
                settlement=s,
                close=s,
                exec_price=s,
                multiplier=np.full(n, 10.0),
                expiry=np.full(n, t),
                margin_rate=base * bump,
                config=bt.Config(initial_cash=2e4, record_trades=True),
            )
            r = book.run(plan, "notional")
            opened.append(np.abs(r.positions[0]).sum())
        assert opened[0] >= opened[1] >= opened[2]


def test_hand_example_forced_close_and_fees():
    # One contract, multiplier 10, margin 10% + 5 pp add-on, fee 1e-4 × 3.
    s = np.array([[100.0], [102.0], [101.0], [99.0]])
    o = np.array([[100.5], [101.0], [101.5], [98.0]])
    book = bt.FuturesBook(
        settlement=s,
        close=s,
        exec_price=o,
        multiplier=[10.0],
        expiry=[3],
        margin_rate=0.10,
        fee_rate=1e-4,
        fee_per_contract=1.0,
        last_hold=[1],
        account=bt.FuturesAccount(margin_add=0.05, fee_scale=3.0),
        costs=bt.Costs(
            commission_rate=0,
            commission_min=None,
            fee_rate_buy=0,
            fee_rate_sell=0,
            slippage=0.001,
        ),
        config=bt.Config(initial_cash=1e5, record_trades=True),
    )
    r = book.run(np.array([[4.0], [np.nan], [np.nan], [np.nan]]), "lots")
    buy = 100.5 * 1.001
    fee0 = 4 * (10 * buy * 3e-4 + 3.0)
    day0 = 1e5 - fee0 + 4 * 10 * (100.0 - buy)
    assert math.isclose(r.equity[0], day0, rel_tol=1e-12)
    assert math.isclose(r.equity[1], day0 + 4 * 10 * 2.0, rel_tol=1e-12)
    # Row 2 is the first row after the last holding row: forced close at
    # that row's settlement, paying the trading fee and the slippage.
    forced = 4 * (10 * 101.0 * (3e-4 + 0.001) + 3.0)
    day2 = day0 + 4 * 10 * 2.0 + 4 * 10 * (101.0 - 102.0) - forced
    assert math.isclose(r.equity[2], day2, rel_tol=1e-12)
    assert r.positions[2, 0] == 0 and r.expiry_events[0].day == 2
    assert book.market.can_buy[2, 0]  # the end row still trades
    assert math.isclose(r.margin_usage[0], 4 * 10 * 100.0 * 0.15 / day0, rel_tol=1e-12)


def test_cash_settled_expiry_pays_delivery_fee_only():
    s = np.array([[100.0], [103.0]])
    book = bt.FuturesBook(
        settlement=s,
        close=s,
        exec_price=s,
        multiplier=[300.0],
        expiry=[1],
        margin_rate=0.12,
        cash_settled=[True],
        delivery_fee_rate=1e-4,
        costs=bt.Costs(
            commission_rate=0,
            commission_min=None,
            fee_rate_buy=0,
            fee_rate_sell=0,
            slippage=0.002,
        ),
        initial_cash=1e6,
    )
    r = book.run(np.array([[-2.0], [np.nan]]), "lots")
    assert r.expiry_events[0].fee == pytest.approx(2 * 300 * 103.0 * 1e-4)
    assert not book.forced_close[0]


def test_locked_days_and_limits_block_one_direction():
    s = np.array([[100.0], [106.0], [106.0]])
    o = np.array([[100.0], [106.0], [104.0]])
    h = np.array([[101.0], [106.0], [107.0]])
    lo = np.array([[99.0], [106.0], [103.0]])
    v = np.array([[10.0], [5.0], [9.0]])
    kw = dict(
        settlement=s,
        close=s,
        open=o,
        high=h,
        low=lo,
        volume=v,
        multiplier=[10.0],
        expiry=[3],
        margin_rate=0.1,
        config=bt.Config(initial_cash=1e5, record_trades=True),
    )
    book = bt.FuturesBook(**kw, infer_locked=True)
    assert book.locked_up[1, 0] and not book.market.can_buy[1, 0]
    assert book.market.can_sell[1, 0]
    r = book.run(np.array([[np.nan], [2.0], [np.nan]]), "lots")
    assert r.positions[1, 0] == 0 and r.positions[2, 0] == 2  # carried
    open_book = bt.FuturesBook(**kw)
    assert open_book.market.can_buy[1, 0]
    capped = bt.FuturesBook(**kw, limit_up=np.full((3, 1), 106.0))
    assert not capped.market.can_buy[1, 0] and capped.market.can_buy[2, 0]


def test_input_errors():
    s = np.array([[100.0], [101.0]])
    base = dict(settlement=s, close=s, exec_price=s, multiplier=[1.0], expiry=[2])
    with pytest.raises(ValueError, match="missing values"):
        bt.FuturesBook(**base, margin_rate=np.array([[0.1], [np.nan]]))
    book = bt.FuturesBook(
        **base, margin_rate=np.array([[0.1], [np.nan]]), missing="previous"
    )
    assert book.margin_rate[1, 0] == 0.1
    book = bt.FuturesBook(**base, margin_rate=np.array([[0.1], [np.nan]]), missing=0.3)
    assert book.margin_rate[1, 0] == 0.3
    with pytest.raises(ValueError, match="ambiguous"):
        bt.FuturesBook(**base, margin_rate=np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="kind"):
        bt.FuturesBook(**base, margin_rate=0.1).targets(np.ones((2, 1)), "weights")
    with pytest.raises(ValueError, match="trading window"):
        bt.FuturesBook(**base, margin_rate=0.1, last_hold=[-1]).targets(
            np.ones((2, 1)), "margin"
        )
    with pytest.raises(ValueError, match="margin_add"):
        bt.FuturesAccount(margin_add=-0.1)


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
def test_readme_book_example_runs(name):
    import re
    from pathlib import Path

    text = (Path(bt.__file__).resolve().parents[2] / name).read_text()
    blocks = [
        b
        for b in re.findall(r"```python\n(.*?)```", text, re.S)
        if "bt.FuturesBook(" in b
    ]
    assert len(blocks) == 1
    ns = {"bt": bt, "np": np}
    exec(blocks[0], ns)
    r = ns["book_result"]
    # 0.5 of equity in margin at 15% effective margin -> notional 3.33x -> 333 lots.
    assert r.trades is None and r.futures_margin[0] > 0
    lots = math.floor(0.5 / 0.15 * 1e5 / (100.0 * 10))
    assert r.futures_margin[0] == pytest.approx(lots * 10 * 100.0 * 0.15)


def test_module_docstring_example():
    s = np.array([[100.0], [101.0], [99.0]])
    book = bt.FuturesBook(
        settlement=s,
        close=s,
        exec_price=s,
        multiplier=[10.0],
        expiry=[3],
        margin_rate=0.1,
        initial_cash=1e5,
    )
    r = book.run(np.array([[10.0], [np.nan], [np.nan]]), kind="lots")
    assert r.equity.tolist() == [100000.0, 100100.0, 99900.0]


def test_mixed_book_equals_hand_built_market():
    # One spot column and one futures column: the book only prepares arrays.
    s = np.array([[10.0, 100.0], [11.0, 102.0], [12.0, 99.0], [11.5, 101.0]])
    costs = bt.Costs(slippage=1e-3)
    config = bt.Config(initial_cash=1e5, min_qty=1, qty_step=1, record_trades=True)
    book = bt.FuturesBook(
        settlement=s,
        close=s,
        exec_price=s,
        multiplier=[1.0, 10.0],
        expiry=[4, 4],
        margin_rate=0.1,
        fee_rate=2e-4,
        is_future=[False, True],
        costs=costs,
        config=config,
    )
    plan = np.array([[0.5, -1.0], [np.nan] * 2, [0.2, -2.0], [np.nan] * 2])
    r = book.run(plan, "notional")
    hand = bt.Market(
        s,
        s,
        np.ones((4, 2), bool),
        np.ones((4, 2), bool),
        futures=bt.Futures(
            is_future=[False, True],
            multiplier=[1.0, 10.0],
            margin_rate=np.array([[0.0, 0.1]] * 4),
            settlement=np.column_stack([np.ones(4), s[:, 1]]),
            expiry=[4, 4],
            fee_rate=np.array([[0.0, 2e-4]] * 4),
        ),
    )
    h = bt.run(hand, bt.Targets.from_dense(plan), costs, config)
    assert np.array_equal(r.equity, h.equity)
    assert np.array_equal(r.positions, h.positions)
    assert r.positions[0, 0] > 0 and r.positions[0, 1] < 0
