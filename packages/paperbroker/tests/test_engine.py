import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import paperbroker
import pytest
from paperbroker import Broker, Costs, Lots, MarketConfig, money
from paperbroker.reference import ReferenceBroker

assert (
    Path(paperbroker.__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[3])
)


def quote(
    symbol=0,
    time=1,
    price=10,
    volume=0,
    *,
    ask=10,
    ask_size=1000,
    bid=10,
    bid_size=1000,
    upper=11,
    lower=9,
    halted=0,
    auction_volume=0,
    auction_kind=0,
):
    return [
        symbol,
        time,
        money(price),
        money(price),
        money(price),
        money(price),
        money(10),
        volume,
        money(price * volume),
        -1 if ask is None else money(ask),
        ask_size,
        -1 if bid is None else money(bid),
        bid_size,
        money(upper),
        money(lower),
        halted,
        auction_volume,
        auction_kind,
    ]


def feed(b, *quotes):
    b.feed(np.array(quotes, dtype=np.int64))


def observe_at_order(b, time, symbol=0):
    """Observe an unchanged volume baseline at submission before testing later fills."""
    row = json.loads(b.dump())["quotes"][symbol]
    row[1] = time
    b.feed(np.array([row], np.int64))


def start(config=None):
    b = Broker(["A", "B", "C"], config=config)
    b.session(0, 20261111, "continuous")
    feed(b, quote(), quote(1), quote(2))
    b.events()
    return b


def fills(b):
    return [e for e in b.events() if e["type"] == "fill"]


def test_depth_volume_shared_and_protection():
    b = start()
    b.order(2, "A", "buy", 1000, 11)
    b.order(2, "A", "buy", 1000, 11, "market")
    observe_at_order(b, 2)
    feed(b, quote(time=3, volume=1000, ask_size=40))
    fs = fills(b)
    assert [f["qty"] for f in fs] == [40]
    assert fs[0]["price"] == money(10)
    feed(b, quote(time=4, volume=1100, ask_size=500))
    assert sum(f["qty"] for f in fills(b)) == 10
    feed(b, quote(time=5, volume=1100))
    assert not fills(b)
    b.cancel(6, 1)
    b.cancel(6, 2)
    assert b.account()["frozen"] == 0
    assert b.account()["cash"] == money(1_000_000 - 500 - 5)
    b.order(7, "A", "buy", 100, 9)
    feed(b, quote(time=8, volume=10000))
    assert not fills(b)


def test_fallback_limit_locks_and_stale():
    b = start()
    b.order(2, "A", "buy", 1000, 11)
    feed(b, quote(time=3, price=11, volume=10000, ask=None))
    assert not fills(b)
    feed(b, quote(time=4, volume=11000, ask=None))
    fs = fills(b)
    assert fs[0]["qty"] == 100 and fs[0]["fallback"]
    feed(b, quote(time=4, volume=100000, ask=None))
    feed(b, quote(time=5, volume=100, ask=None))
    assert not fills(b)
    feed(b, quote(time=6, volume=12000, ask=0))
    assert not fills(b)


def test_t1_expiry_split_dividend_costs_and_sell_tax():
    b = start()
    b.order(2, "A", "buy", 200, 11)
    observe_at_order(b, 2)
    feed(b, quote(time=3, volume=1000))
    b.order(4, "A", "sell", 100, 9)
    events = b.events()
    assert events[-1]["type"] == "rejected"
    assert events[-1]["reason"] == "sellable"
    b.session(5, 20261111, "closed")
    b.settle(
        6,
        20261111,
        [
            dict(
                symbol=0,
                cash_per_share=money(1),
                split_num=2,
                split_den=1,
                reference_price=None,
            )
        ],
    )
    state = json.loads(b.dump())
    assert state["positions"][0] == dict(
        total=200, sellable=200, frozen=0, cost=money(905)
    )
    assert state["cash"] == money(999095)
    assert state["frozen"] == 0
    assert state["orders"][0]["status"] == "expired"
    assert state["marks"][0] == money(4.5)
    b.session(7, 20261112, "continuous")
    assert b.account()["market_value"] == money(900)
    feed(b, quote(time=8, price=4.5, ask=4.5, bid=4.5, upper=5, lower=4))
    b.order(9, "A", "sell", 200, 4)
    feed(b, quote(time=10, price=4, ask=4, bid=None, volume=10000, upper=5, lower=4))
    assert not [f for f in fills(b) if f["side"] == "sell"]
    feed(b, quote(time=11, price=4.5, ask=4.5, bid=4.5, volume=20000, upper=5, lower=4))
    fs = fills(b)
    assert fs[0]["fees"] == dict(commission=money(5), tax=money(0.45), other=0)
    assert b.account()["cash"] == money(999989.55)
    assert b.account()["positions"][0]["total"] == 0


def test_auction_once_and_late_orders():
    b = start()
    b.session(2, 20261111, "open_auction")
    b.order(3, "A", "buy", 1000, 11)
    feed(
        b,
        quote(
            time=5, price=10.5, ask=0, volume=10000, auction_volume=2000, auction_kind=1
        ),
    )
    assert not fills(b)
    b.order(6, "A", "buy", 100, 11)
    b.session(7, 20261111, "continuous")
    fs = fills(b)
    assert (
        len(fs) == 1
        and fs[0]["qty"] == 200
        and fs[0]["price"] == money(10.5)
        and fs[0]["auction"]
    )
    b.session(8, 20261111, "continuous")
    assert not fills(b)
    b.session(9, 20261111, "close_auction")
    feed(
        b, quote(time=10, price=10.7, volume=11000, auction_volume=1000, auction_kind=2)
    )
    b.session(11, 20261111, "closed")
    assert fills(b)[0]["price"] == money(10.7)


def test_market_config_t0_lots_slippage():
    config = MarketConfig(
        t_plus_one=False,
        tick=50,
        enforce_bands=False,
        lots=Lots(1),
        slippage_ticks=1,
        slippage_ppm=1000,
    )
    b = start(config)
    b.order(2, "A", "buy", 3, 20)
    observe_at_order(b, 2)
    feed(b, quote(time=3, volume=100, ask=10.005))
    fs = fills(b)
    assert fs[0]["price"] == money(10.025)
    assert b.account()["positions"][0]["sellable"] == 3
    b.order(4, "A", "sell", 3, 1)
    observe_at_order(b, 4)
    feed(b, quote(time=5, volume=200))
    assert fills(b)[0]["qty"] == 3


def test_invalid_inputs_are_atomic():
    b = start()
    before = b.dump()
    with pytest.raises(ValueError):
        feed(b, quote(time=2), quote(symbol=99, time=3))
    assert b.dump() == before
    with pytest.raises(ValueError):
        b.feed(np.zeros((18, 3), np.int64).T)
    with pytest.raises(ValueError):
        Broker(["A"], config=MarketConfig(participation_ppm=1_000_001))
    with pytest.raises(ValueError):
        b.apply(dict(type="order", time=2, side="short"))


def assert_invariants(state, previous, events, coverage, interest=0):
    positions, orders = state["positions"], state["orders"]
    assert state["cash"] >= 0 and state["frozen"] >= 0
    expected_frozen = 0
    for o in orders:
        if o["status"] == "open" and o["side"] == "buy":
            value = o["notional"] + o["remaining"] * o["price"]
            c = state["config"]["instruments"][o["symbol"]]["costs"]
            future_fee = max(
                c["minimum"], (value * c["commission_ppm"] + 999999) // 1000000
            )
            reserve = o["remaining"] * o["price"] + future_fee - sum(o["fees"].values())
            assert o["reserve"] == reserve
            expected_frozen += reserve
    assert state["frozen"] == expected_frozen
    for i, p in enumerate(positions):
        assert 0 <= p["sellable"] <= p["total"]
        assert 0 <= p["frozen"] <= p["total"] - p["sellable"]
        assert p["frozen"] == sum(
            o["remaining"]
            for o in orders
            if o["status"] == "open" and o["side"] == "sell" and o["symbol"] == i
        )
    fill_events = [e for e in events if e["type"] == "fill"]
    for e in events:
        coverage[e["type"]] += 1
        if e["type"] == "rejected":
            coverage["reject:" + e["reason"]] += 1
        if e["type"] == "ended":
            coverage[e["status"]] += 1
        if e["type"] == "corporate":
            coverage["cash_dividend"] += e["dividend"] > 0
            coverage["split_applied"] += e["after"] != e["before"]
        if e["type"] == "fill":
            coverage["auction" if e["auction"] else "continuous"] += 1
            coverage["fallback"] += e["fallback"]
            coverage["partial"] += (
                state["orders"][e["id"] - state["order_offset"] - 1]["remaining"] > 0
            )
    # Exact self-financing check, including executions away from the marking price.
    old_equity = (
        previous["cash"]
        + previous["frozen"]
        + sum(
            p["total"] * m
            for p, m in zip(previous["positions"], previous["marks"], strict=True)
        )
    )
    new_equity = (
        state["cash"]
        + state["frozen"]
        + sum(p["total"] * m for p, m in zip(positions, state["marks"], strict=True))
    )
    pnl = sum(
        p["total"] * (new - old)
        for p, new, old in zip(
            previous["positions"], state["marks"], previous["marks"], strict=True
        )
    )
    pnl += sum(
        (1 if f["side"] == "buy" else -1)
        * f["qty"]
        * (state["marks"][f["symbol"]] - f["price"])
        - sum(f["fees"].values())
        for f in fill_events
    )
    pnl += sum(
        e["dividend"] + (e["after"] - e["before"]) * state["marks"][e["symbol"]]
        for e in events
        if e["type"] == "corporate"
    )
    if any(e["type"] == "settled" for e in events):
        pnl += interest
        coverage["interest_credit"] += interest > 0
        coverage["interest_debit"] += interest < 0
    assert new_equity - old_equity == pnl + state["flows"] - previous["flows"]
    if not any(e["type"] in ("corporate", "settled") for e in events):
        for i, (old, new) in enumerate(
            zip(previous["positions"], positions, strict=True)
        ):
            expected = old["sellable"] + old["frozen"]
            for f in fill_events:
                if f["symbol"] == i:
                    if f["side"] == "sell":
                        expected -= f["qty"]
                    elif not state["config"]["instruments"][i]["t_plus_one"]:
                        expected += f["qty"]
            assert new["sellable"] + new["frozen"] == expected
    for symbol in {e["symbol"] for e in fill_events}:
        fs = [e for e in fill_events if e["symbol"] == symbol]
        q, old = state["quotes"][symbol], previous["quotes"][symbol]
        volume = q[16] if fs[0]["auction"] else (q[7] - old[7] if old else 0)
        assert (
            sum(f["qty"] for f in fs)
            <= volume * state["config"]["participation_ppm"] // 1_000_000
        )
        for side, price_index, depth_index in (("buy", 9, 10), ("sell", 11, 12)):
            side_fills = [f for f in fs if f["side"] == side]
            if not fs[0]["auction"] and q[price_index] >= 0:
                assert sum(f["qty"] for f in side_fills) <= q[depth_index]
            trade_price = q[3] if fs[0]["auction"] and q[17] == 1 else q[2]
            if (side == "buy" and q[13] > 0 and trade_price >= q[13]) or (
                side == "sell" and q[14] > 0 and trade_price <= q[14]
            ):
                if q[price_index] <= 0 or q[depth_index] == 0:
                    assert not side_fills
            for f in side_fills:
                order = state["orders"][f["id"] - state["order_offset"] - 1]
                assert f["quote_time"] > order["time"]
                assert (
                    f["price"] <= order["price"]
                    if side == "buy"
                    else f["price"] >= order["price"]
                )
                if not f["auction"]:
                    assert previous["quote_received"][symbol] >= order["time"]
                if side == "sell":
                    costs = state["config"]["instruments"][symbol]["costs"]
                    assert (
                        f["fees"]["tax"]
                        >= (f["qty"] * f["price"] * costs["sell_tax_ppm"]) // 1_000_000
                    )


def corpus(seed, config):
    rng = np.random.default_rng(seed)
    b, reference = (
        Broker(["A", "B", "C"], config=config),
        ReferenceBroker(["A", "B", "C"], config=config),
    )
    count = Counter()
    t = 0

    def apply(event=None, quotes=None):
        previous = json.loads(b.dump())
        if event is not None:
            b.apply(event)
            reference.apply(event)
        else:
            a = np.array(quotes, np.int64)
            b.feed(a)
            reference.feed(a)
        actual = json.loads(b.dump())
        assert actual == reference.state
        events = b.events()
        assert events == reference.events()
        assert b.account() == reference.account()
        assert_invariants(
            actual, previous, events, count, event.get("interest", 0) if event else 0
        )

    def event(event_type, **kwargs):
        nonlocal t
        t += 1
        apply(dict(type=event_type, time=t, **kwargs))

    def order(**kwargs):
        event(
            "order",
            **(
                dict(symbol=0, side="buy", qty=100, price=money(11), kind="limit")
                | kwargs
            ),
        )

    for day in range(1, 7):
        event("session", day=day, phase="preopen")
        order()
        event("session", day=day, phase="open_auction")
        order()
        t += 1
        apply(quotes=[quote(i, t) for i in range(3)])
        order(symbol=99)
        order(qty=1)
        order(price=100001)
        order(price=money(12))
        order(qty=100_000_000)
        order(side="sell", qty=100_000_000)
        order(qty=1000)
        t += 1
        apply(quotes=[quote(0, t, volume=10000, auction_volume=5000, auction_kind=1)])
        event("session", day=day, phase="continuous")
        volumes = [10000, 0, 0]
        for _ in range(140):
            choice = int(rng.integers(0, 10))
            symbol = int(rng.integers(0, 3))
            if choice <= 4:
                t += 1
                volumes[symbol] += int(rng.integers(0, 10000))
                price = int(rng.choice([9, 10, 11]))
                book = int(rng.integers(0, 4))
                ask = None if book == 0 else 0 if book == 1 else price
                apply(
                    quotes=[
                        quote(
                            symbol,
                            t,
                            price,
                            volumes[symbol],
                            ask=ask,
                            bid=ask,
                            ask_size=int(rng.integers(0, 800)),
                            bid_size=int(rng.integers(0, 800)),
                            halted=int(choice == 4),
                        )
                    ]
                )
            elif choice <= 7:
                side = str(rng.choice(["buy", "sell"]))
                order(
                    symbol=symbol,
                    side=side,
                    qty=int(rng.choice([100, 200, 500, 1000])),
                    price=money(11 if side == "buy" else 9),
                    kind=str(rng.choice(["market", "limit"])),
                )
            elif choice == 8:
                event(
                    "cancel",
                    id=int(rng.integers(0, len(json.loads(b.dump())["orders"]) + 2)),
                )
            else:
                event("transfer", amount=int(rng.choice([-10000, 10000, -(10**18)])))
        event("session", day=day, phase="close_auction")
        t += 1
        apply(
            quotes=[
                quote(
                    i, t, volume=volumes[i] + 10000, auction_volume=5000, auction_kind=2
                )
                for i in range(3)
            ]
        )
        event("session", day=day, phase="closed")
        event(
            "settle",
            day=day,
            actions=[
                dict(
                    symbol=0,
                    cash_per_share=100,
                    split_num=2,
                    split_den=1,
                    reference_price=None,
                )
            ],
            interest=int(rng.integers(-5000, 5000)),
        )
        event("settle", day=day, actions=[], interest=0)
        event("corporate", actions=[])
        apply(dict(type="transfer", time=0, amount=0))
    return count


def test_random_reference_and_properties():
    count = Counter()
    for seed in range(16):
        config = replace(
            MarketConfig(),
            t_plus_one=seed % 2 == 0,
            participation_ppm=[100000, 370000, 1000000][seed % 3],
            costs=Costs(250 + seed * 111, 50000, 500),
        )
        count.update(corpus(seed, config))
    expected = [
        "fill",
        "partial",
        "cancelled",
        "expired",
        "auction",
        "corporate",
        "fallback",
        "filled",
        "continuous",
        "cash_dividend",
        "split_applied",
        "interest_credit",
        "interest_debit",
    ]
    expected += [
        "reject:" + reason
        for reason in (
            "session",
            "unknown_symbol",
            "no_quote",
            "suspended",
            "quantity",
            "tick",
            "price_band",
            "cash",
            "sellable",
            "unknown_order",
            "time",
            "already_settled",
            "invalid_action",
        )
    ]
    assert all(count[k] > 0 for k in expected), count
    print("PAPER_COVERAGE=" + json.dumps(dict(sorted(count.items())), sort_keys=True))


def test_batch_equals_single_and_determinism():
    brokers = [start() for _ in range(3)]
    for b in brokers:
        for symbol in range(3):
            b.order(2, symbol, "buy", 1000, 11)
    rows = np.array([quote(i, 3 + i, volume=2000) for i in range(3)], np.int64)
    brokers[0].feed(rows)
    for row in rows:
        brokers[1].feed(row.reshape(1, -1))
    brokers[2].feed(rows)
    # The first post-order observations establish eligibility. The next batch
    # must actually fill, so equivalence cannot pass on an empty execution path.
    later = np.array([quote(i, 6 + i, volume=4000) for i in range(3)], np.int64)
    brokers[0].feed(later)
    for row in later:
        brokers[1].feed(row.reshape(1, -1))
    brokers[2].feed(later)
    assert brokers[0].dump() == brokers[1].dump() == brokers[2].dump()
    outputs = [b.events() for b in brokers]
    assert outputs[0] == outputs[1] == outputs[2]
    executions = [e for e in outputs[0] if e["type"] == "fill"]
    assert len(executions) == 3 and sum(e["qty"] for e in executions) == 600


def test_corporate_idempotence_and_auction_not_reused():
    b = start()
    b.order(2, "A", "buy", 1000, 11)
    b.session(3, 20261111, "open_auction")
    feed(b, quote(time=4, volume=1000, auction_volume=1000, auction_kind=1))
    b.session(5, 20261111, "continuous")
    assert sum(f["qty"] for f in fills(b)) == 100
    b.session(6, 20261111, "open_auction")
    b.session(7, 20261111, "continuous")
    assert not fills(b)
    b.session(8, 20261111, "closed")
    b.settle(9, 20261111)
    b.session(10, 20261112, "preopen")
    action = dict(
        symbol=0,
        cash_per_share=money(1),
        split_num=2,
        split_den=1,
        reference_price=None,
    )
    b.apply(dict(type="corporate", time=11, actions=[action]))
    before = b.account()
    b.apply(dict(type="corporate", time=12, actions=[action]))
    assert b.events()[-1]["reason"] == "invalid_action"
    assert b.account() == before
    assert before["positions"][0]["total"] == 200
    assert before["market_value"] == money(900)


def test_target_orders_and_minimum_commission_per_order():
    b = start()
    b.target_weights(2, {"A": 0.01})
    state = json.loads(b.dump())
    assert state["orders"][0]["qty"] == 1000
    observe_at_order(b, 2)
    feed(b, quote(time=3, volume=1000))
    assert fills(b)[0]["fees"]["commission"] == money(5)
    feed(b, quote(time=4, volume=2000))
    assert fills(b)[0]["fees"]["commission"] == 0
    before = b.dump()
    with pytest.raises(ValueError):
        b.target_weights(5, {"A": float("nan")})
    assert b.dump() == before
    b.target_weights(5, {"B": 0.02})
    state = json.loads(b.dump())
    assert state["orders"][0]["status"] == "cancelled"
    assert state["orders"][-1]["symbol"] == 1
