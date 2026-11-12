"""Synthetic regression cases with independently calculated execution results."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import paperbroker
import pytest
from paperbroker import Broker, Costs, Instrument, Lots, MarketConfig, money
from test_engine import feed, fills, observe_at_order, quote, start

assert (
    Path(paperbroker.__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[3])
)


def position_broker(config=None):
    b = start(config or MarketConfig(t_plus_one=False))
    assert b.order(2, "A", "buy", 1000, 11)["accepted"]
    observe_at_order(b, 2)
    feed(b, quote(time=3, volume=10_000))
    assert sum(f["qty"] for f in fills(b)) == 1000
    return b


@pytest.mark.parametrize(
    "phase,kind,end",
    [("open_auction", 1, "continuous"), ("close_auction", 2, "closed")],
)
@pytest.mark.parametrize("side,price", [("buy", 11), ("sell", 9)])
@pytest.mark.parametrize("book", [0, None, "priced_empty"])
def test_auction_limit_lock(phase, kind, end, side, price, book):
    b = position_broker()
    b.session(4, 20261111, phase)
    b.order(5, "A", side, 1000, price)
    row = quote(
        time=6,
        price=price,
        volume=20_000,
        ask=(price if book == "priced_empty" else book) if side == "buy" else price,
        bid=(price if book == "priced_empty" else book) if side == "sell" else price,
        ask_size=0 if side == "buy" else 1000,
        bid_size=0 if side == "sell" else 1000,
        auction_volume=10_000,
        auction_kind=kind,
    )
    feed(b, row)
    b.session(7, 20261111, end)
    assert fills(b) == []
    assert b.account()["positions"][0]["total"] == 1000


def test_open_auction_uses_open_not_last_and_t1():
    b = start()
    b.session(2, 20261111, "open_auction")
    b.order(3, "A", "buy", 1000, 11)
    q = quote(time=4, price=10.4, volume=50_000, auction_volume=2000, auction_kind=1)
    q[3] = money(10.2)
    feed(b, q)
    b.session(5, 20261111, "continuous")
    f = fills(b)
    assert [(e["price"], e["qty"]) for e in f] == [(money(10.2), 200)]
    assert b.account()["positions"][0]["sellable"] == 0
    assert b.order(6, "A", "sell", 100, 9)["events"][0]["reason"] == "sellable"


def test_receipt_clock_and_day_guard():
    def stamp(iso):
        return int(datetime.fromisoformat(iso).timestamp() * 1000)

    now = stamp("2026-11-12T09:30:00+08:00")
    b = Broker(["A"])
    b.session(now, 20261112, "continuous")
    b.feed(
        np.array([quote(time=now - 86400000, volume=50_000_000)], np.int64),
        received_at=now,
    )
    assert json.loads(b.dump())["quotes"] == [None]
    b.feed(np.array([quote(time=now + 2000, volume=1000)], np.int64), received_at=now)
    assert b.order(now + 1, "A", "buy", 100, 11)["accepted"]
    assert b.session(now + 2, 20261112, "closed")["accepted"]
    assert b.settle(now + 3, 20261112)["accepted"]
    assert json.loads(b.dump())["settled_day"] == 20261112


def test_volume_before_submission_is_excluded():
    b = start()
    b.order(1_000_000, "A", "buy", 40_000, 11)
    feed(b, quote(time=1_000_001, volume=5_000_000, ask_size=1_000_000))
    assert not fills(b)
    feed(b, quote(time=1_000_002, volume=5_001_000, ask_size=1_000_000))
    assert [f["qty"] for f in fills(b)] == [100]


def test_equal_time_decreasing_volume_and_nonmarketable_sell():
    b = position_broker()
    b.order(4, "A", "sell", 300, 10.5)
    observe_at_order(b, 4)
    feed(b, quote(time=5, volume=20_000, bid=10.4))
    assert not fills(b)
    feed(b, quote(time=6, volume=1, bid=10.6))
    assert not fills(b)
    assert json.loads(b.dump())["quotes"][0][7] == 20_000
    b.order(7, "A", "sell", 100, 9)
    feed(b, quote(time=7, volume=30_000, bid=10.6))
    fs = fills(b)
    assert all(f["id"] != 3 for f in fs)
    assert sum(f["qty"] for f in fs) == 300


def test_sell_slippage_tax_and_cancel_release():
    b = position_broker(MarketConfig(t_plus_one=False, slippage_ticks=2))
    b.order(4, "A", "sell", 100, 9)
    observe_at_order(b, 4)
    feed(b, quote(time=5, volume=20_000))
    fs = fills(b)
    assert fs[0]["price"] == money(9.98)
    assert fs[0]["fees"] == dict(commission=money(5), tax=money(0.499), other=0)
    before = b.account()["cash"]
    b.order(6, "B", "buy", 100, 11)
    assert b.account()["frozen"] == money(1105)
    b.cancel(7, 3)
    assert b.account()["cash"] == before and b.account()["frozen"] == 0


def test_sell_band_odd_lot_and_split():
    b = position_broker()
    b.session(4, 20261111, "closed")
    b.settle(
        5,
        20261111,
        [
            dict(
                symbol=0,
                cash_per_share=1000,
                split_num=103,
                split_den=100,
                reference_price=None,
            )
        ],
    )
    assert b.account()["positions"][0]["total"] == 1030
    b.session(6, 20261112, "continuous")
    feed(b, quote(time=7))
    assert b.order(8, "A", "sell", 100, 8.9)["events"][0]["reason"] == "price_band"
    assert b.order(9, "A", "sell", 150, 9)["events"][0]["reason"] == "quantity"
    assert b.order(10, "A", "sell", 1030, 9)["accepted"]
    observe_at_order(b, 10)
    feed(b, quote(time=11, volume=30_000, bid_size=2000))
    assert sum(f["qty"] for f in fills(b)) == 1030


def test_insufficient_cash_for_sell_fees_never_creates_debt():
    b = Broker(["A"], 20, MarketConfig(t_plus_one=False))
    b.session(0, 1, "continuous")
    feed(b, quote(time=1, price=0.14, ask=0.14, upper=0.15, lower=0.01))
    b.order(2, "A", "buy", 100, 0.15)
    observe_at_order(b, 2)
    feed(b, quote(time=3, price=0.14, ask=0.14, volume=1000, upper=0.15, lower=0.01))
    b.transfer(4, -0.99)
    b.order(5, "A", "sell", 100, 0.01)
    observe_at_order(b, 5)
    feed(b, quote(time=6, price=0.04, bid=0.04, volume=2000, upper=0.15, lower=0.01))
    assert b.account()["cash"] == money(0.01)
    assert b.account()["positions"][0]["total"] == 100
    assert json.loads(b.dump())["orders"][-1]["status"] == "unaffordable_fees"


def test_per_security_profiles_and_append_recovery(tmp_path):
    profiles = dict(
        STAR=Instrument(lots=Lots(1, 200), profile_id="star-v1"),
        BJ=Instrument(lots=Lots(1, 100), profile_id="bse-v1"),
        ETF=Instrument(tick=10, costs=Costs(0, 0, 0), profile_id="etf-v1"),
    )
    config = MarketConfig(instruments=profiles)
    with Broker(["STAR", "BJ"], config=config, journal=tmp_path) as b:
        b.session(0, 1, "continuous")
        feed(b, quote(0), quote(1))
        assert not b.order(2, "STAR", "buy", 199, 10)["accepted"]
        assert b.order(3, "STAR", "buy", 201, 10)["accepted"]
        assert b.order(4, "BJ", "buy", 101, 10)["accepted"]
        assert b.append_symbols(["ETF"])["accepted"]
        feed(b, quote(2, 5))
        assert b.order(6, "ETF", "buy", 100, 10.001)["accepted"]
        assert not b.order(7, "STAR", "buy", 200, 10.001)["accepted"]
        observe_at_order(b, 8, 2)
        feed(b, quote(2, 9, volume=2000))
        assert fills(b)[-1]["fees"] == dict(commission=0, tax=0, other=0)
        b.checkpoint()
        expected = b.dump()
        history = b.history
    with Broker(["BJ", "STAR", "ETF"], config=config, journal=tmp_path) as b:
        assert b.symbols == ["STAR", "BJ", "ETF"]
        assert b.dump() == expected
        assert b.events() == history
    assert json.loads(expected)["version"] == 3


def test_targets_proportional_and_rotation_converge():
    symbols = [str(i) for i in range(10)]
    b = Broker(symbols, config=MarketConfig(t_plus_one=False))
    b.session(0, 1, "continuous")
    b.feed(np.array([quote(i) for i in range(10)], np.int64), received_at=1)
    b.target_weights(1, dict.fromkeys(symbols, 0.1))
    orders = json.loads(b.dump())["orders"]
    assert max(o["qty"] for o in orders) - min(o["qty"] for o in orders) <= 100
    feed(b, *[quote(i, 2, volume=10_000_000, ask_size=100_000) for i in range(10)])
    assert b.account()["market_value"] / b.account()["equity"] > 0.98
    for t in range(3, 15, 2):
        b.target_weights(t, {"1": 0.99})
        feed(
            b,
            *[
                quote(i, t, volume=t * 10_000_000, ask_size=100_000, bid_size=100_000)
                for i in range(10)
            ],
        )
        feed(
            b,
            *[
                quote(
                    i,
                    t + 1,
                    volume=(t + 1) * 10_000_000,
                    ask_size=100_000,
                    bid_size=100_000,
                )
                for i in range(10)
            ],
        )
    assert (
        b.account()["positions"][1]["total"] * money(10) / b.account()["equity"] > 0.98
    )
    assert all(
        p["total"] == 0 for i, p in enumerate(b.account()["positions"]) if i != 1
    )


def test_request_results_and_finite_weights():
    b = start()
    rejected = b.order(2, "missing", "buy", 100, 10)
    accepted = b.order(3, "A", "buy", 100, 10)
    assert rejected["accepted"] is False and rejected["order_id"] is None
    assert accepted["accepted"] and accepted["order_id"] == 1
    assert rejected["request_id"] != accepted["request_id"]
    names = [str(i) for i in range(20)]
    Broker(names).target_weights(0, dict.fromkeys(names, 0.05))


def test_auction_quote_at_submission_time_is_not_future_liquidity():
    b = start()
    b.session(2, 20261111, "open_auction")
    b.order(3, "A", "buy", 1000, 11)
    feed(b, quote(time=3, volume=10000, auction_volume=10000, auction_kind=1))
    b.session(4, 20261111, "continuous")
    assert fills(b) == []


def test_native_buffered_outputs_retain_emission_day():
    b = Broker(["A"])
    for event in [
        dict(type="session", time=1, day=1, phase="continuous"),
        dict(type="session", time=2, day=1, phase="closed"),
        dict(type="settle", time=3, day=1, actions=[], interest=0),
        dict(type="session", time=4, day=2, phase="preopen"),
    ]:
        b._core.apply(json.dumps(event))
    events = json.loads(b._core.events())
    assert events[0]["day"] == 1
    assert next(e for e in events if e["type"] == "settled")["day"] == 1
    assert events[-1]["day"] == 2
    with pytest.raises(ValueError):
        Broker(["A"], config=MarketConfig(utc_offset_seconds=-(2**63)))
