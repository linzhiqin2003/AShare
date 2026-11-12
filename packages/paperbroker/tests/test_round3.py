"""Independent regression cases for release overflow, cursors and retention."""

import hashlib
import json
from pathlib import Path

import numpy as np
import paperbroker
import paperbroker._core
import pytest
from paperbroker import Broker, Costs, Instrument, Lots, MarketConfig, money
from test_engine import quote, start

ROOT = Path(__file__).resolve().parents[3]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(paperbroker._core.__file__).resolve().is_relative_to(ROOT)


def test_receipt_backwards_without_journal_and_unsettled_rollover():
    b = start()
    b.feed(np.array([quote(time=20)], np.int64), received_at=10)
    before = b.dump()
    with pytest.raises(ValueError, match="receipt time"):
        b.feed(np.array([quote(time=21)], np.int64), received_at=9)
    assert b.dump() == before
    result = b.session(11, 20261112, "preopen")
    assert not result["accepted"]
    assert result["events"][0]["reason"] == "session"
    assert json.loads(b.dump())["day"] == 20261111


def huge_broker(directory=None, dividends=8):
    cfg = MarketConfig(
        tick=1,
        participation_ppm=1_000_000,
        costs=Costs(0, 0, 0),
        lots=Lots(1, 0),
        enforce_bands=False,
    )
    b = Broker(["A"], 10**14, cfg, journal=directory)

    def q(t, v, price=1):
        return np.array(
            [
                [
                    0,
                    t,
                    price,
                    price,
                    price,
                    price,
                    price,
                    v,
                    0,
                    price,
                    10**12,
                    price,
                    10**12,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            np.int64,
        )

    b.session(0, 1, "continuous")
    b.feed(q(1, 0), received_at=1)
    b.order(2, "A", "buy", 10**9, 0.0001)
    b.feed(q(3, 0), received_at=3)
    b.feed(q(4, 10**11), received_at=4)
    assert b.account()["positions"][0]["total"] == 10**9
    b.session(6, 1, "closed")
    action = dict(
        symbol=0, cash_per_share=10**9, split_num=1, split_den=1, reference_price=None
    )
    for day in range(1, dividends + 1):
        if day > 1:
            b.session(day * 10, day, "closed")
        assert b.settle(day * 10 + 1, day, [action])["accepted"]
    return b, cfg, q, action


def test_release_extension_dividend_overflow_is_atomic():
    b, _, _, action = huge_broker()
    b.session(90, 9, "closed")
    before = b.dump()
    with pytest.raises(OverflowError, match="accounting overflow"):
        b.settle(91, 9, [action])
    assert b.dump() == before
    assert b.account()["cash"] == 8_999_999_999_000_000_000
    print(
        "OVERFLOW_NATIVE_SHA256="
        + hashlib.sha256(Path(paperbroker._core.__file__).read_bytes()).hexdigest()
    )


def test_feed_overflow_preflight_does_not_poison_wal(tmp_path):
    b, cfg, q, _ = huge_broker(tmp_path)
    b.session(90, 9, "continuous")
    b.feed(q(91, 0, 10**9), received_at=91)
    assert b.order(92, "A", "sell", 10**9, 100000)["accepted"]
    b.feed(q(93, 0, 10**9), received_at=93)
    before = b.dump()
    wal = (tmp_path / "events.jsonl").read_bytes()
    files = set((tmp_path / "quotes").iterdir())
    with pytest.raises(OverflowError, match="accounting overflow"):
        b.feed(q(94, 10**12, 10**9), received_at=94)
    assert b.dump() == before
    assert (tmp_path / "events.jsonl").read_bytes() == wal
    assert set((tmp_path / "quotes").iterdir()) == files
    b.close()
    with Broker(["A"], 10**14, cfg, journal=tmp_path) as restored:
        assert restored.dump() == before


@pytest.mark.parametrize(
    "qty,accepted",
    [
        (50, True),
        (150, True),
        (250, True),
        (30, False),
        (130, False),
        (100, True),
        (200, True),
    ],
)
def test_mainboard_residual_lot(qty, accepted):
    b = start(MarketConfig(t_plus_one=False, participation_ppm=1_000_000))
    b.order(2, "A", "buy", 300, 11)
    b.feed(np.array([quote(time=3, volume=0)], np.int64), received_at=2)
    b.feed(np.array([quote(time=4, volume=250)], np.int64), received_at=4)
    b.cancel(5, 1)
    assert b.account()["positions"][0]["total"] == 250
    assert b.order(6, "A", "sell", qty, 9)["accepted"] is accepted


@pytest.mark.parametrize("minimum", [100, 200])
def test_step_one_residual_and_target_reconciliation(minimum):
    cfg = MarketConfig(
        t_plus_one=False,
        participation_ppm=1_000_000,
        instruments={"A": Instrument(t_plus_one=False, lots=Lots(1, minimum))},
    )
    b = Broker(["A"], config=cfg)
    b.session(0, 1, "continuous")
    b.feed(np.array([quote(time=1)], np.int64), received_at=1)
    b.order(2, "A", "buy", minimum + 51, 11)
    b.feed(np.array([quote(time=3, volume=0)], np.int64), received_at=2)
    b.feed(np.array([quote(time=4, volume=1000)], np.int64), received_at=4)
    assert not b.order(5, "A", "sell", minimum - 1, 9)["accepted"]
    assert b.order(6, "A", "sell", minimum, 9)["accepted"]
    assert b.order(7, "A", "sell", 51, 9)["accepted"]


def test_target_sells_residual_without_dropping_it():
    b = start(MarketConfig(t_plus_one=False, participation_ppm=1_000_000))
    b.order(2, "A", "buy", 300, 11)
    b.feed(np.array([quote(time=3, volume=0)], np.int64), received_at=2)
    b.feed(np.array([quote(time=4, volume=250)], np.int64), received_at=4)
    b.cancel(5, 1)
    equity = b.account()["equity"]
    result = b.target_weights(6, {"A": 100 * money(10) / equity})
    accepted = [r for r in result if r["order_id"]]
    assert len(accepted) == 1
    assert json.loads(b.dump())["orders"][-1]["qty"] == 150


def test_bounded_snapshots_archived_events_and_gc(tmp_path):
    b = Broker(["A"], journal=tmp_path, snapshot_retention=3, quote_retention_days=2)
    snapshots = []
    all_events = []
    for day in range(1, 7):
        b.session(day * 10, day, "continuous")
        b.feed(
            np.array([quote(time=day * 10 + 1, volume=1000)], np.int64),
            received_at=day * 10 + 1,
        )
        b.order(day * 10 + 2, "A", "buy", 100, 11)
        b.session(day * 10 + 3, day, "closed")
        b.settle(day * 10 + 4, day)
        all_events.extend(b.events())
        b.checkpoint()
        snapshots.append(max(p.stat().st_size for p in tmp_path.glob("state-*.json")))
        assert len(list(tmp_path.glob("state-*.json"))) <= 3
        assert len(b.daily_states) == 1
        assert all(e["day"] == day for e in b.history)
        assert len(json.loads(b.dump())["orders"]) == 1
    assert max(snapshots) < min(snapshots) * 1.15
    result = b.journal.gc()
    assert result["quotes_deleted"] == 4
    assert len(list((tmp_path / "quotes").iterdir())) == 2
    assert [e["seq"] for e in all_events] == list(range(1, len(all_events) + 1))
    assert b.events(after=0) == all_events
    cursor = all_events[-8]["seq"]
    assert b.events(after=cursor) == all_events[-7:]
    expected = b.dump()
    b.close()
    with Broker(["A"], journal=tmp_path) as restored:
        assert restored.dump() == expected
        assert restored.events(after=cursor) == all_events[-7:]
    # A corrupt newest cache can still fall back to the other retained anchor.
    latest = sorted(tmp_path.glob("state-*.json"))[-1]
    latest.write_bytes(b"broken")
    with Broker(["A"], journal=tmp_path) as restored:
        assert restored.dump() == expected


def test_historical_action_idempotence_and_consecutive_dates():
    b = start(MarketConfig(t_plus_one=False, participation_ppm=1_000_000))
    assert b.order(2, "A", "buy", 100, 11)["accepted"]
    b.feed(np.array([quote(time=3, volume=0)], np.int64), received_at=2)
    b.feed(np.array([quote(time=4, volume=10000)], np.int64), received_at=4)
    b.session(5, 20261111, "closed")
    b.settle(6, 20261111)
    b.session(7, 20261113, "preopen")
    action = dict(
        symbol=0, cash_per_share=0, split_num=2, split_den=1, reference_price=None
    )
    event = dict(type="corporate", time=8, day=20261112, actions=[action])
    assert b.apply(event)["accepted"]
    assert b.account()["positions"][0]["total"] == 200
    repeated = b.apply(event | dict(time=9))
    assert not repeated["accepted"]
    assert repeated["events"][0]["reason"] == "invalid_action"
    assert b.account()["positions"][0]["total"] == 200
    assert b.apply(event | dict(time=10, day=20261113))["accepted"]
    assert b.account()["positions"][0]["total"] == 400
