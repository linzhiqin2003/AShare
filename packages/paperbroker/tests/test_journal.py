import json
import shutil
from pathlib import Path

import numpy as np
import paperbroker
import pytest
from paperbroker import Broker
from paperbroker._core import BrokerCore
from paperbroker.journal import NumpyQuoteStore
from test_engine import feed, quote

assert (
    Path(paperbroker.__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[3])
)


def recorded(directory, snapshots=True):
    b = Broker(["A"], journal=directory)
    b.session(0, 1, "continuous")
    feed(b, quote())
    for i in range(2, 42, 2):
        b.order(i, "A", "buy", 100, 11)
        feed(b, quote(time=i + 1, volume=i * 1000))
        if snapshots and i % 10 == 0:
            b.checkpoint()
    b.session(50, 1, "closed")
    b.settle(51, 1)
    state = b.dump()
    b.close()
    return state


def test_recovery_snapshot_replay_and_random_truncation(tmp_path):
    directory = tmp_path / "original"
    expected = recorded(directory)
    with Broker(["A"], journal=directory) as b:
        assert b.dump() == expected
    raw = (directory / "events.jsonl").read_bytes()
    offsets = np.random.default_rng(120).integers(0, len(raw) + 1, 60).tolist()
    offsets += [0, 1, len(raw), len(raw) - 1] + [
        i + 1 for i, char in enumerate(raw) if char == 10
    ]
    store = NumpyQuoteStore(directory / "quotes")
    for case, offset in enumerate(offsets):
        copy = tmp_path / f"cut-{case}"
        shutil.copytree(directory, copy)
        (copy / "events.jsonl").write_bytes(raw[:offset])
        reference = Broker(["A"])
        complete = raw[:offset].rsplit(b"\n", 1)[0] if b"\n" in raw[:offset] else b""
        for line in complete.splitlines():
            body = json.loads(line)["body"]
            if body["kind"] == "quote":
                reference.feed(store.get(body["reference"]))
            else:
                reference.apply(json.loads(body["event"]))
        with Broker(["A"], journal=copy) as b:
            assert (copy / "events.jsonl").read_bytes() == raw[:offset][
                : raw[:offset].rfind(b"\n") + 1
            ]
            assert b.dump() == reference.dump(), offset
            b.transfer(1000, 1)
        tail = (copy / "events.jsonl").read_bytes()
        assert tail.endswith(b"\n")
        for line in tail.splitlines():
            json.loads(line)
    for path in directory.glob("state-*.json"):
        path.unlink()
    with Broker(["A"], journal=directory) as b:
        assert b.dump() == expected


def test_corruption_missing_quotes_and_writer_lock(tmp_path):
    recorded(tmp_path)
    with Broker(["A"], journal=tmp_path):
        with pytest.raises(BlockingIOError):
            Broker(["A"], journal=tmp_path)
    with Broker(["A", "B"], journal=tmp_path) as extended:
        assert extended.symbols == ["A", "B"]
    with pytest.raises(ValueError, match="configuration"):
        Broker(["A"], initial_cash=1, journal=tmp_path)
    for path in tmp_path.glob("state-*.json"):
        path.unlink()
    q = next((tmp_path / "quotes").glob("*.npz"))
    q.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        Broker(["A"], journal=tmp_path)


def test_wal_precedes_execution_and_invalid_input_not_logged(tmp_path, monkeypatch):
    b = Broker(["A"], journal=tmp_path)
    with pytest.raises(ValueError):
        b.apply(dict(type="nonsense"))
    assert not (tmp_path / "events.jsonl").read_bytes()
    original = b._core

    class Crash:
        def dump(self):
            return original.dump()

        def apply(self, event):
            raise RuntimeError("simulated process crash")

    b._core = Crash()
    with pytest.raises(RuntimeError):
        b.session(1, 1, "continuous")
    b.close()
    with Broker(["A"], journal=tmp_path) as recovered:
        expected = Broker(["A"])
        expected.session(1, 1, "continuous")
        assert recovered.dump() == expected.dump()
        clone = BrokerCore.restore(recovered.dump().decode())
        assert clone.dump().encode() == recovered.dump()


def test_corrupt_snapshot_falls_back_and_append_checks_newline(tmp_path):
    expected = recorded(tmp_path)
    newest = sorted(tmp_path.glob("state-*.json"))[-1]
    newest.write_bytes(b'{"truncated":')
    with Broker(["A"], journal=tmp_path) as b:
        assert b.dump() == expected
        with (tmp_path / "events.jsonl").open("ab") as writer:
            writer.write(b"incomplete")
        with pytest.raises(ValueError, match="newline"):
            b.transfer(100, 1)
    with Broker(["A"], journal=tmp_path) as b:
        assert b.dump() == expected


def test_middle_line_tamper_and_delete_fail_closed(tmp_path):
    recorded(tmp_path, snapshots=False)
    log = tmp_path / "events.jsonl"
    lines = log.read_bytes().splitlines(keepends=True)
    original = b"".join(lines)
    changed = json.loads(lines[2])
    event = json.loads(changed["body"]["event"])
    event["qty"] = 900
    changed["body"]["event"] = json.dumps(event)
    from paperbroker.journal import canonical

    log.write_bytes(
        b"".join(lines[:2]) + canonical(changed) + b"\n" + b"".join(lines[3:])
    )
    with pytest.raises(ValueError, match="checksum"):
        Broker(["A"], journal=tmp_path)
    log.write_bytes(b"".join(lines[:2] + lines[3:]))
    with pytest.raises(ValueError, match="checksum"):
        Broker(["A"], journal=tmp_path)
    log.write_bytes(original)


def test_snapshot_future_head_cannot_cross_truncation(tmp_path):
    expected = recorded(tmp_path)
    log = tmp_path / "events.jsonl"
    lines = log.read_bytes().splitlines(keepends=True)
    # A valid snapshot hash is insufficient: its history must still be in the WAL.
    log.write_bytes(b"".join(lines[:3]))
    with Broker(["A"], journal=tmp_path) as b:
        assert b.dump() != expected
        assert b.account()["positions"][0]["total"] == 0
        assert json.loads(b.dump())["orders"][0]["remaining"] == 100


def test_outputs_restore_before_and_after_checkpoint(tmp_path):
    recorded(tmp_path)
    with Broker(["A"], journal=tmp_path) as b:
        expected = b.events()
        assert any(e["type"] == "fill" for e in expected)
        assert any(e["type"] == "settled" for e in expected)
        b.checkpoint()
        daily = b.daily_states
    with Broker(["A"], journal=tmp_path) as b:
        assert b.events() == expected
        assert b.daily_states == daily


def test_fullfsync_is_used_and_errors_are_not_suppressed(tmp_path, monkeypatch):
    import fcntl

    import paperbroker.journal as journal

    monkeypatch.setattr(journal.sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, op: calls.append(op))
    journal.durable_write(tmp_path / "durable", b"data")
    assert calls == [getattr(fcntl, "F_FULLFSYNC", 51)]

    def fail(fd, op):
        raise OSError("unsupported device")

    monkeypatch.setattr(fcntl, "fcntl", fail)
    with pytest.raises(OSError, match="unsupported"):
        journal.durable_write(tmp_path / "durable", b"new")
    assert (tmp_path / "durable").read_bytes() == b"data"
