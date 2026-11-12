"""Measure a bounded account over 40 days and a full-market day's actual WAL."""

import ctypes
import gc
import json
import shutil
import time
from pathlib import Path

import paperbroker
from paperbroker import Broker

from ashare.paper.storage import ParquetStore, disk_usage
from bench.paper.benchmark import quotes

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)


def rss_bytes():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    buffer = (ctypes.c_uint32 * 12)()
    count = ctypes.c_uint32(12)
    task = ctypes.c_uint32.in_dll(lib, "mach_task_self_").value
    result = lib.task_info(task, 20, buffer, ctypes.byref(count))
    if result != 0 or count.value != 12:
        raise RuntimeError(f"mach_task_basic_info failed: {result}, {count.value}")
    return int.from_bytes(bytes(buffer)[8:16], "little")


def run():
    directory = ROOT / ".cache/round3/retention-account"
    if directory.exists():
        shutil.rmtree(directory)
    names = [f"S{i:05d}" for i in range(5500)]
    store = ParquetStore(directory / "quotes")
    b = Broker(names, 1_000_000_000, journal=directory, quote_store=store)
    records = []
    t = 0
    for day in range(1, 41):
        t += 10
        b.session(t, day, "continuous")
        a = quotes(5500)
        for k in range(4):
            t += 3
            a[:, 1] = t
            a[:, 7] = (k + 1) * 10**6
            store.received_at = store.sent_at = t
            store.source = "synthetic_retention"
            b.feed(a, received_at=t)
            if k == 0:
                for i in range(200):
                    t += 1
                    assert b.order(t, i, "buy", 100, 10)["accepted"]
        t += 5
        b.session(t, day, "closed")
        t += 1
        b.settle(t, day)
        assert sum(e["qty"] for e in b.history if e["type"] == "fill") == 20000
        b.journal.gc()
        gc.collect()
        snapshots = sorted(directory.glob("state-*.json"))
        record = dict(
            day=day,
            snapshot_bytes=snapshots[-1].stat().st_size,
            snapshot_count=len(snapshots),
            quote_files=len(list((directory / "quotes").iterdir())),
            history_events=len(b.history),
            daily_states=len(b.daily_states),
            rss_bytes=rss_bytes(),
            disk_bytes=disk_usage(directory),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    expected = b.dump()
    b.close()
    with Broker(
        names, 1_000_000_000, journal=directory, quote_store=store
    ) as recovered:
        assert recovered.dump() == expected
    assert all(x["snapshot_count"] <= 3 and x["daily_states"] == 1 for x in records)
    assert all(x["quote_files"] == 80 for x in records[19:])
    assert (
        max(x["snapshot_bytes"] for x in records) < records[0]["snapshot_bytes"] * 1.05
    )
    # Genuine daily reports and immutable WAL accumulate by design; caches must plateau.
    assert (
        records[-1]["disk_bytes"]["snapshots"]
        < records[19]["disk_bytes"]["snapshots"] * 1.05
    )
    assert (
        records[-1]["disk_bytes"]["quotes"] < records[19]["disk_bytes"]["quotes"] * 1.05
    )
    return records


def full_day():
    directory = ROOT / ".cache/round3/full-day-wal"
    if directory.exists():
        shutil.rmtree(directory)
    names = [f"S{i:05d}" for i in range(5500)]
    store = ParquetStore(directory / "quotes")
    started = time.perf_counter()
    with Broker(names, 1_000_000_000, journal=directory, quote_store=store) as b:
        b.session(0, 1, "continuous")
        a = quotes(5500)
        for batch in range(2400):
            t = (batch + 1) * 6000
            a[:, 1] = t
            a[:, 7] = (batch + 1) * 10000
            store.received_at = store.sent_at = t
            store.source = "synthetic_full_market"
            b.feed(a, received_at=t)
            if batch % 12 == 0:
                assert b.order(t, batch // 12, "buy", 100, 10)["accepted"]
            if b.journal.sequence % 100 == 0:
                b.checkpoint()
            if batch % 400 == 0:
                print(f"full day batch {batch}/2400", flush=True)
        b.session(t + 1, 1, "closed")
        b.settle(t + 2, 1)
        b.journal.gc()
        assert sum(e["type"] == "fill" for e in b.history) == 200
        entries = b.journal.sequence
    return dict(
        symbols=5500,
        batches=2400,
        interval_seconds=6,
        orders=200,
        wal_entries=entries,
        elapsed_seconds=time.perf_counter() - started,
        disk_bytes=disk_usage(directory),
    )


def main():
    result = dict(retention_40_days=run(), full_day_persistence=full_day())
    path = ROOT / "bench/paper/benchmark-results.json"
    baseline = json.loads(path.read_bytes()) if path.exists() else {}
    path.write_text(json.dumps(baseline | result, indent=2) + "\n")
    print(json.dumps(result["full_day_persistence"], indent=2))


if __name__ == "__main__":
    main()
