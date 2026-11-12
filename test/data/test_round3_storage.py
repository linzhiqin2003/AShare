"""读取、垃圾回收、测试隔离与人工确认的边界。"""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data as ad
from ashare.data.catalog import DATASETS
from ashare.data.maintenance import collect_garbage
from ashare.data.panel import build
from ashare.data.sources.base import FetchResult
from ashare.data.sources.tushare import TokenBucket, Tushare
from ashare.data.store import Store, data_root
from ashare.data.sync_policy import month_final, should_refresh

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_read_only_and_expired_reader(tables, make_store, monkeypatch, tmp_path):
    store = make_store(tables)
    build(store)
    names = {str(p.relative_to(store.root)) for p in store.root.rglob("*")}
    reader = ad.open(store.root.parent)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader._store._manifest.db.execute("CREATE TABLE forbidden_write (x)")
    old = reader.panel(["close"])
    # 读取和列投影不能调用数据根内任何写文件入口。
    import ashare.data.manifest as module

    monkeypatch.setattr(
        module.Manifest, "put", lambda *a, **k: pytest.fail("reader wrote manifest")
    )
    projected = reader.panel(["open"], symbols=["000003.SZ", "000001.SZ"])
    assert (
        isinstance(projected["open"], np.memmap)
        and not projected["open"].flags.writeable
    )
    assert {str(p.relative_to(store.root)) for p in store.root.rglob("*")} == names
    monkeypatch.undo()
    protected = list(store.root.rglob("*")) + [store.root]
    for path in protected:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        readonly = ad.open(store.root.parent)
        assert readonly.calendar().collect().height == 4
        assert readonly.table("dividend").collect().height == 1
        # 即使运行者有提权或 ACL 绕过 chmod，连接本身也必须拒绝写事务。
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly._store._manifest.db.execute("CREATE TABLE forbidden_write (x)")
    finally:
        for path in reversed(protected):
            path.chmod(0o755 if path.is_dir() else 0o644)
    build(store, end="20260106")
    assert old["close"][0, 0] == 10
    with pytest.raises(RuntimeError, match="重新.*open"):
        reader.panel(["open"])
    with pytest.raises(RuntimeError, match="重新.*open"):
        reader.calendar()
    # 所有祖先下的真实默认 data 路径都属于禁止目标，且检查不产生文件。
    repository = Path(ashare.__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="测试禁止"):
        data_root(repository / "data")
    assert data_root(tmp_path / "safe") == tmp_path / "safe"


def test_gc_does_not_remove_live_write(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    entered, release = threading.Event(), threading.Event()
    original = pl.DataFrame.write_parquet
    paths = []
    errors = []

    def paused(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        paths.append(Path(path))
        entered.set()
        assert release.wait(5)
        return original(self, path, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", paused)

    def write():
        try:
            store.write(
                "daily",
                "20260102",
                {},
                FetchResult(
                    pl.DataFrame(
                        dict(ts_code=["A"], trade_date=["20260102"], close=[1.0])
                    ),
                    "now",
                    1,
                ),
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    assert entered.wait(5)
    try:
        collect_garbage(store)
        assert paths[0].exists()
    finally:
        release.set()
        thread.join(5)
    assert not errors and store.complete("daily/20260102")


def test_suspect_confirmation_order_and_old_revision(tmp_path):
    def result(day, n, value=1):
        return FetchResult(
            pl.DataFrame(
                dict(
                    ts_code=[f"S{i}" for i in range(n)],
                    trade_date=[day] * n,
                    close=[value] * n,
                )
            ),
            "now",
            1,
        )

    for reverse in (False, True):
        store = Store(tmp_path / str(reverse))
        days = [("20260102", 100), ("20260105", 60)]
        for day, n in days[::-1] if reverse else days:
            store.write("daily", day, {}, result(day, n))
        store.validate_partitions(["daily"])
        assert store.manifest()["daily/20260105"]["status"] == "suspect"
        entry = store.confirm("daily/20260105", "人工核对：当日大面积停牌")
        assert entry["status"] == "complete" and store.complete("daily/20260105")
        store.validate_partitions(["daily"])
        assert store.complete("daily/20260105")
    store = Store(tmp_path / "revisions")
    first = store.write("daily", "20260102", {}, result("20260102", 1, 1))
    path = store.root / first["file"]
    before = path.read_bytes()
    store.write("daily", "20260102", {}, result("20260102", 1, 2))
    store.write("daily", "20260102", {}, result("20260102", 1, 1))
    assert path.read_bytes() == before and len(list(path.parent.glob("*.parquet"))) == 2


def test_zero_month_legacy_freeze_and_repeated_rate():
    entry = dict(
        params={"end_date": "20261031"}, fetched_at="2026-11-10T00:00:00Z", rows=0
    )
    assert not month_final(entry)
    entry["rows"] = 300
    assert month_final(entry)
    assert not should_refresh(
        DATASETS["index_member"], "801010.SI_N", entry, [], "20261111"
    )
    clock = [100.0]
    bucket = TokenBucket(300, clock=lambda: clock[0])
    bucket.slow_down()
    assert bucket.rate == 150
    clock[0] += 20
    bucket.slow_down()
    assert bucket.rate == 150
    clock[0] += 40
    bucket.slow_down()
    assert bucket.rate == 75


def test_rate_wait_does_not_hold_other_api_and_corrupt_ledger(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async with Tushare(
            concurrency=1,
            state_path=tmp_path / "account.json",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json={"code": 0, "data": {"fields": ["x"], "items": [[1]]}}
                )
            ),
        ) as source:
            source.state_path.write_text("{partial")

            async def slow():
                entered.set()
                await release.wait()

            source.bucket("slow").acquire = slow
            task = asyncio.create_task(source.fetch("slow", {}))
            await entered.wait()
            try:
                result = await source.fetch("fast", {}, timeout=0.5)
                assert len(result.table) == 1
            finally:
                release.set()
                await task
            assert set(json.loads(source.state_path.read_text())["interfaces"]) == {
                "slow",
                "fast",
            }

    asyncio.run(run())
