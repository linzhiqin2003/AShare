import asyncio
from pathlib import Path

import polars as pl
import pytest

import ashare
from ashare.data.catalog import DATASETS
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.data.sync import sync

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def result(value):
    return FetchResult(
        pl.DataFrame(
            dict(ts_code=["A"], trade_date=["20260102"], close=[float(value)])
        ),
        "now",
        1,
    )


def test_atomic_and_hashes(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.write("daily", "20260102", {}, result(1))
    assert store.complete("daily/20260102")
    before = store.manifest_path.read_bytes()

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    with monkeypatch.context() as ctx:
        ctx.setattr(pl.DataFrame, "write_parquet", crash)
        with pytest.raises(RuntimeError):
            store.write("daily", "20260102", {}, result(2))
    assert store.manifest_path.read_bytes() == before
    assert not list(store.root.rglob("*.tmp"))
    entry = store.entries("daily")["daily/20260102"]
    (store.root / entry["file"]).write_bytes(b"bad")
    assert not store.complete("daily/20260102")


def test_resume_failed_force_and_recent(tmp_path):
    store = Store(tmp_path)
    calls = []

    class Source:
        broken = True

        async def fetch(self, api, params, **kwargs):
            calls.append((api, params.copy()))
            if api == "trade_cal":
                year = params["start_date"][:4]
                days = [f"{year}010{i}" for i in range(1, 9)]
                return FetchResult(
                    pl.DataFrame(dict(cal_date=days, is_open=[1] * 8)), "now", 1
                )
            if params["trade_date"] == "20260102" and self.broken:
                raise RuntimeError("crash")
            return result(1)

    source = Source()
    with pytest.raises(RuntimeError):
        asyncio.run(sync(source, store, "20260101", "20260108", ["daily"]))
    assert store.manifest()["daily/20260102"]["status"] == "failed"
    calls.clear()
    source.broken = False
    asyncio.run(sync(source, store, "20260101", "20260108", ["daily"]))
    days = [p["trade_date"] for api, p in calls if api == "daily"]
    assert "20260101" not in days
    assert "20260102" in days
    calls.clear()
    asyncio.run(sync(source, store, "20260101", "20260108", ["daily"]))
    assert sorted(p["trade_date"] for api, p in calls if api == "daily") == [
        "20260104",
        "20260105",
        "20260106",
        "20260107",
        "20260108",
    ]
    calls.clear()
    asyncio.run(sync(source, store, "20260101", "20260108", ["daily"], force=True))
    assert len([api for api, p in calls if api == "daily"]) == 8


def test_catalog_financial_and_secret(tmp_path, monkeypatch):
    store = Store(tmp_path)
    for name in (
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "forecast",
        "express",
    ):
        assert DATASETS[name].partition == "report"
        assert DATASETS[name].api.endswith("_vip")
    monkeypatch.setenv("TUSHARE_TOKEN", "SENSITIVE_FIXTURE_SENTINEL")
    with pytest.raises(ValueError, match="鉴权"):
        store.write(
            "daily",
            "bad",
            {},
            FetchResult(
                pl.DataFrame({"ts_code": ["SENSITIVE_FIXTURE_SENTINEL"]}), "now", 1
            ),
        )


def test_snapshot_preserves_previous_revision(tmp_path):
    store = Store(tmp_path)
    first = result(1)
    store.write("daily", "20260102", {}, first)
    snapshot = store.snapshot()
    second = FetchResult(result(2).table, "later", 1)
    store.write("daily", "20260102", {}, second)
    assert store.read("daily")["close"][0] == 2
    assert snapshot.read("daily")["close"][0] == 1
    assert snapshot.version != store.version


def test_crash_between_file_and_manifest(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.mark(
        "daily/20260102",
        dict(dataset="daily", partition="20260102", params={}, status="pending"),
    )
    with monkeypatch.context() as context:
        context.setattr(
            store,
            "mark",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        with pytest.raises(RuntimeError):
            store.write("daily", "20260102", {}, result(1))
    assert not store.complete("daily/20260102")
    store.write("daily", "20260102", {}, result(1))
    assert store.complete("daily/20260102")
