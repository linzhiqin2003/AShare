import asyncio
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest
from test_source import Clock, response
from test_store import result

import ashare
from ashare.data.catalog import DATASETS
from ashare.data.sources import SourceError, Tushare
from ashare.data.sources.base import FetchResult
from ashare.data.sources.tushare import TokenBucket, classify
from ashare.data.store import Store, atomic_json
from ashare.data.sync import sync
from ashare.data.sync_policy import month_final, should_refresh

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_migration_unchanged_content_and_partial_write(tmp_path, monkeypatch):
    root = tmp_path / "cn"
    root.mkdir()
    atomic_json(
        root / "manifest.json",
        {"daily/old": dict(dataset="daily", partition="old", status="failed")},
    )
    legacy = (root / "manifest.json").read_bytes()
    store = Store(tmp_path)
    assert store.manifest()["daily/old"]["status"] == "failed"
    first = store.write("daily", "20260102", {}, result(1))
    original_file_bytes = (store.root / first["file"]).read_bytes()
    again = store.write(
        "daily", "20260102", {}, FetchResult(result(1).table, "later", 1)
    )
    assert again["file"] == first["file"]
    assert (store.root / first["file"]).read_bytes() == original_file_bytes
    assert again["fetched_at"] == "later"
    assert len(list(root.rglob("*.parquet"))) == 1
    before = dict(store.manifest())

    def partial(self, path, **kwargs):
        Path(path).write_bytes(b"PAR1 partial interrupted write")
        raise RuntimeError("crash halfway")

    with monkeypatch.context() as ctx:
        ctx.setattr(pl.DataFrame, "write_parquet", partial)
        with pytest.raises(RuntimeError):
            store.write("daily", "20260102", {}, result(2))
    assert store.manifest() == before
    assert len(list(root.rglob("*.parquet"))) == 1
    assert store.complete("daily/20260102")
    assert (root / "manifest.json").read_bytes() == legacy
    store.export_manifest(root / "rollback.json")
    assert json.loads((root / "rollback.json").read_text()) == store.manifest()


def test_month_finality_report_window_and_refresh_zero():
    spec = DATASETS["index_weight"]
    entry = dict(
        params={"end_date": "20261027"}, fetched_at="2026-11-05T00:00:00Z", rows=300
    )
    assert should_refresh(spec, "000300.SH_202610", entry, [], "20261102")
    entry["params"]["end_date"] = "20261031"
    entry["fetched_at"] = "2026-11-01T00:00:00Z"
    assert not month_final(entry)
    entry["fetched_at"] = "2026-11-04T00:00:00Z"
    assert month_final(entry)
    assert not should_refresh(spec, "000300.SH_202610", entry, [], "20261105")
    assert not should_refresh(
        replace(DATASETS["daily"], refresh=0), "20260102", {}, ["20260102"], "20260102"
    )
    assert not should_refresh(
        DATASETS["income"], "20191231", {}, [], "20261102", now=date(2026, 11, 12)
    )
    assert should_refresh(
        DATASETS["income"], "20260930", {}, [], "20261102", now=date(2026, 11, 12)
    )


def test_refresh_failure_keeps_previous(make_store, tables):
    store = make_store(tables)
    old = store.manifest()["daily/20260102"]

    class Broken:
        async def fetch(self, api, params, **kwargs):
            if api == "trade_cal":
                return FetchResult(tables["trade_cal"], "now", 1)
            raise RuntimeError("network")

    with pytest.raises(RuntimeError):
        asyncio.run(sync(Broken(), store, "20260102", "20260102", ["daily"]))
    entry = store.manifest()["daily/20260102"]
    assert entry["status"] == "complete" and entry["file"] == old["file"]
    assert entry["refresh_error"] == "RuntimeError"


def test_rate_fixture_daily_quota_and_recovery(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
    real = json.loads((Path(__file__).parent / "fixtures/rate-40203.json").read_text())
    assert classify(real["code"], real["msg"]) == "rate_limit"
    assert classify(40203, "每天最多访问该接口20000次") == "daily_quota"
    clock = Clock()
    bucket = TokenBucket(300, clock=clock.time, sleep=clock.sleep)
    replies = [httpx.Response(200, json=real), response([[1]])]

    async def run():
        async with Tushare(
            transport=httpx.MockTransport(lambda _: replies.pop(0)),
            limiter=bucket,
            sleep=clock.sleep,
        ) as client:
            assert (await client.fetch("trade_cal", {})).retries == 1
        assert sum(clock.waits) >= 60 and bucket.rate == 150
        for _ in range(256):
            bucket.success()
        assert bucket.rate == 300
        async with Tushare(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, json={"code": 40203, "msg": "每天最多访问该接口20000次"}
                )
            ),
            limiter=bucket,
            sleep=clock.sleep,
        ) as client:
            with pytest.raises(SourceError, match="daily_quota"):
                await client.fetch("daily", {})
            assert client.calls == 1

    asyncio.run(run())


def test_budget_per_run_api_buckets_and_total_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
    state = tmp_path / "account.json"
    state.write_text(json.dumps(dict(calls=999999, rate=1, next_at=0)))

    async def run():
        clock = Clock()
        for _ in range(2):
            async with Tushare(
                state_path=state,
                budget=1,
                transport=httpx.MockTransport(lambda _: response([])),
                limiter=TokenBucket(clock=clock.time, sleep=clock.sleep),
                sleep=clock.sleep,
            ) as client:
                await client.fetch("a", {})
                with pytest.raises(SourceError, match="预算"):
                    await client.fetch("b", {})
        async with (
            Tushare(transport=httpx.MockTransport(lambda _: response([]))) as a,
            Tushare(transport=httpx.MockTransport(lambda _: response([]))) as b,
        ):
            assert a.bucket("a") is b.bucket("a")
            assert a.bucket("a") is not a.bucket("b")

        async def hanging(request):
            await asyncio.sleep(10)
            return response([])

        async with Tushare(transport=httpx.MockTransport(hanging)) as client:
            with pytest.raises(SourceError, match="timeout"):
                await client.fetch("daily", {}, timeout=0.01)

    asyncio.run(run())


def test_sync_concurrent_and_resume(make_store, tables):
    store = make_store(tables)
    active = maximum = calls = 0

    class Source:
        async def fetch(self, api, params, **kwargs):
            nonlocal active, maximum, calls
            active += 1
            maximum = max(maximum, active)
            calls += 1
            await asyncio.sleep(0.001)
            active -= 1
            if api == "trade_cal":
                return FetchResult(tables["trade_cal"], "now", 1)
            return FetchResult(
                tables[api].filter(pl.col("trade_date") == params["trade_date"]),
                "now",
                1,
            )

    asyncio.run(
        sync(
            Source(),
            store,
            "20260102",
            "20260107",
            ["daily", "adj_factor"],
            force=True,
            workers=4,
        )
    )
    assert maximum == 4
    calls = 0
    asyncio.run(
        sync(
            Source(),
            store,
            "20260102",
            "20260107",
            ["daily", "adj_factor"],
            refresh=False,
        )
    )
    assert calls == 0


def test_row_collapse_quarantines_new_partition(tmp_path):
    store = Store(tmp_path)
    rows = pl.DataFrame(
        dict(
            ts_code=[str(i) for i in range(200)],
            trade_date=["20260102"] * 200,
            close=[1.0] * 200,
        )
    )
    store.write("daily", "20260102", {}, FetchResult(rows, "now", 1))
    small = rows.head(100).with_columns(pl.lit("20260105").alias("trade_date"))
    entry = store.write("daily", "20260105", {}, FetchResult(small, "now", 1))
    assert entry["status"] == "suspect" and entry["quality_state"] == "row_collapse"
    assert not store.complete("daily/20260105")
