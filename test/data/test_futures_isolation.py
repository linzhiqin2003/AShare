"""共用清单加入期货分区（含未完成、失败的）前后，股票侧的输出逐位不变。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
from futures_fixtures import FakeTushare

import ashare
from ashare.data.catalog import DAILY_DATASETS, DATASETS
from ashare.data.futures_sync import sync_futures
from ashare.data.panel import build
from ashare.data.panel_quality import required
from ashare.data.sources.base import FetchResult
from ashare.data.sync import plan
from ashare.data.sync_policy import should_refresh
from ashare.data.verify import verify

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

VOLATILE = {"generated_at", "elapsed_seconds"}


def stock_plan(store, start, end):
    manifest = store.manifest()
    days = sorted(
        store.read("trade_cal")
        .filter(
            (pl.col("is_open") == 1)
            & pl.col("cal_date").is_between(pl.lit(start), pl.lit(end))
        )["cal_date"]
        .to_list()
    )
    tasks = []
    for name in ["trade_cal", *DAILY_DATASETS]:
        spec = DATASETS[name]
        try:
            parts = list(plan(spec, start, end, days, store))
        except FileNotFoundError:
            tasks.append((name, "依赖未采集"))
            continue
        for partition, params in parts:
            key = f"{name}/{partition}"
            if store.complete(key, verify_hash=False) and not should_refresh(
                spec, partition, manifest.get(key, {}), days, end, now=None
            ):
                continue
            tasks.append((name, partition, json.dumps(params, sort_keys=True)))
    return tasks


def status(root):
    output = subprocess.run(
        [sys.executable, "-m", "ashare.data", "--root", str(root), "status"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    value = json.loads(output)
    value.pop("futures", None)
    usage = value.pop("disk_usage")
    usage.pop("futures", None)
    # 清单是共用的一个 SQLite 文件，它的字节数必然随期货条目增长；其余各项逐位比较。
    usage.pop("manifest")
    return value, usage


def snapshot(store, root, tmp_path, tag):
    report = verify(store, output=tmp_path / f"verify-{tag}")
    for key in VOLATILE:
        report.pop(key, None)
    return dict(
        plan=stock_plan(store, "20260101", "20260110"),
        required=required(store, ["20260102", "20260105", "20260106", "20260107"]),
        status=status(root),
        verify=json.dumps(report, sort_keys=True, ensure_ascii=False, default=str),
        version=store.version,
        panel=build(store)["version"],
    )


def test_stock_outputs_identical_with_futures_partitions(tables, make_store, tmp_path):
    store = make_store(tables)
    build(store)
    before = snapshot(store, tmp_path / "root", tmp_path, "a")
    futures_before = store.futures_version
    asyncio.run(sync_futures(FakeTushare(), store, "20260105", "20260213"))
    # 一个中断留下的 pending 分区、一个失败分区。
    store.mark(
        "futures_daily/20260216",
        dict(
            dataset="futures_daily", partition="20260216", params={}, status="pending"
        ),
    )
    store.mark(
        "futures_settle/20260216",
        dict(
            dataset="futures_settle",
            partition="20260216",
            params={},
            status="failed",
            error="RuntimeError",
        ),
    )
    after = snapshot(store, tmp_path / "root", tmp_path, "b")
    for key in before:
        assert before[key] == after[key], key
    assert store.futures_version != futures_before
    assert len(store.entries()) > len(store.entries("daily")) + 50


def test_futures_day_refresh_keeps_good_revision(tmp_path):
    """刷新返回空表或截断时拒绝覆盖已完成的期货日分区（复核 P1-6）。"""
    from ashare.data.store import Store

    store = Store(tmp_path / "r")
    asyncio.run(sync_futures(FakeTushare(), store, "20260105", "20260213"))
    key = "futures_daily/20260213"
    good = store.manifest()[key]
    assert good["rows"] > 0 and good["status"] == "complete"
    empty = FetchResult(pl.DataFrame(), "2026-02-15T00:00:00+00:00", 1)
    try:
        store.write("futures_daily", "20260213", {}, empty)
    except ValueError as exc:
        assert "保留旧修订" in str(exc)
    else:
        raise AssertionError("空表覆盖了完好的期货分区")
    assert store.manifest()[key]["file_hash"] == good["file_hash"]
    # 同步路径：源只对最后一天返回空表，旧修订保留并记 refresh_error。
    source = FakeTushare()
    source.tables["fut_daily"] = source.tables["fut_daily"].filter(
        pl.col("trade_date") != "20260213"
    )
    try:
        asyncio.run(sync_futures(source, store, "20260105", "20260213"))
    except ValueError:
        pass
    else:
        raise AssertionError("同步没有报告刷新失败")
    entry = store.manifest()[key]
    assert entry["file_hash"] == good["file_hash"] and entry["status"] == "complete"
    assert entry["refresh_error"] == "ValueError"
    assert len(store.read("futures_daily", "20260213", "20260213")) == good["rows"]


def test_futures_day_refresh_rejects_collapse(tmp_path):
    from ashare.data.store import Store

    store = Store(tmp_path / "r")
    spec_rows = 200
    table = pl.DataFrame(
        {"ts_code": [f"A{i}.SHF" for i in range(spec_rows)]}
    ).with_columns(pl.lit("20260105").alias("trade_date"))
    store.write("futures_daily", "20260105", {}, FetchResult(table, "t", 1))
    try:
        store.write(
            "futures_daily", "20260105", {}, FetchResult(table.head(100), "t", 1)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("行数骤降没有被拒绝")
    # 行数增加照常写入。
    more = pl.concat([table, table.with_columns(pl.lit("B").alias("ts_code"))])
    assert (
        store.write("futures_daily", "20260105", {}, FetchResult(more, "t", 1))["rows"]
        == 400
    )
