"""共用清单加入基金、可转债分区（含未完成、失败的）前后，股票侧与期货侧的输出逐位不变。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from futures_fixtures import FakeTushare
from test_fund_fixtures import FakeFundTushare
from test_futures_isolation import VOLATILE, stock_plan

import ashare
from ashare.data.fund_sync import sync_funds
from ashare.data.futures_sync import sync_futures
from ashare.data.panel import build
from ashare.data.panel_quality import required
from ashare.data.verify import verify

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def status(root):
    output = subprocess.run(
        [sys.executable, "-m", "ashare.data", "--root", str(root), "status"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    value = json.loads(output)
    funds = value.pop("funds", None)
    usage = value.pop("disk_usage")
    fund_usage = usage.pop("funds", None)
    usage.pop("manifest")  # 共用的清单文件字节数必然随条目增长
    return value, usage, funds, fund_usage


def snapshot(store, root, tmp_path, tag):
    report = verify(store, output=tmp_path / f"verify-{tag}")
    for key in VOLATILE:
        report.pop(key, None)
    value, usage, funds, fund_usage = status(root)
    return dict(
        plan=stock_plan(store, "20260101", "20260110"),
        required=required(store, ["20260102", "20260105", "20260106", "20260107"]),
        status=value,
        usage=usage,
        verify=json.dumps(report, sort_keys=True, ensure_ascii=False, default=str),
        version=store.version,
        futures_version=store.futures_version,
        panel=build(store)["version"],
    ), (funds, fund_usage)


def test_stock_and_futures_identical_with_fund_partitions(tables, make_store, tmp_path):
    store = make_store(tables)
    build(store)
    asyncio.run(sync_futures(FakeTushare(), store, "20260105", "20260213"))
    before, fund_before = snapshot(store, tmp_path / "root", tmp_path, "a")
    assert fund_before == (None, None)
    version_before = store.fund_version
    asyncio.run(sync_funds(FakeFundTushare(), store, "20260105", "20260213"))
    # 中断留下的 pending 分区、一个失败分区。
    store.mark(
        "fund_daily/20260216",
        dict(dataset="fund_daily", partition="20260216", params={}, status="pending"),
    )
    store.mark(
        "cb_daily/20260216",
        dict(
            dataset="cb_daily",
            partition="20260216",
            params={},
            status="failed",
            error="RuntimeError",
        ),
    )
    after, fund_after = snapshot(store, tmp_path / "root", tmp_path, "b")
    for key in before:
        assert before[key] == after[key], key
    # 基金部分确实写进了同一份清单、有自己的状态和版本。
    funds, fund_usage = fund_after
    assert funds["fund_daily"]["failed"] == 1 and funds["cb_daily"]["failed"] == 1
    assert fund_usage["files"] > 10
    assert store.fund_version != version_before
    assert len(store.entries("fund_daily")) > 20
