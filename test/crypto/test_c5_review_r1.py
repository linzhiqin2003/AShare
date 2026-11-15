"""C5 第一轮审查的四个反例；网络完全由合成 HTTP 响应替代。"""

import copy
import hashlib
import json

import httpx
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ashare.data import crypto
from ashare.data.crypto.archive import archive_key
from ashare.data.crypto.events import normalize_event
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.sync import sync
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame
from ashare.data.crypto.verify_raw import verify_raw
from test.crypto.conftest import FIXTURES
from test.crypto.test_pipeline import zip_rows
from test.crypto.test_recent import NAME, START
from test.crypto.test_recent import recent as recent


def field_hashes(frame):
    """浮点逐位、其他列含 null 与类型；基准由 HEAD 旧模块离线产出。"""
    return {
        col.name: hashlib.sha256(
            str(col.dtype).encode()
            + (
                col.to_numpy().tobytes()
                if col.dtype == pl.Float64
                else json.dumps(col.to_list(), ensure_ascii=False).encode()
            )
        ).hexdigest()
        for col in frame.sort(["time", "instrument"])
    }


def assert_legacy(frame, scenario):
    expected = json.loads((FIXTURES / "c5_strict_hashes.json").read_text())
    assert field_hashes(frame) == expected["scenarios"][scenario]


def ended_case(plan, data, kind="settlement"):
    other = "usdm:ENDEDUSDT"
    p = dict(plan["profiles"][0], instrument=other, symbol="ENDEDUSDT", coin="ENDED")
    plan["profiles"].append(p)
    data[other] = copy.deepcopy(data[NAME])
    for dataset in ("klines", "markPriceKlines", "indexPriceKlines"):
        data[other][dataset] = data[other][dataset][:20]
    data[other]["funding_prices"] = data[other]["funding_prices"][:3]
    data[NAME]["fundingRate"] += [
        dict(time=START + h * HOUR, interval=8, rate=0.0001, source="archive")
        for h in (24, 32, 40, 48)
    ]
    plan["events"] = [
        normalize_event(
            dict(
                event_id="synthetic-end",
                instrument=other,
                kind=kind,
                known_at=START,
                effective_at=START + 20 * HOUR,
                source_url="https://example.invalid/exit",
                original_text="synthetic exit",
                settlement_price=100,
            )
        )
    ]


@pytest.mark.parametrize("kind", ["settlement", "delist"])
@pytest.mark.parametrize("inferred", [False, True])
def test_ended_column_does_not_cap_panel(tmp_path, recent, inferred, kind):
    """来源 P1-1：20h 退出列不能截短另一列的 48h；独立 HEAD 摘要。"""
    plan, data = recent
    ended_case(plan, data, kind)
    ds = crypto.build(tmp_path, allow_inferred_funding=inferred)
    assert ds.meta["end"] == START + 48 * HOUR
    assert_legacy(ds.read(), "ended_" + kind)
    assert ds.read(instruments=[NAME])["funding"].sum() == pytest.approx(0.0006)
    assert crypto.verify(tmp_path, ds.version)["errors"] == []


@pytest.mark.parametrize("inferred", [False, True])
def test_verifier_rejects_ended_column_truncation(tmp_path, recent, inferred):
    """来源 P1-1：独立校验也必须发现退出列导致的错误共同终点。"""
    plan, data = recent
    ended_case(plan, data)
    ds = crypto.build(tmp_path, allow_inferred_funding=inferred)
    cut = ds.read().filter(pl.col("time") < START + 19 * HOUR)
    assert any(
        "terminal" in e["check"] for e in verify_frame(cut, ds.inputs())["errors"]
    )


@pytest.fixture
def archive_server(tmp_path, fixture):
    """六月末至七月末；先缺七月资金费文件，后发布，REST 可有价格修订。"""
    start, end = millis("2024-06-30"), millis("2024-08-01")
    p = dict(fixture["profiles"][0], listed_at=start)
    config = dict(start=start, end=end, instruments=[NAME], profiles={NAME: p})
    state = dict(published=False, missing_last=False, revised_price="100", calls=[])
    payloads = {}
    for label, hours in [("2024-06", range(24)), ("2024-07", range(24, 768))]:
        for dataset in ("klines", "markPriceKlines", "indexPriceKlines"):
            payloads[archive_key("usdm", dataset, p["symbol"], label)] = zip_rows(
                [
                    [
                        start + h * HOUR,
                        100,
                        102,
                        99,
                        101,
                        10,
                        start + (h + 1) * HOUR - 1,
                        1000,
                        1,
                        0,
                        500,
                        0,
                    ]
                    for h in hours
                ]
            )
        payloads[archive_key("usdm", "fundingRate", p["symbol"], label)] = zip_rows(
            [[start + h * HOUR, 8, 0.0001] for h in hours if h % 8 == 0]
        )
    july = archive_key("usdm", "fundingRate", p["symbol"], "2024-07")

    def handle(req):
        path = req.url.path
        if path.endswith("exchangeInfo"):
            # 显式历史档案，合成交易对不必在当前 exchangeInfo 中。
            return httpx.Response(200, json=dict(symbols=[]))
        if path.endswith("fundingInfo"):
            return httpx.Response(200, json=[])
        if path == "/data.binance.vision":
            prefix = req.url.params["prefix"]
            return httpx.Response(
                200,
                text=(
                    "<ListBucketResult><IsTruncated>false</IsTruncated>"
                    f"<CommonPrefixes><Prefix>{prefix}{p['symbol']}/</Prefix>"
                    "</CommonPrefixes></ListBucketResult>"
                ),
            )
        if path.endswith("/fundingRate"):
            lo, hi = (int(req.url.params[k]) for k in ("startTime", "endTime"))
            state["calls"].append((lo, hi))
            return httpx.Response(
                200,
                json=[
                    dict(
                        fundingTime=start + h * HOUR,
                        fundingRate="0.0001",
                        markPrice=state["revised_price"] if h == 56 else "100",
                    )
                    for h in range(0, 785, 8)
                    if lo <= start + h * HOUR <= hi
                ],
            )
        key = path.lstrip("/").removesuffix(".CHECKSUM")
        if key not in payloads or (key == july and not state["published"]):
            return httpx.Response(404)
        payload = payloads[key]
        if key == july and state["missing_last"]:
            payload = zip_rows(
                [[start + h * HOUR, 8, 0.0001] for h in range(24, 760, 8)]
            )
        if path.endswith(".CHECKSUM"):
            return httpx.Response(
                200,
                text=hashlib.sha256(payload).hexdigest()
                + "  "
                + key.rsplit("/", 1)[-1],
            )
        return httpx.Response(200, content=payload)

    def synchronize():
        with PublicClient(
            transport=httpx.MockTransport(handle), sleep=lambda _: None
        ) as client:
            sync(
                tmp_path,
                config=config,
                client=client,
                now=end + 9 * HOUR,
                announcements=False,
            )

    state.update(start=start, end=end, sync=synchronize, config=config)
    return state


def test_sync_refetches_prices_when_archive_published(tmp_path, archive_server):
    """来源 P1-2：走真实同步链路；新归档需要新 REST 价格快照，旧版可复现。"""
    server = archive_server
    server["sync"]()
    old = crypto.build(tmp_path, allow_inferred_funding=True)
    old_frame, old_inputs = old.read(), old.inputs()
    server.update(published=True, revised_price="120")
    server["calls"].clear()
    server["sync"]()
    changed = server["start"] + 56 * HOUR
    assert any(lo <= changed <= hi for lo, hi in server["calls"])
    new = crypto.build(tmp_path)
    row = new.read().filter(pl.col("time") == changed - HOUR).row(0, named=True)
    assert row["funding_price"] == 120
    assert row["funding"] == pytest.approx(0.00012)
    report = crypto.verify(tmp_path, new.version)
    assert report["errors"] == []
    diff = report["funding_reconciliations"][0]["differences"]
    assert len(diff) == 1 and diff[0]["time"] == changed
    assert diff[0]["inferred"]["price"] == 100 and diff[0]["archive"]["price"] == 120
    assert_frame_equal(
        crypto.open(tmp_path, old.version).read(), old_frame, check_exact=True
    )
    assert crypto.open(tmp_path, old.version).inputs() == old_inputs
    assert crypto.verify(tmp_path, old.version)["errors"] == []
    server["calls"].clear()
    server["sync"]()
    assert not any(lo <= changed <= hi for lo, hi in server["calls"])


def test_sync_archive_refetch_uses_full_declared_month(tmp_path, archive_server):
    """返工自检/P1-2：请求起点后移也不能用旧缓存审计该月更早的推断。"""
    server = archive_server
    server["sync"]()
    old = crypto.build(tmp_path, allow_inferred_funding=True)
    server.update(published=True, revised_price="120")
    server["config"]["start"] = server["start"] + 14 * 24 * HOUR
    server["calls"].clear()
    server["sync"]()
    changed = server["start"] + 56 * HOUR
    assert any(lo <= changed <= hi for lo, hi in server["calls"])
    strict = crypto.build(tmp_path)
    report = crypto.verify(tmp_path, strict.version)
    assert report["errors"] == []
    assert report["funding_reconciliations"][0]["inferred_version"] == old.version
    diff = report["funding_reconciliations"][0]["differences"]
    assert len(diff) == 1 and diff[0]["time"] == changed
    assert diff[0]["archive"]["price"] == 120


def test_reconciliation_audits_declared_month_tail(tmp_path, archive_server):
    """来源 P2-3：七月最后一笔缺失不能缩小审计范围；八月零点不属七月。"""
    server = archive_server
    server["sync"]()
    crypto.build(tmp_path, allow_inferred_funding=True)
    server.update(published=True, missing_last=True)
    server["sync"]()
    strict = crypto.build(tmp_path)
    assert strict.meta["end"] == server["start"] + 759 * HOUR
    report = crypto.verify(tmp_path, strict.version)
    assert report["errors"] == []
    assert len(report["funding_reconciliations"]) == 1
    differences = report["funding_reconciliations"][0]["differences"]
    assert len(differences) == 1
    assert differences[0]["time"] == server["start"] + 760 * HOUR
    assert differences[0]["archive"] is None
    assert crypto.build(tmp_path).version == strict.version


@pytest.mark.parametrize(
    "field,value",
    [
        ("markPrice", "200"),
        ("fundingRate", "0.0002"),
        ("fundingTime", 1),
    ],
)
def test_verify_rest_input_against_frozen_response(
    tmp_path, archive_server, field, value
):
    """来源 P2-4：规范化输入与面板一起改错，原始对象和摘要仍正确。"""
    server = archive_server
    server["sync"]()
    ds = crypto.build(tmp_path, allow_inferred_funding=True)
    inputs = ds.inputs()
    target = server["start"] + 56 * HOUR
    source = inputs["data"][NAME]
    raw = next(r for r in source["funding_prices"] if r["fundingTime"] == target)
    raw[field] = target + value if field == "fundingTime" else value
    if field == "fundingRate":
        next(r for r in source["fundingRate"] if r["time"] == target)["rate"] = float(
            value
        )
    frame, _ = build_frame(
        inputs["profiles"],
        inputs["data"],
        inputs["events"],
        ds.meta["start"],
        ds.meta["end"],
    )
    assert verify_frame(frame, inputs)["errors"] == []
    errors, _ = verify_raw(tmp_path, inputs)
    assert any(e["check"] == "funding_rest_decode" for e in errors)
