"""来源 C5：月文件遗漏整日、日归档校验和、禁止 REST K 线补尾。"""

import hashlib
import json

import httpx
import pytest

from ashare.data.crypto.archive import archive_key, download_series, read_series
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from test.crypto.test_pipeline import zip_rows


def test_daily_only_missing_days_and_no_rest(tmp_path):
    start = millis("2024-07-01")
    month = archive_key("usdm", "klines", "FAKEUSDT", "2024-07")
    daily = archive_key("usdm", "klines", "FAKEUSDT", "2024-07-02", frequency="daily")

    def rows(day):
        return [
            [
                start + day * DAY + h * HOUR,
                100,
                102,
                99,
                101,
                10,
                start + day * DAY + (h + 1) * HOUR - 1,
                1000,
                1,
                0,
                500,
                0,
            ]
            for h in range(24)
        ]

    payloads = {month: zip_rows(rows(0) + rows(2)), daily: zip_rows(rows(1))}
    calls = []

    def handle(req):
        key = req.url.path.lstrip("/")
        calls.append(key)
        assert "/fapi/" not in req.url.path
        base = key.removesuffix(".CHECKSUM")
        if base not in payloads:
            return httpx.Response(404)
        if key.endswith(".CHECKSUM"):
            return httpx.Response(
                200,
                text=hashlib.sha256(payloads[base]).hexdigest()
                + "  "
                + base.rsplit("/", 1)[-1],
            )
        return httpx.Response(200, content=payloads[base])

    store = RawStore(tmp_path)
    try:
        with PublicClient(
            transport=httpx.MockTransport(handle), sleep=lambda _: None
        ) as c:
            keys = download_series(
                c,
                store,
                "usdm",
                "klines",
                "FAKEUSDT",
                start,
                start + 4 * DAY,
                now=start + 5 * DAY,
            )
        assert set(keys) == {month, daily}
        assert len(read_series(store, keys, "usdm", "klines")) == 72
        assert not any("daily" in k and ("07-01" in k or "07-03" in k) for k in calls)
        assert any("07-04" in k for k in calls)
    finally:
        store.close()


def test_daily_bad_checksum_raises(tmp_path):
    start = millis("2024-07-01")
    key = archive_key("spot", "klines", "FAKEUSDT", "2024-07-01", frequency="daily")

    def handle(req):
        if "/monthly/" in req.url.path:
            return httpx.Response(404)
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text="0" * 64 + "  " + key.rsplit("/", 1)[-1])
        return httpx.Response(200, content=b"synthetic corrupt zip")

    store = RawStore(tmp_path)
    try:
        with PublicClient(
            transport=httpx.MockTransport(handle), sleep=lambda _: None
        ) as c:
            with pytest.raises(ValueError, match="CHECKSUM 不匹配"):
                download_series(
                    c,
                    store,
                    "spot",
                    "klines",
                    "FAKEUSDT",
                    start,
                    start + DAY,
                    now=start + 2 * DAY,
                )
    finally:
        store.close()


def test_funding_snapshot_exact_bytes_and_incremental(tmp_path):
    """来源 C5：原始快照字节不可重编码；资金费只请求旧覆盖之后。"""
    from ashare.data.crypto.sync import sync_funding_info, sync_funding_prices

    store = RawStore(tmp_path)
    raw = (
        b'[ { "symbol" : "FAKEUSDT", "fundingIntervalHours" : 8, "updateTime": 9 } ]\n'
    )
    calls = []

    def handle(req):
        calls.append(req)
        if req.url.path.endswith("fundingInfo"):
            return httpx.Response(200, content=raw)
        return httpx.Response(
            200,
            json=[dict(fundingTime=24 * HOUR, fundingRate="0.001", markPrice="100")],
        )

    try:
        with PublicClient(
            transport=httpx.MockTransport(handle), sleep=lambda _: None
        ) as c:
            info = sync_funding_info(c, store, 25 * HOUR)
            assert store.get(info["key"]) == raw
            assert info["fetched_at"] == 25 * HOUR
            assert info["sha256"] == hashlib.sha256(raw).hexdigest()
            previous = "funding_prices/FAKEUSDT/old"
            old = [dict(fundingTime=16 * HOUR, fundingRate="0.001", markPrice="100")]
            store.put(previous, json.dumps(old).encode(), source="rest")
            key = sync_funding_prices(
                c, store, "FAKEUSDT", 0, 25 * HOUR, previous=previous
            )
            assert int(calls[-1].url.params["startTime"]) == 16 * HOUR + 1
            assert json.loads(store.get(key)) == old + [
                dict(fundingTime=24 * HOUR, fundingRate="0.001", markPrice="100")
            ]
    finally:
        store.close()
