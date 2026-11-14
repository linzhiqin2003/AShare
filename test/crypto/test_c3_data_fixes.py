"""来源：C3 任务书，CMS 第 61 页限流与真实归档的停机/时间戳形态。"""

import copy

import httpx
import polars as pl
import pytest

from ashare.data.crypto.announcements import crawl_announcements
from ashare.data.crypto.formats import parse_klines
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.sources import CMS, PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame


def kline(t, close_t, scale=1):
    return f"{t * scale},100,102,99,101,10,{close_t * scale},1000,10,5,500,0\n".encode()


@pytest.mark.parametrize("market,scale", [("spot", 1), ("spot", 1000), ("usdm", 1)])
@pytest.mark.parametrize(
    "offset,flag",
    [
        (-999, None),
        (-1000, None),
        (-1001, "partial_halt"),
        (-1_500_000, "partial_halt"),
        (-HOUR, "invalid_close_time"),
    ],
)
def test_kline_end_anomalies_preserved(market, scale, offset, flag):
    t = millis("2020-01-01")
    end = t + HOUR - 1 + offset
    rows = parse_klines(kline(t, end, scale), market)
    assert len(rows) == 1 and rows[0]["close"] == 101
    assert rows[0]["close_time"] == end
    assert rows[0]["time_quality"] == (flag or "normal")


def halt_fixture(fixture, market):
    f = copy.deepcopy(fixture)
    p = f["profiles"][0]
    old = p["instrument"]
    p.update(
        instrument=f"{market}:TESTUSDT",
        symbol="TESTUSDT",
        market=market,
        listed_at=f["start"],
    )
    p["contract_type"] = "SPOT" if market == "spot" else "PERPETUAL"
    f["profiles"] = [p]
    d = f["data"][old]
    f["data"] = {p["instrument"]: d}
    f["events"] = []
    for field in ("klines", "markPriceKlines", "indexPriceKlines"):
        bars = d[field]
        first = dict(
            bars[0],
            time=f["start"],
            close_time=f["start"] + HOUR // 2,
            time_quality="partial_halt",
        )
        d[field] = [first, *(b for b in bars if b["time"] >= f["start"] + 3 * HOUR)]
    return f


@pytest.mark.parametrize("market", ["spot", "usdm"])
def test_partial_then_gap_is_self_evidenced_halt(fixture, market):
    f = halt_fixture(fixture, market)
    frame, notes = build_frame(**f)
    assert frame["quality"][0] & 512
    for i in (1, 2):
        assert frame["quality"][i] & 1024
        assert not frame["tradable"][i]
        assert frame["close"][i] == frame["close"][0]
    assert frame["tradable"][3]
    assert notes[f["profiles"][0]["instrument"]]["halts"]
    report = verify_frame(frame, f)
    assert report["verdict"] == "pass", report["errors"]
    assert any(d["check"] == "inferred_halt" for d in report["diagnostics"])
    bad = frame.with_columns((pl.col("quality") & ~1024).alias("quality"))
    assert verify_frame(bad, f)["verdict"] == "fail"
    # 追加恢复之后的数据不回改已观察到的停机小时。
    short = dict(f, end=f["start"] + 3 * HOUR)
    prefix, _ = build_frame(**short)
    assert prefix.equals(frame.head(3))


def test_bad_end_is_reported_and_not_halt_evidence(fixture):
    f = halt_fixture(fixture, "spot")
    b = f["data"][f["profiles"][0]["instrument"]]["klines"][0]
    b.update(close_time=b["time"] - 1, time_quality="invalid_close_time")
    with pytest.raises(ValueError, match="缺少"):
        build_frame(**f)
    f["end"] = f["start"] + HOUR
    frame, _ = build_frame(**f)
    assert frame["quality"][0] & 2048 and not frame["tradable"][0]
    report = verify_frame(frame, f)
    assert any(
        d["check"] == "invalid_close_time" and d["review"]
        for d in report["diagnostics"]
    )


def test_cms_long_backoff_recovers_and_is_configurable():
    calls, waits, now = [], [], [0.0]

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    def handle(request):
        calls.append(now[0])
        return httpx.Response(429 if len(calls) <= 5 else 200, json={})

    with PublicClient(
        transport=httpx.MockTransport(handle),
        sleep=sleep,
        clock=lambda: now[0],
        cms_attempts=7,
        cms_backoff=30,
        cms_backoff_cap=120,
    ) as c:
        assert c.json(CMS + "/list/query") == {}
    assert calls == [0, 30, 90, 210, 330, 450]
    assert max(waits) <= 60


def test_cms_interrupted_pages_resume_without_fetching_page_one(tmp_path):
    calls = []
    broken = True

    class Client:
        def json(self, url, params):
            page = params["pageNo"]
            calls.append(page)
            if page == 3 and broken:
                raise RuntimeError("interrupted")
            articles = (
                []
                if page == 4
                else [
                    dict(
                        code=str(page),
                        title="Ordinary news",
                        releaseDate=millis("2024-01-01"),
                    )
                ]
            )
            return dict(success=True, data=dict(catalogs=[dict(articles=articles)]))

    store = RawStore(tmp_path)
    args = dict(since="2020-01-01", fetched_at="2026-01-01", catalogs=(161,))
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            crawl_announcements(Client(), store, **args)
        broken = False
        crawl_announcements(Client(), store, **args)
        assert calls == [1, 2, 3, 3, 4]
        # 成功跑完后下一次同步从新首页开始，不能永远漏新公告。
        calls.clear()
        crawl_announcements(Client(), store, **args)
        assert calls == [1, 2, 3, 4]
    finally:
        store.close()


def test_announcement_resume_after_body_saved_before_events(tmp_path, monkeypatch):
    """开发自检：body 已落盘、事件写入中断时，不能凭 body hash 跳过解析。"""
    from ashare.data.crypto.events import EventLedger
    from test.crypto.test_announcements import PUBLISHED, article

    store = RawStore(tmp_path)

    class Client:
        def json(self, url, params):
            if "list" in url:
                items = (
                    [
                        dict(
                            code="synthetic",
                            title="Binance Will Delist FAKE",
                            releaseDate=PUBLISHED,
                        )
                    ]
                    if params["pageNo"] == 1
                    else []
                )
                return dict(success=True, data=dict(catalogs=[dict(articles=items)]))
            return dict(
                success=True,
                data=article(
                    publishDate=PUBLISHED,
                    body="The mock venue will conduct an automatic settlement on FAKEUSDT at 2024-06-17 03:00 (UTC).",
                ),
            )

    append = EventLedger.append

    def crash(self, event):
        raise RuntimeError("event write interrupted")

    args = dict(since="2020-01-01", fetched_at="2026-11-15", catalogs=(161,))
    monkeypatch.setattr(EventLedger, "append", crash)
    with pytest.raises(RuntimeError, match="event write"):
        crawl_announcements(Client(), store, **args)
    monkeypatch.setattr(EventLedger, "append", append)
    crawl_announcements(Client(), store, **args)
    assert len(EventLedger(tmp_path).read()) == 1
    store.close()


def test_cms_body_reversion_is_a_new_observed_revision(tmp_path):
    """续抓自检：回退到旧正文也是新修订，不能因旧版本 hash 存在就漏掉。"""
    from ashare.data.crypto.events import EventLedger
    from test.crypto.test_announcements import article, crawl

    a = article(
        body="The mock venue will conduct an automatic settlement on FAKEUSDT at 2024-06-11 09:00 (UTC)."
    )
    store = RawStore(tmp_path)
    try:
        crawl(store, a, "2024-06-04")
        a["body"] = a["body"].replace("2024-06-11", "2024-06-12")
        crawl(store, a, "2024-06-05")
        a["body"] = a["body"].replace("2024-06-12", "2024-06-11")
        crawl(store, a, "2024-06-06")
        events = EventLedger(tmp_path).read()
        assert len(events) == 3
        assert events[-1]["known_at"] == millis("2024-06-06")
        assert events[-1]["effective_at"] == millis("2024-06-11T09:00")
        assert events[-1]["supersedes"] == events[-2]["version_id"]
    finally:
        store.close()


def test_cms_partial_revision_resume_keeps_first_observation_time(
    tmp_path, monkeypatch
):
    """续抓自检：同页多个事件中途失败，已写修订不重复，未写修订保留首次观察时间。"""
    from ashare.data.crypto.events import EventLedger
    from test.crypto.test_announcements import article, crawl

    a = article(
        body="The mock venue will conduct an automatic settlement on FAKEUSDT and DUMMYUSDT at 2024-06-11 09:00 (UTC)."
    )
    store = RawStore(tmp_path)
    try:
        crawl(store, a, "2024-06-04")
        a["body"] = a["body"].replace("2024-06-11", "2024-06-12")
        append = EventLedger.append
        calls = [0]

        def fail_second(self, event):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("second event")
            return append(self, event)

        monkeypatch.setattr(EventLedger, "append", fail_second)
        with pytest.raises(RuntimeError, match="second event"):
            crawl(store, a, "2024-06-05")
        monkeypatch.setattr(EventLedger, "append", append)
        crawl(store, a, "2024-06-06")
        events = EventLedger(tmp_path).read()
        assert len(events) == 4
        revised = [e for e in events if e["supersedes"]]
        assert {e["known_at"] for e in revised} == {millis("2024-06-05")}
    finally:
        store.close()
