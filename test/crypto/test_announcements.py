"""来源：Claude C2 R1 时间规则；R2 将正文换为自撰的 FAKE 合成公告。"""

import json
from pathlib import Path

import httpx
import pytest

from ashare.data.crypto.announcements import (
    article_version,
    crawl_announcements,
    extract_events,
)
from ashare.data.crypto.events import EventLedger, asof
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore, encoded
from ashare.data.crypto.timeutil import millis

FIXTURE = Path(__file__).parents[1] / "fixtures/crypto/announcements/waves.json"
PUBLISHED = millis("2024-06-03T01:02:03.004Z")
FETCHED = "2026-11-15"


def article(**kwargs):
    return dict(code="synthetic", title="Binance Will Delist FAKE", **kwargs)


@pytest.mark.parametrize(
    "dates",
    [
        {"publishDate": PUBLISHED},
        {"releaseDate": PUBLISHED},
        {"publishDate": None, "releaseDate": PUBLISHED},
        {"publishDate": 0, "releaseDate": PUBLISHED},
    ],
)
def test_no_update_evidence_uses_publication(dates):
    # 生效日不是更新证据；没有更新说明时允许采用明确标注的首发假设。
    a = article(**dates, body="Trading ceases at 2024-06-17 03:00 (UTC).")
    v = article_version(a, FETCHED)
    assert v["known_at"] == v["published_at"] == PUBLISHED
    assert v["known_at_basis"] == "publish_no_update_evidence"
    assert v["version_unknown"] is True
    assert article_version(a, "2027-01-01") == v


@pytest.mark.parametrize("dates", [{}, {"publishDate": None}, {"publishDate": 0}])
def test_fetch_fallback_only_when_all_time_evidence_missing(dates):
    v = article_version(article(**dates, body="No date metadata."), FETCHED)
    assert v["known_at"] == millis(FETCHED)
    assert v["published_at"] is None
    assert v["known_at_basis"] == "fetched_no_time_evidence"


@pytest.mark.parametrize(
    "marker",
    [
        "Update (2024-06-05)",
        "Updated on 2024-06-05",
        "(Updated 2024-06-05)",
        "Update: 2024-06-05",
        "Updated: 2024-06-05",
        "UPDATE (2024-06-05)",
        "Updated on&nbsp;2024-06-05",
    ],
)
def test_synthetic_update_marker_variants(marker):
    """合成兼容性测试：除 updated on 外，不声称这些变体来自本地实测。"""
    a = article(
        publishDate=PUBLISHED,
        body=json.dumps(dict(node="root", child=[dict(node="text", text=marker)])),
    )
    v = article_version(a, FETCHED)
    assert v["known_at"] == millis("2024-06-06")
    assert v["known_at_basis"] == "latest_update_evidence"


@pytest.mark.parametrize(
    "cms_update,expected",
    [
        (millis("2024-06-04"), "2024-06-07T12:34:56Z"),
        (millis("2024-06-08T01:02:03Z"), "2024-06-08T01:02:03Z"),
    ],
)
def test_latest_update_wins_across_title_body_and_cms(cms_update, expected):
    a = article(
        publishDate=PUBLISHED,
        body="Updated on 2024-06-05. Update: 2024-06-07 12:34:56 (UTC).",
        lastUpdateTime=cms_update,
    )
    a["title"] += " (Updated 2024-06-06)"
    v = article_version(a, FETCHED)
    assert v["known_at"] == millis(expected)
    assert v["known_at_basis"] == "latest_update_evidence"
    assert article_version(a, "2027-01-01") == v


def test_title_update_and_update_without_publication():
    a = article(body="Original text.")
    a["title"] += " Update (2024-06-05)"
    assert article_version(a, FETCHED)["known_at"] == millis("2024-06-06")
    v = article_version(article(lastUpdateTime=millis("2024-06-07")), FETCHED)
    assert v["known_at"] == millis("2024-06-07")
    assert v["known_at_basis"] == "latest_update_evidence"


def test_date_only_update_is_bounded_by_observation():
    a = article(publishDate=PUBLISHED, body="Updated on 2024-06-03")
    assert article_version(a, "2024-06-03T15:00:00Z")["known_at"] == millis(
        "2024-06-03T15:00:00Z"
    )


def test_synthetic_cms_update_marker():
    a = json.loads(FIXTURE.read_text())
    assert a["lastUpdateTime"] == 0
    v = article_version(a, FETCHED)
    assert "This invented notice was updated on 2024-06-03" in v["original_text"]
    assert v["published_at"] == PUBLISHED
    assert v["known_at"] == 1717459200000  # 2024-06-04 00:00 UTC，只有日期。
    assert v["known_at_basis"] == "latest_update_evidence"
    assert article_version(a, "2027-01-01") == v


def crawl(store, a, fetched=FETCHED):
    def handle(request):
        if "detail" in request.url.path:
            return httpx.Response(200, json=dict(success=True, data=a))
        entries = (
            [dict(code=a["code"], title=a["title"], releaseDate=PUBLISHED)]
            if request.url.params["pageNo"] == "1"
            else []
        )
        return httpx.Response(
            200, json=dict(success=True, data=dict(catalogs=[dict(articles=entries)]))
        )

    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        return crawl_announcements(
            c, store, since="2024-06-01", fetched_at=fetched, catalogs=(161,)
        )


def test_synthetic_public_crawl_creates_historically_known_spot_exit(tmp_path):
    store = RawStore(tmp_path)
    try:
        crawl(store, json.loads(FIXTURE.read_text()))
        ledger = EventLedger(tmp_path)
        versions = ledger.read()
        fake = [e for e in versions if e["instrument"].endswith(":FAKEUSDT")]
        assert len(fake) == 2
        assert asof(versions, "2024-06-03T23:59:59Z") == []
        assert len(asof(versions, "2024-06-04")) == 5
        for e in fake:
            assert e["known_at"] == millis("2024-06-04")
            assert e["known_at_basis"] == "latest_update_evidence"
            assert e["manual"] is False
        by_kind = {e["kind"]: e for e in fake}
        assert by_kind["delist"]["effective_at"] == millis("2024-06-17T03:00:00Z")
        assert by_kind["settlement"]["effective_at"] == millis("2024-06-11T09:00:00Z")
    finally:
        store.close()


def test_list_publication_fallback_and_observed_revision_preserve_history(tmp_path):
    """P1 排查：列表首发时间不能丢失，后来观察到的静默修改不能回写首发。"""
    a = article(
        body="The mock venue will conduct an automatic settlement on FAKEUSDT at "
        "2024-06-11 09:00 (UTC)."
    )
    store = RawStore(tmp_path)
    try:
        crawl(store, a, "2024-06-04")
        ledger = EventLedger(tmp_path)
        first = ledger.read()[0]
        assert first["known_at"] == PUBLISHED
        assert first["known_at_basis"] == "publish_no_update_evidence"
        old_file = tmp_path / "events" / f"{first['version_id']}.json"
        original = old_file.read_bytes()
        assert crawl(store, a, "2024-06-05")["imported"] == 0
        a["body"] = a["body"].replace("2024-06-11", "2024-06-12")
        assert crawl(store, a, "2024-06-06")["imported"] == 1
        revised = asof(ledger.read(), "2024-06-06")[0]
        assert revised["known_at"] == millis("2024-06-06")
        assert revised["known_at_basis"] == "observed_revision"
        assert revised["supersedes"] == first["version_id"]
        assert revised["effective_at"] == millis("2024-06-12T09:00:00Z")
        assert asof(ledger.read(), "2024-06-05") == [first]
        assert old_file.read_bytes() == original
    finally:
        store.close()


def test_list_metadata_does_not_turn_cached_body_into_a_revision(tmp_path):
    """R1 开发回归：给解析器补列表日期不能改变已缓存原文的 hash。"""
    a = article(
        publishDate=PUBLISHED,
        body="The mock venue will conduct an automatic settlement on FAKEUSDT at "
        "2024-06-11 09:00 (UTC).",
    )
    store = RawStore(tmp_path)
    try:
        key = "announcements/article/synthetic"
        store.put(key, encoded(a), source="cms", fetched_at=millis("2024-06-04"))
        ledger = EventLedger(tmp_path)
        initial = ledger.append(extract_events(a, "2024-06-04")[0])
        assert crawl(store, a)["imported"] == 0
        assert store.get(key) == encoded(a)
        assert ledger.read() == [initial]
    finally:
        store.close()
