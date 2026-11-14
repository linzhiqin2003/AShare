import json

import httpx
import pytest

from ashare.data.crypto.announcements import (
    article_version,
    body_text,
    crawl_announcements,
    extract_events,
)
from ashare.data.crypto.events import EventLedger, asof, normalize_event
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.timeutil import millis


def event(**kwargs):
    return dict(
        event_id="OMG",
        instrument="usdm:OMGUSDT",
        kind="settlement",
        known_at="2024-12-14",
        effective_at="2024-12-30",
        source_url="https://www.binance.com/en/support/announcement/detail/eedac7aff5b849d58efda6c4b3cec01b",
        original_text="Synthetic version structure based on R3 V3, not a recovered original.",
        **kwargs,
    )


def test_omg_revision_cancellation_and_append_only(tmp_path):
    """来源：R3 V3，12/20 更新不得覆盖 12/14 已知的 12/30。"""
    ledger = EventLedger(tmp_path)
    first = ledger.append(event())
    old_bytes = (tmp_path / "events" / f"{first['version_id']}.json").read_bytes()
    revision = dict(
        event(),
        known_at="2024-12-20",
        effective_at="2025-01-31",
        supersedes=first["version_id"],
    )
    second = ledger.append(revision)
    assert asof(ledger.read(), "2024-12-19")[0]["effective_at"] == millis("2024-12-30")
    assert asof(ledger.read(), "2024-12-20")[0]["effective_at"] == millis("2025-01-31")
    assert (
        tmp_path / "events" / f"{first['version_id']}.json"
    ).read_bytes() == old_bytes
    ledger.append(
        dict(
            revision,
            known_at="2024-12-21",
            supersedes=second["version_id"],
            status="cancelled",
        )
    )
    assert asof(ledger.read(), "2024-12-22") == []
    assert len(ledger.read()) == 3
    with pytest.raises(ValueError, match="supersedes"):
        ledger.append(dict(event(), known_at="2024-12-25"))


def test_migration_and_no_unproven_backdating():
    e = normalize_event(
        dict(
            event(),
            kind="migration",
            replacement="POL",
            ratio=1,
            effective_at="2024-09-13",
            known_at="2024-08-28",
        )
    )
    assert e["ratio"] == 1 and e["replacement"] == "POL"
    version = article_version(
        dict(
            code="x",
            publishDate=millis("2024-12-14"),
            body="Updated on 2024-12-20: now January 31",
        ),
        "2026-11-15",
    )
    assert version["known_at"] == millis("2024-12-21")
    assert version["version_unknown"]


def test_cms_child_tree_regression():
    """开发实测 CMS JSON 使用 child；用虚构交易对复现该结构。"""
    body = json.dumps(
        dict(
            node="root",
            child=[dict(node="element", child=[dict(node="text", text="FAKEUSDT")])],
        )
    )
    assert body_text(body) == "FAKEUSDT"


def test_crawl_public_lists_and_originals(tmp_path):
    text = "The mock venue will close all positions and conduct an automatic settlement on FAKEUSDT at 2024-06-11 09:00 (UTC)."
    article = dict(
        code="synthetic-fake",
        title="Binance Will Delist FAKE",
        publishDate=millis("2024-06-03"),
        body=text,
    )
    assert extract_events(article, "2026-11-15")[0]["effective_at"] == millis(
        "2024-06-11T09:00:00Z"
    )
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if "detail" in request.url.path:
            return httpx.Response(200, json=dict(success=True, data=article))
        articles = (
            [
                dict(
                    code="synthetic-fake",
                    title=article["title"],
                    releaseDate=millis("2024-06-03"),
                )
            ]
            if request.url.params["pageNo"] == "1"
            else []
        )
        return httpx.Response(
            200, json=dict(success=True, data=dict(catalogs=[dict(articles=articles)]))
        )

    store = RawStore(tmp_path)
    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        r = crawl_announcements(
            c, store, since="2024-06-01", fetched_at="2026-11-15", catalogs=(161,)
        )
    assert r == dict(pages=2, imported=1, pending_review=1)
    assert len(EventLedger(tmp_path).read()) == 1
    assert len(calls) == 3
    store.close()
