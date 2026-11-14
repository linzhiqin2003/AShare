import hashlib
import io
import json
import zipfile

import httpx
import pytest

from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore


def zipped(text="a,b\n1,2\n"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("sample.csv", text)
    return buf.getvalue()


def test_checksum_resume_and_revisions(tmp_path):
    """来源：6.10 REST/归档替换必须保留旧版本；坏校验不提交清单。"""
    payload = [zipped()]
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200, text=hashlib.sha256(payload[0]).hexdigest() + "  sample.zip\n"
            )
        return httpx.Response(200, content=payload[0])

    store = RawStore(tmp_path)
    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        first = c.archive(store, "sample.zip")
        old_version = store.freeze()
        assert c.archive(store, "sample.zip") == first
        assert len(calls) == 2
        payload[0] = zipped("a,b\n3,4\n")
        assert c.archive(store, "sample.zip", refresh=True) != first
        assert store.freeze() != old_version
        entry = store.manifest.read()["sample.zip"]
        assert len(entry["revisions"]) == 2
        assert store.get("sample.zip", sha=entry["revisions"][0]["sha256"]) == first
    store.close()


def test_bad_checksum_and_resume_corruption(tmp_path):
    store = RawStore(tmp_path)

    def handle(request):
        return httpx.Response(
            200,
            content=b"0" * 64 + b"  x.zip"
            if request.url.path.endswith("CHECKSUM")
            else zipped(),
        )

    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        with pytest.raises(ValueError, match="CHECKSUM 不匹配"):
            c.archive(store, "x.zip")
    assert store.manifest.read()["x.zip"]["status"] == "failed"
    e = store.put("x.zip", b"valid", source="rest")
    (tmp_path / "objects" / e["sha256"]).write_bytes(b"bad")
    assert not store.complete("x.zip")
    store.close()


@pytest.mark.parametrize("prefixes", [False, True])
def test_s3_pagination(prefixes):
    """来源：经验文件与 R1-16；超过 1000 项必须翻页。"""

    def handle(request):
        marker = request.url.params.get("marker")
        if not marker:
            tag = (
                "<CommonPrefixes><Prefix>a/</Prefix></CommonPrefixes>"
                if prefixes
                else "<Contents><Key>a</Key></Contents>"
            )
            body = f"<ListBucketResult xmlns='urn:s3'><IsTruncated>true</IsTruncated>{tag}</ListBucketResult>"
        else:
            assert marker == ("a/" if prefixes else "a")
            tag = (
                "<CommonPrefixes><Prefix>b/</Prefix></CommonPrefixes>"
                if prefixes
                else "<Contents><Key>b</Key></Contents>"
            )
            body = f"<ListBucketResult><IsTruncated>false</IsTruncated>{tag}</ListBucketResult>"
        return httpx.Response(200, text=body)

    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        assert list(c.listing("data/", delimiter="/" if prefixes else None)) == (
            ["a/", "b/"] if prefixes else ["a", "b"]
        )


def test_s3_no_progress_rejected():
    response = "<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>"
    with PublicClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=response)),
        sleep=lambda _: None,
    ) as c:
        with pytest.raises(ValueError, match="前进"):
            list(c.listing("x"))


def test_rate_limit_retry_timeout_and_budget():
    waits, requests = [], []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=[])

    with PublicClient(
        transport=httpx.MockTransport(handle), sleep=waits.append, clock=lambda: 0
    ) as c:
        c.funding("BTCUSDT", 0, 100)
        c.funding("BTCUSDT", 0, 100)
    assert 2 in waits and 0.65 in waits
    assert all(
        r.method == "GET"
        and "X-MBX-APIKEY" not in r.headers
        and "signature" not in r.url.params
        for r in requests
    )
    with PublicClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"123")),
        max_bytes=2,
        sleep=lambda _: None,
    ) as c:
        with pytest.raises(RuntimeError, match="max_bytes"):
            c.get("https://data.binance.vision/a")

    def timeout(request):
        raise httpx.ReadTimeout("fixture", request=request)

    with PublicClient(
        transport=httpx.MockTransport(timeout), sleep=lambda _: None
    ) as c:
        with pytest.raises(httpx.ReadTimeout):
            c.get("https://data.binance.vision/a")


def test_rest_pagination():
    calls = []

    def handle(request):
        start = int(request.url.params["startTime"])
        calls.append(start)
        if start == 0:
            return httpx.Response(
                200,
                json=[
                    dict(fundingTime=i, fundingRate="0", markPrice="1")
                    for i in range(1000)
                ],
            )
        return httpx.Response(
            200, json=[dict(fundingTime=1000, fundingRate="0", markPrice="1")]
        )

    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        assert len(c.funding("X", 0, 1000)) == 1001
    assert calls == [0, 1000]


def test_frozen_content_manifest(tmp_path):
    store = RawStore(tmp_path)
    store.put("rest/kline", b"old", source="rest")
    v = store.freeze()
    store.put("rest/kline", b"new", source="archive")
    old = json.loads((tmp_path / "raw_versions" / f"{v}.json").read_text())
    assert store.get("rest/kline", sha=old["rest/kline"]["sha256"]) == b"old"
    store.close()


def test_corrupt_object_repaired_on_resume(tmp_path):
    """开发自检：内容地址文件损坏后续跑必须重写损坏对象，而非只更新清单。"""
    store = RawStore(tmp_path)
    payload = zipped()
    entry = store.put("x.zip", payload, source="archive")
    (tmp_path / "objects" / entry["sha256"]).write_bytes(b"broken")

    def handle(request):
        return httpx.Response(
            200,
            content=(hashlib.sha256(payload).hexdigest() + "  x.zip").encode()
            if request.url.path.endswith("CHECKSUM")
            else payload,
        )

    with PublicClient(transport=httpx.MockTransport(handle), sleep=lambda _: None) as c:
        c.archive(store, "x.zip")
    assert store.complete("x.zip")
    store.close()
