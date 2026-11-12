import asyncio
import json
from pathlib import Path

import httpx
import pytest

import ashare
from ashare.data.sources import SourceError, Tushare
from ashare.data.sources.tushare import TokenBucket

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

    def time(self):
        return self.now


def response(rows, fields=("n",)):
    return httpx.Response(
        200, json={"code": 0, "data": {"fields": fields, "items": rows}}
    )


def test_exact_page_and_single_poll(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "fake-secret-not-a-real-token")
    seen = []

    def handler(request):
        params = json.loads(request.content)["params"]
        seen.append(params)
        return response([[1], [2]] if params.get("offset", 0) == 0 else [])

    async def run():
        clock = Clock()
        async with Tushare(
            transport=httpx.MockTransport(handler),
            limiter=TokenBucket(200, clock=clock.time, sleep=clock.sleep),
        ) as client:
            result = await client.fetch("example", {}, page_size=2)
            assert result.table["n"].to_list() == [1, 2]
            assert result.calls == 2
            await client.fetch("example", {})

    asyncio.run(run())
    assert seen == [{"limit": 2, "offset": 0}, {"limit": 2, "offset": 2}, {}]


def test_rate_limit_retries_and_slowdown(monkeypatch, caplog):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
    replies = [
        httpx.Response(200, json={"code": -2001, "msg": "每分钟最多访问该接口200次"}),
        response([[1]]),
    ]
    clock = Clock()
    limiter = TokenBucket(200, clock=clock.time, sleep=clock.sleep)

    async def run():
        async with Tushare(
            transport=httpx.MockTransport(lambda _: replies.pop(0)),
            limiter=limiter,
            sleep=clock.sleep,
        ) as client:
            result = await client.fetch("daily", {})
            assert result.retries == 1

    asyncio.run(run())
    assert limiter.rate == 100
    assert clock.waits[0] >= 1
    assert "test-secret" not in caplog.text


@pytest.mark.parametrize(
    "failure", ["network", "server", "permission", "api_name", "unknown", "malformed"]
)
def test_errors(monkeypatch, failure):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            return response([[1]])
        if failure == "network":
            raise httpx.ConnectError("test-secret", request=request)
        if failure == "server":
            return httpx.Response(503, text="test-secret")
        if failure == "malformed":
            return httpx.Response(200, text="test-secret")
        msg = {
            "permission": "没有权限 test-secret",
            "api_name": "请指定正确的接口名",
            "unknown": "未知错误",
        }[failure]
        return httpx.Response(200, json={"code": 40101, "msg": msg})

    async def run():
        clock = Clock()
        async with Tushare(
            transport=httpx.MockTransport(handler),
            limiter=TokenBucket(clock=clock.time, sleep=clock.sleep),
            sleep=clock.sleep,
        ) as client:
            if failure in ("network", "server", "malformed"):
                result = await client.fetch("example", {})
                assert result.retries == 1
                assert clock.waits[0] == 1
            else:
                with pytest.raises(SourceError) as exc:
                    await client.fetch("example", {})
                assert "example" in str(exc.value)
                assert "test-secret" not in str(exc.value)
                assert calls == 1

    asyncio.run(run())


def test_bucket_and_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")

    async def run():
        clock = Clock()
        bucket = TokenBucket(120, clock=clock.time, sleep=clock.sleep)
        await asyncio.gather(*(bucket.acquire() for _ in range(4)))
        assert clock.now == 1.5
        async with Tushare(
            transport=httpx.MockTransport(lambda _: response([])),
            limiter=bucket,
            state_path=tmp_path / "budget.json",
            budget=1,
        ) as client:
            await client.fetch("example", {})
            with pytest.raises(SourceError, match="预算"):
                await client.fetch("example", {})

    asyncio.run(run())
    assert "test-secret" not in (tmp_path / "budget.json").read_text()


def test_repeated_page_detected(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")

    async def run():
        clock = Clock()
        async with Tushare(
            transport=httpx.MockTransport(lambda _: response([[1]])),
            limiter=TokenBucket(clock=clock.time, sleep=clock.sleep),
        ) as client:
            with pytest.raises(SourceError, match="offset"):
                await client.fetch("example", {}, page_size=1)

    asyncio.run(run())


def test_handwritten_protocol_response(monkeypatch):
    """手写协议行覆盖空值和字段映射，不把探查响应放进仓库。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")

    async def run():
        async with Tushare(
            transport=httpx.MockTransport(
                lambda _: response(
                    [["EXAMPLE", None, 1.25]], ("symbol", "optional", "value")
                )
            )
        ) as client:
            result = await client.fetch("example", {})
            assert result.table.to_dicts() == [
                dict(symbol="EXAMPLE", optional=None, value=1.25)
            ]

    asyncio.run(run())


def test_concurrency_cap_and_exponential_backoff(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
    active = maximum = 0

    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response([])

    async def run():
        clock = Clock()
        async with Tushare(
            transport=httpx.MockTransport(handler),
            concurrency=2,
            limiter=TokenBucket(clock=clock.time, sleep=clock.sleep),
        ) as client:
            results = await asyncio.gather(
                *(client.fetch("example", {}) for _ in range(8))
            )
            assert all(result.calls == 1 for result in results)
        assert maximum == 2
        clock = Clock()
        async with Tushare(
            transport=httpx.MockTransport(lambda _: httpx.Response(503)),
            retries=3,
            sleep=clock.sleep,
            limiter=TokenBucket(clock=clock.time, sleep=clock.sleep),
        ) as client:
            with pytest.raises(SourceError):
                await client.fetch("example", {})
        assert clock.waits == [1, 2, 4]

    asyncio.run(run())
