"""大模型研究循环：OpenAI 兼容客户端（重试、友好报错、密钥不外泄、用量与费用）。不联网。

来源：LLM 挖掘任务书（2026-11-14）——429 / 5xx / 传输错误有限次退避重试，用完给友好的
错误；密钥只从环境变量读、任何报错都不带它；费用按每次 200 响应的 usage 计。
"""

import json

import httpx
import pytest

from ashare.factor.mining.llm import ChatClient, LLMError, Pricing, Usage
from ashare.factor.mining.llm.client import parse_usage

KEY = "sk-test-SECRET-0123456789"
MSG = [{"role": "user", "content": "hi"}]


def _ok(text="{}", usage=None):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    if usage is not False:
        body["usage"] = usage or {"prompt_tokens": 1000, "completion_tokens": 200,
                                  "prompt_cache_hit_tokens": 400}  # fmt: skip
    return httpx.Response(200, json=body)


def _client(handler, **kw):
    sleeps = []
    c = ChatClient(api_key=KEY, transport=httpx.MockTransport(handler),
                   sleep=sleeps.append, **kw)  # fmt: skip
    return c, sleeps


def test_request_shape_and_cost():
    seen = []

    def handler(request):
        seen.append(request)
        return _ok('{"factors": []}')

    c, _ = _client(handler, pricing=Pricing(input=1.0, cached_input=0.1, output=2.0))
    reply = c.complete(MSG, json_mode=True, seed=5)
    req = seen[0]
    assert req.url == "https://api.deepseek.com/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "deepseek-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert "seed" not in body
    assert reply.text == '{"factors": []}'
    assert reply.usage == Usage(1000, 200, 400)
    # 600 未命中 × 1 + 400 命中 × 0.1 + 200 输出 × 2，每百万
    assert reply.cost == pytest.approx((600 * 1.0 + 400 * 0.1 + 200 * 2.0) / 1e6)
    assert reply.attempts == 1
    c.send_seed = True
    c.complete(MSG, seed=5)
    assert json.loads(seen[-1].content)["seed"] == 5


def test_retries_429_and_5xx_then_succeeds():
    codes = iter([429, 503, 500])

    def handler(request):
        code = next(codes, 200)
        if code == 200:
            return _ok()
        return httpx.Response(code, headers={"retry-after": "7"} if code == 429 else {},
                              json={"error": {"message": "busy"}})  # fmt: skip

    c, sleeps = _client(handler, retries=3, backoff=2.0)
    reply = c.complete(MSG)
    assert reply.attempts == 4
    assert sleeps == [7.0, 2.0, 4.0]  # Retry-After 优先（取大），其余指数退避


def test_gives_up_with_friendly_error_after_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(502, json={"error": {"message": f"bad gateway {KEY}"}})

    c, sleeps = _client(handler, retries=2)
    with pytest.raises(LLMError) as info:
        c.complete(MSG)
    assert len(calls) == 3 and len(sleeps) == 2
    assert "502" in str(info.value) and "重试 2 次" in str(info.value)
    assert KEY not in str(info.value)


def test_transport_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused", request=request)
        return _ok()

    c, _ = _client(handler, retries=3)
    assert c.complete(MSG).attempts == 3
    calls.clear()

    def always(request):
        raise httpx.ReadTimeout("slow", request=request)

    c, _ = _client(always, retries=1)
    with pytest.raises(LLMError, match="网络错误"):
        c.complete(MSG)


def test_client_errors_are_not_retried_and_key_is_redacted():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": f"invalid key {KEY}"}})

    c, sleeps = _client(handler)
    with pytest.raises(LLMError) as info:
        c.complete(MSG)
    assert calls == [1] and sleeps == []
    assert "401" in str(info.value)
    assert KEY not in str(info.value) and "***" in str(info.value)


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("ASHARE_TEST_LLM_KEY", raising=False)
    c = ChatClient(api_key_env="ASHARE_TEST_LLM_KEY",
                   transport=httpx.MockTransport(lambda r: _ok()))  # fmt: skip
    with pytest.raises(LLMError, match="ASHARE_TEST_LLM_KEY"):
        c.complete(MSG)
    monkeypatch.setenv("ASHARE_TEST_LLM_KEY", KEY)
    assert c.complete(MSG).text == "{}"
    assert KEY not in json.dumps(c.describe())


def test_unparseable_200_is_an_error():
    c, _ = _client(lambda r: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(LLMError, match="无法解析"):
        c.complete(MSG)


def test_missing_usage_is_estimated_high():
    c, _ = _client(lambda r: _ok("x" * 300, usage=False))
    reply = c.complete(MSG)
    assert reply.usage.estimated
    assert reply.usage.completion_tokens >= 150 and reply.cost > 0
    # OpenAI 写法的缓存命中字段也认
    u = parse_usage({"prompt_tokens": 10, "completion_tokens": 2,
                     "prompt_tokens_details": {"cached_tokens": 4}}, MSG, "")  # fmt: skip
    assert u == Usage(10, 2, 4)


def test_whole_request_deadline(monkeypatch):
    """正文慢慢滴：超过整次请求的截止时间就算传输失败（重试用完报错）。"""
    from ashare.factor.mining.llm import client as mod

    clock = iter(range(0, 10_000, 50))
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clock))

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                yield b" "

    c, _ = _client(lambda r: httpx.Response(200, stream=Drip()), retries=0, timeout=60)
    with pytest.raises(LLMError, match="TimeoutError"):
        c.complete(MSG)


# ---- 思考与实际扣费（2026-11-14 首次真实试跑：OpenRouter 上 deepseek-v4.1-flash 默认思考，
# 旧的 4096 输出上限三轮全被思考占满、正文为空）---------------------------------------------

OPENROUTER = "https://openrouter.ai/api/v1"


def test_reasoning_tokens_parsed():
    u = parse_usage({"prompt_tokens": 10, "completion_tokens": 4096,
                     "completion_tokens_details": {"reasoning_tokens": 4096}}, MSG, "")  # fmt: skip
    assert u.reasoning_tokens == 4096
    assert u.completion_tokens == 4096


@pytest.mark.parametrize("level, field", [("off", {"enabled": False}),
                                          ("low", {"effort": "low"}),
                                          ("high", {"effort": "high"})])  # fmt: skip
def test_reasoning_sent_on_openrouter(level, field):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _ok()

    c, _ = _client(handler, base_url=OPENROUTER, reasoning=level)
    c.complete(MSG)
    assert seen[0]["reasoning"] == field
    assert c.describe()["reasoning"] == level


def test_reasoning_default_sends_nothing_and_unverified_hosts_refuse():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _ok()

    c, _ = _client(handler)
    c.complete(MSG)
    assert "reasoning" not in seen[0]
    assert seen[0]["max_tokens"] >= 16384  # 4096 装不下默认思考加回答
    with pytest.raises(ValueError, match="OpenRouter"):
        ChatClient(api_key=KEY, reasoning="low")
    with pytest.raises(ValueError, match="reasoning"):
        ChatClient(api_key=KEY, base_url=OPENROUTER, reasoning="max")


@pytest.mark.parametrize("reported, expect", [(0.00109872, "reported"), (None, "pricing"),
                                              (-1.0, "pricing"), (float("nan"), "pricing"),
                                              ("0.5", "pricing"), (True, "pricing")])  # fmt: skip
def test_reported_cost_used_when_valid(reported, expect):
    usage = {"prompt_tokens": 1000, "completion_tokens": 200}
    if reported is not None:
        usage["cost"] = reported

    def handler(request):
        body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": usage, "provider": "DeepInfra"}  # fmt: skip
        return httpx.Response(200, content=json.dumps(body).encode())

    c, _ = _client(handler, pricing=Pricing(input=1.0, cached_input=0.1, output=2.0))
    reply = c.complete(MSG)
    assert reply.cost_source == expect
    assert reply.provider == "DeepInfra"
    if expect == "reported":
        assert reply.cost == pytest.approx(reported)
    else:
        assert reply.cost == pytest.approx((1000 * 1.0 + 200 * 2.0) / 1e6)
