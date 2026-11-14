"""OpenAI 兼容的聊天接口（默认 DeepSeek 官方），只用 httpx。

- 密钥只从构造参数或环境变量（默认 ``DEEPSEEK_API_KEY``）读，第一次请求时才读；不写日志、
  不进转写记录，所有报错消息都先把密钥替换掉；
- ``seed`` 只在 ``send_seed=True`` 时放进请求（不是所有兼容接口都接受这个字段）；
- 不用流式；``timeout`` 管整个请求（连接、等响应头、读完正文），超时算一次传输错误；
- 429、5xx、传输错误按 ``backoff`` 指数退避重试 ``retries`` 次（有数字型 Retry-After 时
  取两者较大，上限 60 秒），其余 4xx 直接报错；重试用完报 `LLMError`，消息只有状态码和
  接口给的简短说明；
- 用量与费用按每次 200 响应里的 ``usage`` 算（DeepSeek 的缓存命中 token 单独计价）；
  响应没有 usage 时按字符数估一个偏高的值并标记 ``estimated``——拿到 200 就已经付费，
  无论内容能不能用。

测试注入假客户端（实现 ``complete(messages, *, json_mode, seed)`` 返回 `Reply`，有 ``model``
属性和 ``describe()``）或给 `ChatClient` 传 ``transport=httpx.MockTransport(...)``。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


class LLMError(RuntimeError):
    """模型接口调用失败（消息里不含密钥）。"""


@dataclass(frozen=True)
class Pricing:
    """每百万 token 的价格（美元）。默认值是 deepseek-flash 的高峰价（2026-11-14 核对官方价目：
    缓存未命中输入 0.3、命中 0.006、输出 1.2；非高峰减半），按高峰价算预算只会偏高。换模型或价目变了要显式传入。"""

    input: float = 0.3
    cached_input: float = 0.006
    output: float = 1.2

    def cost(self, usage):
        hit = usage.cached_tokens
        miss = max(usage.prompt_tokens - hit, 0)
        return (
            miss * self.input
            + hit * self.cached_input
            + usage.completion_tokens * self.output
        ) / 1e6


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    estimated: bool = False

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Reply:
    """一次成功调用：正文、用量、费用（美元）、实际尝试次数。"""

    text: str
    usage: Usage
    cost: float
    attempts: int = 1
    finish_reason: str | None = None


def _int(x):
    return (
        int(x)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0
        else 0
    )


def parse_usage(raw, messages, text):
    """响应里的 usage → `Usage`；缺失时按字符数估（每 2 个字符算 1 个 token，偏高）。"""
    if isinstance(raw, dict) and "prompt_tokens" in raw:
        cached = raw.get("prompt_cache_hit_tokens")
        if cached is None:
            cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        return Usage(_int(raw.get("prompt_tokens")), _int(raw.get("completion_tokens")),
                     _int(cached))  # fmt: skip
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return Usage(chars // 2 + 1, len(text) // 2 + 1, 0, True)


class ChatClient:
    """OpenAI 兼容的 ``/chat/completions`` 客户端。"""

    def __init__(
        self,
        model=DEFAULT_MODEL,
        *,
        base_url=DEFAULT_BASE_URL,
        api_key=None,
        api_key_env=DEFAULT_KEY_ENV,
        pricing=None,
        timeout=180.0,
        retries=3,
        backoff=2.0,
        temperature=1.0,
        max_tokens=4096,
        send_seed=False,
        transport=None,
        sleep=time.sleep,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self._key = api_key
        self.api_key_env = api_key_env
        self.pricing = pricing or Pricing()
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.send_seed = bool(send_seed)
        self._transport = transport
        self._sleep = sleep

    def describe(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "send_seed": self.send_seed,
            "pricing": asdict(self.pricing),
        }

    def _api_key(self):
        if self._key is None:
            self._key = os.environ.get(self.api_key_env) or None
        if not self._key:
            raise LLMError(f"没有找到 API 密钥：设置环境变量 {self.api_key_env}")
        return self._key

    def _redact(self, text):
        text = str(text)
        if self._key:
            text = text.replace(self._key, "***")
        return text

    def complete(self, messages, *, json_mode=True, seed=None):
        """发一次请求，返回 `Reply`；失败抛 `LLMError`。"""
        import httpx

        key = self._api_key()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if seed is not None and self.send_seed:
            body["seed"] = int(seed)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        last = None
        for attempt in range(1, self.retries + 2):
            wait = None
            try:
                status, raw, retry_after = self._post(httpx, url, headers, body)
            except (httpx.TransportError, TimeoutError) as exc:
                last = f"网络错误（{type(exc).__name__}）"
                wait = self.backoff ** (attempt - 1)
            else:
                if status == 200:
                    return self._reply(raw, messages, attempt)
                detail = self._detail(raw)
                last = f"模型接口返回 {status}" + (f"：{detail}" if detail else "")
                if status == 429 or status >= 500:
                    wait = self.backoff ** (attempt - 1)
                    if retry_after is not None:
                        wait = max(wait, retry_after)
                else:
                    raise LLMError(self._redact(last))
            if attempt <= self.retries:
                self._sleep(min(wait, 60.0))
        raise LLMError(self._redact(f"重试 {self.retries} 次后仍失败：{last}"))

    def _post(self, httpx, url, headers, body):
        """整个请求一个截止时间：逐块读正文，超时即放弃。"""
        deadline = time.monotonic() + self.timeout
        timeout = httpx.Timeout(self.timeout)
        with httpx.Client(transport=self._transport, timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                chunks = []
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    if time.monotonic() > deadline:
                        raise TimeoutError("整个请求超时")
                retry_after = resp.headers.get("retry-after")
                try:
                    retry_after = float(retry_after) if retry_after else None
                except ValueError:
                    retry_after = None
                return resp.status_code, b"".join(chunks), retry_after

    def _detail(self, raw):
        try:
            err = json.loads(raw).get("error")
            msg = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            return ""
        return self._redact(str(msg))[:200] if msg else ""

    def _reply(self, raw, messages, attempt):
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("模型接口返回了无法解析的响应") from None
        usage = parse_usage(data.get("usage"), messages, text)
        return Reply(str(text), usage, self.pricing.cost(usage), attempt,
                     choice.get("finish_reason"))  # fmt: skip
