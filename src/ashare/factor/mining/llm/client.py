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
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENCODE_GO_URL = "https://opencode.ai/inference/go/openai/v1"
# 自己的客户端名：OpenCode Go 要求调用方用自己的 User-Agent 表明身份，不用通用 HTTP 库的名字
USER_AGENT = "ashare-llm-mining/1.0"


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
    reasoning_tokens: int = 0  # 输出里用于思考的部分（已含在 completion_tokens 里）

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
    cost_source: str = "pricing"  # "reported"：平台在 usage.cost 里给了实际扣费
    provider: str | None = None  # OpenRouter 实际转到的服务商


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
        details = raw.get("completion_tokens_details")
        thinking = (
            details.get("reasoning_tokens") if isinstance(details, dict) else None
        )
        return Usage(_int(raw.get("prompt_tokens")), _int(raw.get("completion_tokens")),
                     _int(cached), reasoning_tokens=_int(thinking))  # fmt: skip
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
        max_tokens=32768,
        reasoning=None,
        send_seed=False,
        key_fallback=None,
        transport=None,
        sleep=time.sleep,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self._key = api_key
        self.api_key_env = api_key_env
        self._key_fallback = key_fallback  # 环境变量没有密钥时再问它（返回密钥或 None）
        self.session = f"ashare-{uuid.uuid4().hex[:16]}"
        self.pricing = pricing or Pricing()
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.reasoning = _reasoning_field(
            reasoning, self.base_url
        )  # (字段名, 值) 或 None
        self.reasoning_name = reasoning
        self.send_seed = bool(send_seed)
        self._transport = transport
        self._sleep = sleep

    def describe(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning_name,
            "send_seed": self.send_seed,
            "pricing": asdict(self.pricing),
        }

    def new_session(self, name):
        """换一个会话编号（每次搜索一个；OpenCode Go 按它路由和缓存，同一次搜索里保持不变）。"""
        self.session = str(name)[:64]

    def _api_key(self):
        if self._key is None:
            self._key = os.environ.get(self.api_key_env) or None
        if self._key is None and self._key_fallback is not None:
            self._key = self._key_fallback() or None
        if not self._key:
            hint = (
                "，或先运行 opencode auth login"
                if self._key_fallback is not None
                else ""
            )
            raise LLMError(f"没有找到 API 密钥：设置环境变量 {self.api_key_env}{hint}")
        return self._key

    def _redact(self, text):
        text = str(text)
        if self._key:
            text = text.replace(self._key, "***")
        return text

    def complete(self, messages, *, json_mode=True, seed=None):
        """发一次请求，返回 `Reply`；失败抛 `LLMError`。"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning is not None:
            field_name, value = self.reasoning
            body[field_name] = value
        if seed is not None and self.send_seed:
            body["seed"] = int(seed)
        raw, attempt = self._call("/chat/completions", body)
        return self._reply(raw, messages, attempt)

    def _call(self, path, body):
        """POST 一次（带重试），返回 (200 的正文, 第几次尝试)；失败抛 `LLMError`。"""
        import httpx

        key = self._api_key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "User-Agent": USER_AGENT}  # fmt: skip
        if _is_opencode_go(self.base_url):
            # 缺这个头 OpenCode Go 直接回 400 MissingSessionID（2026-11-14 实测）
            headers["x-opencode-session"] = self.session
        url = f"{self.base_url}{path}"
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
                    return raw, attempt
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
        raw_usage = data.get("usage")
        usage = parse_usage(raw_usage, messages, text)
        cost, source = self.pricing.cost(usage), "pricing"
        raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
        reported = _money(raw_usage.get("cost"))
        oc = raw_usage.get("opencode")
        allowance = _money(oc.get("allowance")) if isinstance(oc, dict) else None
        # OpenCode Go 订阅内的调用 usage.cost 是 0，实际扣的是套餐额度（usage.opencode.allowance）；
        # 超出额度转按量付费时扣在 cost 上。预算按两者较大的算，不会因为 cost=0 永远不停。
        if allowance is not None:
            cost, source = max(allowance, reported or 0.0), "allowance"
        elif reported is not None:
            cost, source = reported, "reported"
        provider = data.get("provider")
        return Reply(str(text), usage, cost, attempt, choice.get("finish_reason"),
                     source, provider if isinstance(provider, str) else None)  # fmt: skip


def _money(x):
    if (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and x >= 0
    ):
        return float(x)
    return None


def _is_opencode_go(base_url):
    return base_url.startswith("https://opencode.ai/inference/go/")


REASONING = ("off", "low", "medium", "high", "max")


def _reasoning_field(reasoning, base_url):
    """``--reasoning`` → 请求体里的 (字段名, 值)。``None`` 不发（用模型默认：DeepSeek Flash 默认思考）。

    只支持实测过的接口（2026-11-14，DeepSeek V4.1 Flash），别的接口写法不同，传了就报错，不猜：
    - OpenRouter：``reasoning`` 对象，off / low / medium / high（off 时思考 0 token，low/high 约 550，
      默认约 1400）；
    - OpenCode Go：``reasoning_effort``，low / high / max（low 约 190，max 约 3900；没有关闭开关，
      ``reasoning`` 对象它不认）。"""
    if reasoning is None:
        return None
    if reasoning not in REASONING:
        raise ValueError(f"reasoning 只能是 {REASONING} 之一，收到 {reasoning!r}")
    if "openrouter.ai" in base_url:
        if reasoning == "max":
            raise ValueError(
                "OpenRouter 上 reasoning 只核实过 off / low / medium / high"
            )
        return "reasoning", (
            {"enabled": False} if reasoning == "off" else {"effort": reasoning}
        )
    if _is_opencode_go(base_url):
        if reasoning not in ("low", "high", "max"):
            raise ValueError("OpenCode Go 的 reasoning 只支持 low / high / max")
        return "reasoning_effort", reasoning
    raise ValueError(
        "reasoning 只在 OpenRouter 和 OpenCode Go 上核实过，其他接口请不传（用模型默认）"
    )


def opencode_auth_key(path=None):
    """读 ``opencode auth login`` 存下的 OpenCode 密钥（Go 订阅用的就是它）。只读；读不到返回 None。

    位置：``OPENCODE_AUTH_FILE`` 或 ``~/.local/share/opencode/auth.json`` 里的 ``opencode.key``
    （``type`` 为 ``api``）。"""
    path = (
        path
        or os.environ.get("OPENCODE_AUTH_FILE")
        or os.path.expanduser("~/.local/share/opencode/auth.json")
    )
    try:
        with open(path, encoding="utf-8") as f:
            entry = json.load(f).get("opencode")
    except (OSError, ValueError, AttributeError):
        return None
    if (
        isinstance(entry, dict)
        and entry.get("type") == "api"
        and isinstance(entry.get("key"), str)
    ):
        return entry["key"] or None
    return None


@dataclass(frozen=True)
class Provider:
    """一个接口预设：地址、默认模型、密钥的环境变量，以及密钥的后备来源。"""

    base_url: str
    model: str
    key_env: str
    key_fallback: object = None


PROVIDERS = {
    "deepseek": Provider(DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_KEY_ENV),
    "openrouter": Provider(
        OPENROUTER_URL, "deepseek/deepseek-v4.1-flash", "OPENROUTER_API_KEY"
    ),
    # OpenCode Go 订阅（DeepSeek V4.1 Flash 每月 100 美元额度，按 5 小时 20%、每周 50% 限流）
    "opencode-go": Provider(
        OPENCODE_GO_URL, "deepseek-v4.1-flash", "OPENCODE_API_KEY", opencode_auth_key
    ),  # fmt: skip
}


def client_for(provider, *, model=None, base_url=None, api_key_env=None, **kw):
    """按预设建 `ChatClient`；显式给的 model / base_url / api_key_env 覆盖预设。"""
    if provider not in PROVIDERS:
        raise ValueError(f"provider 只能是 {sorted(PROVIDERS)} 之一，收到 {provider!r}")
    p = PROVIDERS[provider]
    return ChatClient(model or p.model, base_url=base_url or p.base_url,
                      api_key_env=api_key_env or p.key_env,
                      key_fallback=p.key_fallback, **kw)  # fmt: skip


DEFAULT_EMBED_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_EMBED_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBED_KEY_ENV = "OPENROUTER_API_KEY"


class EmbeddingClient(ChatClient):
    """OpenAI 兼容的 ``/embeddings`` 客户端（默认 OpenRouter 上的 qwen3-embedding-8b，4096 维），
    与聊天客户端共用密钥处理、超时和重试。``embed(texts)`` 返回 (float32 向量列表, 费用美元)。

    费用：响应 ``usage.cost`` 有效时用它，否则按 ``price_per_m``（每百万输入 token 美元，
    2026-11-14 OpenRouter 价目 0.01）估。"""

    def __init__(self, model=DEFAULT_EMBED_MODEL, *, base_url=DEFAULT_EMBED_BASE_URL,
                 api_key_env=DEFAULT_EMBED_KEY_ENV, price_per_m=0.01, timeout=60.0,
                 **kw):  # fmt: skip
        super().__init__(model, base_url=base_url, api_key_env=api_key_env,
                         timeout=timeout, **kw)  # fmt: skip
        self.price_per_m = float(price_per_m)

    def describe(self):
        return {"model": self.model, "base_url": self.base_url}

    def embed(self, texts):
        import numpy as np

        texts = [str(t) for t in texts]
        if not texts:
            return [], 0.0
        raw, _ = self._call("/embeddings", {"model": self.model, "input": texts})
        try:
            data = json.loads(raw)
            items = sorted(data["data"], key=lambda d: d["index"])
            vectors = [np.asarray(d["embedding"], dtype=np.float32) for d in items]
        except (ValueError, KeyError, TypeError):
            raise LLMError("向量接口返回了无法解析的响应") from None
        if len(vectors) != len(texts) or any(
            v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() for v in vectors
        ):
            raise LLMError("向量接口返回的条数或数值不对")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        reported = usage.get("cost")
        if (isinstance(reported, (int, float)) and not isinstance(reported, bool)
                and math.isfinite(reported) and reported >= 0):  # fmt: skip
            cost = float(reported)
        else:
            tokens = _int(usage.get("prompt_tokens")) or sum(len(t) for t in texts)
            cost = tokens * self.price_per_m / 1e6
        return vectors, cost
