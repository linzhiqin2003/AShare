"""异步 Tushare 客户端：单次轮询和分页采集共享账号节流器。"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
import weakref
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from .base import FetchResult, SourceError

LOG = logging.getLogger(__name__)
URL = "https://api.tushare.pro"


class TokenBucket:
    """容量为一的令牌桶，避免突发；同一事件循环中的客户端共享实例。"""

    def __init__(self, rate=300.0, *, clock=time.monotonic, sleep=asyncio.sleep):
        if rate <= 0:
            raise ValueError("rate 必须大于零")
        self.rate, self.clock, self.sleep = rate, clock, sleep
        self.ceiling, self.successes = rate, 0
        self.last_reduction = -float("inf")
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            delay = max(0, self.next_at - self.clock())
            if delay:
                await self.sleep(delay)
            self.next_at = self.clock() + 60 / self.rate

    def slow_down(self):
        if self.clock() - self.last_reduction >= 60:
            self.rate = max(1, self.rate / 2)
            self.last_reduction = self.clock()
        self.next_at = max(self.next_at, self.clock() + 60)
        self.successes = 0

    def success(self):
        self.successes += 1
        if self.successes >= 64:
            self.rate = min(self.ceiling, self.rate * 1.25)
            self.successes = 0


_BUCKETS = weakref.WeakKeyDictionary()


def classify(code, message):
    """实测接口名错误为 40101；其他错误按消息保守分类，未知错误不重试。"""
    if "每天最多" in message or "每日上限" in message or "每天限" in message:
        return "daily_quota"
    if code == 40203 or any(
        word in message for word in ("每分钟", "频率超限", "每小时")
    ):
        return "rate_limit"
    if any(word in message for word in ("权限", "积分", "token", "Token")):
        return "permission"
    if any(
        word in message
        for word in ("接口不存在", "接口名", "无效的接口", "正确的接口", "没有找到")
    ):
        return "api_name"
    return "api_error"


class Tushare:
    def __init__(
        self,
        *,
        rate=300,
        concurrency=4,
        timeout=30,
        retries=4,
        transport=None,
        limiter=None,
        sleep=asyncio.sleep,
        state_path: Path | None = None,
        budget=3000,
    ):
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise SourceError(
                "configuration", "permission", message="请设置环境变量 TUSHARE_TOKEN"
            )
        self._token = token
        self._key = hashlib.sha256(token.encode()).hexdigest()[:16]
        self.limiter = limiter
        self.limiters = {}
        self.rate, self.sleep, self.retries = rate, sleep, retries
        self.state_path, self.budget = state_path, budget
        self.calls = 0
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": "AShare-Research/0.1"},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.http.aclose()

    def safe(self, value):
        return str(value).replace(self._token, "[REDACTED]")

    def bucket(self, api):
        if self.limiter is not None:
            return self.limiter
        if api not in self.limiters:
            buckets = _BUCKETS.setdefault(
                asyncio.get_running_loop(), weakref.WeakValueDictionary()
            )
            key = (self._key, api)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.rate, sleep=self.sleep)
                buckets[key] = bucket
            self.limiters[api] = bucket
        return self.limiters[api]

    async def _acquire(self, api):
        limiter = self.bucket(api)
        await limiter.acquire()
        if self.calls >= self.budget:
            raise SourceError(
                api, "budget", message=f"本次运行调用预算 {self.budget} 已耗尽"
            )
        self.calls += 1
        if self.state_path is not None:
            # 仅持久化各接口的预约时间；旧账本的终身 calls 和 rate 不再影响本次运行。
            import fcntl

            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.with_suffix(".lock").open("a+") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                try:
                    state = (
                        json.loads(self.state_path.read_text())
                        if self.state_path.exists()
                        else {}
                    )
                except (ValueError, OSError):
                    state = {}  # 旧账本被截断时重新建预约，不再永久阻塞客户端。
                slots = state.get("interfaces", {})
                reserved = max(time.time(), slots.get(api, 0))
                slots[api] = reserved + 60 / limiter.rate
                temporary = self.state_path.with_name(
                    f".{self.state_path.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("w") as stream:
                        stream.write(json.dumps({"interfaces": slots}))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.state_path)
                finally:
                    temporary.unlink(missing_ok=True)
            await self.sleep(max(0, reserved - time.time()))

    async def _page(self, api, params, fields):
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            retry_category = None
            await self._acquire(api)
            async with self.sem:
                try:
                    response = await self.http.post(
                        URL,
                        json=dict(
                            api_name=api,
                            token=self._token,
                            params=params,
                            fields=fields,
                        ),
                    )
                except httpx.TransportError:
                    retry_category = "network"
                else:
                    if response.status_code >= 500 or response.status_code == 429:
                        retry_category = (
                            "rate_limit" if response.status_code == 429 else "server"
                        )
                    elif response.status_code != 200:
                        raise SourceError(api, "http", response.status_code) from None
                    else:
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = {"code": None, "msg": "响应不是 JSON"}
                        code = payload.get("code")
                        if code != 0:
                            message = self.safe(payload.get("msg", ""))
                            category = (
                                "protocol" if code is None else classify(code, message)
                            )
                            if category not in ("rate_limit", "protocol"):
                                raise SourceError(
                                    api, category, code, message
                                ) from None
                            retry_category = category
                        else:
                            data = payload.get("data") or {}
                            columns, rows = (
                                data.get("fields") or [],
                                data.get("items") or [],
                            )
                            if self._token in json.dumps(data, ensure_ascii=False):
                                raise SourceError(
                                    api, "protocol", message="响应含鉴权信息，拒绝保存"
                                )
                            table = pl.DataFrame(
                                rows,
                                schema=columns,
                                orient="row",
                                infer_schema_length=None,
                            )
                            LOG.info(
                                "api=%s params=%s rows=%d elapsed=%.3f retries=%d",
                                api,
                                self.safe(params),
                                len(table),
                                time.monotonic() - started,
                                attempt,
                            )
                            self.bucket(api).success()
                            return table, attempt
            if retry_category == "rate_limit":
                self.bucket(api).slow_down()
            if attempt == self.retries:
                raise SourceError(
                    api, retry_category, message=f"已重试 {attempt} 次"
                ) from None
            await self.sleep(
                max(
                    60 if retry_category == "rate_limit" else 0,
                    min(30, 2**attempt),
                )
            )
        raise AssertionError("不可达")

    async def fetch(self, api, params, *, fields="", page_size=None, timeout=None):
        """timeout 是包含节流、重试及翻页的总时限，适合实时轮询。"""
        try:
            async with asyncio.timeout(timeout):
                return await self._fetch(
                    api, params, fields=fields, page_size=page_size
                )
        except TimeoutError:
            raise SourceError(api, "timeout", message="单次调用总时限已到") from None

    async def _fetch(self, api, params, *, fields="", page_size=None):
        """page_size=None 为单次轮询；恰好满页时必须再读一页。"""
        pages, retry_count, calls = [], 0, 0
        offset = 0
        hashes = set()
        while True:
            query = dict(params)
            if page_size:
                query.update(limit=page_size, offset=offset)
            table, retried = await self._page(api, query, fields)
            retry_count += retried
            calls += retried + 1
            pages.append(table)
            if not page_size or len(table) < page_size:
                break
            digest = hashlib.sha256(table.write_json().encode()).hexdigest()
            if digest in hashes:
                raise SourceError(
                    api, "pagination", message="源忽略 offset，拒绝静默截断"
                )
            hashes.add(digest)
            offset += len(table)
        table = pl.concat(pages, how="diagonal_relaxed")
        return FetchResult(table, datetime.now(UTC).isoformat(), calls, retry_count)
