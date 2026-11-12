"""行情源、单一网络出口、令牌桶与不含接口原文的结构化告警。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import polars as pl
from paperbroker import COLUMNS, QuoteArray

from .market import RT_FIELDS, TZ, normalize, rows, timestamp

LOG = logging.getLogger("ashare.paper")


def log(event: str, level: int = logging.INFO, **values: object) -> None:
    LOG.log(
        level,
        json.dumps(dict(event=event, **values), ensure_ascii=False, sort_keys=True),
    )


class Clock(Protocol):
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(TZ)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0, seconds))


class TokenBucket:
    def __init__(
        self, per_minute: float = 40, burst: float = 1, clock: Clock | None = None
    ):
        if not 0 < per_minute <= 500 or burst < 1 or burst > per_minute:
            raise ValueError("无效令牌桶配置")
        self.rate, self.capacity = per_minute / 60, burst
        self.clock = clock or SystemClock()
        self.tokens, self.updated = burst, self.clock.monotonic()

    def acquire(self) -> None:
        while True:
            now = self.clock.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + max(0, now - self.updated) * self.rate
            )
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            self.clock.sleep(max(0.001, (1 - self.tokens) / self.rate))


class SourceError(RuntimeError):
    pass


class TushareClient:
    """同进程只允许一个实例；跨进程通过同一锁文件排他，所有接口串行调用。"""

    _guard = threading.Lock()
    _active = False

    def __init__(
        self,
        lock_path: Path,
        *,
        clock: Clock | None = None,
        transport: httpx.BaseTransport | None = None,
        retries: int = 3,
        per_minute: float = 40,
    ):
        import fcntl

        if retries < 0:
            raise ValueError("重试次数不能为负")
        self.clock = clock or SystemClock()
        if per_minute > 50:
            raise ValueError("rt_k 每分钟限额不能高于 50")
        self.bucket = TokenBucket(per_minute, clock=self.clock)
        self.buckets: dict[str, TokenBucket] = {"rt_k": self.bucket}
        self._closed = False
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise SourceError("TUSHARE_TOKEN 未设置")
        self.retries = retries
        self._serial = threading.Lock()
        with self._guard:
            if self._active:
                raise SourceError("行情出口已占用")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = lock_path.open("a+b")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._client = httpx.Client(
                    transport=transport,
                    timeout=20,
                    headers={"User-Agent": "AShare-Paper/0.1"},
                )
            except BaseException:
                self._lock.close()
                raise SourceError("行情出口锁不可用") from None
            type(self)._active = True
        self._token = token
        self.failures = 0

    def call(
        self, api_name: str, params: dict | None = None, fields: str = ""
    ) -> list[dict]:
        with self._serial:
            for attempt in range(self.retries + 1):
                self.buckets.setdefault(
                    api_name,
                    TokenBucket(
                        40 if api_name.startswith("rt_") else 400, clock=self.clock
                    ),
                ).acquire()
                category = "transport"
                try:
                    response = self._client.post(
                        "https://api.tushare.pro",
                        json=dict(
                            api_name=api_name,
                            token=self._token,
                            params=params or {},
                            fields=fields,
                        ),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("code") == 0:
                        result = rows(payload)
                        self.failures = 0
                        return result
                    msg = str(payload.get("msg", ""))
                    limited = payload.get("code") == 40203 or any(
                        phrase in msg
                        for phrase in (
                            "频率超限",
                            "每分钟最多",
                            "限流",
                            "rate limit exceeded",
                            "频率超过",
                        )
                    )
                    category = "rate_limit" if limited else "api_error"
                    if not limited:
                        self.failures += 1
                        log(
                            "source_failure",
                            logging.ERROR,
                            category=category,
                            consecutive=self.failures,
                        )
                        raise SourceError(
                            f"Tushare API 拒绝请求: api={api_name}, code={payload.get('code')}"
                        )
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    category = "transport_or_schema"
                self.failures += 1
                log(
                    "source_failure",
                    logging.WARNING,
                    category=category,
                    attempt=attempt + 1,
                    consecutive=self.failures,
                )
                if self.failures >= 3:
                    log("source_outage", logging.ERROR, consecutive=self.failures)
                if attempt < self.retries:
                    self.clock.sleep(
                        max(60 if category == "rate_limit" else 0, 2**attempt)
                    )
            raise SourceError("行情请求重试耗尽") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()
        self._lock.close()
        with self._guard:
            type(self)._active = False

    def __enter__(self) -> TushareClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass
class Batch:
    received_at: int
    quotes: QuoteArray
    source: str
    sent_at: int | None = None
    symbols: list[str] | None = None


class MarketSource(Protocol):
    def poll(self) -> Batch | None: ...


class LiveSource:
    def __init__(
        self,
        client: TushareClient,
        symbols: list[str],
        *,
        full_market: bool = False,
        interval: float | None = None,
    ):
        self.client, self.symbols = client, symbols
        self.full_market = full_market
        self.patterns = (
            ["6*.SH", "3*.SZ,0*.SZ", "9*.BJ,8*.BJ,4*.BJ"]
            if full_market
            else [",".join(symbols)]
        )
        self.interval = max(
            6 if full_market else 2, interval or (6 if full_market else 3)
        )
        self.limits: dict[str, tuple[float, float]] = {}
        self.suspended: set[str] = set()
        self.suspension_intervals: dict[str, list[tuple[str, str]]] = {}
        self.day: str | None = None

    def poll(self) -> Batch:
        sent_at = timestamp(self.client.clock.now())
        records = []
        for pattern in self.patterns:
            records.extend(self.client.call("rt_k", {"ts_code": pattern}, RT_FIELDS))
        if self.full_market:
            self.symbols = self.symbols + sorted(
                {r["ts_code"] for r in records} - set(self.symbols)
            )
        return Batch(
            timestamp(self.client.clock.now()),
            normalize(
                records,
                self.symbols,
                limits=self.limits,
                suspended=self.suspended,
                suspension_intervals=self.suspension_intervals,
                day=self.day or self.client.clock.now().strftime("%Y%m%d"),
            ),
            "rt_k",
            sent_at,
            list(self.symbols),
        )


class ReplaySource:
    def __init__(self, directory: Path, day: str | None = None):
        self.paths = iter(sorted(directory.glob("*.parquet")))
        self.day = day

    def poll(self) -> Batch | None:
        for path in self.paths:
            table = pl.read_parquet(path)
            if table.height == 0:
                continue
            if "schema_version" not in table.columns or table["schema_version"][0] != 2:
                raise ValueError("录制格式版本不支持")
            received = table["received_at"][0]
            if (
                self.day
                and datetime.fromtimestamp(received / 1000, TZ).strftime("%Y%m%d")
                != self.day
            ):
                continue
            return Batch(
                received,
                np.ascontiguousarray(table.select(COLUMNS).to_numpy(), dtype=np.int64),
                str(table["source"][0]),
                int(table["sent_at"][0]) if "sent_at" in table.columns else None,
            )
        return None


class MinuteSource:
    """分钟收盘时发快照，不把本分钟尚未结束的价格提前用于成交。"""

    def __init__(
        self,
        records: list[dict],
        symbols: list[str],
        *,
        limits: dict | None = None,
        suspended: set | None = None,
    ):
        self.symbols, self.limits, self.suspended = symbols, limits, suspended
        self.groups: dict[str, list[dict]] = {}
        for r in records:
            self.groups.setdefault(r["trade_time"], []).append(r)
        self.times: Iterator[str] = iter(sorted(self.groups))
        self.totals: dict[str, tuple[float, float]] = {}
        self.opens: dict[str, float] = {}
        self.highs: dict[str, float] = {}
        self.lows: dict[str, float] = {}

    def poll(self) -> Batch | None:
        when = next(self.times, None)
        if when is None:
            return None
        records = []
        for raw in sorted(self.groups[when], key=lambda r: r["ts_code"]):
            r = raw.copy()
            s = r["ts_code"]
            volume, amount = self.totals.get(s, (0, 0))
            self.totals[s] = volume + r["vol"], amount + r["amount"]
            self.opens.setdefault(s, r["open"])
            self.highs[s] = max(self.highs.get(s, r["high"]), r["high"])
            self.lows[s] = min(self.lows.get(s, r["low"]), r["low"])
            r.update(
                vol=self.totals[s][0],
                amount=self.totals[s][1],
                open=self.opens[s],
                high=self.highs[s],
                low=self.lows[s],
            )
            records.append(r)
        return Batch(
            timestamp(when),
            normalize(
                records, self.symbols, limits=self.limits, suspended=self.suspended
            ),
            "stk_mins_synthetic",
        )


class Monitor:
    def __init__(self, clock: Clock, stale_seconds: float = 60):
        self.clock, self.threshold = clock, stale_seconds
        self.seen: dict[int, tuple[int, float]] = {}
        self.last_alert = float("-inf")
        self.alerting = False

    def observe(self, batch: Batch) -> None:
        now = self.clock.monotonic()
        for row in batch.quotes:
            symbol, ts = int(row[0]), int(row[1])
            if row[15]:
                self.seen.pop(symbol, None)
                continue
            previous, changed = self.seen.get(symbol, (-1, now))
            if ts > previous:
                changed = now
            self.seen[symbol] = max(previous, ts), changed
        stale = [
            s
            for s, (ts, changed) in self.seen.items()
            if now - changed >= self.threshold
            or batch.received_at - ts >= self.threshold * 1000
        ]
        unhealthy = bool(stale) or not len(batch.quotes)
        if unhealthy and (not self.alerting or now - self.last_alert >= 300):
            self.last_alert = now
            log(
                "stale_quotes",
                logging.ERROR,
                count=len(stale),
                empty=len(batch.quotes) == 0,
            )

        self.alerting = unhealthy
