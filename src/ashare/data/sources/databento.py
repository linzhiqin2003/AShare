"""Databento 历史行情的取数层（官方客户端 ``databento``）。**未用真实账号验证。**

拿到账号后第一次该跑的核对清单见 ``docs/FOREIGN_FUTURES.md`` 的「首次核对清单」，最短的三条：

1. ``DatabentoSource(budget_usd=0).estimate("GLBX.MDP3", "definition", ["CL.FUT"], "2026-11-02",
   "2026-11-07", stype_in="parent")`` 只估价不取数，确认费用量级；
2. 给一个小预算（如 1 美元）取 CL 2026-11-02 至 11-06 的 ``definition``、``statistics``、``ohlcv-1h``，
   核对 CLZ6 的 ``expiration`` 是 CME 公布的最后交易日、结算价与 CME 当日公布的结算价逐日相同；
3. 同样取 ``IFEU.IMPACT`` 的 Brent，核对结算价与 ICE 公布值相同、价格单位是美元/桶（不是美分）。

这一层只做三件事：从环境变量 ``DATABENTO_API_KEY`` 取密钥建客户端（不写进任何文件、不打印）；
**取数前先对整个请求计划调官方的费用估算接口** ``metadata.get_cost``，累计估算超过预算就一个都不取并报出
差额；已取到的原始表在本次会话内缓存，重试不重复计费；
把客户端返回的 ``DBNStore`` 转成 polars 表（字段保留 Databento 原名，价格已由客户端按 1e-9 缩放成小数）。
默认预算为 0，即**默认只估不取**，调用方显式给预算才会产生计费请求。字段口径换算在
``ashare.data.futures_foreign`` 里做。

不导入其他 ashare 模块。没装可选依赖时，建客户端会报错并提示 ``pip install "ashare[databento]"``。
"""

import math
import os
import threading
from dataclasses import dataclass, field

import polars as pl

KEY_ENV = "DATABENTO_API_KEY"
EXTRA_HINT = '没有安装 databento；请安装可选依赖：pip install "ashare[databento]"（或 uv sync --extra databento）'


class BudgetExceeded(RuntimeError):
    """估算费用超过预算，**一个请求都没有发出**。

    ``plan_total`` 是完成整个请求计划（不含本次会话里已经取到、缓存着的请求）的估算总价，``spent`` 是此前已
    批准的累计估算，``budget`` 是预算，``shortfall`` = ``spent + plan_total − budget``，即预算至少还要加多少。
    ``requests`` 是 [(请求, 估算美元)]。
    """

    def __init__(self, requests, plan_total, spent, budget):
        self.requests, self.plan_total, self.spent, self.budget = (
            list(requests),
            plan_total,
            spent,
            budget,
        )
        self.shortfall = spent + plan_total - budget
        # 兼容只取一个请求的旧写法
        self.estimate = plan_total
        lines = "；".join(f"{label} {cost:.4f}" for label, cost in self.requests)
        super().__init__(
            f"估算费用不够：完成这次取数还需 {plan_total:.4f} 美元（{len(self.requests)} 个请求：{lines}），"
            f"已批准 {spent:.4f}，预算 {budget:.4f}，至少还差 {self.shortfall:.4f}；未发出任何取数请求。"
            f"确实要取，把 budget_usd 调到不低于 {spent + plan_total:.4f} 后重试（已取到的不会重取）"
        )


def require():
    """导入官方客户端；没装时给出一句看得懂的报错。"""
    try:
        import databento
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return databento


def default_client():
    """用环境变量里的密钥建 ``databento.Historical``；密钥不出现在任何输出里。"""
    databento = require()
    key = os.environ.get(KEY_ENV)
    if not key:
        raise RuntimeError(
            f"环境变量 {KEY_ENV} 没有设置；Databento 密钥只从环境变量读取"
        )
    return databento.Historical(key=key)


def _check_budget(value):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"budget_usd 必须是数（美元），收到 {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"budget_usd 必须是有限的非负数，收到 {value!r}")
    return float(value)


def _check_cost(value, label):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label}: get_cost 返回的不是数：{value!r}；未取数")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: get_cost 返回 {value!r}，不是有限的非负数；未取数")
    return float(value)


@dataclass(frozen=True)
class Request:
    """一次取数请求的参数，名字与 ``timeseries.get_range`` / ``metadata.get_cost`` 的参数一致。"""

    dataset: str
    schema: str
    symbols: tuple
    start: str
    end: str
    stype_in: str = "raw_symbol"

    def label(self):
        return (
            f"{self.dataset} {self.schema} {list(self.symbols)} {self.start}~{self.end}"
        )

    def kwargs(self):
        return dict(
            dataset=self.dataset,
            symbols=list(self.symbols),
            schema=self.schema,
            start=self.start,
            end=self.end,
            stype_in=self.stype_in,
        )


def _as_request(r):
    if isinstance(r, Request):
        return r
    dataset, schema, symbols, start, end, *rest = r
    return Request(dataset, schema, tuple(symbols), start, end, *rest)


@dataclass
class DatabentoSource:
    """带费用闸门的 Databento 历史取数。

    - ``budget_usd`` 是这个对象整个生命周期的累计预算（美元，按 ``get_cost`` 的估算计，必须是有限的非负数），
      默认 0：只估价、不取数。
    - ``fetch_plan(requests)`` 先对计划里**全部**未取过的请求逐个估价（免费），合计加上已批准的累计额不超过
      预算才开始取第一个；否则抛 ``BudgetExceeded``，一个都不取，并给出差额。
    - 取到的原始表缓存在对象里（按请求参数），同一会话里重试或重复请求不会再取、不会再计费。
    - 检查、记账和缓存在一把锁里，多线程共用一个对象时预算不会被并发越过。``spent_usd`` 只应由本对象改。
    - ``client`` 可以传入任何实现了 ``metadata.get_cost``、``timeseries.get_range``、``symbology.resolve``
      的对象（测试用假的）；不传时第一次用到才用环境变量里的密钥建官方客户端。不要 pickle 本对象（客户端里有密钥）。

    例::

        src = DatabentoSource(budget_usd=0)
        src.estimate("GLBX.MDP3", "ohlcv-1h", ["CL.FUT"], "2026-11-02", "2026-11-07", stype_in="parent")
    """

    budget_usd: float = 0.0
    client: object = None
    spent_usd: float = 0.0
    log: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        self.budget_usd = _check_budget(self.budget_usd)

    def _client(self):
        if self.client is None:
            self.client = default_client()
        return self.client

    def _cost(self, request):
        return _check_cost(
            self._client().metadata.get_cost(**request.kwargs()), request.label()
        )

    def estimate(self, dataset, schema, symbols, start, end, *, stype_in="raw_symbol"):
        """只调费用估算接口（不计费），返回美元。"""
        return self._cost(
            Request(dataset, schema, tuple(symbols), start, end, stype_in)
        )

    def fetch_plan(self, requests):
        """按计划取数：先对全部未缓存的请求估价，总价过闸门后才逐个取。返回与 ``requests`` 同序的 polars 表。

        预算为 0 时永远不取数（哪怕估算为 0）：默认不产生任何计费请求。
        """
        requests = [_as_request(r) for r in requests]
        with self._lock:
            budget = _check_budget(self.budget_usd)
            # spent_usd 是公开属性；被改成负数或非有限值等于凭空多出额度，取数前拦下。
            if isinstance(self.spent_usd, bool) or not isinstance(
                self.spent_usd, int | float
            ):
                raise TypeError(f"spent_usd 必须是数，收到 {self.spent_usd!r}")
            if not math.isfinite(self.spent_usd) or self.spent_usd < 0:
                raise ValueError(
                    f"spent_usd 必须是有限的非负数，收到 {self.spent_usd!r}；它只应由本对象改"
                )
            pending = [r for r in dict.fromkeys(requests) if r not in self._cache]
            quotes = [(r, self._cost(r)) for r in pending]
            total = sum(c for _, c in quotes)
            for r, c in quotes:
                self.log.append(dict(request=r.label(), estimate=c))
            if pending and (budget <= 0 or self.spent_usd + total > budget):
                raise BudgetExceeded(
                    [(r.label(), c) for r, c in quotes], total, self.spent_usd, budget
                )
            for r, c in quotes:
                # 先记账再取：请求发出后即使失败也可能已计费，宁可多记。
                self.spent_usd += c
                self._cache[r] = frame_of(
                    self._client().timeseries.get_range(**r.kwargs())
                )
            return [self._cache[r] for r in requests]

    def fetch(self, dataset, schema, symbols, start, end, *, stype_in="raw_symbol"):
        """单个请求的 ``fetch_plan``。"""
        return self.fetch_plan(
            [Request(dataset, schema, tuple(symbols), start, end, stype_in)]
        )[0]

    def resolve(self, dataset, symbols, stype_in, stype_out, start_date, end_date=None):
        """符号映射（``symbology.resolve``，官方说明 symbology 类调用免费），用于把连续合约记法对到具体合约；
        原样返回官方的字典。"""
        return self._client().symbology.resolve(
            dataset=dataset,
            symbols=list(symbols),
            stype_in=stype_in,
            stype_out=stype_out,
            start_date=start_date,
            end_date=end_date,
        )


def frame_of(store):
    """``DBNStore`` → polars：``to_df()`` 默认价格为小数、时间为 UTC 的 datetime，索引列放回成普通列。"""
    df = store.to_df()
    return pl.from_pandas(df.reset_index())


def mapping_frame(resolved):
    """``symbology.resolve`` 的返回 → 长表 (continuous, start_date, end_date, instrument_id)。

    官方返回形如 ``{"result": {"CL.c.0": [{"d0": "2026-11-02", "d1": "2026-11-19", "s": "42"}]}, ...}``；
    d0 含、d1 不含（以官方文档为准，未用真实账号验证）。
    """
    rows = [
        dict(
            continuous=symbol,
            start_date=item["d0"],
            end_date=item["d1"],
            instrument_id=int(item["s"]),
        )
        for symbol, items in resolved.get("result", {}).items()
        for item in items
        if str(item.get("s", "")).isdigit()
    ]
    return pl.DataFrame(
        rows,
        schema=dict(
            continuous=pl.String,
            start_date=pl.String,
            end_date=pl.String,
            instrument_id=pl.Int64,
        ),
    )


__all__ = [
    "BudgetExceeded",
    "DatabentoSource",
    "KEY_ENV",
    "Request",
    "frame_of",
    "mapping_frame",
    "require",
]
