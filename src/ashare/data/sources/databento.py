"""Databento 历史行情的取数层（官方客户端 ``databento``）。**未用真实账号验证。**

拿到账号后第一次该跑的核对清单见 ``docs/FOREIGN_FUTURES.md`` 的「首次核对清单」，最短的三条：

1. ``DatabentoSource(budget_usd=0).estimate("GLBX.MDP3", "definition", ["CL.FUT"], "2026-11-02",
   "2026-11-07", stype_in="parent")`` 只估价不取数，确认费用量级；
2. 给一个小预算（如 1 美元）取 CL 2026-11-02 至 11-06 的 ``definition``、``statistics``、``ohlcv-1h``，
   核对 CLZ6 的 ``expiration`` 是 CME 公布的最后交易日、结算价与 CME 当日公布的结算价逐日相同；
3. 同样取 ``IFEU.IMPACT`` 的 Brent，核对结算价与 ICE 公布值相同、价格单位是美元/桶（不是美分）。

这一层只做三件事：从环境变量 ``DATABENTO_API_KEY`` 取密钥建客户端（不写进任何文件、不打印）；
**每次取数前先调官方的费用估算接口** ``metadata.get_cost``，累计估算超过预算就不取并报出估算值；
把客户端返回的 ``DBNStore`` 转成 polars 表（字段保留 Databento 原名，价格已由客户端按 1e-9 缩放成小数）。
默认预算为 0，即**默认只估不取**，调用方显式给预算才会产生计费请求。字段口径换算在
``ashare.data.futures_foreign`` 里做。

不导入其他 ashare 模块。没装可选依赖时，建客户端会报错并提示 ``pip install "ashare[databento]"``。
"""

import os
from dataclasses import dataclass, field

import polars as pl

KEY_ENV = "DATABENTO_API_KEY"
EXTRA_HINT = '没有安装 databento；请安装可选依赖：pip install "ashare[databento]"（或 uv sync --extra databento）'


class BudgetExceeded(RuntimeError):
    """估算费用超过预算，请求没有发出。``estimate`` 是本次请求的估算（美元），``spent`` 是此前已批准的累计估算。"""

    def __init__(self, request, estimate, spent, budget):
        self.request, self.estimate, self.spent, self.budget = (
            request,
            estimate,
            spent,
            budget,
        )
        super().__init__(
            f"{request}: 估算费用 {estimate:.4f} 美元，已批准 {spent:.4f}，预算 {budget:.4f}；"
            "未发出取数请求。确实要取，请把 budget_usd 调到不低于两者之和"
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


@dataclass
class Request:
    """一次取数请求的参数，名字与 ``timeseries.get_range`` / ``metadata.get_cost`` 的参数一致。"""

    dataset: str
    schema: str
    symbols: list
    start: str
    end: str
    stype_in: str = "raw_symbol"

    def label(self):
        return f"{self.dataset} {self.schema} {self.symbols} {self.start}~{self.end}"

    def kwargs(self):
        return dict(
            dataset=self.dataset,
            symbols=list(self.symbols),
            schema=self.schema,
            start=self.start,
            end=self.end,
            stype_in=self.stype_in,
        )


@dataclass
class DatabentoSource:
    """带费用闸门的 Databento 历史取数。

    ``budget_usd`` 是这个对象整个生命周期的累计预算（美元，按 ``get_cost`` 的估算计），默认 0：
    ``fetch`` 只估价、不取数，抛 ``BudgetExceeded`` 并报出估算值。``client`` 可以传入任何实现了
    ``metadata.get_cost``、``timeseries.get_range``、``symbology.resolve`` 的对象（测试用假的）；不传时
    第一次用到才用环境变量里的密钥建官方客户端。

    例::

        src = DatabentoSource(budget_usd=0)
        src.estimate("GLBX.MDP3", "ohlcv-1h", ["CL.FUT"], "2026-11-02", "2026-11-07", stype_in="parent")
    """

    budget_usd: float = 0.0
    client: object = None
    spent_usd: float = 0.0
    log: list = field(default_factory=list)

    def _client(self):
        if self.client is None:
            self.client = default_client()
        return self.client

    def estimate(self, dataset, schema, symbols, start, end, *, stype_in="raw_symbol"):
        """只调费用估算接口（不计费），返回美元。"""
        request = Request(dataset, schema, list(symbols), start, end, stype_in)
        return float(self._client().metadata.get_cost(**request.kwargs()))

    def fetch(self, dataset, schema, symbols, start, end, *, stype_in="raw_symbol"):
        """先估价，累计估算不超过预算才调 ``timeseries.get_range``；返回 polars 表（原字段名）。

        预算为 0 时永远不取数（哪怕估算为 0）：默认不产生任何计费请求。
        """
        request = Request(dataset, schema, list(symbols), start, end, stype_in)
        cost = float(self._client().metadata.get_cost(**request.kwargs()))
        self.log.append(dict(request=request.label(), estimate=cost))
        if self.budget_usd <= 0 or self.spent_usd + cost > self.budget_usd:
            raise BudgetExceeded(request.label(), cost, self.spent_usd, self.budget_usd)
        self.spent_usd += cost
        store = self._client().timeseries.get_range(**request.kwargs())
        return frame_of(store)

    def resolve(self, dataset, symbols, stype_in, stype_out, start_date, end_date=None):
        """符号映射（``symbology.resolve``），用于把连续合约记法对到具体合约；原样返回官方的字典。"""
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
    "frame_of",
    "mapping_frame",
    "require",
]
