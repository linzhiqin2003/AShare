"""境外期货（WTI、布伦特等）的公共部分：品种表、代码解析、交易日归属、分钟线字段、落盘与接回测。

两个来源的适配器在 ``futures_foreign_databento``（Databento）和 ``futures_foreign_bloomberg``（Bloomberg）；
它们交出与境内期货相同的统一字段（``futures_schema``），所以面板、连续合约、``panelbt_inputs`` 都不用改。
**两个适配器都未用真实账号验证**，口径、出处和首次核对清单见 ``docs/FOREIGN_FUTURES.md``。

最短例子（合成数据，不联网）::

    from ashare.data.futures_foreign import build_panel
    fut = build_panel(adapter, "/tmp/foreign")        # adapter 是两个适配器之一
    fut.continuous("CL", rule="open_interest")        # 由本仓库的换月规则拼连续合约

合约标识：品种 + 四位年月 + . + 交易所，``CL2612.NYMEX``、``B2612.ICE``、``BZ2612.NYMEX``。
"""

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .futures_schema import Column, conform

MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class Session:
    """交易时段归属规则：本地时间 ``cutoff`` 点（含）以后开始的成交归下一个交易日（再顺延过周末和休市日）。"""

    name: str
    tz: str
    cutoff_hour: int


# CME Globex：周日至周五 17:00（芝加哥时间）开盘，次日 16:00 收盘，交易日是收盘那天。
CME = Session("cme_globex", "America/Chicago", 17)
# ICE Futures Europe 布伦特：伦敦时间 01:00–23:00，周日 23:00 开盘归周一（futures_rules.md 第二部分）。
ICE_LONDON = Session("ice_london", "Europe/London", 23)


@dataclass(frozen=True)
class Product:
    code: str  # 统一品种代码
    exchange: str  # 统一交易所缩写
    name: str
    multiplier: float  # 每手对应的报价单位数量（桶）
    quote_unit: str
    tick_size: float
    settlement_type: str
    currency: str
    session: Session
    databento_dataset: str | None
    databento_root: str | None  # Databento 母代码（parent 记法 ``<root>.FUT``）
    bloomberg_root: str | None  # Bloomberg 代码前缀（``CLZ6 Comdty`` 的 CL）


PRODUCTS = {
    p.code: p
    for p in (
        Product(
            "CL",
            "NYMEX",
            "WTI 原油（NYMEX Light Sweet Crude）",
            1000.0,
            "美元/桶",
            0.01,
            "physical",
            "USD",
            CME,
            "GLBX.MDP3",
            "CL",
            "CL",
        ),
        Product(
            "BZ",
            "NYMEX",
            "布伦特原油（NYMEX Brent Last Day Financial）",
            1000.0,
            "美元/桶",
            0.01,
            "cash",
            "USD",
            CME,
            "GLBX.MDP3",
            "BZ",
            None,
        ),
        Product(
            "B",
            "ICE",
            "布伦特原油（ICE Brent Crude）",
            1000.0,
            "美元/桶",
            0.01,
            "cash",
            "USD",
            ICE_LONDON,
            "IFEU.IMPACT",
            "BRN",
            "CO",
        ),
    )
}


def unified_symbol(product, year, month):
    return f"{product}{year % 100:02d}{month:02d}.{PRODUCTS[product].exchange}"


def resolve_year(digits, month, ref):
    """一位或两位年份 → 四位年份：取「交割月不早于参照日所在月」的最小年份。

    ``ref`` 是合约在场的某一天（``datetime.date`` 或 YYYYMMDD）：最好用最后交易日，因为交割月最多晚于
    最后交易日两个月（WTI 一个月、布伦特两个月），这样十年回绕不会错；用观测日时，十年以上的远月合约
    与同名近月合约无法区分（Bloomberg、CME 一位年份代码的固有歧义）。
    """
    ref = _as_date(ref)
    modulus = 10 ** len(digits)
    year = ref.year - ref.year % modulus + int(digits)  # 与参照年同一个十年（或百年）
    if (year, month) < (ref.year, ref.month):
        year += modulus
    return year


def _as_date(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).replace("-", "")
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def trade_dates(ts, session, open_days=None, known=None):
    """UTC 时刻 → 交易日（YYYYMMDD），按 ``session`` 的时区和切换点（polars 向量化）。

    1. 本地时间 ≥ cutoff 的归本地日期的下一天，再顺延过周六周日，得到「自然交易日」。CME 周日 17:00（芝加哥）
       开盘的成交归周一，与结算价的交易日（Databento ``ts_ref``）一致。
    2. 给了 ``open_days``（开市日集合）时，在已知日历范围 ``known = (首日, 末日)``（含两端，不给就取开市日集合的
       最早、最晚一天）之内顺延到下一个开市日（休市日的成交归下一开市日）。**自然交易日落在已知范围之外、或顺延
       越过末日的，返回空**：不知道那几天开不开市，就不猜，由调用方丢弃并计数（不会被顺延到范围的第一天）。

    ``ts`` 是 ``datetime`` 列表（无时区的按 UTC）或 polars 的时间列；返回同长度的字符串列表（空值仍为空）。
    """
    series = (
        ts
        if isinstance(ts, pl.Series)
        else pl.Series("ts", list(ts), dtype=pl.Datetime("us", "UTC"))
    )
    if series.dtype.time_zone is None:
        series = series.dt.replace_time_zone("UTC")
    local = pl.col("ts").dt.convert_time_zone(session.tz)
    weekday = pl.col("day").dt.weekday()  # 周一 = 1 … 周日 = 7
    frame = (
        pl.DataFrame({"ts": series})
        .select(
            day=local.dt.date()
            + pl.duration(days=(local.dt.hour() >= session.cutoff_hour).cast(pl.Int64))
        )
        .with_columns(
            day=pl.col("day")
            + pl.duration(
                days=pl.when(weekday == 6)
                .then(2)
                .when(weekday == 7)
                .then(1)
                .otherwise(0)
            )
        )
    )
    text = frame.select(pl.col("day").dt.strftime("%Y%m%d"))["day"]
    if open_days is None:
        return text.to_list()
    ordered = sorted(open_days)
    if known is None:
        if not ordered:
            return [None] * len(text)
        known = (ordered[0], ordered[-1])
    first, last = known
    keyed = pl.DataFrame({"key": text, "row": range(len(text))})
    inside = keyed.filter(
        pl.col("key").is_not_null()
        & pl.col("key").is_between(pl.lit(first), pl.lit(last))
    )
    found = (
        inside.sort("key")
        .join_asof(
            pl.DataFrame({"open": ordered}, schema={"open": pl.String}).with_columns(
                on=pl.col("open")
            ),
            left_on="key",
            right_on="on",
            strategy="forward",
        )
        .with_columns(open=pl.when(pl.col("open") <= last).then(pl.col("open")))
    )
    return (
        keyed.join(found.select("row", "open"), on="row", how="left")
        .sort("row")["open"]
        .to_list()
    )


# ---------------------------------------------------------------- 分钟线（统一字段新增，境内表里还没有）
MINUTE = {
    c.name: c
    for c in (
        Column("ts", pl.Datetime("us", "UTC"), "K 线开始时刻，UTC", "K 线结束后"),
        Column(
            "date",
            pl.String,
            "YYYYMMDD；所属交易日（交易所口径，CME 前一天傍晚起的成交归次日）",
            "—",
        ),
        Column("symbol", pl.String, "统一合约标识", "—"),
        Column("open", pl.Float64, "报价单位", "K 线结束后"),
        Column("high", pl.Float64, "报价单位", "K 线结束后"),
        Column("low", pl.Float64, "报价单位", "K 线结束后"),
        Column("close", pl.Float64, "报价单位", "K 线结束后"),
        Column(
            "volume",
            pl.Float64,
            "手，单边；只含电子盘成交（不含大宗、EFP）",
            "K 线结束后",
        ),
    )
}


def conform_minute(frame):
    """分钟线按 ``MINUTE`` 检查、排列、转型；缺列或多列报错。"""
    missing = [k for k in MINUTE if k not in frame.columns]
    extra = sorted(set(frame.columns) - MINUTE.keys())
    if missing or extra:
        raise ValueError(f"minute: 缺少必填列 {missing}，多余列 {extra}")
    return frame.select([pl.col(c.name).cast(c.dtype) for c in MINUTE.values()]).sort(
        "symbol", "ts"
    )


# ---------------------------------------------------------------- 统一表的公共片段
def calendar_from_open_days(exchange, open_days, start, end):
    """开市日集合 → 日历表（区间内每个自然日一行）。来源没有日历接口时，开市日取「有结算价的日子」。"""
    first, last = _as_date(start), _as_date(end)
    opened = set(open_days)
    days = [first + dt.timedelta(days=k) for k in range((last - first).days + 1)]
    texts = [d.strftime("%Y%m%d") for d in days]
    return conform(
        "calendar",
        pl.DataFrame(
            dict(
                exchange=[exchange] * len(texts),
                date=texts,
                is_open=[t in opened for t in texts],
            )
        ),
    )


def previous_settlement(daily, contracts, calendar):
    """加 ``previous_settlement``：该合约所在交易所**上一开市日**的结算价（按日历，不是合约自己的上一行）；
    上一开市日没有该合约的结算价则为空（新合约首日同样为空，来源没有挂牌基准价）。"""
    open_days = (
        calendar.filter(pl.col("is_open"))
        .select("exchange", "date")
        .sort("exchange", "date")
    )
    prev = open_days.with_columns(prev=pl.col("date").shift(1).over("exchange"))
    keyed = daily.join(
        contracts.select("symbol", "exchange"), on="symbol", how="left"
    ).join(prev, on=["exchange", "date"], how="left")
    lookup = daily.select(
        prev="date", symbol="symbol", previous_settlement="settlement"
    )
    return keyed.join(lookup, on=["prev", "symbol"], how="left").drop(
        "prev", "exchange"
    )


def check_settlements(daily, contracts, *, require=True):
    """结算价完整性：返回每个品种区间内结算价为空的行数（进诊断）。

    空结算价不往下传（日历推不出开市日、``previous_settlement`` 全空、按持仓量换月失灵，都是静默的）：

    - 某个品种区间内有行情行，但一个结算价都没有（比如来源的结算价字段或标志位口径与预期不同）：总是报错；
    - 某个合约某天有成交（``volume > 0`` 或有开盘价），却没有最终结算价：``require=True``（默认）时报错。
      确实要放行（比如结算价还没公布的当天），传 ``require=False``，空值计数仍在诊断里。
    """
    keyed = daily.join(contracts.select("symbol", "product"), on="symbol", how="left")
    counts = keyed.group_by("product").agg(
        rows=pl.len(), missing=pl.col("settlement").is_null().sum()
    )
    nulls = {r["product"]: int(r["missing"]) for r in counts.iter_rows(named=True)}
    empty = sorted(
        r["product"]
        for r in counts.iter_rows(named=True)
        if r["rows"] > 0 and r["missing"] == r["rows"]
    )
    if empty:  # 不受 require 控制：整个品种没有结算价时日历都推不出来
        raise ValueError(
            f"品种 {empty} 在区间内一个结算价都没有：来源的结算价口径可能与预期不同，"
            "先按 docs/FOREIGN_FUTURES.md 的首次核对清单核对"
        )
    if require:
        traded = keyed.filter(
            pl.col("settlement").is_null()
            & ((pl.col("volume") > 0) | pl.col("open").is_not_null())
        )
        if len(traded):
            sample = traded.select("symbol", "date").sort("symbol", "date").head(10)
            raise ValueError(
                f"{len(traded)} 个有成交的合约日没有最终结算价（前几个：{sample.rows()}）；"
                "确实要放行传 require_settlement=False"
            )
    return nulls


def params_rows(daily, contracts, margin):
    """统一参数表：来源没有交易所逐日保证金和费用。``margin`` 是调用方给的保证金率（标量，或「品种 → 比例」），
    不给就留空；费用一律留空，回测里显式传。"""
    base = daily.select("date", "symbol").join(
        contracts.select("symbol", "product"), on="symbol"
    )
    if isinstance(margin, dict):
        rate = pl.col("product").replace_strict(
            margin, default=None, return_dtype=pl.Float64
        )
    else:
        rate = pl.lit(margin, dtype=pl.Float64)
    return conform(
        "params",
        base.with_columns(margin_long=rate, margin_short=rate).drop("product"),
    ).sort("symbol", "date")


def inside_listing(frame, contracts):
    """只留上市期内的行；返回 (留下的, 丢掉的行数)。"""
    span = frame.join(
        contracts.select("symbol", "listing_date", "last_trade_date"),
        on="symbol",
        how="inner",
    )
    keep = span.filter(
        pl.col("date").is_between(pl.col("listing_date"), pl.col("last_trade_date"))
    )
    return keep.drop("listing_date", "last_trade_date"), len(frame) - len(keep)


def content_name(prefix, tables):
    """适配器名字带上输出内容的哈希：``build_futures`` 的版本号由清单（境外为空）、代码和来源名组成，
    同名来源换了数据也会复用旧面板；把内容哈希放进名字就不会。"""
    digest = hashlib.sha256()
    for name in sorted(tables):
        digest.update(name.encode())
        digest.update(tables[name].write_ipc(None).getvalue())
    return f"{prefix} {digest.hexdigest()[:12]}"


class ForeignAdapter:
    """两个境外适配器的公共外壳：子类实现 ``_compute()``，返回 dict(contracts, daily, calendar,
    mapping?, source_continuous?, diagnostics)。只算一次。"""

    source_name = "foreign"

    def __init__(self, margin=None):
        self.margin = margin
        self._tables = None

    def _all(self):
        if self._tables is None:
            self._tables = self._compute()
        return self._tables

    def contracts(self):
        return self._all()["contracts"]

    def daily(self, start=None, end=None):
        out = self._all()["daily"]
        if start is not None:
            out = out.filter(pl.col("date") >= start)
        if end is not None:
            out = out.filter(pl.col("date") <= end)
        return out

    def params(self, start=None, end=None):
        return params_rows(self.daily(start, end), self.contracts(), self.margin)

    def calendar(self):
        return self._all()["calendar"]

    def mapping(self, start=None, end=None):
        out = self._all().get("mapping")
        if out is None:
            out = conform(
                "mapping",
                pl.DataFrame(
                    schema=dict(
                        date=pl.String,
                        continuous=pl.String,
                        kind=pl.String,
                        symbol=pl.String,
                    )
                ),
            )
        if start is not None:
            out = out.filter(pl.col("date") >= start)
        if end is not None:
            out = out.filter(pl.col("date") <= end)
        return out

    def source_continuous(self):
        """来源自带的连续序列，只用于对账，不进统一字段。"""
        out = self._all().get("source_continuous")
        return out if out is not None else pl.DataFrame()

    def diagnostics(self):
        return self._all().get("diagnostics", {})

    @property
    def name(self):
        tables = {
            k: v
            for k, v in self._all().items()
            if isinstance(v, pl.DataFrame) and not k.startswith("_")
        }
        return content_name(self.source_name, tables)

    def tables(self):
        """统一字段的全部表（含 params），供落盘。"""
        out = dict(
            contracts=self.contracts(),
            daily=self.daily(),
            params=self.params(),
            calendar=self.calendar(),
            mapping=self.mapping(),
        )
        return out


def write_parquet(adapter, folder, *, minute=None):
    """把统一字段的表写成 ``<folder>/<表名>.parquet``，返回写出的路径。数据按来源许可只存本地（如仓库下
    ``data/foreign/``，该目录不入库），不要写进仓库的其他目录。``minute`` 给分钟线表时一并写出。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = adapter.tables()
    if minute is not None:
        tables["minute"] = conform_minute(minute)
    paths = []
    for name, table in tables.items():
        path = folder / f"{name}.parquet"
        table.write_parquet(path)
        paths.append(path)
    (folder / "diagnostics.json").write_text(
        json.dumps(adapter.diagnostics(), ensure_ascii=False, indent=1, default=str)
    )
    return paths


def build_panel(adapter, root):
    """适配器 → 期货面板（与境内同一套 ``build_futures``）→ 读取入口 ``FuturesData``。

    ``root`` 是**单独的**数据根（境外与境内不要共用一个根：面板目录只有一个 ``current.json``）。
    之后 ``continuous``、``roll_schedule``、``futures_panelbt.panelbt_inputs`` 照境内用法。
    """
    from . import open as open_data
    from .futures_panel import build_futures
    from .store import Store

    build_futures(Store(root), adapter)
    return open_data(root).futures()


__all__ = [
    "CME",
    "ICE_LONDON",
    "MINUTE",
    "MONTH_CODES",
    "PRODUCTS",
    "ForeignAdapter",
    "build_panel",
    "calendar_from_open_days",
    "check_settlements",
    "conform_minute",
    "resolve_year",
    "trade_dates",
    "unified_symbol",
    "write_parquet",
]
