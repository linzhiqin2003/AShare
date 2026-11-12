"""境外期货适配器的合成夹具（不是真实行情）。

一条合成的月度合约链：NYMEX WTI（CLV6 至 CLG7）和 ICE 布伦特（X26 至 H27）。合成数据覆盖
2026-09-01 至 2027-01-15（``DATA_FIRST``～``DATA_LAST``），比测试常用的区间 ``START``～``END``
（2026-10-01 至 12-31）两头都宽：start 之前和 end 之后都有 K 线与统计，适配器多取的那几天不能混进区间
（复核 P0-1）。测试可以取数据范围里的**任意** start/end，期望值由 ``expected_frame(data, volume, start, end)``
给出（start 当天的 ``previous_settlement`` 是数据里上一开市日的结算价）。

形状照官方文档（Databento 的 DBN 记录经 ``DBNStore.to_df()`` 后的列名和类型、Bloomberg ``Message.toPy()``
的嵌套字典），数值是确定性公式，**期望的统一字段由这里按「交易时段 = 前一开市日的切换点到当天的切换点」
直接循环算出**，与适配器按 UTC 时刻归属交易日的写法不共用代码。定义（definition）照官方说明每天 UTC 0 点
重发一次（只发当时还没到期的合约）。

假客户端只实现适配器用到的方法，方法名和参数名严格照官方签名（databento 0.90.0、blpapi 3.24.11 文档），
传了文档里没有的参数会因签名不符直接报 TypeError；请求里的取值（数据集、schema、stype、请求元素名）
也按文档的取值范围检查。
"""

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UTC = dt.UTC
CHICAGO = ZoneInfo("America/Chicago")
LONDON = ZoneInfo("Europe/London")
START, END = "20261001", "20261231"
DATA_FIRST, DATA_LAST = "20260901", "20270115"
INT64_MAX = 2**63 - 1


def _d(text):
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _weekdays(first, last, closed):
    day, out = _d(first), []
    while day <= _d(last):
        if day.weekday() < 5 and day.strftime("%Y%m%d") not in closed:
            out.append(day.strftime("%Y%m%d"))
        day += dt.timedelta(days=1)
    return out


# 开市日：CME 劳动节、感恩节、圣诞、元旦休市；ICE 圣诞、元旦休市。第一个开市日只作「前一开市日」，没有行情。
CME_DAYS = _weekdays(
    "20260831", DATA_LAST, {"20260907", "20261126", "20261225", "20270101"}
)
ICE_DAYS = _weekdays("20260831", DATA_LAST, {"20261225", "20270101"})

# (统一代码, 原始代码, instrument_id, 最后交易日, 交割年月, Bloomberg 代码)
CL = [
    ("CL2610.NYMEX", "CLV6", 1000, "20260922", (2026, 10), "CLV6 Comdty"),
    ("CL2611.NYMEX", "CLX6", 1001, "20261020", (2026, 11), "CLX6 Comdty"),
    ("CL2612.NYMEX", "CLZ6", 1002, "20261119", (2026, 12), "CLZ6 Comdty"),
    ("CL2701.NYMEX", "CLF7", 1003, "20261217", (2027, 1), "CLF7 Comdty"),
    ("CL2702.NYMEX", "CLG7", 1004, "20270120", (2027, 2), "CLG7 Comdty"),
]
BRENT = [
    ("B2611.ICE", "BRN FMX0026!", 2000, "20260930", (2026, 11), "COX6 Comdty"),
    ("B2612.ICE", "BRN FMZ0026!", 2001, "20261030", (2026, 12), "COZ6 Comdty"),
    ("B2701.ICE", "BRN FMF0027!", 2002, "20261130", (2027, 1), "COF7 Comdty"),
    ("B2702.ICE", "BRN FMG0027!", 2003, "20261231", (2027, 2), "COG7 Comdty"),
    ("B2703.ICE", "BRN FMH0027!", 2004, "20270129", (2027, 3), "COH7 Comdty"),
]
SPREAD = ("CLZ6-CLF7", 5001)
UNKNOWN_ID = 9999
ACTIVATION = dt.datetime(2024, 1, 2, 23, 0, tzinfo=UTC)
# 一个合约日的持仓量记录没有 ts_ref（适配器要按发布时刻的上一开市日归属），另一个合约日的持仓量被删除。
OI_WITHOUT_REF = ("CL2612.NYMEX", "20261104")
OI_DELETED = ("CL2701.NYMEX", "20261105")
# 分钟线覆盖的 CLZ6 交易日（含周一，它的时段从周日傍晚开始）。
MINUTE_DAYS = ("20261102", "20261103", "20261104", "20261105")
MINUTE_DAY = ("CL2612.NYMEX", "20261104")


def utc(year, month, day, hour=0, minute=0):
    return dt.datetime(year, month, day, hour, minute, tzinfo=UTC)


def _local(day, hour, zone):
    return dt.datetime.combine(_d(day), dt.time(hour), zone).astimezone(UTC)


def _cme_closed(t):
    """周五 16:00 收盘到周日 17:00 开盘之间（芝加哥时间）没有成交。"""
    local = t.astimezone(CHICAGO)
    return (
        local.weekday() == 5
        or (local.weekday() == 4 and local.hour >= 16)
        or (local.weekday() == 6 and local.hour < 17)
    )


def _hours(prev, day, kind):
    """交易日 day 的各根小时线开始时刻（UTC）。CME：prev 17:00 至 day 15:00（芝加哥），中间跳过周末；
    休市日（如感恩节）那段时间照样有成交，归下一开市日。ICE：prev 23:00 至 day 22:00（伦敦），跳过周末。"""
    if kind == "cme":
        start, end = _local(prev, 17, CHICAGO), _local(day, 16, CHICAGO)
        out, t = [], start
        while t < end:
            if not _cme_closed(t):
                out.append(t)
            t += dt.timedelta(hours=1)
        return out
    start, end = _local(prev, 23, LONDON), _local(day, 23, LONDON)
    out, t = [], start
    while t < end:
        local = t.astimezone(LONDON)
        # 周末（周五 23:00 之后到周日 23:00 之前）没有成交
        weekend = local.weekday() == 5 or (local.weekday() == 6 and local.hour < 23)
        weekend = weekend or (local.weekday() == 4 and local.hour >= 23)
        if not weekend:
            out.append(t)
        t += dt.timedelta(hours=1)
    return out


def fixed(x):
    """按 DBN 的定点价格（1 单位 = 1e-9）取整后再还原成小数，与官方 ``to_df()`` 读出的值逐位相同。"""
    return round(x * 1e9) / 1e9


def _price(j, t, h):
    return round(60.0 + 3.0 * j + 0.11 * t + 0.013 * h, 6)


def _definition(kind, raw, iid, last, y, m, ts_recv):
    return dict(
        ts_recv=ts_recv,
        instrument_id=iid,
        raw_symbol=raw,
        security_update_action="A",
        instrument_class="F",
        min_price_increment=0.01,
        display_factor=1.0,
        expiration=_local(last, 13, CHICAGO if kind == "cme" else LONDON),
        activation=ACTIVATION,
        unit_of_measure_qty=1000.0,
        maturity_year=y,
        maturity_month=m,
        currency="USD",
        unit_of_measure="BBL",
        asset=raw[:3] if kind == "ice" else "CL",
        exchange="XNYM" if kind == "cme" else "IFEU",
        security_type="FUT",
    )


def build():
    """返回合成数据：原始记录（三种 schema）与期望的统一字段（数据范围内的全部合约日）。"""
    bars, stats, defs, expected = [], [], [], {}
    for kind, chain, days in (("cme", CL, CME_DAYS), ("ice", BRENT, ICE_DAYS)):
        for j, (symbol, raw, iid, last, (y, m), _bbg) in enumerate(chain):
            # 定义每天 UTC 0 点重发，直到最后交易日。
            day = _d(DATA_FIRST)
            while day <= min(_d(last), _d(DATA_LAST)):
                stamp = dt.datetime.combine(day, dt.time(0, 5), UTC)
                defs.append(_definition(kind, raw, iid, last, y, m, stamp))
                day += dt.timedelta(days=1)
            prev_settle = None
            for t in range(1, len(days)):
                day, prev = days[t], days[t - 1]
                if day > last:
                    continue
                hours = _hours(prev, day, kind)
                prices = [_price(j, t, h) for h in range(len(hours))]
                for h, ts in enumerate(hours):
                    p = prices[h]
                    bars.append(
                        dict(
                            ts_event=ts,
                            instrument_id=iid,
                            open=fixed(p),
                            high=fixed(p + 0.05),
                            low=fixed(p - 0.04),
                            close=fixed(p + 0.01),
                            volume=10 + h + j,
                        )
                    )
                settle = fixed(prices[-1] + 0.005)
                # 持仓量：离最后交易日 8 个开市日以上的近月最大，之后转到下一合约。
                left = sum(1 for x in days if day < x <= last)
                oi = 50000.0 - 1000.0 * j if left > 8 else 1000.0 + left
                cleared = 2000.0 + 7 * t + j
                ref = dt.datetime.combine(_d(day), dt.time(), UTC)
                publish = dt.datetime.combine(
                    _d(day) + dt.timedelta(days=1), dt.time(10), UTC
                )
                if _d(day).weekday() == 4:
                    publish += dt.timedelta(days=2)  # 周五的持仓量周日发布
                close_ts = hours[-1] + dt.timedelta(minutes=40)
                stats += [
                    # 初步结算
                    _stat(
                        close_ts - dt.timedelta(minutes=5),
                        close_ts,
                        iid,
                        ref,
                        3,
                        fixed(settle + 1.0),
                        None,
                        2,
                    ),
                    _stat(close_ts, close_ts, iid, ref, 3, settle, None, 3),  # 最终结算
                    # 日内结算，ts_recv 更晚
                    _stat(
                        close_ts + dt.timedelta(minutes=1),
                        close_ts,
                        iid,
                        ref,
                        3,
                        fixed(settle + 9.0),
                        None,
                        11,
                    ),
                    _stat(publish, publish, iid, ref, 6, None, cleared, 0),
                ]
                oi_ref = None if (symbol, day) == OI_WITHOUT_REF else ref
                stats.append(_stat(publish, publish, iid, oi_ref, 9, None, oi, 0))
                if (symbol, day) == OI_DELETED:
                    stats.append(
                        _stat(
                            publish + dt.timedelta(minutes=1),
                            publish,
                            iid,
                            ref,
                            9,
                            None,
                            oi,
                            0,
                            action=2,
                        )
                    )
                expected[(symbol, day)] = dict(
                    open=fixed(prices[0]),
                    high=max(fixed(p + 0.05) for p in prices),
                    low=min(fixed(p - 0.04) for p in prices),
                    close=fixed(prices[-1] + 0.01),
                    settlement=settle,
                    previous_settlement=prev_settle,
                    cleared_volume=cleared,
                    electronic_volume=float(sum(10 + h + j for h in range(len(hours)))),
                    open_interest=None if (symbol, day) == OI_DELETED else oi,
                )
                prev_settle = settle
    # 价差合约和一条对不上合约的记录：适配器要丢掉并计数。
    defs.append(
        dict(
            defs[0], instrument_id=SPREAD[1], raw_symbol=SPREAD[0], instrument_class="S"
        )
    )
    for iid in (SPREAD[1], UNKNOWN_ID):
        bars.append(
            dict(
                ts_event=_local("20261104", 9, CHICAGO),
                instrument_id=iid,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1,
            )
        )
    return SimpleNamespace(bars=bars, stats=stats, defs=defs, expected=expected)


def _stat(ts_recv, ts_event, iid, ref, stat_type, price, quantity, flags, action=1):
    return dict(
        ts_recv=ts_recv,
        ts_event=ts_event,
        instrument_id=iid,
        ts_ref=ref,
        price=np.nan if price is None else price,
        quantity=INT64_MAX if quantity is None else int(quantity),
        stat_type=stat_type,
        update_action=action,
        stat_flags=flags,
    )


def minute_bars():
    """CLZ6 在 ``MINUTE_DAYS`` 各交易日的 1 分钟线：上一开市日 17:00 至当天 16:00（芝加哥），跳过周末。
    返回记录列表；每条另带夹具自己算的交易日 ``_date``（假客户端发出前去掉）。"""
    out = []
    for day in MINUTE_DAYS:
        prev = CME_DAYS[CME_DAYS.index(day) - 1]
        t, end, k = _local(prev, 17, CHICAGO), _local(day, 16, CHICAGO), 0
        while t < end:
            if not _cme_closed(t):
                p = fixed(61.0 + 0.001 * k + 0.5 * MINUTE_DAYS.index(day))
                out.append(
                    dict(
                        ts_event=t,
                        instrument_id=1002,
                        open=p,
                        high=fixed(p + 0.01),
                        low=fixed(p - 0.01),
                        close=p,
                        volume=1 + k % 5,
                        _date=day,
                    )
                )
                k += 1
            t += dt.timedelta(minutes=1)
    return out


def expected_frame(data, volume="cleared", start=START, end=END):
    """期望的统一日线（[start, end] 内，行按 symbol、date 排序），供与适配器输出逐位比较。"""
    rows = []
    for (symbol, day), v in sorted(data.expected.items()):
        if not start <= day <= end:
            continue
        rows.append(
            dict(
                date=day,
                symbol=symbol,
                open=v["open"],
                high=v["high"],
                low=v["low"],
                close=v["close"],
                settlement=v["settlement"],
                previous_settlement=v["previous_settlement"],
                volume=v["cleared_volume"]
                if volume == "cleared"
                else v["electronic_volume"],
                amount=None,
                open_interest=v["open_interest"],
            )
        )
    return rows


# ---------------------------------------------------------------- 假 Databento（形状照 DBNStore.to_df()）
DATASETS = {"GLBX.MDP3", "IFEU.IMPACT"}
SCHEMAS = {"definition", "statistics", "ohlcv-1h", "ohlcv-1m", "ohlcv-1d", "trades"}
STYPES = {"raw_symbol", "instrument_id", "parent", "continuous"}
# 母代码覆盖的 instrument_id：CL 的单月合约、价差和一条对不上合约表的记录（9999）；布伦特的单月合约。
PARENTS = {
    "CL.FUT": (range(1000, 1100), range(5000, 5100), range(9999, 10000)),
    "BRN.FUT": (range(2000, 2100),),
}
RTYPE = {"ohlcv-1h": 34, "ohlcv-1m": 33}
# 每 GB 美元的假单价只用于让估算非零；真实单价见 metadata.list_unit_prices。
PRICE = {"definition": 0.1, "statistics": 0.2, "ohlcv-1h": 0.3, "ohlcv-1m": 2.0}


DEFINITION_COLUMNS = [
    "ts_recv",
    "instrument_id",
    "raw_symbol",
    "security_update_action",
    "instrument_class",
    "min_price_increment",
    "display_factor",
    "expiration",
    "activation",
    "unit_of_measure_qty",
    "maturity_year",
    "maturity_month",
    "currency",
    "unit_of_measure",
    "asset",
    "exchange",
    "security_type",
]


class _Default:
    """模仿 databento 的 ``Default`` 哨兵（get_cost 的 mode 默认值）。"""


DEFAULT = _Default()


class FakeStore:
    """只实现 ``to_df()``：索引与列名、类型照真实 ``DBNStore.to_df()``（价格已缩放成小数、时间为 UTC）。"""

    def __init__(self, frame, index):
        self._frame, self._index = frame, index

    def to_df(
        self,
        price_type="float",
        pretty_ts=True,
        map_symbols=True,
        schema=None,
        tz=DEFAULT,
        count=None,
    ):
        if price_type != "float" or not pretty_ts or count is not None:
            raise NotImplementedError("假对象只实现默认参数")
        return self._frame.set_index(self._index)


def _frame(rows, schema):
    utc = "datetime64[ns, UTC]"
    if schema in RTYPE:
        df = pd.DataFrame(
            rows,
            columns=[
                "ts_event",
                "instrument_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        df = df.astype(
            {
                "ts_event": utc,
                "instrument_id": "uint32",
                "volume": "uint64",
                **dict.fromkeys(["open", "high", "low", "close"], "float64"),
            }
        )
        df.insert(1, "rtype", np.uint8(RTYPE[schema]))
        df.insert(2, "publisher_id", np.uint16(1))
        df["symbol"] = None
        return FakeStore(df, "ts_event")
    if schema == "statistics":
        df = pd.DataFrame(
            rows,
            columns=[
                "ts_recv",
                "ts_event",
                "instrument_id",
                "ts_ref",
                "price",
                "quantity",
                "stat_type",
                "update_action",
                "stat_flags",
            ],
        )
        df = df.astype(
            {
                "ts_recv": utc,
                "ts_event": utc,
                "ts_ref": utc,
                "instrument_id": "uint32",
                "price": "float64",
                "quantity": "int64",
                "stat_type": "uint16",
                "update_action": "uint8",
                "stat_flags": "uint8",
            }
        )
        df["symbol"] = None
        return FakeStore(df, "ts_recv")
    df = pd.DataFrame(rows, columns=DEFINITION_COLUMNS)
    df = df.astype(
        {
            "ts_recv": utc,
            "expiration": utc,
            "activation": utc,
            "instrument_id": "uint32",
            "maturity_year": "uint16",
            "maturity_month": "uint8",
            "raw_symbol": "string",
            "security_update_action": "string",
            "instrument_class": "string",
            "currency": "string",
            "unit_of_measure": "string",
            "asset": "string",
            "exchange": "string",
            "security_type": "string",
        }
    )
    df["ts_event"] = df["ts_recv"]
    df["symbol"] = None
    return FakeStore(df, "ts_recv")


def _check(dataset, schema, stype_in):
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset}")
    if schema not in SCHEMAS:
        raise ValueError(f"未知 schema {schema}")
    if stype_in not in STYPES:
        raise ValueError(f"未知 stype_in {stype_in}")


def _range(value):
    return pd.Timestamp(value, tz="UTC")


class FakeMetadata:
    def __init__(self, owner):
        self.owner = owner

    def get_cost(
        self,
        dataset,
        start,
        end=None,
        mode=DEFAULT,
        symbols=None,
        schema="trades",
        stype_in="raw_symbol",
        limit=None,
    ):
        _check(dataset, schema, stype_in)
        self.owner.calls.append(("get_cost", dataset, schema, tuple(symbols or ())))
        days = (_range(end) - _range(start)).days if end else 1
        return round(PRICE.get(schema, 1.0) * max(days, 1) / 10, 6)


class FakeTimeseries:
    def __init__(self, owner):
        self.owner = owner

    def get_range(
        self,
        dataset,
        start,
        end=None,
        symbols=None,
        schema="trades",
        stype_in="raw_symbol",
        stype_out="instrument_id",
        limit=None,
        path=None,
    ):
        _check(dataset, schema, stype_in)
        if stype_in != "parent":
            raise NotImplementedError("假对象只实现 parent 记法")
        self.owner.calls.append(("get_range", dataset, schema, tuple(symbols)))
        wanted = [rg for sym in symbols for rg in PARENTS[sym]]
        lo, hi = _range(start), _range(end)
        key = "ts_event" if schema in RTYPE else "ts_recv"
        source = {
            "definition": self.owner.data.defs,
            "statistics": self.owner.data.stats,
            "ohlcv-1h": self.owner.data.bars,
            "ohlcv-1m": self.owner.minute,
        }[schema]
        rows = [
            r
            for r in source
            if any(r["instrument_id"] in rg for rg in wanted) and lo <= r[key] < hi
        ]
        return _frame(rows, schema)


class FakeSymbology:
    def __init__(self, owner):
        self.owner = owner

    def resolve(self, dataset, symbols, stype_in, stype_out, start_date, end_date=None):
        self.owner.calls.append(("resolve", dataset, tuple(symbols)))
        result = {}
        for code in symbols:
            result[code] = self.owner.continuous.get((dataset, code), [])
        return dict(
            result=result,
            symbols=list(symbols),
            stype_in=stype_in,
            stype_out=stype_out,
            start_date=start_date,
            end_date=end_date,
            partial=[],
            not_found=[],
            message="OK",
            status=0,
        )


def continuous_map(data):
    """假 ``symbology.resolve`` 的连续合约区间：c.0 = 未到期合约里最后交易日最早的（到期次日才换），
    n.0 = 前一开市日持仓量最大的。返回 {(dataset, 代码): [{"d0", "d1", "s"}]}，d1 不含。"""
    out = {}
    for dataset, root, chain, days in (
        ("GLBX.MDP3", "CL", CL, CME_DAYS),
        ("IFEU.IMPACT", "BRN", BRENT, ICE_DAYS),
    ):
        picks = {"c": [], "n": [], "v": []}
        for t in range(1, len(days)):
            day, prev = days[t], days[t - 1]
            if day < START:
                continue
            live = [c for c in chain if c[3] >= day]
            picks["c"].append((day, min(live, key=lambda c: c[3])[2]))
            oi = {
                c[2]: data.expected.get((c[0], prev), {}).get("open_interest") or 0
                for c in live
            }
            best = max(oi, key=lambda k: (oi[k], -k))
            picks["n"].append((day, best))
            picks["v"].append((day, best))
        for rule, seq in picks.items():
            runs = []
            for day, iid in seq:
                iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
                nxt = (_d(day) + dt.timedelta(days=1)).isoformat()
                if runs and runs[-1]["s"] == str(iid):
                    runs[-1]["d1"] = nxt
                else:
                    runs.append(dict(d0=iso, d1=nxt, s=str(iid)))
            out[(dataset, f"{root}.{rule}.0")] = runs
    return out


class FakeHistorical:
    """``databento.Historical`` 的替身：``metadata``、``timeseries``、``symbology`` 三个子接口，``calls`` 记下每次调用。"""

    def __init__(self, data=None, continuous=None):
        self.data = data or build()
        self.minute = minute_bars()
        self.continuous = (
            continuous if continuous is not None else continuous_map(self.data)
        )
        self.calls = []
        self.metadata = FakeMetadata(self)
        self.timeseries = FakeTimeseries(self)
        self.symbology = FakeSymbology(self)

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


# ---------------------------------------------------------------- 假 blpapi（形状照 Message.toPy()）
REQUEST_ELEMENTS = {
    "HistoricalDataRequest": {
        "securities",
        "fields",
        "startDate",
        "endDate",
        "periodicitySelection",
        "overrides",
    },
    "ReferenceDataRequest": {"securities", "fields", "overrides"},
    "IntradayBarRequest": {
        "security",
        "eventType",
        "interval",
        "startDateTime",
        "endDateTime",
    },
}
ARRAYS = {"securities", "fields", "overrides"}
EXPIRED_TICKER = (
    "CLZ16 Comdty"  # 已到期的两位年份写法（2016-12），在区间外，适配器要丢掉
)


class FakeElement:
    def __init__(self, store):
        self.store = store

    def appendElement(self):
        item = {}
        self.store.append(item)
        return FakeElement(item)

    def setElement(self, name, value):
        if name not in ("fieldId", "value"):
            raise ValueError(f"覆盖项没有元素 {name}")
        self.store[name] = value


class FakeRequest:
    def __init__(self, operation):
        self.operation = operation
        self.values = {}

    def _allowed(self, name):
        if name not in REQUEST_ELEMENTS[self.operation]:
            raise ValueError(f"{self.operation} 没有元素 {name}")

    def set(self, name, value):
        self._allowed(name)
        if name in ARRAYS:
            raise ValueError(f"{name} 是数组，要用 append")
        self.values[name] = value

    def append(self, name, value):
        self._allowed(name)
        if name not in ARRAYS:
            raise ValueError(f"{name} 不是数组")
        self.values.setdefault(name, []).append(value)

    def getElement(self, name):
        self._allowed(name)
        return FakeElement(self.values.setdefault(name, []))


class FakeService:
    def __init__(self, owner):
        self.owner = owner

    def createRequest(self, operationName):
        if operationName not in REQUEST_ELEMENTS:
            raise ValueError(f"//blp/refdata 没有请求 {operationName}")
        return FakeRequest(operationName)


class FakeMessage:
    def __init__(self, kind, body):
        self.kind, self.body = kind, body

    def messageType(self):
        return self.kind

    def toPy(self):
        return self.body


class FakeEvent:
    def __init__(self, kind, messages):
        self.kind, self.messages = kind, messages

    def eventType(self):
        return self.kind

    def __iter__(self):
        return iter(self.messages)


class FakeSessionOptions:
    def __init__(self):
        self.host, self.port = "127.0.0.1", 8194

    def setServerHost(self, serverHost):
        self.host = serverHost

    def setServerPort(self, serverPort):
        self.port = serverPort


class FakeBlpapi:
    """假 ``blpapi`` 模块：``SessionOptions``、``Session``、``Event`` 常量。``requests`` 记下发出的请求。"""

    class Event:
        (
            SESSION_STATUS,
            REQUEST_STATUS,
            RESPONSE,
            PARTIAL_RESPONSE,
            SERVICE_STATUS,
            TIMEOUT,
        ) = 2, 4, 5, 6, 9, 10

    def __init__(self, data=None):
        self.data = data or build()
        self.minute = minute_bars()
        self.requests = []
        # 按证券注入的错误：{代码: {"securityError": {...}} 或 {"fieldExceptions": [...]}}，并进该证券的应答。
        self.faults = {}
        # 应答前对每个证券的字段做的改动：{代码: {字段: 值或 None（删掉）}}。
        self.patches = {}
        api = self

        class Session:
            def __init__(self, options=None, eventHandler=None, eventDispatcher=None):
                self.options, self.queue, self.open = options, [], set()

            def start(self):
                return True

            def stop(self):
                return True

            def openService(self, serviceName):
                self.open.add(serviceName)
                return serviceName == "//blp/refdata"

            def getService(self, serviceName):
                if serviceName not in self.open:
                    raise RuntimeError("InvalidStateException: service not open")
                return FakeService(api)

            def sendRequest(
                self,
                request,
                identity=None,
                correlationId=None,
                eventQueue=None,
                requestLabel=None,
            ):
                api.requests.append((request.operation, dict(request.values)))
                bodies = api.respond(request)
                # 先给 PARTIAL_RESPONSE，最后一条放 RESPONSE。
                events = [
                    FakeEvent(
                        api.Event.PARTIAL_RESPONSE, [FakeMessage(request.operation, b)]
                    )
                    for b in bodies[:-1]
                ]
                events.append(
                    FakeEvent(
                        api.Event.RESPONSE,
                        [FakeMessage(request.operation, bodies[-1])] if bodies else [],
                    )
                )
                self.queue = [FakeEvent(api.Event.REQUEST_STATUS, [])] + events

            def nextEvent(self, timeout=0):
                return (
                    self.queue.pop(0)
                    if self.queue
                    else FakeEvent(api.Event.TIMEOUT, [])
                )

        self.Session = Session
        self.SessionOptions = FakeSessionOptions

    def respond(self, request):
        v = request.values
        tickers = {
            b: (symbol, last, ym)
            for chain in (CL, BRENT)
            for (symbol, _r, _i, last, ym, b) in chain
        }
        if request.operation == "ReferenceDataRequest":
            out = []
            for sec in v["securities"]:
                fields = {}
                if "FUT_CHAIN" in v["fields"]:
                    root = sec[:2]
                    chain = [b for b in tickers if b.startswith(root)]
                    if root == "CL" and {
                        "fieldId": "INCLUDE_EXPIRED_CONTRACTS",
                        "value": "Y",
                    } in v.get("overrides", []):
                        chain = [EXPIRED_TICKER, *chain]
                    fields["FUT_CHAIN"] = [{"Security Description": t} for t in chain]
                if sec in tickers or sec == EXPIRED_TICKER:
                    symbol, last, (y, m) = tickers.get(
                        sec, ("CL1612.NYMEX", "20161121", (2016, 12))
                    )
                    full = dict(
                        NAME=f"NAME {sec}",
                        FUT_CONT_SIZE=1000.0,
                        FUT_TICK_SIZE=0.01,
                        FUT_FIRST_TRADE_DT=dt.date(2024, 1, 3),
                        LAST_TRADEABLE_DT=_d(last),
                        FUT_DLV_DT_LAST=dt.date(y, m, 28),
                        FUT_NOTICE_FIRST=dt.date(y, m, 1),
                        CRNCY="USD",
                    )
                    fields.update({f: full[f] for f in v["fields"] if f in full})
                for f, x in self.patches.get(sec, {}).items():
                    if x is None:
                        fields.pop(f, None)
                    else:
                        fields[f] = x
                out.append(
                    {
                        **dict(
                            security=sec,
                            eidData=[],
                            fieldExceptions=[],
                            sequenceNumber=len(out),
                            fieldData=fields,
                        ),
                        **self.faults.get(sec, {}),
                    }
                )
            return [dict(securityData=out[:1])] + (
                [dict(securityData=out[1:])] if len(out) > 1 else []
            )
        if request.operation == "HistoricalDataRequest":
            bodies = []
            for sec in v["securities"]:
                rows = []
                if sec in tickers:
                    symbol = tickers[sec][0]
                    for (s, day), e in sorted(self.data.expected.items()):
                        if s == symbol and v["startDate"] <= day <= v["endDate"]:
                            rows.append(
                                dict(
                                    date=_d(day),
                                    PX_OPEN=e["open"],
                                    PX_HIGH=e["high"],
                                    PX_LOW=e["low"],
                                    PX_LAST=e["close"],
                                    PX_SETTLE=e["settlement"],
                                    PX_VOLUME=e["cleared_volume"],
                                    OPEN_INT=e["open_interest"],
                                )
                            )
                    rows = [
                        {k: x for k, x in r.items() if k == "date" or k in v["fields"]}
                        for r in rows
                    ]
                    for f, x in self.patches.get(sec, {}).items():
                        for r in rows:
                            r[f] = x
                    rows = [
                        {k: x for k, x in r.items() if x is not None} for r in rows
                    ]  # 缺值的字段不出现
                else:  # 通用代码：给一个确定性的价格序列
                    rank = int(sec[2])
                    rows = [
                        dict(date=_d(day), PX_SETTLE=70.0 + rank + 0.1 * k)
                        for k, day in enumerate(CME_DAYS)
                        if v["startDate"] <= day <= v["endDate"]
                    ]
                bodies.append(
                    dict(
                        securityData={
                            **dict(
                                security=sec,
                                eidData=[],
                                sequenceNumber=len(bodies),
                                fieldExceptions=[],
                                fieldData=rows,
                            ),
                            **self.faults.get(sec, {}),
                        }
                    )
                )
            return bodies
        if request.operation == "IntradayBarRequest":
            lo, hi = v["startDateTime"], v["endDateTime"]
            if lo.tzinfo is not None or hi.tzinfo is not None:
                raise ValueError(
                    "IntradayBarRequest 的时间要写成 UTC 的 naive datetime"
                )
            bars = []
            if v["security"] == "CLZ6 Comdty":
                for b in self.minute:
                    t = b["ts_event"].replace(tzinfo=None)
                    if lo <= t < hi:
                        bars.append(
                            dict(
                                time=t,
                                open=b["open"],
                                high=b["high"],
                                low=b["low"],
                                close=b["close"],
                                volume=b["volume"],
                                numEvents=1,
                                value=b["close"] * b["volume"] * 1000,
                            )
                        )
            return [
                dict(barData=dict(eidData=[], delayedSecurity=False, barTickData=bars))
            ]
        raise ValueError(request.operation)
