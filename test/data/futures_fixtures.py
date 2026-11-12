"""合成的期货原始数据（Tushare 接口形状，不含真实行情）和一个假的 Tushare 源。"""

import polars as pl

from ashare.data.catalog import FUTURES_EXCHANGES
from ashare.data.sources.base import FetchResult

DAYS = [
    f"2026{m:02}{d:02}"
    for m, d in [
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 12),
        (1, 13),
        (1, 14),
        (1, 15),
        (1, 16),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 26),
        (1, 27),
        (1, 28),
        (1, 29),
        (1, 30),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 9),
        (2, 10),
        (2, 11),
        (2, 12),
        (2, 13),
    ]
]
CLOSED = ["20260110", "20260111"]

# 代码, 交易所, 品种, 每手单位(per_unit), 乘数(multiplier), 最小变动, 上市行, 到期行, 基价
CONTRACTS = [
    ("RB2602.SHF", "SHFE", "RB", 10.0, None, "1人民币元/吨", 0, 9, 3000.0),
    ("RB2603.SHF", "SHFE", "RB", 10.0, None, "1人民币元/吨", 0, 19, 3050.0),
    ("RB2604.SHF", "SHFE", "RB", 10.0, None, "1人民币元/吨", 0, 29, 3100.0),
    ("IF2602.CFX", "CFFEX", "IF", 1.0, 300.0, "0.2指数点", 0, 14, 4000.0),
    ("IF2603.CFX", "CFFEX", "IF", 1.0, 300.0, "0.2指数点", 0, 29, 4010.0),
]


def calendar(exchange):
    rows = [(d, 1) for d in DAYS] + [(d, 0) for d in CLOSED]
    rows.sort()
    return pl.DataFrame(
        dict(
            exchange=[exchange] * len(rows),
            cal_date=[r[0] for r in rows],
            is_open=[r[1] for r in rows],
            pretrade_date=[None] * len(rows),
        )
    )


def contracts():
    rows = []
    for code, ex, product, per_unit, mult, tick, first, last, _ in CONTRACTS:
        rows.append(
            dict(
                ts_code=code,
                symbol=code.split(".")[0],
                exchange=ex,
                name=code,
                fut_code=product,
                multiplier=mult,
                trade_unit="吨",
                per_unit=per_unit,
                quote_unit="人民币元/吨",
                quote_unit_desc=tick,
                d_mode_desc="现金交割" if ex == "CFFEX" else "实物交割",
                list_date=DAYS[first],
                delist_date=DAYS[last],
                d_month="202602",
                last_ddate=DAYS[last],
            )
        )
    return pl.DataFrame(rows, schema_overrides=dict(multiplier=pl.Float64))


def price(j, t):
    return CONTRACTS[j][8] + 2.0 * t + (j % 3)


def open_interest(j, t):
    """持仓在到期前 9 个交易日见顶：RB 在第 6 行按持仓换月，IF 到期前强制换月。"""
    peak = CONTRACTS[j][7] - 9
    return float(3000 - 100 * abs(t - peak) + j)


def daily_rows():
    rows = []
    for j, (code, ex, _p, per_unit, mult, *_rest) in enumerate(CONTRACTS):
        first, last = CONTRACTS[j][6], CONTRACTS[j][7]
        unit = mult or per_unit
        for t in range(first, last + 1):
            settle = price(j, t)
            vol = float(101 + 3 * t + j)
            rows.append(
                dict(
                    ts_code=code,
                    trade_date=DAYS[t],
                    pre_close=settle - 1,
                    pre_settle=price(j, t - 1) if t > first else settle - 2.0,
                    open=settle - 0.4 if ex == "SHFE" else settle - 0.2,
                    high=settle + 3,
                    low=settle - 3,
                    close=settle + 0.6 if ex == "SHFE" else settle + 0.2,
                    settle=settle,
                    change1=0.0,
                    change2=0.0,
                    vol=vol,
                    amount=vol * settle * unit / 10000,
                    oi=open_interest(j, t),
                    oi_chg=0.0,
                )
            )
    return pl.DataFrame(rows)


def settle_rows():
    rows = []
    for j, (code, ex, *_rest) in enumerate(CONTRACTS):
        first, last = CONTRACTS[j][6], CONTRACTS[j][7]
        for t in range(first, last + 1):
            cffex = ex == "CFFEX"
            rows.append(
                dict(
                    ts_code=code,
                    trade_date=DAYS[t],
                    settle=None if cffex else price(j, t),
                    trading_fee_rate=0.23 if cffex else 0.05,
                    trading_fee=0.23 if cffex else 0.0,
                    delivery_fee=0.5 if cffex else 0.0,
                    b_hedging_margin_rate=10.0 if cffex else 0.09,
                    s_hedging_margin_rate=10.0 if cffex else 0.09,
                    long_margin_rate=12.0 if cffex else 0.1 + 0.01 * (t > 20),
                    short_margin_rate=12.0 if cffex else 0.1,
                    offset_today_fee=1000.0 if cffex else None,
                    exchange=ex,
                )
            )
    return pl.DataFrame(
        rows, schema_overrides=dict(settle=pl.Float64, offset_today_fee=pl.Float64)
    )


def mapping_rows(daily):
    """来源主力：前一日持仓量最大（只向后换）；连续：最近月。"""
    rows = []
    for product, suffix in (("RB", "SHF"), ("IF", "CFX")):
        members = [c for c in CONTRACTS if c[2] == product]
        held = None
        for t, day in enumerate(DAYS):
            alive = [c for c in members if c[6] <= t <= c[7]]
            if t > 0:
                prev = daily.filter(pl.col("trade_date") == DAYS[t - 1])
                oi = dict(zip(prev["ts_code"], prev["oi"], strict=False))
                best = max(
                    (c for c in alive if c[7] > t),
                    key=lambda c: oi.get(c[0], -1),
                    default=None,
                )
                if (
                    held is None
                    or held[7] <= t
                    or (
                        best
                        and best[7] > held[7]
                        and oi.get(best[0], -1) > oi.get(held[0], -1)
                    )
                ):
                    held = best
            else:
                held = alive[0]
            if held:
                rows.append(
                    dict(
                        ts_code=f"{product}.{suffix}",
                        trade_date=day,
                        mapping_ts_code=held[0],
                    )
                )
            near = min(alive, key=lambda c: c[7], default=None)
            if near:
                rows.append(
                    dict(
                        ts_code=f"{product}L.{suffix}",
                        trade_date=day,
                        mapping_ts_code=near[0],
                    )
                )
    return pl.DataFrame(rows)


class FakeTushare:
    """按 (接口, 参数) 返回合成数据；记录调用，可指定某些分区失败一次。"""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)
        self.daily = daily_rows()
        main = mapping_rows(self.daily)
        cont = (
            self.daily.join(
                main.filter(~pl.col("ts_code").str.contains("L\\.")).rename(
                    {"ts_code": "cont", "mapping_ts_code": "ts_code"}
                ),
                on=["ts_code", "trade_date"],
            )
            .drop("ts_code")
            .rename({"cont": "ts_code"})
        )
        self.tables = dict(
            fut_daily=pl.concat([self.daily, cont.select(self.daily.columns)]),
            fut_settle=settle_rows(),
            fut_mapping=main,
            fut_basic=contracts(),
        )

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls.append((api, dict(params)))
        key = (api, params.get("trade_date") or params.get("exchange"))
        if key in self.fail:
            self.fail.discard(key)
            raise RuntimeError("故意失败")
        if api == "trade_cal":
            table = calendar(params["exchange"])
        elif api == "fut_basic":
            table = self.tables[api].filter(pl.col("exchange") == params["exchange"])
        else:
            table = self.tables[api].filter(
                pl.col("trade_date") == params["trade_date"]
            )
        return FetchResult(table, "2026-02-14T00:00:00+00:00", 1)


assert set(FUTURES_EXCHANGES) >= {"SHFE", "CFFEX"}
