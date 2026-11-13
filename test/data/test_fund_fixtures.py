"""合成的基金、可转债原始数据（Tushare 接口形状，不含真实行情）和一个假的 Tushare 源。

场景（交易日取期货测试同一份 30 天日历）：
- 510001.SH 股票 ETF：第 10 个交易日 1 拆 2（前收盘减半、复权因子翻倍），第 15 天每份分红 0.05，
  第 7 天开盘一字涨停。
- 511001.SH 债券 ETF：平稳。
- 159001.SZ 创业板 ETF（20% 档）：基金表上市日比第一笔行情晚两天（转型后改了上市日），第 5 天
  涨 15%，退市日为第 19 天（最后一笔行情在第 18 天）。
- 161022A.SZ：日线里有、基金表里没有。
- 113001.SH 可转债：转股价 10 → 9（第 6 天生效、第 8 天才公告）→ 8.5（第 9 天公告、第 12 天生效）；
  第 11 天公告实施强赎，赎回登记日第 17 天，最后交易日第 14 天，之后是成交量 0 的占位行。
- 123001.SZ 可转债：上市前 5 天是成交量 0、价格 0 的占位行。
"""

import polars as pl
from futures_fixtures import CLOSED, DAYS

from ashare.data.sources.base import FetchResult

FUND_COLUMNS = (
    "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,issue_date,delist_date,"
    "issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,exp_return,benchmark,status,invest_type,"
    "type,trustee,purc_startdate,redm_startdate,market"
).split(",")
SPLIT_DAY, DIV_DAY, LIMIT_DAY = 10, 15, 7
DIVIDEND = 0.05


def calendar():
    rows = sorted([(d, 1) for d in DAYS] + [(d, 0) for d in CLOSED])
    return pl.DataFrame(
        dict(
            exchange=["SSE"] * len(rows),
            cal_date=[r[0] for r in rows],
            is_open=[r[1] for r in rows],
            pretrade_date=[None] * len(rows),
        )
    )


def fund_basic():
    rows = [
        ("510001.SH", "测试沪深300ETF", "股票型", "20200101", None, "L"),
        ("511001.SH", "测试国债ETF", "债券型", "20200101", None, "L"),
        ("159001.SZ", "测试创业板ETF", "股票型", DAYS[2], DAYS[19], "D"),
    ]
    data = {c: [None] * len(rows) for c in FUND_COLUMNS}
    for i, (code, name, kind, listing, delisting, status) in enumerate(rows):
        data["ts_code"][i], data["name"][i], data["fund_type"][i] = code, name, kind
        data["list_date"][i], data["delist_date"][i], data["status"][i] = (
            listing,
            delisting,
            status,
        )
        data["market"][i], data["m_fee"][i], data["c_fee"][i] = "E", 0.5, 0.1
    return pl.DataFrame(
        data,
        schema={
            c: (
                pl.Float64
                if c
                in (
                    "m_fee",
                    "c_fee",
                    "issue_amount",
                    "duration_year",
                    "p_value",
                    "min_amount",
                    "exp_return",
                )
                else pl.String
            )
            for c in FUND_COLUMNS
        },
    )


def fund_bars():
    """返回 (日线, 复权因子, 份额) 三张表。"""
    daily, adj, share = [], [], []
    # 510001.SH
    close, factor = 4.0, 1.0
    for k, day in enumerate(DAYS):
        pre = close
        if k == SPLIT_DAY:
            pre = close / 2
            factor *= 2
        if k == DIV_DAY:
            factor *= close / (close - DIVIDEND)
            pre = close - DIVIDEND
        new = round(pre * (1 + 0.01 * ((k % 5) - 2)), 3)
        if k == LIMIT_DAY:
            new = round(pre * 1.1, 3)
            o = h = lo = new
        else:
            o, h, lo = pre, max(pre, new) + 0.01, min(pre, new) - 0.01
        vol = 1000.0 + k
        daily.append(
            (
                "510001.SH",
                day,
                pre,
                o,
                h,
                lo,
                new,
                new - pre,
                (new / pre - 1) * 100,
                vol,
                round(vol * 100 * (h + lo) / 2 / 1000, 3),
            )
        )
        adj.append(("510001.SH", day, factor))
        share.append(("510001.SH", day, 50000.0 + k))
        close = new
    # 511001.SH
    close = 100.0
    for k, day in enumerate(DAYS):
        pre = close
        new = round(pre * (1 + 0.0005 * ((k % 3) - 1)), 3)
        h, lo = max(pre, new) + 0.002, min(pre, new) - 0.002
        daily.append(
            (
                "511001.SH",
                day,
                pre,
                pre,
                h,
                lo,
                new,
                new - pre,
                0.0,
                500.0,
                round(500 * 100 * (h + lo) / 2 / 1000, 3),
            )
        )
        adj.append(("511001.SH", day, 1.0))
        share.append(("511001.SH", day, 3000.0))
        close = new
    # 159001.SZ：第 0 天起有行情，第 18 天最后一笔
    close = 2.0
    for k, day in enumerate(DAYS[:19]):
        pre = close
        new = round(pre * (1.15 if k == 5 else 1.0 + 0.002 * (k % 2)), 3)
        h, lo = max(pre, new) + 0.001, min(pre, new) - 0.001
        daily.append(
            (
                "159001.SZ",
                day,
                pre,
                pre,
                h,
                lo,
                new,
                new - pre,
                0.0,
                200.0,
                round(200 * 100 * (h + lo) / 2 / 1000, 3),
            )
        )
        adj.append(("159001.SZ", day, 1.0))
        close = new
    daily.append(("161022A.SZ", DAYS[3], 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.1))
    columns = [
        "ts_code",
        "trade_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    return (
        pl.DataFrame(daily, schema=columns, orient="row"),
        pl.DataFrame(adj, schema=["ts_code", "trade_date", "adj_factor"], orient="row"),
        pl.DataFrame(
            [(c, d, s, "ETF", c[-2:]) for c, d, s in share],
            schema=["ts_code", "trade_date", "fd_share", "fund_type", "market"],
            orient="row",
        ),
    )


def fund_div():
    columns = "ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,ex_date,pay_date,earpay_date,net_ex_date,div_cash,base_unit,ear_distr,ear_amount,account_date,base_year".split(
        ","
    )
    rows = [
        dict(
            ts_code="510001.SH",
            ann_date=DAYS[DIV_DAY - 4],
            div_proc="实施",
            ex_date=DAYS[DIV_DAY],
            div_cash=DIVIDEND,
        ),
        dict(
            ts_code="510001.SH",
            ann_date=DAYS[DIV_DAY - 8],
            div_proc="预案",
            ex_date=DAYS[DIV_DAY],
            div_cash=DIVIDEND,
        ),
        dict(
            ts_code="000001.OF",
            ann_date=DAYS[3],
            div_proc="实施",
            ex_date=DAYS[DIV_DAY],
            div_cash=0.1,
        ),
    ]
    data = {c: [r.get(c) for r in rows] for c in columns}
    return pl.DataFrame(
        data,
        schema={
            c: (
                pl.Float64
                if c in ("div_cash", "base_unit", "ear_distr", "ear_amount")
                else pl.String
            )
            for c in columns
        },
    )


CB_BASIC_COLUMNS = "ts_code,bond_full_name,bond_short_name,cb_type,cb_code,stk_code,stk_short_name,maturity,par,issue_price,issue_size,remain_size,value_date,maturity_date,rate_type,coupon_rate,add_rate,pay_per_year,list_date,delist_date,exchange,conv_start_date,conv_end_date,conv_stop_date,first_conv_price,conv_price,rate_clause".split(
    ","
)
NUMERIC = {
    "maturity",
    "par",
    "issue_price",
    "issue_size",
    "remain_size",
    "coupon_rate",
    "add_rate",
    "pay_per_year",
    "first_conv_price",
    "conv_price",
}
STOCK = {"600001.SH": 9.5, "000001.SZ": 21.0}  # 正股基价（测试里用 monkeypatch 供给）


def stock_price(code, k):
    return round(STOCK[code] * (1 + 0.01 * k), 3)


def cb_basic():
    rows = [
        dict(
            ts_code="113001.SH",
            bond_short_name="测试转债",
            cb_type="CB",
            stk_code="600001.SH",
            par=100.0,
            issue_size=1e9,
            value_date="20250101",
            maturity_date="20310101",
            list_date=DAYS[0],
            delist_date=DAYS[18],
            first_conv_price=10.0,
            conv_price=8.5,
            exchange="SH",
        ),
        dict(
            ts_code="123001.SZ",
            bond_short_name="测试二转债",
            cb_type="CB",
            stk_code="000001.SZ",
            par=100.0,
            issue_size=5e8,
            value_date="20251201",
            maturity_date="20311201",
            list_date=DAYS[5],
            delist_date=None,
            first_conv_price=20.0,
            conv_price=20.0,
            exchange="SZ",
        ),
    ]
    return pl.DataFrame(
        {c: [r.get(c) for r in rows] for c in CB_BASIC_COLUMNS},
        schema={
            c: (pl.Float64 if c in NUMERIC else pl.String) for c in CB_BASIC_COLUMNS
        },
    )


# 113001.SH 的转股价：源（按生效日）与当时可知（max(生效日, 公告次日)）
CB_CHANGES = [
    # 公告行, 生效行, 新转股价
    (8, 6, 9.0),
    (9, 12, 8.5),
]
CALL_ANN, CALL_RECORD, LAST_TRADE = 11, 17, 14


def source_conversion_price(code, k):
    if code == "123001.SZ":
        return 20.0
    price = 10.0
    for _, eff, value in CB_CHANGES:
        if k >= eff:
            price = value
    return price


def cb_daily():
    rows = []
    for code, base in (("113001.SH", 105.0), ("123001.SZ", 110.0)):
        close = 100.0
        for k, day in enumerate(DAYS):
            placeholder = (code == "113001.SH" and k > LAST_TRADE) or (
                code == "123001.SZ" and k < 5
            )
            if code == "113001.SH" and k > 18:
                continue  # 摘牌次日起源不再出行（摘牌日当天仍有一行占位）
            if placeholder:
                stale = close if code == "113001.SH" else 0.0
                rows.append(
                    (
                        code,
                        day,
                        close if code == "113001.SH" else 100.0,
                        0.0,
                        0.0,
                        0.0,
                        stale,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                    )
                )
                continue
            pre = close
            new = round(base * (1 + 0.004 * (k % 4)), 3)
            h, lo = max(pre, new) + 0.5, min(pre, new) - 0.5
            vol = 1000.0 + k  # 手
            amount = round(vol * 10 * (h + lo) / 2 / 1e4, 4)  # 万元
            cv = round(
                100 / source_conversion_price(code, k) * stock_price(STOCK_OF[code], k),
                2,
            )
            rows.append(
                (
                    code,
                    day,
                    pre,
                    pre,
                    h,
                    lo,
                    new,
                    new - pre,
                    0.0,
                    vol,
                    amount,
                    95.0,
                    10.0,
                    cv,
                    (new / cv - 1) * 100,
                )
            )
            close = new
    columns = "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate".split(
        ","
    )
    return pl.DataFrame(rows, schema=columns, orient="row")


STOCK_OF = {"113001.SH": "600001.SH", "123001.SZ": "000001.SZ"}


def cb_price_chg():
    rows = [("113001.SH", "测试转债", DAYS[0], DAYS[0], 10.0, None, None)]
    rows += [
        ("113001.SH", "测试转债", DAYS[p], DAYS[e], 10.0, None, v)
        for p, e, v in CB_CHANGES
    ]
    return pl.DataFrame(
        rows,
        schema="ts_code,bond_short_name,publish_date,change_date,convert_price_initial,convertprice_bef,convertprice_aft".split(
            ","
        ),
        orient="row",
    )


def cb_call():
    rows = [
        (
            "113001.SH",
            "强赎",
            "已满足强赎条件",
            DAYS[CALL_ANN - 2],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "113001.SH",
            "强赎",
            "公告实施强赎",
            DAYS[CALL_ANN],
            DAYS[CALL_RECORD + 1],
            100.3,
            100.24,
            None,
            None,
            DAYS[CALL_RECORD + 2],
            DAYS[CALL_RECORD],
        ),
    ]
    return pl.DataFrame(
        rows,
        schema="ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,call_vol,call_amount,payment_date,call_reg_date".split(
            ","
        ),
        orient="row",
    )


def cb_share():
    rows = [
        (
            "113001.SH",
            "测试转债",
            DAYS[4],
            DAYS[3],
            1e9,
            10.0,
            10.0,
            1e8,
            1e7,
            10.0,
            1e8,
            1e7,
            10.0,
            9e8,
            1e9,
        ),
        (
            "113001.SH",
            "测试转债",
            DAYS[13],
            DAYS[12],
            1e9,
            10.0,
            8.5,
            3e8,
            3e7,
            30.0,
            4e8,
            4e7,
            40.0,
            6e8,
            1e9,
        ),
    ]
    return pl.DataFrame(
        rows,
        schema="ts_code,bond_short_name,publish_date,end_date,issue_size,convert_price_initial,convert_price,convert_val,convert_vol,convert_ratio,acc_convert_val,acc_convert_vol,acc_convert_ratio,remain_size,total_shares".split(
            ","
        ),
        orient="row",
    )


def cb_rate():
    rows = [
        ("113001.SH", 1, "20250101", "20251231", 0.2),
        ("113001.SH", 1, "20260101", "20261231", 0.4),
        ("123001.SZ", 1, "20251201", "20261130", 0.3),
    ]
    return pl.DataFrame(
        rows,
        schema="ts_code,rate_freq,rate_start_date,rate_end_date,coupon_rate".split(","),
        orient="row",
    )


def cb_issue():
    return pl.DataFrame(
        {
            "ts_code": ["113001.SH", "123001.SZ"],
            "ann_date": ["20241201", "20251120"],
            "issue_size": [1e9, 5e8],
        }
    )


class FakeFundTushare:
    """按接口与参数返回合成数据；记录调用；可指定 (接口, 分区参数) 失败一次。"""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)
        daily, adj, share = fund_bars()
        self.tables = dict(
            trade_cal=calendar(),
            fund_basic=fund_basic(),
            fund_daily=daily,
            fund_adj=adj,
            fund_share=share,
            fund_div=fund_div(),
            cb_basic=cb_basic(),
            cb_daily=cb_daily(),
            cb_price_chg=cb_price_chg(),
            cb_call=cb_call(),
            cb_share=cb_share(),
            cb_rate=cb_rate(),
            cb_issue=cb_issue(),
        )

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls.append((api, dict(params)))
        key = (
            api,
            params.get("trade_date") or params.get("ex_date") or params.get("ts_code"),
        )
        if key in self.fail:
            self.fail.discard(key)
            raise RuntimeError("故意失败")
        table = self.tables[api]
        if "trade_date" in params:
            table = table.filter(pl.col("trade_date") == params["trade_date"])
        if "ex_date" in params:
            table = table.filter(pl.col("ex_date") == params["ex_date"])
        if "ts_code" in params:
            table = table.filter(pl.col("ts_code").is_in(params["ts_code"].split(",")))
        return FetchResult(table, "2026-02-14T00:00:00+00:00", 1)


def test_fake_source_shapes():
    import asyncio

    source = FakeFundTushare()
    result = asyncio.run(
        source.fetch("cb_price_chg", {"ts_code": "113001.SH,123001.SZ"})
    )
    assert len(result.table) == 3
    daily = source.tables["fund_daily"]
    assert daily.filter(pl.col("ts_code") == "510001.SH").height == len(DAYS)
