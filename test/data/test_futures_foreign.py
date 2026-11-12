"""境外期货适配器（Databento、Bloomberg）：代码解析、十年回绕、交易日归属、单位、统一字段、接回测。

全部用合成夹具和假客户端（``futures_foreign_fixtures``），不联网、不产生计费请求。

这些测试证明的：在**形状与官方文档一致**的输入上，换算逻辑（代码 → 统一标识、UTC 时刻 → 交易日、
结算价 final / 初步 / 日内的取舍、持仓量的归属、单位、上市期过滤、前结算）得到与夹具独立算出的期望
逐位相同的统一字段，并且能走通面板 → 连续合约 → panelbt 输入。没证明的：真实账号返回的数据是否就是
这个形状和口径（字段含义、ts_ref 的时区、ICE 代码写法、Bloomberg 的日期与持仓量是否滞后一天等），这些
列在 ``docs/FOREIGN_FUTURES.md`` 的「未核实清单」和「首次核对清单」里。
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from futures_foreign_fixtures import (
    BRENT,
    CL,
    CME_DAYS,
    END,
    EXPIRED_TICKER,
    ICE_DAYS,
    OI_DELETED,
    OI_WITHOUT_REF,
    START,
    FakeBlpapi,
    FakeHistorical,
    build,
    expected_frame,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from ashare.data.futures_foreign import (
    CME,
    ICE_LONDON,
    MINUTE,
    PRODUCTS,
    build_panel,
    resolve_year,
    trade_dates,
    unified_symbol,
    write_parquet,
)
from ashare.data.futures_foreign_bloomberg import (
    BloombergFutures,
    parse_generic,
    parse_ticker,
)
from ashare.data.futures_foreign_databento import (
    DatabentoFutures,
    parse_continuous,
    parse_raw_symbol,
    settlement_rows,
)
from ashare.data.futures_schema import TABLES
from ashare.data.sources.bloomberg import BloombergSource
from ashare.data.sources.databento import DatabentoSource

UTC = dt.UTC


# ---------------------------------------------------------------- 十年回绕
def reference_year(digits, month, ref):
    """直白参考：在 [参照年 − 200, 参照年 + 200] 里逐年找末尾数字相同、交割月不早于参照月的最小年份。"""
    modulus = 10 ** len(digits)
    for year in range(ref.year - 200, ref.year + 200):
        if year % modulus == int(digits) and (year, month) >= (ref.year, ref.month):
            return year
    raise AssertionError


@pytest.mark.parametrize(
    ("digits", "month", "ref", "year"),
    [
        ("6", 12, dt.date(2026, 11, 19), 2026),  # CLZ6 在 2026-11 到期
        ("6", 11, dt.date(2026, 11, 3), 2026),  # 参照日就在交割月：仍是本年
        ("5", 10, dt.date(2026, 11, 3), 2035),  # 交割月已过：只能是下一个十年
        ("7", 1, dt.date(2026, 12, 17), 2027),  # CLF7
        ("0", 1, dt.date(2029, 12, 18), 2030),  # 跨十年：2029 年底到期的 F0 是 2030 年
        ("9", 12, dt.date(2029, 11, 19), 2029),
        ("0", 2, dt.date(2019, 12, 31), 2020),  # 布伦特 G0 在交割月前两个月到期
        ("16", 12, dt.date(2016, 11, 21), 2016),  # 两位年份
        ("99", 12, dt.date(1999, 11, 19), 1999),
        ("00", 1, dt.date(1999, 12, 20), 2000),  # 两位年份跨世纪
    ],
)
def test_resolve_year_cases(digits, month, ref, year):
    assert resolve_year(digits, month, ref) == year
    assert resolve_year(digits, month, ref.strftime("%Y%m%d")) == year


@settings(max_examples=300, deadline=None)
@given(
    year=st.integers(1985, 2090),
    month=st.integers(1, 12),
    lead=st.integers(0, 2),
    day=st.integers(1, 28),
    width=st.sampled_from([1, 2]),
)
def test_resolve_year_matches_reference_at_last_trade_date(
    year, month, lead, day, width
):
    """以最后交易日（交割月前 1–2 个月）为参照日，一位、两位年份都解回真实年份，并与直白参考一致。"""
    k = year * 12 + (month - 1) - lead
    ref = dt.date(k // 12, k % 12 + 1, day)
    digits = f"{year % 10**width:0{width}d}"
    assert (
        resolve_year(digits, month, ref) == year == reference_year(digits, month, ref)
    )


# ---------------------------------------------------------------- 代码解析
def test_parse_databento_codes():
    ref = dt.date(2026, 11, 19)
    assert parse_raw_symbol("CLZ6", ref) == ("CL", 2026, 12)
    assert parse_raw_symbol("CLF7", dt.date(2026, 12, 17)) == ("CL", 2027, 1)
    assert parse_raw_symbol("BZZ6", ref) == ("BZ", 2026, 12)
    assert parse_raw_symbol("BRN FMZ0026!", dt.date(2026, 10, 30)) == ("BRN", 2026, 12)
    assert parse_raw_symbol("BRN FMF0027", dt.date(2026, 11, 30)) == ("BRN", 2027, 1)
    for spread in ("CLZ6-CLF7", "CL:C1 Z6-F7", "UD:1V: VT 1104"):
        assert parse_raw_symbol(spread, ref) is None
    assert parse_continuous("CL.c.0") == ("CL", "c", 0)
    assert parse_continuous("CL.v.1") == ("CL", "v", 1)
    assert parse_continuous("BRN.n.0") == ("BRN", "n", 0)
    with pytest.raises(ValueError):
        parse_continuous("CL.x.0")


def test_parse_bloomberg_codes():
    assert parse_ticker("CLZ6 Comdty", dt.date(2026, 11, 19)) == ("CL", 2026, 12)
    assert parse_ticker("CLZ16 Comdty", dt.date(2016, 11, 21)) == ("CL", 2016, 12)
    assert parse_ticker("COF7 Comdty", dt.date(2026, 11, 30)) == ("B", 2027, 1)
    # 一位年份在十年边界：2029-12 到期的 CLF0 是 2030 年 1 月合约。
    assert parse_ticker("CLF0 Comdty", dt.date(2029, 12, 18)) == ("CL", 2030, 1)
    assert parse_ticker("CL1 Comdty", dt.date(2026, 1, 1)) is None
    assert parse_ticker("CLZ6 Index", dt.date(2026, 1, 1)) is None
    assert parse_generic("CL1 Comdty") == ("CL", 1)
    assert parse_generic("CO12 Comdty") == ("B", 12)
    assert parse_generic("CLZ6 Comdty") is None
    assert unified_symbol("CL", 2026, 12) == "CL2612.NYMEX"
    assert unified_symbol("B", 2027, 1) == "B2701.ICE"


# ---------------------------------------------------------------- 交易日归属
def _utc(y, m, d, h, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=UTC)


@pytest.mark.parametrize(
    ("session", "ts", "day"),
    [
        (CME, _utc(2026, 11, 1, 23), "20261102"),  # 周日 17:00 CST 开盘 → 周一
        (CME, _utc(2026, 11, 1, 22, 59), "20261102"),  # 周日 16:59 CST：周末顺延到周一
        (CME, _utc(2026, 11, 6, 21, 59), "20261106"),  # 周五 15:59 CST → 周五
        (CME, _utc(2026, 7, 8, 21, 59), "20260708"),  # 夏令时 16:59 CDT → 当天
        (CME, _utc(2026, 7, 8, 22, 0), "20260709"),  # 夏令时 17:00 CDT → 次日
        (CME, _utc(2026, 11, 2, 23, 0), "20261103"),  # 冬令时 17:00 CST → 次日
        (
            CME,
            _utc(2026, 11, 2, 22, 30),
            "20261102",
        ),  # 冬令时 16:30 CST → 当天（同一 UTC 钟点夏令时会是次日）
        (ICE_LONDON, _utc(2026, 11, 1, 23), "20261102"),  # 周日 23:00 伦敦开盘 → 周一
        (ICE_LONDON, _utc(2026, 7, 5, 22), "20260706"),  # 夏令时周日 23:00 BST
        (ICE_LONDON, _utc(2026, 11, 3, 22, 59), "20261103"),
        (ICE_LONDON, _utc(2026, 11, 3, 1), "20261103"),
    ],
)
def test_trade_date_cases(session, ts, day):
    assert trade_dates([ts], session) == [day]


def test_trade_date_rolls_over_holiday():
    open_days = set(CME_DAYS)
    # 感恩节（2026-11-26 休市）当天的成交归 11-27。
    assert trade_dates([_utc(2026, 11, 26, 15)], CME, open_days) == ["20261127"]
    assert trade_dates([_utc(2026, 11, 26, 15)], CME) == ["20261126"]
    assert trade_dates([None], CME) == [None]


def reference_trade_date(ts, session, open_days):
    """直白参考：交易日 = 满足「时刻早于 D 当地 cutoff 点」的最早开市日 D（与适配器的「≥cutoff 归次日再顺延」写法不同）。"""
    zone = ZoneInfo(session.tz)
    for day in sorted(open_days):
        boundary = dt.datetime.combine(
            dt.date(int(day[:4]), int(day[4:6]), int(day[6:])),
            dt.time(session.cutoff_hour),
            zone,
        )
        if ts < boundary:
            return day
    return None


@settings(max_examples=400, deadline=None)
@given(minutes=st.integers(0, 80 * 24 * 60), which=st.sampled_from(["cme", "ice"]))
def test_trade_dates_match_reference(minutes, which):
    session, days = (CME, CME_DAYS) if which == "cme" else (ICE_LONDON, ICE_DAYS)
    ts = _utc(2026, 10, 1, 0) + dt.timedelta(minutes=minutes)
    assert trade_dates([ts], session, set(days)) == [
        reference_trade_date(ts, session, days)
    ]


# ---------------------------------------------------------------- Databento 适配器
@pytest.fixture(scope="module")
def data():
    return build()


def _databento(products=("CL", "B"), budget=100.0, client=None, **kwargs):
    client = client or FakeHistorical()
    source = DatabentoSource(budget_usd=budget, client=client)
    return DatabentoFutures(source, list(products), START, END, **kwargs), client


def _expected(fake, volume="cleared"):
    return pl.DataFrame(
        expected_frame(fake.data, volume),
        schema={c.name: c.dtype for c in TABLES["daily"].values()},
    )


def test_databento_daily_matches_independent_expectation():
    adapter, fake = _databento()
    daily = adapter.daily()
    assert daily.equals(_expected(fake)), "日线与夹具独立算出的期望不一致"
    # 用例确实覆盖了：两个交易所、周一（含周末前的夜盘）、感恩节后的周五、持仓量缺 ts_ref 与被删除。
    assert set(daily["symbol"].str.extract(r"\.(\w+)$")) == {"NYMEX", "ICE"}
    row = daily.filter(
        (pl.col("symbol") == OI_WITHOUT_REF[0]) & (pl.col("date") == OI_WITHOUT_REF[1])
    )
    assert (
        row["open_interest"][0] == fake.data.expected[OI_WITHOUT_REF]["open_interest"]
    )
    row = daily.filter(
        (pl.col("symbol") == OI_DELETED[0]) & (pl.col("date") == OI_DELETED[1])
    )
    assert row["open_interest"][0] is None
    assert "20261126" not in set(
        daily.filter(pl.col("symbol").str.ends_with("NYMEX"))["date"]
    )
    assert "20261126" in set(
        daily.filter(pl.col("symbol").str.ends_with("ICE"))["date"]
    )


def test_databento_electronic_volume_option():
    adapter, fake = _databento(volume="electronic")
    assert adapter.daily().equals(_expected(fake, "electronic"))
    with pytest.raises(ValueError, match="volume"):
        _databento(volume="both")


def test_databento_contract_table_units():
    adapter, _ = _databento()
    contracts = adapter.contracts()
    cl = contracts.filter(pl.col("symbol") == "CL2612.NYMEX").row(0, named=True)
    assert (
        cl["multiplier"] == 1000.0
        and cl["currency"] == "USD"
        and cl["tick_size"] == 0.01
    )
    assert cl["quote_unit"] == "美元/桶" and cl["settlement_type"] == "physical"
    assert cl["last_trade_date"] == "20261119" and cl["delivery_month"] == "202612"
    assert cl["listing_date"] == "20240103"  # 2024-01-02 23:00 UTC = 17:00 CST → 次日
    brent = contracts.filter(pl.col("symbol") == "B2701.ICE").row(0, named=True)
    assert brent["exchange"] == "ICE" and brent["last_trade_date"] == "20261130"
    # 区间里有行情的合约：CLV6（09-22 到期）、B2611（09-30 到期）在 start 之前，不出现。
    assert set(contracts["symbol"]) == {c[0] for c in CL if c[3] >= START} | {
        c[0] for c in BRENT if c[3] >= START
    }
    diag = adapter.diagnostics()
    assert diag["unknown_instrument_bar_rows"] == 2  # 价差和对不上合约表的那条
    assert diag["outside_listing_rows"] == 0 and diag["maturity_conflicts"] == []


def test_settlement_takes_final_not_preliminary_or_intraday(data):
    stats = pl.DataFrame(
        [
            dict(
                instrument_id=1,
                ts_recv=_utc(2026, 11, 4, 19, 0),
                ts_ref=_utc(2026, 11, 4, 0),
                price=60.0,
                stat_type=3,
                update_action=1,
                stat_flags=2,
            ),
            dict(
                instrument_id=1,
                ts_recv=_utc(2026, 11, 4, 19, 30),
                ts_ref=_utc(2026, 11, 4, 0),
                price=61.0,
                stat_type=3,
                update_action=1,
                stat_flags=3,
            ),
            dict(
                instrument_id=1,
                ts_recv=_utc(2026, 11, 4, 19, 45),
                ts_ref=_utc(2026, 11, 4, 0),
                price=99.0,
                stat_type=3,
                update_action=1,
                stat_flags=11,
            ),
            dict(
                instrument_id=1,
                ts_recv=_utc(2026, 11, 5, 19, 0),
                ts_ref=_utc(2026, 11, 5, 0),
                price=62.0,
                stat_type=3,
                update_action=1,
                stat_flags=2,
            ),
            dict(
                instrument_id=2,
                ts_recv=_utc(2026, 11, 4, 19, 30),
                ts_ref=_utc(2026, 11, 4, 0),
                price=70.0,
                stat_type=3,
                update_action=1,
                stat_flags=3,
            ),
            dict(
                instrument_id=2,
                ts_recv=_utc(2026, 11, 4, 19, 40),
                ts_ref=_utc(2026, 11, 4, 0),
                price=70.0,
                stat_type=3,
                update_action=2,
                stat_flags=3,
            ),
        ]
    )
    final = {
        (r["instrument_id"], r["date"]): r["settlement"]
        for r in settlement_rows(stats).iter_rows(named=True)
    }
    assert final == {(1, "20261104"): 61.0, (2, "20261104"): None}
    anyway = {
        (r["instrument_id"], r["date"]): r["settlement"]
        for r in settlement_rows(stats, settlement="any").iter_rows(named=True)
    }
    assert anyway[(1, "20261105")] == 62.0 and anyway[(1, "20261104")] == 61.0


def test_databento_calendar_from_settlements():
    adapter, _ = _databento()
    cal = adapter.calendar()
    closed = cal.filter(~pl.col("is_open") & (pl.col("date") >= "20261101"))
    nymex = set(closed.filter(pl.col("exchange") == "NYMEX")["date"])
    ice = set(closed.filter(pl.col("exchange") == "ICE")["date"])
    assert "20261126" in nymex and "20261126" not in ice
    assert "20261225" in nymex and "20261225" in ice
    assert set(
        cal.filter(pl.col("is_open") & (pl.col("exchange") == "NYMEX"))["date"]
    ) == {d for d in CME_DAYS if START <= d <= END}
    assert cal["date"].min() == START and cal["date"].max() == END


def test_params_margin_given_or_empty():
    adapter, _ = _databento()
    params = adapter.params()
    assert len(params) == len(adapter.daily())
    assert params["margin_long"].null_count() == len(params) and params[
        "fee_open_rate"
    ].null_count() == len(params)
    adapter, _ = _databento(margin={"CL": 0.12})
    params = adapter.params().join(
        adapter.contracts().select("symbol", "product"), on="symbol"
    )
    assert (params.filter(pl.col("product") == "CL")["margin_long"] == 0.12).all()
    assert params.filter(pl.col("product") == "B")["margin_long"].null_count() > 0


def test_databento_minute_bars():
    adapter, fake = _databento(products=("CL",))
    minute = adapter.minute("20261104", "20261104")
    assert list(minute.columns) == list(MINUTE)
    assert minute.height == 23 * 60 and set(minute["date"]) == {"20261104"}
    assert minute["ts"].min() == dt.datetime(
        2026, 11, 3, 23, 0, tzinfo=UTC
    )  # 前一天 17:00 CST
    assert (
        fake.count("get_range") == 4
    )  # definition、statistics、ohlcv-1h、ohlcv-1m 各一次


def test_databento_source_continuous_and_mapping():
    adapter, _ = _databento(reference_continuous=True)
    source = adapter.source_continuous()
    assert set(source["continuous"]) == {
        "CL.c.0",
        "CL.n.0",
        "CL.v.0",
        "BRN.c.0",
        "BRN.n.0",
        "BRN.v.0",
    }
    mapping = adapter.mapping()
    nearby = mapping.filter(
        (pl.col("continuous") == "CL.c.0") & (pl.col("date") == "20261119")
    )
    after = mapping.filter(
        (pl.col("continuous") == "CL.c.0") & (pl.col("date") == "20261120")
    )
    assert (
        nearby["symbol"][0] == "CL2612.NYMEX" and after["symbol"][0] == "CL2701.NYMEX"
    )
    assert set(mapping["kind"]) == {"main", "nearby"}
    # 默认不调符号映射接口
    plain, fake = _databento()
    plain.daily()
    assert fake.count("resolve") == 0 and plain.mapping().is_empty()


# ---------------------------------------------------------------- Bloomberg 适配器
def _bloomberg(products=("CL", "B")):
    api = FakeBlpapi()
    return BloombergFutures(BloombergSource(api), list(products), START, END), api


def test_bloomberg_daily_matches_independent_expectation():
    adapter, api = _bloomberg()
    assert adapter.daily().equals(_expected(api))
    contracts = adapter.contracts()
    assert EXPIRED_TICKER not in adapter.diagnostics()["skipped_tickers"]
    assert "CL1612.NYMEX" not in set(contracts["symbol"])  # 到期合约在区间外
    row = contracts.filter(pl.col("symbol") == "CL2612.NYMEX").row(0, named=True)
    assert row["delivery_date"] == "20261228" and row["multiplier"] == 1000.0
    overrides = [
        v.get("overrides")
        for op, v in api.requests
        if op == "ReferenceDataRequest" and "FUT_CHAIN" in v["fields"]
    ]
    assert overrides == [[{"fieldId": "INCLUDE_EXPIRED_CONTRACTS", "value": "Y"}]]
    generic = adapter.source_continuous()
    assert set(generic["continuous"]) == {
        "CL1 Comdty",
        "CL2 Comdty",
        "CO1 Comdty",
        "CO2 Comdty",
    }


def test_bloomberg_minute_matches_databento():
    adapter, _ = _bloomberg(("CL",))
    minute = adapter.minute("20261104", "20261104")
    other, _ = _databento(products=("CL",))
    assert minute.equals(other.minute("20261104", "20261104"))


def test_both_sources_give_same_unified_tables():
    """同一份合成市场，经两家的形状进来，统一字段的日线、日历逐位相同（合约表只差来源独有的字段）。"""
    a, _ = _databento()
    b, _ = _bloomberg()
    assert a.daily().equals(b.daily())
    assert (
        a.calendar()
        .sort("exchange", "date")
        .equals(b.calendar().sort("exchange", "date"))
    )
    keep = [
        "symbol",
        "exchange",
        "product",
        "last_trade_date",
        "delivery_month",
        "multiplier",
        "currency",
        "tick_size",
    ]
    assert a.contracts().select(keep).equals(b.contracts().select(keep))


# ---------------------------------------------------------------- 落盘与接回测
def test_write_parquet_roundtrip(tmp_path):
    adapter, _ = _databento(products=("CL",))
    paths = write_parquet(
        adapter, tmp_path / "foreign", minute=adapter.minute("20261104", "20261104")
    )
    names = sorted(p.stem for p in paths)
    assert names == ["calendar", "contracts", "daily", "mapping", "minute", "params"]
    assert pl.read_parquet(tmp_path / "foreign/daily.parquet").equals(adapter.daily())
    assert (tmp_path / "foreign/diagnostics.json").exists()


def test_adapter_to_continuous_to_panelbt(tmp_path):
    """合成月度合约链：适配器 → 面板 → 本仓库的换月规则拼连续合约 → panelbt 输入数组。"""
    adapter, _ = _databento(products=("CL",))
    fut = build_panel(adapter, tmp_path)
    daily = adapter.daily()
    cont = fut.continuous("CL", rule="open_interest")
    # 直白参考：第 t 日持有「第 t−1 日持仓量最大且最后交易日晚于第 t 日」的合约（夹具里持仓量只向后转移）。
    last = dict(
        zip(
            adapter.contracts()["symbol"],
            adapter.contracts()["last_trade_date"],
            strict=True,
        )
    )
    days = sorted(set(daily["date"]))
    oi = {
        (r["symbol"], r["date"]): r["open_interest"] or 0
        for r in daily.iter_rows(named=True)
    }
    expected = {}
    for prev, day in zip(days, days[1:], strict=False):
        live = [s for s in last if last[s] > day and (s, prev) in oi]
        expected[day] = max(live, key=lambda s: (oi[(s, prev)], -int(s[2:6])))
    got = dict(zip(cont["date"], cont["symbol"], strict=True))
    assert got == expected
    settle = {
        (r["symbol"], r["date"]): r["settlement"] for r in daily.iter_rows(named=True)
    }
    assert cont["price"].to_list() == [
        settle[(s, d)] for d, s in zip(cont["date"], cont["symbol"], strict=True)
    ]
    rolls = fut.roll_schedule("CL", rule="open_interest")
    assert rolls["to_symbol"].to_list() == [
        "CL2612.NYMEX",
        "CL2701.NYMEX",
        "CL2702.NYMEX",
    ]

    from ashare.data.futures_panelbt import panelbt_inputs

    inputs = panelbt_inputs(
        fut,
        products=["CL"],
        missing={"margin_rate": 0.1, "fee_rate": 0.0, "fee_per_contract": 2.0},
    )
    j = list(inputs["symbols"]).index("CL2612.NYMEX")
    t = list(inputs["dates"]).index("20261104")
    assert inputs["futures"].multiplier[j] == 1000.0
    assert inputs["futures"].settlement[t, j] == settle[("CL2612.NYMEX", "20261104")]
    ltd = list(inputs["dates"]).index("20261119")
    assert inputs["futures"].settlement[ltd, j] == settle[("CL2612.NYMEX", "20261119")]
    assert np.isfinite(inputs["market"]["close"]).any()
    import panelbt as bt

    market = bt.Market(
        **inputs["market"], futures=inputs["futures"]
    )  # 引擎接受这些数组
    assert market is not None


def test_panel_version_follows_content(tmp_path):
    """面板版本号随适配器输出内容变化（境外没有原始库清单，名字里带内容哈希）。"""
    a, _ = _databento(products=("CL",))
    data = build()
    data.stats = [
        dict(r, price=r["price"] + 1.0) if r["stat_type"] == 3 else r
        for r in data.stats
    ]
    b, _ = _databento(products=("CL",), client=FakeHistorical(data))
    assert a.name != b.name and a.name == _databento(products=("CL",))[0].name
    first = build_panel(a, tmp_path).version
    second = build_panel(b, tmp_path).version
    assert first != second


def test_product_table():
    cl, brent = PRODUCTS["CL"], PRODUCTS["B"]
    assert (cl.multiplier, cl.currency, cl.exchange, cl.session.tz) == (
        1000.0,
        "USD",
        "NYMEX",
        "America/Chicago",
    )
    assert (
        brent.multiplier,
        brent.exchange,
        brent.session.tz,
        brent.session.cutoff_hour,
    ) == (1000.0, "ICE", "Europe/London", 23)
    with pytest.raises(ValueError, match="不认识"):
        DatabentoFutures(DatabentoSource(), ["XX"], START, END)
    with pytest.raises(ValueError, match="不认识"):
        BloombergFutures(BloombergSource(FakeBlpapi()), ["BZ"], START, END)
