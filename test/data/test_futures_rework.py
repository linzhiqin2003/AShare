"""期货数据线复核返工的回归测试：代码对齐、手续费清洗、按日乘数、参数不前视、换月与到期边界。

每个测试用手写的小原始库（Tushare 接口形状，非真实行情）。
"""

import asyncio
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from futures_fixtures import CONTRACTS, DAYS, FakeTushare

import ashare
import ashare.data as ad
from ashare.data.futures_continuous import select_held
from ashare.data.futures_panel import assemble, build_futures
from ashare.data.futures_panelbt import panelbt_inputs
from ashare.data.futures_sync import sync_futures
from ashare.data.futures_tushare import TushareFutures
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

CONTRACT_COLUMNS = dict(
    symbol=None,
    exchange=None,
    name="x",
    multiplier=None,
    trade_unit="吨",
    quote_unit="人民币元/吨",
    quote_unit_desc="1人民币元/吨",
    d_mode_desc="实物交割",
    d_month=None,
    last_ddate=None,
)


def contract(code, product, per_unit, first, last, **extra):
    row = dict(CONTRACT_COLUMNS, **extra)
    row.update(
        ts_code=code,
        symbol=code.split(".")[0],
        fut_code=product,
        per_unit=per_unit,
        list_date=first,
        delist_date=last,
    )
    return row


def write_days(store, name, rows):
    frame = pl.DataFrame(rows)
    for day in sorted(set(frame["trade_date"])):
        store.write(
            name,
            day,
            {},
            FetchResult(frame.filter(pl.col("trade_date") == day), "now", 1),
        )


def daily_row(code, day, settle, vol, unit, **extra):
    row = dict(
        ts_code=code,
        trade_date=day,
        pre_close=settle,
        pre_settle=settle,
        open=settle,
        high=settle + 5,
        low=settle - 5,
        close=settle,
        settle=settle,
        change1=0.0,
        change2=0.0,
        vol=vol,
        amount=None if unit is None else vol * settle * unit / 10000,
        oi=100.0,
        oi_chg=0.0,
    )
    row.update(extra)
    return row


def settle_row(code, day, **extra):
    row = dict(
        ts_code=code,
        trade_date=day,
        settle=100.0,
        trading_fee_rate=None,
        trading_fee=3.0,
        delivery_fee=None,
        b_hedging_margin_rate=0.08,
        s_hedging_margin_rate=0.08,
        long_margin_rate=0.1,
        short_margin_rate=0.1,
        offset_today_fee=6.0,
        exchange="X",
    )
    row.update(extra)
    return row


# ---------------------------------------------------------------- 代码对齐与手续费


def czce_store(tmp_path):
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [
            contract("AP2003.ZCE", "AP", 10.0, "20190301", "20200313"),
            contract("WH1001.ZCE", "WH", 20.0, "20090101", "20100114"),
            contract("WH2001.ZCE", "WH", 20.0, "20190101", "20200114"),
            contract("BC2407.INE", "BC", 5.0, "20230701", "20240715"),
            contract("SCTAS2407.INE", "SCTAS", 1000.0, "20230701", "20240615"),
        ],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    rows = [
        # 源把 AP2003 写成 AP1003、WH2001 写成 WH1001（与 2010 年的真合约同名）
        settle_row("AP1003.ZCE", "20190603", offset_today_fee=-1.0),
        settle_row("WH1001.ZCE", "20190603", offset_today_fee=-2.0),
        settle_row("WH1001.ZCE", "20091201", offset_today_fee=6.0, trading_fee=2.0),
        # 能源中心的品种写成上期所后缀，另有一行正确后缀
        settle_row("BC2407.SHF", "20240705", trading_fee_rate=0.01, trading_fee=None),
        settle_row("BC2407.INE", "20240705", trading_fee_rate=0.02, trading_fee=None),
        settle_row("BC2407.SHF", "20240708", trading_fee_rate=0.01, trading_fee=None),
        # 对不上的行：已过最后交易日
        settle_row("BC2407.INE", "20240801", trading_fee_rate=0.01, trading_fee=None),
        settle_row("SCTAS2407.INE", "20240705"),
    ]
    write_days(store, "futures_settle", rows)
    return store


def test_codes_resolved_by_listing_window(tmp_path):
    store = czce_store(tmp_path)
    adapter = TushareFutures(store)
    params = adapter.params().sort("date", "symbol")
    keys = params.select("date", "symbol").rows()
    assert ("20190603", "AP2003.CZCE") in keys
    assert ("20190603", "WH2001.CZCE") in keys
    assert ("20091201", "WH1001.CZCE") in keys
    # 同一天错后缀与正确后缀并存：保留正确后缀那行（费率 0.02‰）。
    bc = params.filter(pl.col("symbol") == "BC2407.INE")
    assert bc["date"].to_list() == ["20240705", "20240708"]
    assert bc["fee_open_rate"][0] == pytest.approx(0.02e-3)
    assert not params["symbol"].str.starts_with("SCTAS").any()
    found = adapter.diagnostics()["params"]
    assert found["rows"]["czce_decade"] == 2
    assert found["rows"]["exchange_suffix"] == 2
    assert found["unresolved"] == [
        dict(code="BC2407.INE", date="20240801", resolution="unmatched")
    ]


def test_fee_cleaning_and_close_today(tmp_path):
    store = czce_store(tmp_path)
    params = TushareFutures(store).params()
    row = lambda s, d: params.filter((pl.col("symbol") == s) & (pl.col("date") == d))  # noqa: E731
    # 郑商所平今费 -1、-2 置空，不是负手续费
    assert (
        row("AP2003.CZCE", "20190603")["fee_close_today_per_contract"].is_null().all()
    )
    assert (
        row("WH2001.CZCE", "20190603")["fee_close_today_per_contract"].is_null().all()
    )
    # 平今按手取 offset_today_fee 本身（不是开仓费）
    old = row("WH1001.CZCE", "20091201")
    assert old["fee_close_today_per_contract"][0] == 6.0
    assert old["fee_open_per_contract"][0] == 2.0
    assert (params.select(pl.col("^fee_.*$")).fill_null(0).min_horizontal() >= 0).all()


PER_HAND = ["A", "B", "C", "CS", "EB", "EG", "L", "P", "PG", "PP"]


def test_zero_fee_day_and_structural_zero(tmp_path):
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [contract(f"{k}2601.DCE", k, 10.0, "20250101", "20260115") for k in PER_HAND]
        + [contract("M2601.DCE", "M", 10.0, "20250101", "20260115")],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    rows = [
        settle_row(f"{k}2601.DCE", "20250928", trading_fee=0.0, trading_fee_rate=0.0)
        for k in PER_HAND
    ] + [settle_row(f"{k}2601.DCE", "20250929", trading_fee=1.0) for k in PER_HAND]
    rows.append(settle_row("M2601.DCE", "20250929", trading_fee=None))
    write_days(store, "futures_settle", rows)
    params = TushareFutures(store).params()
    # 交易所整天手续费全为 0 是源错误：置空；次日的真值照常。
    zero_day = params.filter(pl.col("date") == "20250928")
    assert len(zero_day) == len(PER_HAND)
    assert zero_day["fee_open_per_contract"].is_null().all()
    next_day = params.filter(pl.col("date") == "20250929")
    assert (
        next_day.filter(pl.col("symbol") != "M2601.DCE")["fee_open_per_contract"] == 1.0
    ).all()
    # 按手收费的品种，比例部分只在源给了费用时才是 0；源为空时两部分都为空。
    m = params.filter(pl.col("symbol") == "M2601.DCE")
    assert (
        m["fee_open_rate"].is_null().all()
        and m["fee_open_per_contract"].is_null().all()
    )


def test_ambiguous_code_not_resolved(tmp_path):
    """同品种同年份末位同月份、当天都在上市期的两个合约：不猜，计为 ambiguous。"""
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [
            contract("XX2003.ZCE", "XX", 10.0, "20190101", "20200313"),
            contract("XX3003.ZCE", "XX", 10.0, "20190101", "20300313"),
        ],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    write_days(store, "futures_settle", [settle_row("XX1003.ZCE", "20190603")])
    adapter = TushareFutures(store)
    assert adapter.params().is_empty()
    assert adapter.diagnostics()["params"]["unresolved"] == [
        dict(code="XX1003.ZCE", date="20190603", resolution="ambiguous")
    ]


def test_daily_rows_outside_listing_dropped(tmp_path):
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [contract("FU1901.SHF", "FU", 10.0, "20180716", "20181228")],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    write_days(
        store,
        "futures_daily",
        [
            daily_row("FU1901.SHF", day, 3000.0, 10.0, 10.0)
            for day in ("20180102", "20180717", "20181228", "20190102")
        ],
    )
    daily = TushareFutures(store).daily()
    assert daily["date"].to_list() == ["20180717", "20181228"]


def test_zero_prices_become_missing(tmp_path):
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [contract("EG2604.DCE", "EG", 10.0, "20250401", "20260415")],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    write_days(
        store,
        "futures_daily",
        [
            daily_row(
                "EG2604.DCE", "20260402", 5220.0, 0.0, 10.0, open=0.0, high=0.0, low=0.0
            ),
            daily_row(
                "EG2604.DCE",
                "20260403",
                5220.0,
                64.0,
                10.0,
                open=0.0,
                high=0.0,
                low=0.0,
            ),
        ],
    )
    daily = TushareFutures(store).daily()
    assert daily["open"].is_null().all() and daily["high"].is_null().all()
    assert daily["settlement"].to_list() == [5220.0, 5220.0]


# ---------------------------------------------------------------- 按日乘数


SHFE_DAYS = ["20130828", "20130829", "20130830", "20130902", "20130903", "20130910"]


def spec_store(tmp_path):
    store = Store(tmp_path)
    basic = pl.DataFrame(
        [
            contract("PB1309.SHF", "PB", 5.0, "20130102", "20130913"),
            contract("PB1310.SHF", "PB", 5.0, "20130102", "20131015"),
            contract("PB1312.SHF", "PB", 5.0, "20130102", "20131216"),  # 从未成交
            contract("RU0712.SHF", "RU", 10.0, "20061215", "20071214"),
            contract("RU0801.SHF", "RU", 10.0, "20070115", "20080115"),
        ],
        infer_schema_length=None,
    )
    store.write("futures_contracts", "ALL", {}, FetchResult(basic, "now", 1))
    rows = []
    for day in SHFE_DAYS:
        unit = 25.0 if day < "20130902" else 5.0
        if day <= "20130903":
            rows.append(daily_row("PB1309.SHF", day, 15000.0, 40.0, unit))
        if day <= "20130830" or day == "20130910":
            rows.append(daily_row("PB1310.SHF", day, 15100.0, 30.0, unit))
    # 2008 年前上期所成交额为空：RU0712 没有任何可反查的行；RU0801 跨入 2008 年可反查为 5。
    rows.append(daily_row("RU0712.SHF", "20071203", 20000.0, 10.0, None))
    rows.append(daily_row("RU0801.SHF", "20071203", 20000.0, 10.0, None))
    rows.append(daily_row("RU0801.SHF", "20080103", 20000.0, 10.0, 5.0))
    rows.append(daily_row("RU0801.SHF", "20080104", 20000.0, 10.0, 5.0))
    write_days(store, "futures_daily", rows)
    calendar = pl.DataFrame(
        dict(
            exchange=["SHFE"] * 9,
            cal_date=[*SHFE_DAYS, "20071203", "20080103", "20080104"],
            is_open=[1] * 9,
            pretrade_date=[None] * 9,
        )
    )
    store.write("futures_calendar", "SHFE", {}, FetchResult(calendar, "now", 1))
    write_days(
        store,
        "futures_settle",
        [
            settle_row(c, d)
            for c, d in (("PB1309.SHF", "20130828"), ("PB1310.SHF", "20130828"))
        ],
    )
    return store


def test_multiplier_segments(tmp_path):
    store = spec_store(tmp_path)
    table = TushareFutures(store).multipliers()
    seg = {
        s: g.sort("start_date").select("start_date", "multiplier", "source").rows()
        for (s,), g in table.group_by("symbol")
    }
    assert seg["PB1309.SHFE"] == [
        ("20130102", 25.0, "implied_from_amount"),
        ("20130902", 5.0, "contract_table"),
    ]
    # PB1310 在 08-30 之后到 09-10 才有新单位的成交：分界日取品种统一的 09-02。
    assert seg["PB1310.SHFE"][1][:2] == ("20130902", 5.0)
    assert seg["PB1312.SHFE"] == [("20130102", 5.0, "contract_table_unverified")]
    assert seg["RU0712.SHFE"][0][1:] == (5.0, "manual")
    assert seg["RU0801.SHFE"] == [("20070115", 5.0, "implied_from_amount")]
    basis = table.filter(pl.col("source") == "manual")["basis"][0]
    assert basis.startswith("未核实")
    info = {
        r["symbol"]: r for r in TushareFutures(store).contracts().iter_rows(named=True)
    }
    assert info["PB1309.SHFE"]["multiplier"] == 5.0  # 最后一段


def test_panel_multiplier_by_date_and_panelbt(tmp_path):
    store = spec_store(tmp_path)
    build_futures(store)
    fut = ad.open(tmp_path).futures()
    p = fut.panel(["multiplier"], symbols=["PB1309.SHFE", "PB1310.SHFE"])
    dates = list(p.dates)
    m = np.asarray(p["multiplier"])
    assert (
        m[dates.index("20130830"), 0] == 25.0 and m[dates.index("20130902"), 0] == 5.0
    )
    assert (
        m[dates.index("20130830"), 1] == 25.0 and m[dates.index("20130903"), 1] == 5.0
    )
    missing = dict(margin_rate=0.1, fee_rate=0.0, fee_per_contract=3.0)
    with pytest.raises(ValueError, match="改过交易单位"):
        panelbt_inputs(
            fut,
            symbols=["PB1309.SHFE", "PB1310.SHFE"],
            start="20130828",
            end="20130910",
            missing=missing,
        )
    split = panelbt_inputs(
        fut,
        symbols=["PB1309.SHFE"],
        start="20130828",
        end="20130910",
        missing=missing,
        multiplier_change="split",
    )
    assert list(split["symbols"]) == ["PB1309.SHFE", "PB1309.SHFE@20130902"]
    assert list(split["futures"].multiplier) == [25.0, 5.0]
    # 前一列在分界日前一行了结，后一列分界日起才能买卖。
    dates = list(split["dates"])
    assert split["futures"].expiry.tolist()[0] == dates.index("20130830")
    can = split["market"]["can_buy"]
    assert (
        not can[: dates.index("20130902"), 1].any() and can[dates.index("20130902"), 1]
    )
    assert not can[dates.index("20130902") :, 0].any()
    # 区间不跨分界日时不需要拆。
    plain = panelbt_inputs(
        fut, symbols=["PB1309.SHFE"], start="20130902", end="20130903", missing=missing
    )
    assert list(plain["futures"].multiplier) == [5.0]


# ---------------------------------------------------------------- 参数不前视


@pytest.fixture
def late_params(tmp_path):
    """结算参数从第 10 行才有（之前缺失），第 20 行起保证金上调。"""
    source = FakeTushare()
    settle = source.tables["fut_settle"]
    source.tables["fut_settle"] = settle.filter(pl.col("trade_date") >= DAYS[10])
    store = Store(tmp_path)
    asyncio.run(sync_futures(source, store, DAYS[0], DAYS[-1]))
    build_futures(store)
    return ad.open(tmp_path).futures()


def test_params_never_backfilled_from_future(late_params):
    fut = late_params
    fees = {"fee_rate": 0.0, "fee_per_contract": 0.0}
    with pytest.raises(ValueError, match="missing"):
        panelbt_inputs(fut, products=["RB"], missing=fees)
    inputs = panelbt_inputs(fut, products=["RB"], missing={"margin_rate": 0.5, **fees})
    margin = inputs["futures"].margin_rate
    names = list(inputs["symbols"])
    j = names.index("RB2604.SHFE")
    # 开头缺失的格子用调用方给的值，不是第 10 行以后的 0.1。
    assert (margin[:10, j] == 0.5).all() and margin[10, j] == pytest.approx(0.1)
    assert margin[25, j] == pytest.approx(0.11)  # max(买 0.11, 卖 0.10)
    # 按品种给缺省值
    by_product = panelbt_inputs(
        fut, products=["RB"], missing={"margin_rate": {"RB": 0.3}, **fees}
    )
    assert (by_product["futures"].margin_rate[:10, j] == 0.3).all()
    # param_fill="none"：第 10 行以后的缺口也不向前填。
    none = panelbt_inputs(
        fut, products=["RB"], missing={"margin_rate": 0.5, **fees}, param_fill="none"
    )
    assert none["futures"].margin_rate[10, j] == pytest.approx(0.1)


def test_params_prefix_unchanged_by_later_data(tmp_path):
    """防前视：只改第 15 行以后的参数，第 15 行以前的 panelbt 输入逐位不变。"""
    results = []
    for k, bump in enumerate((0.0, 0.3)):
        source = FakeTushare()
        settle = source.tables["fut_settle"]
        source.tables["fut_settle"] = settle.with_columns(
            long_margin_rate=pl.when(pl.col("trade_date") >= DAYS[15])
            .then(pl.col("long_margin_rate") + bump)
            .otherwise(pl.col("long_margin_rate"))
        ).filter(pl.col("trade_date") != DAYS[3])  # 中间有一天缺失，靠向前填
        store = Store(tmp_path / str(k))
        asyncio.run(sync_futures(source, store, DAYS[0], DAYS[-1]))
        build_futures(store)
        fut = ad.open(tmp_path / str(k)).futures()
        results.append(
            panelbt_inputs(fut, products=["RB", "IF"])["futures"].margin_rate
        )
    assert np.array_equal(results[0][:15], results[1][:15])
    assert not np.array_equal(results[0][15:], results[1][15:])


# ---------------------------------------------------------------- 换月与到期边界


def test_not_held_on_last_trade_date():
    """可选合约要求最后交易日晚于当天：指标再大也不能在最后交易日当天持有。"""
    dates = np.array(["d0", "d1", "d2", "d3"])
    expiry = np.array(["d2", "d9"])  # 第 0 列在第 2 行到期
    metric = np.array([[10.0, 1.0]] * 4)
    eligible = np.ones((4, 2), dtype=bool)
    eligible &= expiry[None, :] > dates[:, None]
    held = select_held(metric, eligible, np.array([0, 1]))
    assert held.tolist() == [-1, 0, 1, 1]


def test_panelbt_margin_is_max_of_long_short(tmp_path):
    source = FakeTushare()
    store = Store(tmp_path)
    asyncio.run(sync_futures(source, store, DAYS[0], DAYS[-1]))
    build_futures(store)
    fut = ad.open(tmp_path).futures()
    inputs = panelbt_inputs(fut, products=["RB"])
    j = list(inputs["symbols"]).index("RB2604.SHFE")
    # 夹具：第 20 行以后买 0.11、卖 0.10
    assert inputs["futures"].margin_rate[25, j] == pytest.approx(0.11)
    assert inputs["futures"].margin_rate[5, j] == pytest.approx(0.10)


def test_expiry_row_when_last_trade_date_is_not_a_trading_day(tmp_path):
    source = FakeTushare()
    # RB2602 最后一行是第 9 行（01-16 周五）；把最后交易日写成 01-17（周六，非交易日）。
    source.tables["fut_basic"] = source.tables["fut_basic"].with_columns(
        delist_date=pl.when(pl.col("ts_code") == "RB2602.SHF")
        .then(pl.lit("20260117"))
        .otherwise(pl.col("delist_date"))
    )
    store = Store(tmp_path)
    asyncio.run(sync_futures(source, store, DAYS[0], DAYS[-1]))
    build_futures(store)
    fut = ad.open(tmp_path).futures()
    inputs = panelbt_inputs(fut, products=["RB"])
    j = list(inputs["symbols"]).index("RB2602.SHFE")
    assert inputs["futures"].expiry[j] == 9  # 最后交易日之前最近的交易日


def test_store_scan_filters_futures_days(tmp_path):
    store = Store(tmp_path)
    asyncio.run(sync_futures(FakeTushare(), store, DAYS[0], DAYS[-1]))
    frame = store.read("futures_daily", DAYS[3], DAYS[5])
    assert sorted(set(frame["trade_date"])) == DAYS[3:6]


def test_assemble_rejects_rows_outside_listing(tmp_path):
    store = Store(tmp_path)
    asyncio.run(sync_futures(FakeTushare(), store, DAYS[0], DAYS[-1]))
    adapter = TushareFutures(store)
    contracts = adapter.contracts().with_columns(
        listing_date=pl.when(pl.col("symbol") == "RB2603.SHFE")
        .then(pl.lit(DAYS[2]))
        .otherwise(pl.col("listing_date"))
    )
    with pytest.raises(ValueError, match="上市期之外"):
        assemble(adapter.daily(), adapter.params(), contracts, DAYS)


def test_fixture_contracts_cover_both_exchanges():
    assert {c[1] for c in CONTRACTS} == {"SHFE", "CFFEX"}


def test_roll_happens_on_last_trade_date_not_after(tmp_path):
    """持仓再大，最后交易日当天也不再持有（可选合约要求最后交易日晚于当天）。"""
    source = FakeTushare()
    daily = source.tables["fut_daily"]
    source.tables["fut_daily"] = daily.with_columns(
        oi=pl.when(pl.col("ts_code") == "IF2602.CFX")
        .then(pl.col("oi") * 100)
        .otherwise(pl.col("oi"))
    )
    store = Store(tmp_path)
    asyncio.run(sync_futures(source, store, DAYS[0], DAYS[-1]))
    build_futures(store)
    fut = ad.open(tmp_path).futures()
    rolls = fut.roll_schedule("IF", rule="open_interest")
    assert rolls["date"].to_list() == [DAYS[14]]  # IF2602 的最后交易日
    series = fut.continuous("IF", rule="open_interest")
    assert series.filter(pl.col("date") == DAYS[14])["symbol"].to_list() == [
        "IF2603.CFFEX"
    ]
    assert series.filter(pl.col("date") == DAYS[13])["symbol"].to_list() == [
        "IF2602.CFFEX"
    ]
