"""境外期货适配器的区间两端、费用闸门的整份计划预检、字段与结算价的完整性检查（复核返工）。

全部用合成夹具和假客户端，不联网、不产生计费请求。夹具数据覆盖 2026-09-01 至 2027-01-15，比每个用例的
[start, end] 两头都宽：取数窗口多取的那几天（start 前的 K 线、end 傍晚之后的 K 线）不能混进区间。
"""

import copy
import threading
import time

import numpy as np
import polars as pl
import pytest
from futures_foreign_fixtures import (
    CLOSED_DAYS,
    CME_DAYS,
    ICE_DAYS,
    MINUTE_DAYS,
    FakeBlpapi,
    FakeHistorical,
    build,
    expected_frame,
    minute_bars,
    utc,
)

from ashare.data.futures_continuous import adjust_prices
from ashare.data.futures_foreign import CME, build_panel, conform_minute, trade_dates
from ashare.data.futures_foreign_bloomberg import BloombergFutures
from ashare.data.futures_foreign_databento import DatabentoFutures
from ashare.data.futures_schema import TABLES
from ashare.data.sources.bloomberg import BloombergError, BloombergSource
from ashare.data.sources.databento import BudgetExceeded, DatabentoSource

DATA = build()

# 每组 (start, end) 各有用意：周一（交易时段从上周五之后的周日傍晚开始）、感恩节后的周五、月初且跨 CME
# 夏令时结束、end 在圣诞前一天、圣诞后的周一且跨元旦、end 是数据的最后一天（之后没有统计）、整段。
WINDOWS = [
    ("20261005", "20261016"),
    ("20261127", "20261211"),
    ("20261102", "20261130"),
    ("20261201", "20261224"),
    ("20261228", "20270108"),
    ("20270104", "20270115"),
    ("20261001", "20261231"),
]


def _expected(start, end, volume="cleared", data=DATA):
    return pl.DataFrame(
        expected_frame(data, volume, start, end),
        schema={c.name: c.dtype for c in TABLES["daily"].values()},
    )


def _databento(start, end, products=("CL", "B"), data=DATA, budget=1e6, **kwargs):
    fake = FakeHistorical(data)
    source = DatabentoSource(budget_usd=budget, client=fake)
    kwargs.setdefault("closed_days", CLOSED_DAYS)
    return DatabentoFutures(source, list(products), start, end, **kwargs), fake


def _bloomberg(start, end, products=("CL", "B"), api=None, **kwargs):
    api = api or FakeBlpapi(DATA)
    return BloombergFutures(
        BloombergSource(api), list(products), start, end, **kwargs
    ), api


# ---------------------------------------------------------------- P0：区间两端
@pytest.mark.parametrize(("start", "end"), WINDOWS)
def test_daily_matches_expectation_for_any_window(start, end):
    """任意 start/end：日线逐位等于夹具独立算出的期望（start 当天的开盘、最低、电子盘成交量不含 start
    之前几天的 K 线；end 傍晚之后归下一交易日的 K 线不算进 end；start 当天的前结算价有值）。"""
    expected = _expected(start, end)
    assert expected.height > 0 and expected["date"].min() == start
    for volume in ("cleared", "electronic"):
        adapter, _ = _databento(start, end, volume=volume)
        assert adapter.daily().equals(_expected(start, end, volume)), volume
    bbg, _ = _bloomberg(start, end)
    assert bbg.daily().equals(expected)
    adapter, _ = _databento(start, end)
    cme = {d for d in CME_DAYS if start <= d <= end}
    ice = {d for d in ICE_DAYS if start <= d <= end}
    for a in (adapter, bbg):
        cal = a.calendar()
        assert (cal["date"].min(), cal["date"].max()) == (start, end)
        opened = cal.filter(pl.col("is_open"))
        assert set(opened.filter(pl.col("exchange") == "NYMEX")["date"]) == cme
        assert set(opened.filter(pl.col("exchange") == "ICE")["date"]) == ice
    assert adapter.diagnostics()["calendar_unknown_days"] == {}


def test_reviewer_case_first_day_open():
    """复核的复现：start = 2026-10-05（周一）时 CLZ6 当天的开盘价、电子盘成交量只来自 10-04 傍晚之后。"""
    adapter, _ = _databento(
        "20261005", "20261009", products=("CL",), volume="electronic"
    )
    row = adapter.daily().filter(
        (pl.col("symbol") == "CL2612.NYMEX") & (pl.col("date") == "20261005")
    )
    want = DATA.expected[("CL2612.NYMEX", "20261005")]
    assert row["open"][0] == want["open"] and row["low"][0] == want["low"]
    assert row["volume"][0] == want["electronic_volume"]
    assert (
        row["previous_settlement"][0]
        == DATA.expected[("CL2612.NYMEX", "20261002")]["settlement"]
    )


def test_trade_dates_outside_known_calendar_are_dropped():
    friday = utc(2026, 10, 2, 20)  # 15:00 CDT 周五，自然交易日 10-02
    assert trade_dates([friday], CME, {"20261005", "20261006"}) == [None]
    # 已知范围内的休市日照样顺延；顺延越过最后一个开市日的为空
    thanksgiving = utc(2026, 11, 26, 15)
    assert trade_dates([thanksgiving], CME, {"20261125", "20261127"}) == ["20261127"]
    late = utc(2026, 11, 27, 23)  # 17:00 CST 周五 → 自然交易日下周一
    assert trade_dates([late], CME, {"20261125", "20261127"}) == [None]
    # 显式给已知范围：范围外不猜
    assert trade_dates(
        [thanksgiving], CME, {"20261127"}, known=("20261127", "20261127")
    ) == [None]


def _minute_expected(days):
    rows = [b for b in minute_bars() if b["_date"] in days]
    frame = pl.DataFrame(
        dict(
            ts=[b["ts_event"] for b in rows],
            date=[b["_date"] for b in rows],
            symbol=["CL2612.NYMEX"] * len(rows),
            open=[b["open"] for b in rows],
            high=[b["high"] for b in rows],
            low=[b["low"] for b in rows],
            close=[b["close"] for b in rows],
            volume=[float(b["volume"]) for b in rows],
        )
    )
    return conform_minute(frame)


@pytest.mark.parametrize(
    ("start", "end"),
    [("20261102", "20261105"), ("20261102", "20261102"), ("20261104", "20261105")],
)
def test_minute_matches_expectation_both_sources(start, end):
    days = {d for d in MINUTE_DAYS if start <= d <= end}
    want = _minute_expected(days)
    adapter, _ = _databento("20261102", "20261113", products=("CL",))
    assert adapter.minute(start, end).equals(want)
    bbg, _ = _bloomberg("20261102", "20261113", products=("CL",))
    assert bbg.minute(start, end).equals(want)


def test_minute_outside_adapter_range_raises_before_spending():
    adapter, fake = _databento("20261105", "20261106", products=("CL",))
    adapter.daily()
    costs, ranges = fake.count("get_cost"), fake.count("get_range")
    with pytest.raises(ValueError, match="要落在适配器的区间"):
        adapter.minute("20261104", "20261104")
    assert (fake.count("get_cost"), fake.count("get_range")) == (costs, ranges)
    bbg, api = _bloomberg("20261105", "20261106", products=("CL",))
    bbg.daily()
    sent = len(api.requests)
    with pytest.raises(ValueError, match="要落在适配器的区间"):
        bbg.minute("20261104", "20261106")
    assert len(api.requests) == sent


# ---------------------------------------------------------------- P1-1、P1-2：费用闸门
def _plan_costs(adapter):
    return [c for _, c in adapter.estimate()["requests"]]


def test_whole_plan_is_checked_before_first_fetch():
    """预算只够前两个请求：第三个请求让累计超出预算，一个都不取，报出整份计划的总价、已花和差额。"""
    adapter, fake = _databento("20261102", "20261113", products=("CL",), budget=0)
    costs = _plan_costs(adapter)
    assert len(costs) == 3
    adapter.source.budget_usd = costs[0] + costs[1] + costs[2] / 2  # 第 3 个请求超出
    estimates = fake.count("get_cost")
    with pytest.raises(BudgetExceeded) as caught:
        adapter.daily()
    err = caught.value
    assert fake.count("get_range") == 0
    assert fake.count("get_cost") - estimates == 3
    assert err.plan_total == pytest.approx(sum(costs)) and err.spent == 0
    assert err.shortfall == pytest.approx(costs[2] / 2)
    text = str(err)
    assert f"{sum(costs):.4f}" in text and f"{costs[2] / 2:.4f}" in text
    # 照报错建议把预算调到「已批准 + 计划总价」，重试成功，每个请求只取一次。
    adapter.source.budget_usd = err.spent + err.plan_total
    assert adapter.daily().height > 0
    assert fake.count("get_range") == 3
    assert adapter.source.spent_usd == pytest.approx(sum(costs))


def test_retry_after_partial_failure_does_not_refetch():
    adapter, fake = _databento("20261102", "20261113", products=("CL",), budget=0)
    costs = _plan_costs(adapter)
    adapter.source.budget_usd = sum(costs)
    original, calls = fake.timeseries.get_range, []

    def flaky(**kwargs):
        calls.append(kwargs["schema"])
        if len(calls) == 3:
            raise ConnectionError("网络断了")
        return original(**kwargs)

    fake.timeseries.get_range = flaky
    with pytest.raises(ConnectionError):
        adapter.daily()
    # 失败的那个请求也已记账（发出后可能已计费，宁可多记）；前两个缓存着。
    assert adapter.source.spent_usd == pytest.approx(sum(costs))
    with pytest.raises(BudgetExceeded) as caught:
        adapter.daily()
    assert caught.value.plan_total == pytest.approx(costs[2])
    assert len(caught.value.requests) == 1
    adapter.source.budget_usd = caught.value.spent + caught.value.plan_total
    assert adapter.daily().height > 0
    assert len(calls) == 4  # 前两个没有重取，只重取了失败的那个
    assert calls.count(calls[0]) == 1 and calls.count(calls[1]) == 1


@pytest.mark.parametrize(
    ("budget", "error"),
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-1.0, ValueError),
        ("5", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_budget_must_be_finite_nonnegative_number(budget, error):
    with pytest.raises(error):
        DatabentoSource(budget_usd=budget, client=FakeHistorical(DATA))
    fake = FakeHistorical(DATA)
    source = DatabentoSource(budget_usd=10, client=fake)
    source.budget_usd = budget  # 事后改坏也拦得住，且不估价、不取数
    with pytest.raises(error):
        source.fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert fake.count("get_cost") == 0 and fake.count("get_range") == 0


@pytest.mark.parametrize(
    ("cost", "error"),
    [
        (float("nan"), ValueError),
        (-0.5, ValueError),
        (float("inf"), ValueError),
        ("0.1", TypeError),
        (True, TypeError),
    ],
)
def test_bad_cost_estimate_blocks_fetch(cost, error):
    fake = FakeHistorical(DATA)
    fake.metadata.get_cost = lambda **kw: cost
    source = DatabentoSource(budget_usd=100, client=fake)
    with pytest.raises(error, match="get_cost"):
        source.fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert fake.count("get_range") == 0 and source.spent_usd == 0


def test_budget_holds_under_concurrency():
    """八个线程共用一个预算只够一个请求的 source：只有一个取到，其余报 BudgetExceeded，花费不超预算。"""
    fake = FakeHistorical(DATA)
    original = fake.metadata.get_cost

    def slow_cost(**kwargs):
        time.sleep(0.02)  # 拉长「估价 → 记账」之间的窗口，没有锁时必然交错
        return original(**kwargs)

    fake.metadata.get_cost = slow_cost
    source = DatabentoSource(budget_usd=0, client=fake)
    one = original(
        dataset="GLBX.MDP3",
        start="2026-11-02",
        end="2026-11-03",
        symbols=["CL.FUT"],
        schema="definition",
        stype_in="parent",
    )
    source.budget_usd = one * 1.5
    barrier, results = threading.Barrier(8), []

    def worker(k):
        barrier.wait()
        try:
            source.fetch(
                "GLBX.MDP3",
                "definition",
                ["CL.FUT"],
                f"2026-11-{2 + k:02d}",
                f"2026-11-{3 + k:02d}",
                stype_in="parent",
            )
            results.append("ok")
        except BudgetExceeded:
            results.append("over")

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert results.count("ok") == 1 and results.count("over") == 7
    assert fake.count("get_range") == 1
    assert source.spent_usd <= source.budget_usd
    # 八个线程同时取同一个请求、预算充足：只取一次、只记一次账（没有锁时都在估价时看到「没缓存」，各取一次）。
    source = DatabentoSource(budget_usd=100, client=fake)
    before = fake.count("get_range")
    barrier, frames = threading.Barrier(8), []

    def same():
        barrier.wait()
        frames.append(
            source.fetch(
                "GLBX.MDP3",
                "definition",
                ["CL.FUT"],
                "2026-11-02",
                "2026-11-03",
                stype_in="parent",
            )
        )

    threads = [threading.Thread(target=same) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(frames) == 8 and fake.count("get_range") - before == 1
    assert source.spent_usd == pytest.approx(one)


# ---------------------------------------------------------------- 定义（合约表）与统计的口径
def test_definition_multiplier_is_used_not_product_table():
    """定义里的 unit_of_measure_qty 与品种表不同时用定义的值，并记进诊断（复核 M5）。"""
    data = copy.copy(DATA)
    data.defs = [
        dict(r, unit_of_measure_qty=500.0) if r["instrument_id"] == 1002 else r
        for r in DATA.defs
    ]
    adapter, _ = _databento("20261102", "20261106", products=("CL",), data=data)
    contracts = adapter.contracts()
    multiplier = dict(zip(contracts["symbol"], contracts["multiplier"], strict=True))
    assert multiplier["CL2612.NYMEX"] == 500.0 and multiplier["CL2701.NYMEX"] == 1000.0
    assert [d["raw"] for d in adapter.diagnostics()["multiplier_mismatch"]] == ["CLZ6"]


@pytest.mark.parametrize(
    "field", ["unit_of_measure_qty", "min_price_increment", "currency"]
)
def test_definition_missing_field_raises(field):
    data = copy.copy(DATA)
    blank = float("nan") if field != "currency" else ""
    data.defs = [
        dict(r, **{field: blank}) if r["instrument_id"] == 1002 else r
        for r in DATA.defs
    ]
    adapter, _ = _databento("20261102", "20261106", products=("CL",), data=data)
    with pytest.raises(ValueError, match=field):
        adapter.daily()


def test_instrument_id_reused_for_another_contract_raises():
    data = copy.copy(DATA)
    extra = [
        dict(r, raw_symbol="CLH7")
        for r in DATA.defs
        if r["instrument_id"] == 1002 and r["ts_recv"] == utc(2026, 11, 4, 0, 5)
    ]
    assert len(extra) == 1
    data.defs = [*DATA.defs, *extra]
    adapter, _ = _databento("20261102", "20261106", products=("CL",), data=data)
    with pytest.raises(ValueError, match="instrument_id"):
        adapter.daily()


def test_dbn_v1_undefined_quantity_is_missing():
    """DBN v1 的统计数量未定义值是 i32::MAX（2147483647），不能当成 21 亿手持仓。"""
    target = ("CL2612.NYMEX", "20261105")
    data = copy.copy(DATA)
    ref = utc(2026, 11, 5, 0)
    data.stats = [
        dict(r, quantity=2**31 - 1)
        if r["instrument_id"] == 1002 and r["stat_type"] == 9 and r["ts_ref"] == ref
        else r
        for r in DATA.stats
    ]
    adapter, _ = _databento("20261102", "20261106", products=("CL",), data=data)
    row = adapter.daily().filter(
        (pl.col("symbol") == target[0]) & (pl.col("date") == target[1])
    )
    assert row["open_interest"][0] is None


# ---------------------------------------------------------------- P1-4：结算价完整性
def _drop_settlement(stats, iid, day):
    ref = utc(int(day[:4]), int(day[4:6]), int(day[6:]), 0)
    return [
        r
        for r in stats
        if not (
            r["instrument_id"] == iid and r["stat_type"] == 3 and r["ts_ref"] == ref
        )
    ]


def test_ice_without_final_settlements_raises():
    """ICE 的结算价若没有 final 位（stat_flags 口径与 GLBX 不同），不能悄悄变成「全部休市」。"""
    data = copy.copy(DATA)
    data.stats = [
        dict(r, stat_flags=2)
        if r["stat_type"] == 3 and 2000 <= r["instrument_id"] < 2100
        else r
        for r in DATA.stats
    ]
    adapter, _ = _databento("20261102", "20261113", data=data)
    with pytest.raises(ValueError, match="ICE.*结算价"):
        adapter.daily()
    relaxed, _ = _databento("20261102", "20261113", data=data, require_settlement=False)
    with pytest.raises(
        ValueError, match="ICE.*结算价"
    ):  # 整个交易所都没有，放宽开关也救不了
        relaxed.daily()


def test_traded_day_without_settlement_raises_unless_relaxed():
    data = copy.copy(DATA)
    data.stats = _drop_settlement(DATA.stats, 2002, "20261110")
    adapter, _ = _databento("20261102", "20261113", data=data)
    with pytest.raises(ValueError, match="B2701.ICE"):
        adapter.daily()
    relaxed, _ = _databento("20261102", "20261113", data=data, require_settlement=False)
    daily = relaxed.daily()
    row = daily.filter(
        (pl.col("symbol") == "B2701.ICE") & (pl.col("date") == "20261110")
    )
    assert row["settlement"][0] is None and row["open"][0] is not None
    after = daily.filter(
        (pl.col("symbol") == "B2701.ICE") & (pl.col("date") == "20261111")
    )
    assert after["previous_settlement"][0] is None
    assert relaxed.diagnostics()["settlement_missing"] == {"CL": 0, "B": 1}


def test_settlement_null_counts_in_diagnostics():
    adapter, _ = _databento("20261102", "20261113")
    adapter.daily()
    assert adapter.diagnostics()["settlement_missing"] == {"CL": 0, "B": 0}
    bbg, _ = _bloomberg("20261102", "20261113")
    bbg.daily()
    assert bbg.diagnostics()["settlement_missing"] == {"CL": 0, "B": 0}


# ---------------------------------------------------------------- P1-3：Bloomberg 字段与证券错误
def test_bloomberg_historical_security_error_raises():
    """历史请求的证券级错误要报出来（复核 M10）。"""
    api = FakeBlpapi(DATA)
    api.faults["CLZ6 Comdty"] = dict(
        securityError=dict(category="BAD_SEC", message="Unknown/Invalid security")
    )
    with pytest.raises(BloombergError, match="CLZ6 Comdty.*Unknown"):
        BloombergSource(api).historical(
            ["CLF7 Comdty", "CLZ6 Comdty"], ["PX_SETTLE"], "20261102", "20261106"
        )


@pytest.mark.parametrize("operation", ["historical", "reference"])
def test_bloomberg_field_exceptions_raise(operation):
    api = FakeBlpapi(DATA)
    api.faults["CLZ6 Comdty"] = dict(
        fieldExceptions=[
            dict(
                fieldId="PX_SETTLE",
                errorInfo=dict(category="BAD_FLD", message="Field not valid"),
            ),
            dict(
                fieldId="OPEN_INT",
                errorInfo=dict(category="BAD_FLD", message="Field not applicable"),
            ),
        ]
    )
    source = BloombergSource(api)
    call = (
        (
            lambda: source.historical(
                ["CLZ6 Comdty"], ["PX_SETTLE", "OPEN_INT"], "20261102", "20261106"
            )
        )
        if operation == "historical"
        else (lambda: source.reference(["CLZ6 Comdty"], ["NAME"]))
    )
    with pytest.raises(BloombergError, match=r"\['OPEN_INT', 'PX_SETTLE'\]"):
        call()
    adapter, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    with pytest.raises(BloombergError, match="字段出错"):
        adapter.daily()


def test_bloomberg_settle_column_all_empty_raises():
    api = FakeBlpapi(DATA)
    for code in ("CLX6", "CLZ6", "CLF7", "CLG7", "CLV6"):
        api.patches[f"{code} Comdty"] = dict(PX_SETTLE=None)
    adapter, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    with pytest.raises(ValueError, match="PX_SETTLE 整列为空"):
        adapter.daily()


def test_bloomberg_contract_without_settlement_raises():
    api = FakeBlpapi(DATA)
    api.patches["CLF7 Comdty"] = dict(PX_SETTLE=None)
    adapter, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    with pytest.raises(ValueError, match="CL2701.NYMEX"):
        adapter.daily()


@pytest.mark.parametrize(
    "field", ["FUT_CONT_SIZE", "FUT_TICK_SIZE", "LAST_TRADEABLE_DT", "CRNCY"]
)
def test_bloomberg_missing_contract_field_raises_not_falls_back(field):
    api = FakeBlpapi(DATA)
    api.patches["CLZ6 Comdty"] = {field: None}
    adapter, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    with pytest.raises(ValueError, match=field):
        adapter.daily()


# ---------------------------------------------------------------- P1-6：负价格与 ratio 复权
def test_ratio_adjust_rejects_nonpositive_prices():
    price = np.array([[10.0, 11.0], [-5.0, 12.0], [9.0, 13.0]])
    held = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="difference"):
        adjust_prices(held, price, "ratio")
    raw, adjusted, rolls = adjust_prices(held, price, "difference")
    assert raw.tolist() == [10.0, -5.0, 13.0] and adjusted.tolist() == [
        10.0,
        -5.0,
        13.0 + 9.0 - 13.0,
    ]
    zero = np.array([[10.0, 11.0], [0.0, 12.0]])
    with pytest.raises(ValueError, match="ratio"):
        adjust_prices(np.array([1, 0]), zero, "ratio")


def test_negative_prices_end_to_end(tmp_path):
    """负价格在适配器、面板里保留；ratio 复权报错，difference 照常。"""
    data = copy.copy(DATA)
    shift = 120.0
    data.stats = [
        dict(r, price=r["price"] - shift) if r["stat_type"] == 3 else r
        for r in DATA.stats
    ]
    data.bars = [
        dict(r, **{k: r[k] - shift for k in ("open", "high", "low", "close")})
        if r["instrument_id"] < 2000
        else r
        for r in DATA.bars
    ]
    adapter, _ = _databento("20261102", "20261130", products=("CL",), data=data)
    daily = adapter.daily()
    assert (daily["settlement"] < 0).any() and adapter.diagnostics()[
        "nonpositive_price_rows"
    ] > 0
    fut = build_panel(adapter, tmp_path)
    with pytest.raises(ValueError, match="difference"):
        fut.continuous("CL", rule="open_interest", adjust="ratio")
    cont = fut.continuous("CL", rule="open_interest", adjust="difference")
    assert (cont["price"] < 0).all() and cont.height == daily[
        "date"
    ].n_unique() - 1  # 首日没有前一日持仓量


# ---------------------------------------------------------------- 差异复核 P1-A：区间中间缺统计的工作日
def _drop_stats(days, types=(3,), ids=range(1000, 1100)):
    data = copy.copy(DATA)
    data.stats = [
        r
        for r in DATA.stats
        if not (
            r["stat_type"] in types
            and r["instrument_id"] in ids
            and r["ts_ref"] is not None
            and r["ts_ref"].strftime("%Y%m%d") in days
        )
    ]
    return data


@pytest.mark.parametrize(
    ("types", "kwargs"),
    [
        ((3,), dict(volume="electronic")),
        ((3,), dict(require_settlement=False)),
        ((3, 6, 9), dict(volume="electronic")),
        ((3, 6, 9), dict()),
    ],
)
def test_weekday_without_statistics_raises(types, kwargs):
    """复核的复现：CL 10-14（周三）缺结算价或整天统计。原来当天被当成休市、K 线并进 10-15 且几乎不报警；
    现在报错并列出日期。"""
    data = _drop_stats({"20261014"}, types)
    adapter, _ = _databento("20261012", "20261016", data=data, **kwargs)
    with pytest.raises(ValueError, match=r"NYMEX.*20261014"):
        adapter.daily()


def test_weekday_without_statistics_merged_when_declared_closed():
    """调用方把 10-14 列进已知休市日后，当天的 K 线按休市并进 10-15：开盘取 10-14 的第一根、成交量相加。"""
    data = _drop_stats({"20261014"}, (3, 6, 9))
    closed = {**CLOSED_DAYS, "NYMEX": [*CLOSED_DAYS["NYMEX"], "20261014"]}
    adapter, _ = _databento(
        "20261012", "20261016", data=data, volume="electronic", closed_days=closed
    )
    daily = adapter.daily()
    assert "20261014" not in set(
        daily.filter(pl.col("symbol").str.ends_with("NYMEX"))["date"]
    )
    row = daily.filter(
        (pl.col("symbol") == "CL2611.NYMEX") & (pl.col("date") == "20261015")
    ).row(0, named=True)
    wed, thu = (DATA.expected[("CL2611.NYMEX", d)] for d in ("20261014", "20261015"))
    assert row["open"] == wed["open"] and row["close"] == thu["close"]
    assert row["low"] == min(wed["low"], thu["low"])
    assert row["volume"] == wed["electronic_volume"] + thu["electronic_volume"]
    assert adapter.diagnostics()["closed_days_with_bars"]["NYMEX"]["20261014"] > 0


def test_real_holiday_needs_declaring():
    """感恩节当天有成交、没有结算价：不声明就报错，声明后按休市归并（诊断里列出）。"""
    adapter, _ = _databento("20261123", "20261130", products=("CL",), closed_days=None)
    with pytest.raises(ValueError, match="20261126"):
        adapter.daily()
    adapter, _ = _databento("20261123", "20261130", products=("CL",))
    assert adapter.daily().equals(
        _expected("20261123", "20261130").filter(
            pl.col("symbol").str.ends_with("NYMEX")
        )
    )
    assert set(adapter.diagnostics()["closed_days_with_bars"]["NYMEX"]) == {"20261126"}
    with pytest.raises(ValueError, match="不认识"):
        _databento("20261123", "20261130", closed_days={"CME": ["20261126"]})
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _databento("20261123", "20261130", closed_days={"NYMEX": ["2026-11-26"]})


def _bloomberg_without(day, *, drop_rows):
    """假 Bloomberg 的历史请求在 ``day`` 那天对 CL 合约不给 PX_SETTLE（``drop_rows`` 时整行不给）。"""
    api = FakeBlpapi(DATA)
    original = api.respond

    def respond(request):
        bodies = original(request)
        if request.operation != "HistoricalDataRequest":
            return bodies
        for body in bodies:
            data = body["securityData"]
            if not data["security"].startswith("CL"):
                continue
            rows = []
            for r in data["fieldData"]:
                if r["date"].strftime("%Y%m%d") == day:
                    if drop_rows:
                        continue
                    r = {k: v for k, v in r.items() if k != "PX_SETTLE"}
                rows.append(r)
            data["fieldData"] = rows
        return bodies

    api.respond = respond
    return api


def test_bloomberg_weekday_without_settle_raises():
    api = _bloomberg_without("20261104", drop_rows=False)
    adapter, _ = _bloomberg(
        "20261102", "20261106", products=("CL",), api=api, require_settlement=False
    )
    with pytest.raises(ValueError, match=r"NYMEX.*20261104"):
        adapter.daily()
    api = _bloomberg_without("20261104", drop_rows=False)
    adapter, _ = _bloomberg(
        "20261102",
        "20261106",
        products=("CL",),
        api=api,
        require_settlement=False,
        closed_days={"NYMEX": ["20261104"]},
    )
    adapter.daily()
    assert adapter.diagnostics()["closed_days_with_bars"] == {"NYMEX": {"20261104": 3}}


def test_bloomberg_minute_on_weekday_missing_from_history_raises():
    """历史请求整天不给（日线里看不出），分钟线却有：分钟线会并进下一开市日，所以报错。"""
    api = _bloomberg_without("20261104", drop_rows=True)
    adapter, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    assert "20261104" not in set(adapter.daily()["date"])
    with pytest.raises(ValueError, match=r"NYMEX.*20261104"):
        adapter.minute("20261103", "20261105")


# ---------------------------------------------------------------- 差异复核 P1-B：首个交易日缺失
def test_missing_first_trade_date_keeps_rows():
    api = FakeBlpapi(DATA)
    api.patches["CLZ6 Comdty"] = dict(FUT_FIRST_TRADE_DT=None)
    bbg, _ = _bloomberg("20261102", "20261106", products=("CL",), api=api)
    want = _expected("20261102", "20261106").filter(
        pl.col("symbol").str.ends_with("NYMEX")
    )
    assert bbg.daily().equals(want)
    assert bbg.diagnostics()["unknown_listing_contracts"] == ["CL2612.NYMEX"]
    listing = bbg.contracts().filter(pl.col("symbol") == "CL2612.NYMEX")["listing_date"]
    assert listing[0] is None

    data = copy.copy(DATA)
    data.defs = [
        dict(r, activation=None) if r["instrument_id"] == 1002 else r for r in DATA.defs
    ]
    adapter, _ = _databento("20261102", "20261106", products=("CL",), data=data)
    assert adapter.daily().equals(want)
    assert adapter.diagnostics()["unknown_listing_contracts"] == ["CL2612.NYMEX"]
    bbg_ok, _ = _bloomberg("20261102", "20261106", products=("CL",))
    bbg_ok.daily()
    assert bbg_ok.diagnostics()["unknown_listing_contracts"] == []


# ---------------------------------------------------------------- 差异复核 P2
@pytest.mark.parametrize(
    ("spent", "error"),
    [
        (-100.0, ValueError),
        (float("nan"), ValueError),
        (float("-inf"), ValueError),
        ("0", TypeError),
    ],
)
def test_tampered_spent_is_rejected_before_fetch(spent, error):
    fake = FakeHistorical(DATA)
    source = DatabentoSource(budget_usd=1, client=fake)
    source.spent_usd = spent
    with pytest.raises(error, match="spent_usd"):
        source.fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert fake.count("get_cost") == 0 and fake.count("get_range") == 0


def test_previous_open_day_before_bar_window_raises():
    """start 的上一开市日早于 K 线窗口起点（start−5 天）时报错，不悄悄少算 start 那个交易日的开头几小时。"""
    data = _drop_stats(
        {"20261020", "20261021", "20261022", "20261023"}, (3,), range(2000, 2100)
    )
    adapter, _ = _databento("20261026", "20261030", products=("B",), data=data)
    with pytest.raises(ValueError, match="早于 K 线窗口起点"):
        adapter.daily()


def test_trade_dates_explicit_known_range_stops_at_its_end():
    """显式给 ``known`` 且末日不是开市日时，顺延不越过它（默认用法下末日本身就是开市日，走不到这里）。"""
    monday = utc(2026, 11, 30, 15)  # 09:00 CST 周一 11-30
    open_days = {"20261127", "20261202"}
    assert trade_dates([monday], CME, open_days) == ["20261202"]
    assert trade_dates([monday], CME, open_days, known=("20261125", "20261130")) == [
        None
    ]
