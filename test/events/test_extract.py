"""内置事件提取：用手写的表行核对定义与口径。

预告、利润表、指数权重按原始库字段名（经 ``_store.scan``），快报和分红按公共表
``Data.table`` 的字段名。
"""

import polars as pl
import pytest

from ashare.events import extract


class FakeStore:
    def __init__(self, tables):
        self.tables = tables

    def scan(self, name):
        return self.tables[name].lazy()


class FakeData:
    def __init__(self, tables, securities=None):
        self._store = FakeStore(tables)
        self._sec = (
            securities
            if securities is not None
            else pl.DataFrame(
                {"symbol": [], "delisting_date": []},
                schema={"symbol": pl.String, "delisting_date": pl.String},
            )
        )

    def securities(self):
        return self._sec.lazy()

    def table(self, name):
        return self._public[name].lazy()


def with_public(data, **tables):
    data._public = tables
    return data


def forecast_rows():
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "A", "B", "B"],
            "ann_date": ["20240110", "20240125", "20240125", "20240112", "20240301"],
            "end_date": ["20231231"] * 4 + ["20240331"],
            "type": ["预增", "预增 ", "预增", "略增", "预减"],
            "p_change_min": [50.0, 70.0, 80.0, None, -60.0],
            "p_change_max": [100.0, 120.0, 120.0, 30.0, -40.0],
            "net_profit_min": [150.0, 180.0, 180.0, None, None],
            "net_profit_max": [200.0, 220.0, 220.0, None, None],
            "last_parent_net": [100.0, 100.0, 100.0, 50.0, 10.0],
            "first_ann_date": ["20240110", "20240110", "20240110", "20240112", None],
            "update_flag": ["0", "0", "1", "0", "0"],
        }
    )


def test_forecasts_first_and_revision():
    ev = extract.forecasts(FakeData({"forecast": forecast_rows()}))
    assert ev.height == 4  # A 的修正公告两行（换码重复）只留一条
    a = ev.filter(pl.col("symbol") == "A").sort("date")
    assert a["is_revision"].to_list() == [False, True]
    assert a["revision"].to_list() == [0, 1]
    assert a["forecast_type"].to_list() == ["预增", "预增"]  # 去掉首尾空白
    assert a["net_income_low"].to_list() == [1_500_000.0, 1_800_000.0]  # 万元 -> 元
    assert a["change_mid"].to_list() == [
        75.0,
        100.0,
    ]  # 同日两行数值不同：取更新标识 1 那行
    b = ev.filter(pl.col("symbol") == "B").sort("date")
    assert b["change_mid"].to_list() == [30.0, -50.0]  # 只给上限时取上限
    assert b["first_date"].to_list() == [
        "20240112",
        "20240301",
    ]  # 源缺首次日期时用本表最早
    assert set(ev["timing"].to_list()) == {"after_close"}


def test_expresses_prior_year_and_forecast_are_point_in_time():
    express = pl.DataFrame(
        {
            "symbol": ["A", "A", "C"],
            "announcement_date": ["20240125", "20240205", "20240120"],
            "period_end": ["20231231", "20231231", "20231231"],
            "revenue": [1.0, 1.0, 1.0],
            "net_income": [2_000_000.0, 2_100_000.0, 500.0],
        }
    )
    income = pl.DataFrame(
        {
            "ts_code": ["A", "A", "C"],
            "end_date": ["20221231", "20221231", "20221231"],
            "ann_date": ["20230410", "20240301", "20230410"],
            "f_ann_date": ["20230410", "20240301", "20230410"],
            "report_type": ["1", "1", "1"],
            "update_flag": ["0", "1", "0"],
            "n_income_attr_p": [1_000_000.0, 9_999_999.0, -10.0],
        }
    )
    ev = extract.expresses(
        with_public(
            FakeData({"income": income, "forecast": forecast_rows()}), express=express
        )
    )
    assert ev.height == 2  # A 的更正快报只用最早一条
    assert "versions" not in ev.columns  # 「之后还有几条更正」是未来信息，不输出
    a = ev.filter(pl.col("symbol") == "A").row(0, named=True)
    assert a["net_income"] == 2_000_000.0
    assert a["prior_net_income"] == 1_000_000.0  # 2024-03-01 的重述晚于快报，不用
    assert a["yoy"] == pytest.approx(1.0)
    # 同一天（01-25）的预告不算「之前」，用 01-10 的首次预告：中点 175 万
    assert a["vs_forecast"] == pytest.approx((2_000_000 - 1_750_000) / 1_000_000)
    assert a["forecast_position"] == pytest.approx(1.0)
    c = ev.filter(pl.col("symbol") == "C").row(0, named=True)
    assert c["yoy"] is None and c["has_forecast"] is False  # 上年同期为负不算同比


def dividend_rows():
    return pl.DataFrame(
        {
            "symbol": ["A", "A", "B", "C", "D", "E", "F", "F"],
            "period_end": [
                "20231231",
                "20231231",
                "20231231",
                "20231231",
                "20231129",
                "20230930",
                "20231231",
                "20231231",
            ],
            "announcement_date": [
                "20240420",
                "20240420",
                "20240410",
                "20240301",
                "20240101",
                "20231215",
                "20240330",
                "20240425",
            ],
            "status": ["实施", "实施", "实施 ", "实施", "实施", "实施", "实施", "实施"],
            "stock_dividend": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            "cash_dividend": [0.5, 0.5, None, 0.1, 0.2, 0.3, 0.4, 0.4],
            "record_date": ["20240510"] * 5 + ["20231225", "20240510", "20240510"],
            "ex_date": ["20240513"] * 4
            + ["20240110", "20231226", "20240513", "20240513"],
            "implementation_announcement_date": [
                "20240506",
                "20240506",
                "20240506",
                "20240506",
                "20240105",
                "20231220",
                "20240505",
                "20240506",
            ],
        }
    )


def report_rows():
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B", "C", "E", "F"],
            "end_date": ["20231231"] * 4 + ["20230930", "20231231"],
            "ann_date": [
                "20240328",
                "20240501",
                "20240330",
                "20240330",
                "20231025",
                "20240330",
            ],
            "f_ann_date": [
                "20240328",
                "20240501",
                "20240330",
                "20240330",
                "20231025",
                "20240330",
            ],
            "report_type": ["1"] * 6,
            "update_flag": ["0", "1", "0", "0", "0", "0"],
            "n_income_attr_p": [1.0] * 6,
        }
    )


def dividend_data():
    return with_public(FakeData({"income": report_rows()}), dividend=dividend_rows())


def test_dividend_report_stage_only_for_annual():
    plan = extract.dividends(dividend_data(), "report")
    # A：去重后一条，年报日 03-28；B：只送转；C：年报晚于决议日，丢弃；D：特别分红没有报告期
    # 财报，落到决议日；E：三季报期的分红不用报告日（会早于预案），落到决议日；
    # F：同一除权日两条实施行（年报日 03-30 被误记成决议日的那条与真正的决议日 04-25），
    # 只算一次
    got = plan.select("symbol", "date", "date_source").sort("symbol").rows()
    assert got == [
        ("A", "20240328", "report"),
        ("B", "20240330", "report"),
        ("D", "20240101", "resolution"),
        ("E", "20231215", "resolution"),
        ("F", "20240330", "report"),
    ]
    assert set(plan["kind"].to_list()) == {"dividend_plan"}
    a = plan.filter(pl.col("symbol") == "A").row(0, named=True)
    assert (a["cash_per_share"], a["stock_per_share"]) == (0.5, 0.0)
    b = plan.filter(pl.col("symbol") == "B").row(0, named=True)
    assert (b["cash_per_share"], b["stock_per_share"]) == (0.0, 0.3)


def test_dividend_non_annual_never_earlier_than_resolution():
    plan = extract.dividends(dividend_data(), "report")
    res = extract.dividends(dividend_data(), "resolution")
    joined = plan.join(
        res.select("symbol", "period_end", "ex_date", pl.col("date").alias("res")),
        on=["symbol", "period_end", "ex_date"],
    )
    non_annual = joined.filter(~pl.col("period_end").str.ends_with("1231"))
    assert non_annual.height == 2
    assert (non_annual["date"] >= non_annual["res"]).all()


def test_dividend_resolution_and_implementation_dedupe_per_stage():
    data = dividend_data()
    res = extract.dividends(data, "resolution")
    assert res.filter(pl.col("symbol") == "F")["date"].to_list() == ["20240425"]
    assert res.filter(pl.col("symbol") == "A")["date"].to_list() == ["20240420"]
    assert res.height == 6  # A、B、C、D、E、F 各一条
    imp = extract.dividends(data, "implementation")
    assert imp.filter(pl.col("symbol") == "F")["date"].to_list() == ["20240505"]
    assert imp.filter(pl.col("symbol") == "D")["date"].item() == "20240105"
    with pytest.raises(ValueError):
        extract.dividends(data, "plan")


def test_index_changes():
    w = pl.DataFrame(
        {
            "index_code": ["000300.SH"] * 7 + ["000905.SH"],
            "con_code": ["A", "B", "A", "C", "C", "D", "D", "B"],
            "trade_date": [
                "20240531",
                "20240531",
                "20240628",
                "20240628",
                "20240731",
                "20240731",
                "20240731",
                "20240531",
            ],
            "weight": [1.0, 2.0, 1.5, 0.5, 0.6, 0.7, 0.7, 9.0],
        }
    )
    sec = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "delisting_date": ["20240715", None, None, None],
        }
    )
    ev = extract.index_changes(FakeData({"index_weight": w}, sec))
    rows = (
        ev.select("symbol", "date", "kind", "regular_review", "delisted_by", "weight")
        .sort("date", "kind", "symbol")
        .rows()
    )
    assert rows == [
        ("C", "20240628", "index_add", True, False, 0.5),
        ("B", "20240628", "index_remove", True, False, 2.0),
        ("D", "20240731", "index_add", False, False, 0.7),
        ("A", "20240731", "index_remove", False, True, 1.5),
    ]
    assert ev["previous_snapshot"].to_list()[0] in ("20240531", "20240628")
    with pytest.raises(ValueError):
        extract.index_changes(FakeData({"index_weight": w}, sec), "hs300")


def test_forecast_first_and_revision_follow_one_rule():
    rows = pl.DataFrame(
        {
            "ts_code": ["G", "G", "H", "H", "K"],
            "ann_date": ["20240115", "20240120", "20240110", "20240120", "20240201"],
            "end_date": ["20231231"] * 5,
            "type": ["预增"] * 5,
            "p_change_min": [60.0] * 5,
            "p_change_max": [80.0] * 5,
            "net_profit_min": [1.0] * 5,
            "net_profit_max": [2.0] * 5,
            "last_parent_net": [1.0] * 5,
            # G：源记的首次预告早于本表最早一条（首次预告不在表里）；H：源把修正那条也记成
            # 自己是首次（源内部不一致）；K：源缺首次日期
            "first_ann_date": ["20231030", "20231030", "20240110", "20240120", None],
            "update_flag": ["0"] * 5,
        }
    )
    ev = extract.forecasts(FakeData({"forecast": rows})).sort("symbol", "date")
    got = ev.select("symbol", "first_date", "revision", "is_revision").rows()
    assert got == [
        ("G", "20231030", 1, True),
        ("G", "20231030", 2, True),
        ("H", "20240110", 0, False),
        ("H", "20240110", 1, True),
        ("K", "20240201", 0, False),
    ]


def test_index_regular_review_months_switch_in_2013():
    dates = ["20121231", "20130131", "20130628", "20130731", "20131231", "20140130"]
    w = pl.DataFrame(
        {
            "index_code": ["000300.SH"] * 6,
            "con_code": ["A", "B", "C", "D", "E", "F"],
            "trade_date": dates,
            "weight": [1.0] * 6,
        }
    )
    sec = pl.DataFrame(
        {"symbol": list("ABCDEF"), "delisting_date": [None] * 6},
        schema={"symbol": pl.String, "delisting_date": pl.String},
    )
    ev = extract.index_changes(FakeData({"index_weight": w}, sec))
    adds = ev.filter(pl.col("kind") == "index_add").sort("date")
    # 2013-11 之前定期调整在 1 月、7 月，之后在 6 月、12 月
    assert adds.select("date", "regular_review").rows() == [
        ("20130131", True),
        ("20130628", False),
        ("20130731", True),
        ("20131231", True),
        ("20140130", False),
    ]
