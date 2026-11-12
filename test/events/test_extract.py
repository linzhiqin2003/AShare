"""内置事件提取：用手写的原始表行（字段名照原始库）核对定义与口径。"""

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


def forecast_rows():
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "A", "B", "B"],
            "ann_date": ["20240110", "20240125", "20240125", "20240112", "20240301"],
            "end_date": ["20231231"] * 4 + ["20240331"],
            "type": ["预增", "预增 ", "预增", "略增", "预减"],
            "p_change_min": [50.0, 80.0, 80.0, None, -60.0],
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
    assert a["change_mid"].to_list() == [75.0, 100.0]
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
            "ts_code": ["A", "A", "C"],
            "ann_date": ["20240125", "20240205", "20240120"],
            "end_date": ["20231231", "20231231", "20231231"],
            "revenue": [1.0, 1.0, 1.0],
            "n_income": [2_000_000.0, 2_100_000.0, 500.0],
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
        FakeData({"express": express, "income": income, "forecast": forecast_rows()})
    )
    assert ev.height == 2  # A 的更正快报只用最早一条
    a = ev.filter(pl.col("symbol") == "A").row(0, named=True)
    assert a["versions"] == 2
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
            "ts_code": ["A", "A", "B", "C", "D"],
            "end_date": ["20231231", "20231231", "20231231", "20231231", "20231129"],
            "ann_date": ["20240420", "20240420", "20240410", "20240301", "20240101"],
            "div_proc": ["实施", "实施", "实施 ", "实施", "实施"],
            "stk_div": [0.0, 0.0, 0.3, 0.0, 0.0],
            "cash_div_tax": [0.5, 0.5, None, 0.1, 0.2],
            "record_date": ["20240510"] * 5,
            "ex_date": ["20240513", "20240513", "20240513", "20240513", "20240110"],
            "imp_ann_date": [
                "20240506",
                "20240506",
                "20240506",
                "20240506",
                "20240105",
            ],
        }
    )


def report_rows():
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B", "C"],
            "end_date": ["20231231"] * 4,
            "ann_date": ["20240328", "20240501", "20240330", "20240330"],
            "f_ann_date": ["20240328", "20240501", "20240330", "20240330"],
            "report_type": ["1"] * 4,
            "update_flag": ["0", "1", "0", "0"],
            "n_income_attr_p": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_dividend_stages():
    data = FakeData({"dividend": dividend_rows(), "income": report_rows()})
    plan = extract.dividends(data, "report")
    # A：去重后一条，报告日 03-28；B：只送转；C：报告晚于决议日，丢弃；D：特别分红没有报告，丢弃
    assert plan.select("symbol", "date", "kind").rows() == [
        ("A", "20240328", "dividend_plan"),
        ("B", "20240330", "dividend_plan"),
    ]
    assert plan["cash_per_share"].to_list() == [0.5, 0.0]
    assert plan["stock_per_share"].to_list() == [0.0, 0.3]
    res = extract.dividends(data, "resolution")
    assert (
        res.height == 4
        and res.filter(pl.col("symbol") == "A")["date"].item() == "20240420"
    )
    imp = extract.dividends(data, "implementation")
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
