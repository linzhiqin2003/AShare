"""来源：D3 任务书；复权基准未证实时只测试原始数据与独立核对，不发布筹码字段。"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ashare.data.catalog import DATASETS, FIELDS
from ashare.data.chip_verify import basis_rows, check_chip_values
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.data.sync import plan


def chip(day="20180102", fetched="2026-01-07T08:00:00Z", **changes):
    return pl.DataFrame(
        [
            dict(
                ts_code="A",
                trade_date=day,
                his_low=2.0,
                his_high=8.0,
                cost_5pct=2.0,
                cost_15pct=3.0,
                cost_50pct=4.0,
                cost_85pct=5.0,
                cost_95pct=6.0,
                weight_avg=4.0,
                winner_rate=65.0,
                _fetched_at=fetched,
            )
            | changes
        ]
    )


def history():
    daily = pl.DataFrame(
        dict(
            ts_code=["A"] * 4,
            trade_date=["20170103", "20180102", "20260107", "20260108"],
            low=[4.0, 5.0, 3.0, 1.0],
            high=[16.0, 14.0, 20.0, 50.0],
        )
    )
    factors = daily.select("ts_code", "trade_date").with_columns(
        pl.Series("adj_factor", [1.0, 1.0, 2.0, 4.0])
    )
    basic = pl.DataFrame(dict(ts_code=["A"], list_date=["20170103"]))
    return daily, factors, basic


def history_calendar(*extra_open):
    """合成日历完整列出自然日；只有显式声明的日期开市。"""
    first, last = datetime(2017, 1, 3), datetime(2018, 1, 2)
    days = [
        (first + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((last - first).days + 1)
    ]
    opened = {"20170103", "20180102", *extra_open}
    return pl.DataFrame(dict(cal_date=days, is_open=[int(d in opened) for d in days]))


def checked_basis(chips, daily, factors, basic):
    return basis_rows(chips, daily, factors, basic, calendar=history_calendar())


def test_catalog_and_coverage_boundary():
    spec = DATASETS["cyq_perf"]
    assert spec.partition == "day" and spec.coverage_start == "20180102"
    assert "收盘后" in spec.known_at
    assert list(plan(spec, "20171229", "20180102", ["20171229", "20180102"], None)) == [
        ("20180102", {"trade_date": "20180102"})
    ]
    # 来源：真实核对发现无法证实基准；不能把候选换算登记成可用于研究的字段。
    assert not any(n.startswith("chip_") for n in FIELDS)


def test_basis_uses_partition_timestamp_and_only_past_extremes():
    result = checked_basis(chip(), *history()).row(0, named=True)
    assert result["expected_low"] == 2.0
    assert result["expected_high"] == 8.0
    assert result["basis_factor"] == 2.0
    assert result["within_tolerance"]
    # 同日更晚的复权因子只对另一次抓取生效，不能沿用全库最新因子。
    later = checked_basis(
        chip(fetched="2026-01-07T16:30:00Z", his_low=1.0, his_high=4.0), *history()
    )
    assert later["basis_factor"][0] == 4.0
    assert later["within_tolerance"][0]


def test_basis_rejects_missing_history_and_different_basis():
    daily, factors, basic = history()
    assert not checked_basis(chip(his_high=12.0), daily, factors, basic)[
        "within_tolerance"
    ][0]
    out = checked_basis(chip(), daily.slice(1), factors, basic)
    assert not out["history_complete"][0] and not out["within_tolerance"][0]
    missing = factors.filter(pl.col("trade_date") != "20170103")
    assert not checked_basis(chip(), daily, missing, basic)["within_tolerance"][0]
    assert not checked_basis(chip(fetched="bad"), daily, factors, basic)[
        "within_tolerance"
    ][0]


def test_basis_grid_tolerance_is_bounded():
    assert checked_basis(chip(his_low=2.04, his_high=7.96), *history())[
        "within_tolerance"
    ][0]
    assert not checked_basis(chip(his_high=8.2), *history())["within_tolerance"][0]


@pytest.mark.parametrize(
    "change,check",
    [
        ({"winner_rate": -1.0}, "chip_winner_range"),
        ({"winner_rate": 101.0}, "chip_winner_range"),
        ({"winner_rate": float("inf")}, "chip_winner_range"),
        ({"cost_50pct": 7.0}, "chip_quantile_order"),
        ({"cost_5pct": -1.0}, "chip_cost_positive"),
        ({"trade_date": "20171229"}, "chip_coverage_start"),
    ],
)
def test_chip_values_negative(change, check):
    checks = check_chip_values(chip(**change)).checks
    assert checks[check]["failed"] == 1


def test_chip_values_positive_and_null():
    checks = check_chip_values(chip()).checks
    assert all(x["failed"] == 0 for x in checks.values())
    nulls = check_chip_values(chip(winner_rate=None)).checks
    assert nulls["chip_winner_range"]["checked"] == 0


def test_refresh_only_replaces_changed_content(tmp_path):
    store = Store(tmp_path)
    raw = chip().drop("_fetched_at")
    first = store.write(
        "cyq_perf", "20180102", {}, FetchResult(raw, "2026-01-07T08:00:00Z", 1)
    )
    path = store.root / first["file"]
    payload, stamp = path.read_bytes(), path.stat().st_mtime_ns
    version = store.version
    same = store.write(
        "cyq_perf", "20180102", {}, FetchResult(raw, "2026-01-08T08:00:00Z", 1)
    )
    assert same["file"] == first["file"] and store.version == version
    assert path.read_bytes() == payload and path.stat().st_mtime_ns == stamp
    assert store.read("cyq_perf")["_fetched_at"][0] == "2026-01-07T08:00:00Z"
    changed = raw.with_columns(pl.col("cost_50pct") + 0.1)
    last = store.write(
        "cyq_perf", "20180102", {}, FetchResult(changed, "2026-01-08T08:00:00Z", 1)
    )
    assert last["file"] != first["file"] and store.version != version
    np.testing.assert_equal(store.read("cyq_perf")["cost_50pct"][0], 4.1)


def test_real_probe_date_literals_regression():
    """来源：真实核对脚本首次运行把日期字符串当列名，抛 ColumnNotFoundError。"""
    from bench.d3_real_check import eligible_codes

    basic = pl.DataFrame(
        dict(
            ts_code=["A", "B", "C", "D"],
            list_date=["20070103", "20070104", "20171231", "20180101"],
        )
    )
    assert eligible_codes(basic) == ["B", "C"]


def test_verify_includes_chip_raw_checks_and_basis(tables, make_store):
    from ashare.data.verify import verify

    store = make_store(tables)
    for day in tables["trade_cal"]["cal_date"]:
        table = chip(day=day, ts_code="000003.SZ", winner_rate=120.0).drop(
            "_fetched_at"
        )
        store.write("cyq_perf", day, {}, FetchResult(table, "2026-01-08T08:00:00Z", 1))
    result = verify(store)
    assert result["checks"]["chip_winner_range"]["failed"] == 4
    assert "chip_adjustment_basis" not in result["checks"]
    assert result["chip_adjustment_basis"]["status"] == "unverified"
    assert result["chip_extrema_diagnostic"]["data_status"] == "insufficient_history"
    assert result["coverage"]["cyq_perf"]["missing_partitions"] == []


@pytest.mark.parametrize("missing", ["20171229", "20180102"])
@pytest.mark.parametrize("extreme", [False, True])
def test_basis_rejects_internal_gap_and_truncated_tail(missing, extreme):
    """来源：独立审查 P1；保留首日，删中间极值日或尾部日不能伪装历史完整。"""
    daily, factors, basic = history()
    daily = pl.concat(
        [
            daily,
            pl.DataFrame(
                dict(
                    ts_code=["A"],
                    trade_date=["20171229"],
                    low=[0.2 if extreme else 5.0],
                    high=[100.0 if extreme else 10.0],
                )
            ),
        ]
    )
    factors = pl.concat(
        [
            factors,
            pl.DataFrame(
                dict(ts_code=["A"], trade_date=["20171229"], adj_factor=[1.0])
            ),
        ]
    )
    full = basis_rows(
        chip(), daily, factors, basic, calendar=history_calendar("20171229")
    )
    assert full["history_complete"][0]
    assert full["within_tolerance"][0] == (not extreme)
    out = basis_rows(
        chip(),
        daily.filter(pl.col("trade_date") != missing),
        factors,
        basic,
        calendar=history_calendar("20171229"),
    )
    assert not out["history_complete"][0]
    assert not out["within_tolerance"][0]
    assert out["missing_trade_days"][0] == 1
    assert out["missing_trade_day_examples"][0].to_list() == [missing]
    assert "不足以核实" in out["history_note"][0]


@pytest.mark.parametrize("missing", [None, "20170104", "20180102"])
def test_basis_requires_complete_calendar(missing):
    """来源：P1 的证明责任；连闭市日都缺失的日历也不能证明行情完整。"""
    calendar = (
        None
        if missing is None
        else history_calendar().filter(pl.col("cal_date") != missing)
    )
    out = basis_rows(chip(), *history(), calendar=calendar)
    assert not out["history_complete"][0] and not out["within_tolerance"][0]
    assert "不足以核实" in out["history_note"][0]


@pytest.mark.parametrize(
    "kind,timing,complete",
    [
        ("S", None, True),
        ("S", "", True),
        ("S", "全天", True),
        ("S", "09:30-10:00", False),
        ("R", None, False),
    ],
)
def test_basis_only_full_day_suspension_explains_gap(kind, timing, complete):
    """来源：P1；缺行情只能由同股同日的显式全天停牌解释。"""
    suspensions = pl.DataFrame(
        dict(
            ts_code=["A"],
            trade_date=["20171229"],
            suspend_type=[kind],
            suspend_timing=[timing],
        )
    )
    out = basis_rows(
        chip(),
        *history(),
        calendar=history_calendar("20171229"),
        suspensions=suspensions,
    )
    assert out["history_complete"][0] == complete
    assert out["within_tolerance"][0] == complete


@pytest.mark.parametrize(
    "listing,delisting",
    [(None, None), ("bad", None), ("20180103", None), ("20170103", "20180102")],
)
def test_basis_requires_known_listing_interval(listing, delisting):
    """来源：P1；上市区间未知或被核对日期不在上市区间时不给完整结论。"""
    daily, factors, _ = history()
    basic = pl.DataFrame(
        dict(ts_code=["A"], list_date=[listing], delist_date=[delisting])
    )
    out = checked_basis(chip(), daily, factors, basic)
    assert not out["history_complete"][0] and not out["within_tolerance"][0]


@pytest.mark.parametrize("dataset", ["stock_basic", "daily", "adj_factor", "trade_cal"])
def test_audit_reports_missing_evidence_as_insufficient(tables, make_store, dataset):
    """来源：P1；缺证明材料须明确返回不足以核实，不能崩溃或默认为完整。"""
    from ashare.data.chip_verify import audit_chip_basis

    store = make_store(tables)
    day = tables["trade_cal"]["cal_date"][0]
    raw = chip(day=day, ts_code="000003.SZ").drop("_fetched_at")
    store.write("cyq_perf", day, {}, FetchResult(raw, "2026-01-08T08:00:00Z", 1))
    for key in store.entries(dataset):
        store.mark(key, dict(status="failed"))
    result = audit_chip_basis(store)
    assert result["status"] == "unverified" and result["failed"] == 0
    diagnostic = result["extrema_diagnostic"]
    assert diagnostic["data_status"] == "insufficient_history"
    assert "不足以核实" in diagnostic["note"]


def test_basis_calendar_conflict_is_insufficient():
    """来源：P1；同一自然日开闭市状态冲突不能证明完整。"""
    calendar = pl.concat(
        [history_calendar(), pl.DataFrame(dict(cal_date=["20180102"], is_open=[0]))]
    )
    out = basis_rows(chip(), *history(), calendar=calendar)
    assert not out["history_complete"][0] and not out["within_tolerance"][0]


def test_basis_excludes_prices_before_listing_interval():
    """来源：P1 实现检查；区间外旧报价不能污染上市以来极值。"""
    daily, factors, basic = history()
    basic = basic.with_columns(pl.lit("20180102").alias("list_date"))
    out = checked_basis(chip(his_low=2.5, his_high=7.0), daily, factors, basic)
    assert out["history_complete"][0]
    assert out["expected_low"][0] == 2.5 and out["expected_high"][0] == 7.0
    assert out["within_tolerance"][0]


def diagnostic_store(**changes):
    """来源：recheck2_d3.md 的内存复现；只有边界改变，其余原始值合法。"""
    daily, factors, basic = history()
    tables = dict(
        daily=daily,
        adj_factor=factors,
        stock_basic=basic,
        trade_cal=history_calendar(),
        suspend_d=pl.DataFrame(),
        cyq_perf=chip(**changes),
    )
    return SimpleNamespace(
        entries=lambda n: {
            "x": dict(
                status="complete",
                rows=1,
                partition="20180102" if n == "cyq_perf" else "20170103",
            )
        },
        read=lambda n, *args: tables[n],
        scan=lambda n: tables[n].lazy(),
    )


@pytest.mark.parametrize("boundary", [8.0, 28.0])
def test_chip_boundaries_are_diagnostic_only(boundary):
    """来源：合并前 P1；极值恰好匹配也不能证实基准，不匹配也不是基准失败。"""
    from ashare.data.chip_verify import audit_chip_basis

    assert all(
        c["failed"] == 0
        for c in check_chip_values(chip(his_high=boundary)).checks.values()
    )
    result = audit_chip_basis(diagnostic_store(his_high=boundary))
    assert result["status"] == "unverified"
    assert result["status_label"] == "未证实"
    assert result["checked"] == result["failed"] == 0
    assert result["pass_rate"] is None
    assert {e["id"] for e in result["missing_evidence"]} == {
        "cyq_chips_distribution",
        "corporate_action_method",
        "cross_ex_date_snapshots",
    }
    diagnostic = result["extrema_diagnostic"]
    assert diagnostic["status"] == "diagnostic"
    assert diagnostic["status_label"] == "诊断（不判定）"
    assert "failed" not in diagnostic and "pass_rate" not in diagnostic
    assert diagnostic["checked"] == 1
    assert diagnostic["within_tolerance"] == int(boundary == 8.0)
    assert diagnostic["outside_tolerance"] == int(boundary != 8.0)
    assert diagnostic["examples"][0]["observed_high"] == boundary
    assert diagnostic["examples"][0]["expected_high"] == 8.0


@pytest.mark.parametrize("empty", ["not_downloaded", "insufficient_sample"])
def test_chip_basis_stays_unverified_without_samples(empty):
    """来源：P1；没有样本不能从固定未证实状态退回通过/失败判定。"""
    from ashare.data.chip_verify import audit_chip_basis

    store = diagnostic_store()
    if empty == "not_downloaded":
        store.entries = lambda n: {}
        result = audit_chip_basis(store)
    else:
        result = audit_chip_basis(store, sample_size=0)
    assert result["status"] == "unverified" and result["failed"] == 0
    assert result["extrema_diagnostic"]["data_status"] == empty
    assert result["missing_evidence"]


@pytest.mark.parametrize("boundary", [None, float("nan"), float("inf")])
def test_chip_diagnostic_nonfinite_boundaries_are_json_safe(boundary):
    """来源：本轮检查诊断序列化；非有限边界仍只作诊断，JSON 不能泄漏 NaN/Infinity。"""
    from ashare.data.chip_verify import audit_chip_basis

    result = audit_chip_basis(diagnostic_store(his_high=boundary))
    json.dumps(result, allow_nan=False)
    assert result["status"] == "unverified"
    assert result["extrema_diagnostic"]["examples"][0]["observed_high"] is None


def test_verify_boundary_changes_do_not_change_failures(tables, make_store):
    """来源：recheck2_d3.md 的 verify 0→1 失败数复现；正式落盘报告也必须与边界解耦。"""
    from ashare.data.verify import verify

    store = make_store(tables)
    day = tables["trade_cal"]["cal_date"][0]
    reports = []
    for boundary in (9.9, 29.9):
        raw = chip(day=day, ts_code="000003.SZ", his_low=8.1, his_high=boundary).drop(
            "_fetched_at"
        )
        store.write("cyq_perf", day, {}, FetchResult(raw, "2026-01-08T08:00:00Z", 1))
        reports.append(verify(store))
    assert sum(c["failed"] for c in reports[0]["checks"].values()) == sum(
        c["failed"] for c in reports[1]["checks"].values()
    )
    assert reports[0]["checks"] == reports[1]["checks"]
    for report in reports:
        assert "chip_adjustment_basis" not in report["checks"]
        assert "chip_extrema_diagnostic" not in report["checks"]
        assert report["chip_adjustment_basis"]["status"] == "unverified"
        assert report["chip_extrema_diagnostic"]["status"] == "diagnostic"
    assert reports[0]["chip_adjustment_basis"] == reports[1]["chip_adjustment_basis"]
    assert reports[0]["chip_extrema_diagnostic"]["within_tolerance"] == 1
    assert reports[1]["chip_extrema_diagnostic"]["outside_tolerance"] == 1
    text = (store.root / "reports/verify.md").read_text()
    assert "未证实" in text and "诊断（不判定）" in text and "cyq_chips" in text


def test_d3_notes_use_repository_relative_paths():
    """来源：合并前 P2；公开笔记不携带本机用户名/绝对路径。"""
    note = (Path(__file__).resolve().parents[2] / "docs/D3_NOTES.md").read_text()
    assert "/Users/" not in note
    assert ".review/review_d3.md" in note and "data/staging/cyq" in note
