from pathlib import Path

import numpy as np
import polars as pl

import ashare
from ashare.data.crosscheck_compare import compare_rows
from ashare.data.rules import limit_expressions, limit_price, limit_rates
from ashare.data.verify import audit_tables

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_cross_source_differences_and_missing():
    key = ("A", "20260102")
    missing = ("B", "20260102")
    rows = {key: dict(open=10.0, high=11.0, low=9.0, close=10.0, vol=2)}
    good = dict(
        symbol="A",
        date="20260102",
        open="10",
        high="11",
        low="9",
        close="10",
        volume="200",
        isST="0",
        tradestatus="1",
    )
    report = compare_rows(rows, {key}, set(), set(), ["20260102"], [good], {key})
    assert (
        report["differences"] == []
        and report["missing_status"] == []
        and report["price_checks"] == 5
        and report["status_checks"] == 2
    )
    bad = {**good, "volume": "20", "tradestatus": "0", "isST": "1"}
    report = compare_rows(
        rows, {key, missing}, set(), set(), ["20260102"], [bad], {key, missing}
    )
    assert len(report["differences"]) == 3
    assert len(report["missing_status"]) == len(report["missing_samples"]) == 1
    assert any("正成交量" in r["judgment"] for r in report["differences"])


def test_vectorized_rules_match_decimal_reference():
    rows = []
    for symbol in ("000001.SZ", "300001.SZ", "688001.SH", "920001.BJ"):
        for day in (
            "20200821",
            "20200824",
            "20230407",
            "20230410",
            "20260703",
            "20260706",
        ):
            for age in (1, 5, 6, 100):
                for st in (True, False):
                    for price in (0.04, 10.05, 10.82, 53.53):
                        rows.append(
                            dict(
                                ts_code=symbol,
                                trade_date=day,
                                list_date=day if age == 1 else "20100101",
                                _age=age,
                                _st=st,
                                _non_reformed=False,
                                pre_close=price,
                            )
                        )
    table = pl.DataFrame(rows).with_columns(*limit_expressions())
    for row in table.iter_rows(named=True):
        rate = limit_rates(
            row["ts_code"], row["trade_date"], row["list_date"], row["_age"], row["_st"]
        )
        assert row["_unlimited"] == (rate is None)
        if rate:
            inward = row["ts_code"].endswith(".BJ")
            np.testing.assert_allclose(
                [row["expected_upper"], row["expected_lower"]],
                [
                    limit_price(row["pre_close"], rate[0], inward=inward),
                    limit_price(row["pre_close"], rate[1], upper=False, inward=inward),
                ],
                rtol=0,
                atol=1e-12,
            )


def test_cross_dataset_missing_cells(tables):
    baseline = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert baseline.checks["daily_vs_daily_basic"]["failed"] == 0
    tables["daily_basic"] = tables["daily_basic"].slice(1)
    report = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert report.checks["daily_vs_daily_basic"]["failed"] == 1


def test_annual_aggregation_keeps_denominators():
    from ashare.data.verify import Audit

    aggregate = Audit()
    for year in ("2007", "2008"):
        audit = Audit()
        audit.add("price_limits", 100, [], failed=2)
        audit.checks["price_limits"].update(
            unequal_days=2,
            positive_reference_checked=80,
            positive_reference_failed=1,
            by_year_board_st=[{"year": year, "checked": 100, "failed": 2}],
            failure_categories=[{"category": "zero", "len": 1}],
        )
        audit.add("unlimited_sessions", 3, [])
        audit.checks["unlimited_sessions"]["pass_rate"] = None
        aggregate.merge(audit)
    check = aggregate.checks["price_limits"]
    assert check["checked"] == 200 and check["failed"] == 4
    assert check["unequal_days"] == 4 and check["positive_reference_checked"] == 160
    assert (
        check["positive_reference_failed"] == 2 and len(check["by_year_board_st"]) == 2
    )
    assert check["failure_categories"] == [{"category": "zero", "len": 2}]
    assert aggregate.checks["unlimited_sessions"]["pass_rate"] is None


def test_sampling_preserves_random_priority_not_date_order():
    import random

    from ashare.data.crosscheck import sample_quotes

    entries = [dict(partition="20200102", rows=50), dict(partition="20260102", rows=50)]

    class Store:
        def read(self, name, start, end):
            offset = 0 if start == "20200102" else 50
            return pl.DataFrame(
                dict(
                    ts_code=[f"{i:06}.SZ" for i in range(offset, offset + 50)],
                    trade_date=[start] * 50,
                )
            )

    rows, sampled = sample_quotes(Store(), entries, [], 10)
    expected = random.Random(20261112).sample(range(100), 20)[:10]
    assert {int(s.split(".")[0]) for s, d in sampled} == set(expected)
    assert len(rows) == 10
