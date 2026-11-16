"""来源：recheck_d3.md P1；首空、旧零行及逐日有效覆盖，全部使用合成数据。"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from ashare.data import fund_verify
from ashare.data.fund_sync import sync_funds
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store

DAYS = ["20261112", "20261113"]
STAMP = "2026-11-16T00:00:00Z"


def sizes(days=DAYS):
    return pl.DataFrame(
        dict(
            trade_date=days,
            ts_code=["ETF"] * len(days),
            total_share=[1.0] * len(days),
            total_size=[2.0] * len(days),
        )
    )


def basic(list_date="20200101", status="L"):
    return pl.DataFrame(
        dict(
            ts_code=["ETF"],
            list_date=[list_date],
            list_status=[status],
            _fetched_at=[STAMP],
        )
    )


def write(store, name, part, frame):
    return store.write(name, part, {}, FetchResult(frame, STAMP, 1))


@pytest.mark.parametrize("old", [None, "unchecked", "confirmed"])
def test_empty_etf_partition_remains_unverified(tmp_path, old, caplog):
    """复核原样复现：首次空响应，以及旧零行 complete/confirmed → 相同哈希零行。"""
    store = Store(tmp_path)
    key = f"etf_share_size/{DAYS[0]}"
    if old:
        write(store, "etf_share_size", DAYS[0], sizes([]))
        store.mark(key, dict(status="complete", quality_state=old))
    entry = write(store, "etf_share_size", DAYS[0], sizes([]))
    assert entry["status"] == "suspect"
    assert entry["quality_state"] == "empty_unverified"
    assert not store.complete(key)
    assert "待核实" in caplog.text


def test_small_etf_snapshot_shrink_is_rejected(tmp_path):
    """来源：复核边界 99 → 1；不能以旧修订不足 100 行放过同日截断。"""
    store = Store(tmp_path)
    frame = pl.DataFrame(
        dict(ts_code=[f"ETF{i}" for i in range(99)], trade_date=[DAYS[0]] * 99)
    )
    first = write(store, "etf_share_size", DAYS[0], frame)
    with pytest.raises(ValueError, match="保留旧修订"):
        write(store, "etf_share_size", DAYS[0], frame.head(1))
    assert store.entries("etf_share_size")[f"etf_share_size/{DAYS[0]}"] == first


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_etf_partition_is_retried_without_refresh(tmp_path, legacy):
    """来源：P1 补抓机会；连旧版 complete 零行也不能被 refresh=False 跳过。"""
    store = Store(tmp_path)
    write(
        store,
        "fund_calendar",
        "SSE",
        pl.DataFrame(dict(exchange=["SSE"] * 2, cal_date=DAYS, is_open=[1, 1])),
    )
    key = f"etf_share_size/{DAYS[0]}"
    write(store, "etf_share_size", DAYS[0], sizes([]))
    if legacy:
        store.mark(key, dict(status="complete", quality_state="unchecked"))
    write(store, "etf_share_size", DAYS[1], sizes(DAYS[1:]))
    calls = []

    class Source:
        async def fetch(self, api, params, **kwargs):
            calls.append((api, params))
            return FetchResult(sizes([params["trade_date"]]), STAMP, 1)

    asyncio.run(
        sync_funds(
            Source(), store, *DAYS, names=["etf_share_size"], refresh=False, workers=2
        )
    )
    assert calls == [("etf_share_size", {"trade_date": DAYS[0]})]
    assert store.complete(key)


def mock_verify(monkeypatch, raw, master=None, funds=None, arrays=None):
    master = basic() if master is None else master
    tables = dict(
        etf_share_size=raw,
        etf_basic=master,
        fund_basic=pl.DataFrame() if funds is None else funds,
    )
    monkeypatch.setattr(
        fund_verify, "_read", lambda s, n: tables.get(n, pl.DataFrame())
    )
    meta = dict(
        dates=DAYS,
        symbols=["ETF"],
        fields={n: dict(known_at="次日可知") for n in ("total_share", "total_size")},
    )
    if arrays is None:
        arrays = {
            n: np.array(
                [[v * 10000] if d in raw["trade_date"] else [np.nan] for d in DAYS]
            )
            for n, v in (("total_share", 1.0), ("total_size", 2.0))
        }
    monkeypatch.setattr(fund_verify, "_load", lambda p, n: arrays[n])
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda p: master.rename({"ts_code": "symbol", "_fetched_at": "fetched_at"}),
    )
    entries = {
        f"etf_share_size/{d}": dict(
            partition=d,
            status="complete",
            rows=raw.filter(pl.col("trade_date") == d).height,
        )
        for d in DAYS
    }
    monkeypatch.setattr(
        fund_verify, "_entries", lambda s, n: entries if n == "etf_share_size" else {}
    )
    return meta, entries


@pytest.mark.parametrize("day", DAYS)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("checker", ["partition", "panel"])
def test_verify_rejects_each_empty_etf_day(monkeypatch, day, missing, checker):
    """复核两日一行复现，并检查首日缺失、未建分区、未知上市日期不得豁免。"""
    raw = sizes([d for d in DAYS if d != day])
    meta, entries = mock_verify(monkeypatch, raw, basic(None))
    if missing:
        entries.pop(f"etf_share_size/{day}")
    register = fund_verify.Register(
        dict(
            entries=[
                dict(check="partition_missing", dataset="etf_share_size", partition=day)
            ]
        )
    )
    if checker == "partition":
        assert (
            fund_verify.check_partitions(None, np.array(DAYS), register)[
                "etf_share_size"
            ]["verdict"]
            == "fail"
        )
        assert not register.used  # 代码判断覆盖，不走人工豁免清单。
    else:
        result = fund_verify.check_etf_panel(None, MagicMock(), meta)
        assert result["verdict"] == "fail"
        assert any(
            p.get("problem") == "no_finite_day" and p.get("date") == day
            for p in result["details"]["examples"]
        )


@pytest.mark.parametrize("field", ["total_share", "total_size"])
@pytest.mark.parametrize("checker", ["partition", "panel"])
def test_verify_rejects_each_nan_etf_day(monkeypatch, field, checker):
    """来源：复核整行 NaN；清单 rows>0 与全表存在有限值均不足以证明逐日覆盖。"""
    raw = sizes().with_columns(
        pl.when(pl.col("trade_date") == DAYS[1])
        .then(None)
        .otherwise(pl.col(field))
        .alias(field)
    )
    arrays = dict(
        total_share=np.array([[10000.0], [10000.0]]),
        total_size=np.array([[20000.0], [20000.0]]),
    )
    arrays[field][1] = np.nan
    meta, _ = mock_verify(monkeypatch, raw, arrays=arrays)
    register = fund_verify.Register({})
    if checker == "partition":
        assert (
            fund_verify.check_partitions(None, np.array(DAYS), register)[
                "etf_share_size"
            ]["verdict"]
            == "fail"
        )
    else:
        result = fund_verify.check_etf_panel(None, MagicMock(), meta)
        assert result["verdict"] == "fail"
        assert (
            dict(field=field, date=DAYS[1], problem="no_finite_day")
            in result["details"]["examples"]
        )


@pytest.mark.parametrize("start,end", [("20261114", None), ("20200101", "20261111")])
def test_proven_no_listed_etf_allows_empty_days(tmp_path, monkeypatch, start, end):
    """正例：全部尚未上市，或有退市日期且全部已退市；不凭当前 D 状态推断历史。"""
    store = Store(tmp_path)
    master = basic(start, "D" if end else "L")
    funds = pl.DataFrame(dict(ts_code=["ETF"], list_date=[start], delist_date=[end]))
    write(store, "etf_basic", "all", master.drop("_fetched_at"))
    write(store, "fund_basic", "all", funds)
    for d in DAYS:
        entry = write(store, "etf_share_size", d, sizes([]))
        assert entry["status"] == "complete"
        assert entry["quality_state"] == "no_listed_etf"
    meta, _ = mock_verify(monkeypatch, sizes([]), master, funds)
    assert (
        fund_verify.check_partitions(None, np.array(DAYS), fund_verify.Register({}))[
            "etf_share_size"
        ]["verdict"]
        == "pass"
    )
    assert fund_verify.check_etf_panel(None, MagicMock(), meta)["verdict"] == "pass"


def test_current_delisted_status_is_not_historical_absence(tmp_path):
    """反例：只有当前 D、没有退市日期，不能将历史空日认作合法。"""
    store = Store(tmp_path)
    write(store, "etf_basic", "all", basic(status="D").drop("_fetched_at"))
    entry = write(store, "etf_share_size", DAYS[0], sizes([]))
    assert entry["status"] == "suspect"


def test_etf_coverage_positive(monkeypatch):
    meta, _ = mock_verify(monkeypatch, sizes())
    assert (
        fund_verify.check_partitions(None, np.array(DAYS), fund_verify.Register({}))[
            "etf_share_size"
        ]["verdict"]
        == "pass"
    )
    assert fund_verify.check_etf_panel(None, MagicMock(), meta)["verdict"] == "pass"


def test_empty_etf_revalidated_when_listing_evidence_changes(tmp_path):
    """来源：本轮实现检查；旧合法空日遇主表更正后须撤回完成状态，人工确认不能代替证据。"""
    store = Store(tmp_path)
    write(store, "etf_basic", "all", basic("20261114").drop("_fetched_at"))
    write(store, "etf_share_size", DAYS[0], sizes([]))
    key = f"etf_share_size/{DAYS[0]}"
    assert store.complete(key)
    write(store, "etf_basic", "all", basic("20200101").drop("_fetched_at"))
    store.confirm(key, "合成旧确认")
    store.validate_partitions(["etf_share_size"])
    entry = store.entries("etf_share_size")[key]
    assert entry["status"] == "suspect" and entry["quality_state"] == "empty_unverified"
    assert not store.complete(key)


@pytest.mark.parametrize("start", [None, "invalid", "20261301"])
def test_unknown_listing_date_does_not_prove_absence(tmp_path, start):
    store = Store(tmp_path)
    write(store, "etf_basic", "all", basic(start).drop("_fetched_at"))
    entry = write(store, "etf_share_size", DAYS[0], sizes([]))
    assert entry["status"] == "suspect"


def test_etf_source_coverage_start_is_respected(monkeypatch):
    """正例：来源任务书，接口覆盖前不要求份额数据。"""
    from ashare.data.etf_coverage import EtfCoverage

    coverage = EtfCoverage(basic("20000101"), pl.DataFrame())
    assert not coverage.required("20090102")
    assert coverage.required("20090105")


def test_etf_partition_window_uses_fund_daily(monkeypatch):
    """来源：实现检查；日历含未来日期，核验仅限行情范围，但 ETF 自己的首尾缺日不能缩窗。"""
    raw = sizes(DAYS[1:])
    _, entries = mock_verify(monkeypatch, raw)
    entries.pop(f"etf_share_size/{DAYS[0]}")
    monkeypatch.setattr(
        fund_verify,
        "_entries",
        lambda s, n: (
            entries
            if n == "etf_share_size"
            else {d: dict(partition=d, status="complete") for d in DAYS}
            if n == "fund_daily"
            else {}
        ),
    )
    result = fund_verify.check_partitions(
        None, np.array([*DAYS, "20261116"]), fund_verify.Register({})
    )["etf_share_size"]
    assert result["verdict"] == "fail"
    assert [p["partition"] for p in result["details"]["examples"]] == [DAYS[0]]
