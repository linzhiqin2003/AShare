"""来源：D3 ETF 专题任务；单位、状态合并、版本依赖和次日可知的回归。"""

import asyncio
import json

import numpy as np
import polars as pl
import pytest
from futures_fixtures import DAYS
from test_fund_fixtures import FakeFundTushare

import ashare.data as ad
from ashare.data.catalog import FUND_DATASETS
from ashare.data.fund_panel import build_funds
from ashare.data.fund_sync import sync_funds
from ashare.data.fund_verify import check_etf
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store


@pytest.fixture
def etf(tmp_path):
    source = FakeFundTushare()
    store = Store(tmp_path)
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    build_funds(store, kinds=("fund",))
    return store, source


def test_catalog_units_and_first_observation():
    assert FUND_DATASETS["etf_basic"].partition == "static"
    spec = FUND_DATASETS["etf_share_size"]
    assert spec.coverage_start == "20090105"
    assert spec.units == {
        "total_share": "万份（实测）",
        "total_size": "万元（实测；净资产规模，不等于份额乘收盘价）",
    }
    assert "次日可知" in spec.known_at


def test_all_statuses_single_partition(etf):
    store, source = etf
    calls = [p["list_status"] for a, p in source.calls if a == "etf_basic"]
    assert sorted(calls) == ["D", "L", "P"]
    assert list(store.entries("etf_basic")) == ["etf_basic/all"]
    assert set(store.read("etf_basic")["list_status"]) == {"L", "D", "P"}
    assert not store.entries("stock_basic")


def test_etf_fields_units_timing_and_index_lookup(etf):
    store, source = etf
    funds = ad.open(store.data_root).funds()
    panel = funds.panel("fund", ["total_share", "total_size", "total_shares"])
    raw = source.tables["etf_share_size"]
    for r in raw.iter_rows(named=True):
        i, j = (
            list(panel.dates).index(r["trade_date"]),
            list(panel.symbols).index(r["ts_code"]),
        )
        assert panel["total_share"][i, j] == r["total_share"] * 10000
        assert panel["total_size"][i, j] == r["total_size"] * 10000
    assert funds.field("fund", "total_share")["unit"] == "份"
    assert funds.field("fund", "total_size")["unit"] == "元"
    assert "次日可知" in funds.field("fund", "total_share")["known_at"]
    assert "次日可知" in funds.field("fund", "total_size")["known_at"]
    table = funds.table("fund", "etf_basic").collect()
    assert "510001.SH" in table.filter(pl.col("index_code") == "000300.SH")["symbol"]
    assert all(n in table.columns for n in ("index_name", "list_status", "fetched_at"))


def test_etf_changes_version_but_not_old_fields(etf):
    store, source = etf
    before = ad.open(store.data_root).funds()
    oldnames = [
        n for n in before.fields("fund") if n not in ("total_share", "total_size")
    ]
    arrays = {n: np.asarray(before.panel("fund", [n])[n]).copy() for n in oldnames}
    version, stock, futures = (
        before.version("fund"),
        store.version,
        store.futures_version,
    )
    day = DAYS[0]
    raw = (
        source.tables["etf_share_size"]
        .filter(pl.col("trade_date") == day)
        .with_columns(pl.col("total_size") * 2)
    )
    store.write("etf_share_size", day, {}, FetchResult(raw, "2026-02-15T00:00:00Z", 1))
    build_funds(store, kinds=("fund",))
    after = ad.open(store.data_root).funds()
    assert after.version("fund") != version
    assert (store.version, store.futures_version) == (stock, futures)
    # 来源：真实 ETF 核对确认规模取净值；只改源规模、保持收盘与份额时必须跟随源规模。
    np.testing.assert_equal(
        np.asarray(after.panel("fund", ["total_size"])["total_size"])[0],
        np.asarray(before.panel("fund", ["total_size"])["total_size"])[0] * 2,
    )
    for n, a in arrays.items():
        b = np.asarray(after.panel("fund", [n])[n])
        assert a.tobytes() == b.tobytes(), n


def test_share_difference_is_reported_not_rejected(etf):
    store, source = etf
    checks = check_etf(store)
    assert checks["etf_share_consistency"]["details"]["matched"] > 0
    assert checks["etf_share_consistency"]["details"]["different"] == 0
    day = DAYS[0]
    raw = (
        source.tables["etf_share_size"]
        .filter(pl.col("trade_date") == day)
        .with_columns(pl.col("total_share") * 2)
    )
    store.write("etf_share_size", day, {}, FetchResult(raw, "2026-02-15T00:00:00Z", 1))
    checks = check_etf(store)
    assert checks["etf_share_consistency"]["verdict"] == "info"
    assert checks["etf_share_consistency"]["details"]["different"] == len(raw)
    assert checks["etf_basic_coverage"]["details"]["covered"] > 0
    assert checks["etf_share_size_coverage"]["details"]["covered"] > 0


def test_failed_status_refresh_keeps_complete_table(etf):
    store, source = etf
    old = store.entries("etf_basic")["etf_basic/all"]
    fetch = source.fetch

    async def fail(api, params, **kwargs):
        if api == "etf_basic" and params["list_status"] == "D":
            raise RuntimeError("D3 故意令退市请求失败")
        return await fetch(api, params, **kwargs)

    source.fetch = fail
    with pytest.raises(RuntimeError):
        asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1], names=["etf_basic"]))
    new = store.entries("etf_basic")["etf_basic/all"]
    assert new["status"] == "complete" and old["content_hash"] == new["content_hash"]


def test_verify_catches_wrong_panel_units(etf):
    from ashare.data.fund_verify import check_etf_panel

    store, _ = etf
    base = store.root / "fund_panel/fund"
    path = base / json.loads((base / "current.json").read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    assert check_etf_panel(store, path, meta)["verdict"] == "pass"
    values = np.load(path / "total_size.npy")
    np.save(path / "total_size.npy", values / 10000)
    assert check_etf_panel(store, path, meta)["verdict"] == "fail"


def test_missing_values_are_not_filled(etf):
    """来源：缺值不可被当成零或前填；旧基金 total_shares 保持原值。"""
    store, source = etf
    before = np.asarray(
        ad.open(store.data_root).funds().panel("fund", ["total_shares"])["total_shares"]
    ).copy()
    day = DAYS[1]
    raw = (
        source.tables["etf_share_size"]
        .filter(pl.col("trade_date") == day)
        .with_columns(pl.lit(None).cast(pl.Float64).alias("total_share"))
    )
    store.write("etf_share_size", day, {}, FetchResult(raw, "2026-02-15T00:00:00Z", 1))
    build_funds(store, kinds=("fund",))
    panel = (
        ad.open(store.data_root).funds().panel("fund", ["total_share", "total_shares"])
    )
    assert np.isnan(panel["total_share"][1]).all()
    assert panel["total_shares"].tobytes() == before.tobytes()
    column = list(panel.symbols).index("510001.SH")
    assert np.isfinite(panel["total_shares"][1, column])


def test_verify_detects_wrong_known_at(etf):
    from ashare.data.fund_verify import check_etf_panel

    store, _ = etf
    base = store.root / "fund_panel/fund"
    path = base / json.loads((base / "current.json").read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    meta["fields"]["total_share"]["known_at"] = "盘前"
    assert check_etf_panel(store, path, meta)["verdict"] == "fail"


@pytest.mark.parametrize(
    "name,part", [("etf_basic", "all"), ("etf_share_size", DAYS[0])]
)
@pytest.mark.parametrize("rows", [0, 50])
def test_successful_empty_or_collapsed_refresh_keeps_revision(
    tmp_path, name, part, rows
):
    """来源：独立审查 P0 的两个成功空响应复现，并覆盖同分区骤降。"""
    store = Store(tmp_path)
    raw = pl.DataFrame(
        dict(ts_code=[f"ETF{i}" for i in range(200)], trade_date=[DAYS[0]] * 200)
    )
    first = store.write(name, part, {}, FetchResult(raw, "2026-01-01T00:00:00Z", 1))
    path = store.root / first["file"]
    contents, version = path.read_bytes(), store.fund_version
    # 人工确认不能绕过刷新保护，前一交易日也不影响同分区保护。
    store.confirm(f"{name}/{part}", "合成完整修订")
    with pytest.raises(ValueError, match="保留旧修订"):
        store.write(
            name, part, {}, FetchResult(raw.head(rows), "2026-02-01T00:00:00Z", 1)
        )
    current = store.entries(name)[f"{name}/{part}"]
    assert current["status"] == "complete"
    assert current["content_hash"] == first["content_hash"]
    assert path.read_bytes() == contents and store.fund_version == version


@pytest.mark.parametrize("rows", [0, 5])
def test_successful_status_shrink_keeps_complete_table(etf, caplog, rows):
    """来源：P0；L 增长掩盖 P 的空响应/缩水，必须在合并前逐状态检查。"""
    store, source = etf
    basic = source.tables["etf_basic"]
    expanded = pl.concat(
        [
            basic.filter(pl.col("list_status") == status)
            .head(1)
            .with_columns(pl.lit(f"{status}{i}").alias("ts_code"))
            for status in ("L", "D", "P")
            for i in range(20)
        ]
    )
    old = store.write(
        "etf_basic", "all", {}, FetchResult(expanded, "2026-02-01T00:00:00Z", 1)
    )
    source.tables["etf_basic"] = pl.concat(
        [
            expanded.filter(pl.col("list_status") != "P"),
            expanded.filter(pl.col("list_status") == "P").head(rows),
            expanded.filter(pl.col("list_status") == "L").with_columns(
                (pl.col("ts_code") + "extra").alias("ts_code")
            ),
        ]
    )
    with pytest.raises(ValueError, match="保留旧修订"):
        asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1], names=["etf_basic"]))
    new = store.entries("etf_basic")["etf_basic/all"]
    assert new["status"] == "complete" and old["content_hash"] == new["content_hash"]
    assert new["refresh_error"] == "ValueError"
    assert "etf_basic/all" in caplog.text


@pytest.mark.parametrize("name", ["etf_share_size", "etf_basic"])
def test_verify_rejects_empty_source_and_empty_panel(etf, monkeypatch, name):
    """来源：P0；源与面板同时丢失，NaN 相等或空表相等不能判通过。"""
    from ashare.data import fund_verify

    store, _ = etf
    base = store.root / "fund_panel/fund"
    path = base / json.loads((base / "current.json").read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    read = fund_verify._read
    monkeypatch.setattr(
        fund_verify,
        "_read",
        lambda s, n: read(s, n).head(0) if n == name else read(s, n),
    )
    if name == "etf_share_size":
        for field in ("total_share", "total_size"):
            values = np.load(path / f"{field}.npy")
            np.save(path / f"{field}.npy", np.full_like(values, np.nan))
    else:
        table = path / "table_etf_basic.parquet"
        pl.read_parquet(table).head(0).write_parquet(table)
    result = fund_verify.check_etf_panel(store, path, meta)
    assert result["verdict"] == "fail"
    assert dict(table=name, problem="empty_source") in result["details"]["examples"]


def test_verify_rejects_all_nan_fields(etf, monkeypatch):
    """来源：P0；非空原始表的整列空值同样不能用 NaN == NaN 冒充有效覆盖。"""
    from ashare.data import fund_verify

    store, _ = etf
    base = store.root / "fund_panel/fund"
    path = base / json.loads((base / "current.json").read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    read = fund_verify._read
    raw = read(store, "etf_share_size").with_columns(
        pl.lit(None).cast(pl.Float64).alias("total_share"),
        pl.lit(None).cast(pl.Float64).alias("total_size"),
    )
    monkeypatch.setattr(
        fund_verify, "_read", lambda s, n: raw if n == "etf_share_size" else read(s, n)
    )
    for field in ("total_share", "total_size"):
        values = np.load(path / f"{field}.npy")
        np.save(path / f"{field}.npy", np.full_like(values, np.nan))
    result = fund_verify.check_etf_panel(store, path, meta)
    assert result["verdict"] == "fail"
    for field in ("total_share", "total_size"):
        assert (
            dict(field=field, problem="no_finite_values")
            in result["details"]["examples"]
        )


def test_successful_empty_share_sync_warns_and_keeps_revision(etf, caplog):
    """来源：P0；同步层必须记录成功空响应为刷新失败，并保留旧修订。"""
    store, source = etf
    key = f"etf_share_size/{DAYS[0]}"
    old = store.entries("etf_share_size")[key]
    source.tables["etf_share_size"] = source.tables["etf_share_size"].head(0)
    with pytest.raises(ValueError, match="保留旧修订"):
        asyncio.run(
            sync_funds(source, store, DAYS[0], DAYS[0], names=["etf_share_size"])
        )
    new = store.entries("etf_share_size")[key]
    assert new["status"] == "complete" and old["content_hash"] == new["content_hash"]
    assert new["refresh_error"] == "ValueError"
    assert key in caplog.text and "保留旧修订" in caplog.text
