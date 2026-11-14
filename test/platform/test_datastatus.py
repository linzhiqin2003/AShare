"""数据页、状态栏行情与默认回测设定接口（2026-11-14 新增）。全部用合成数据目录，不读真实行情。"""

import gc
import json
import os
import time

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from ashare.platform import datastatus, market
from ashare.platform.app import create_app
from bench.platform.synthetic import write_indexes, write_panel

ORIGIN = {"Origin": "http://127.0.0.1"}


def tree(root):
    """数据目录下全部文件的 (路径, 大小, mtime)，用来证明接口只读。"""
    output = {}
    for folder, _, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            stat = os.stat(path)
            output[path] = (stat.st_size, stat.st_mtime_ns)
    return output


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    source = write_panel(root)
    write_indexes(root, source, codes=["000001.SH", "000300.SH"])
    # 合成数据的写入器留下处在引用环里的可写清单连接，只有循环垃圾回收才会关闭它们；
    # 关闭时 SQLite 把 WAL 合并进 manifest.sqlite3 并删掉 -wal/-shm。不在这里回收，
    # 回收就会落在测试中途的任意时刻（全量并行时更常见），「数据目录没被改动」的断言随之偶发失败。
    gc.collect()
    return root, source


@pytest.fixture
def client(tmp_path, data):
    app = create_app(tmp_path / "platform", data_root=str(data[0]))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


def wait_fields(client, deadline=120):
    started = time.monotonic()
    while time.monotonic() - started < deadline:
        body = client.get("/api/data/fields").json()
        if body["state"] != "pending":
            return body
        time.sleep(0.05)
    raise AssertionError("字段统计超时")


# ---- 纯函数 --------------------------------------------------------------


def test_field_stats_coverage_counts_only_listed_and_finite():
    dates = np.array(["20240130", "20240131", "20240201", "20240202"])
    listed = np.array(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
            [True, False, True],
        ]
    )
    values = np.array(
        [
            [np.nan, np.nan, 5.0],  # 第三只未上市，不计入
            [1.0, np.nan, np.nan],
            [1.0, 2.0, np.nan],
            [1.0, 9.0, 3.0],  # 第二只已退市，不计入
        ]
    )
    result = datastatus.field_stats(values, "pe", dates, listed)
    assert result["start"] == "2024-01-31"
    assert result["latest"] == 1.0
    assert result["latest_count"] == 2
    assert result["overall"] == pytest.approx(
        (0 + 1 + 2 + 2) / (2 + 3 + 3 + 2), abs=1e-4
    )
    assert result["monthly"] == [
        ["202401", pytest.approx(1 / 6, abs=1e-4)],
        ["202402", pytest.approx((2 / 3 + 1) / 2, abs=1e-4)],
    ]
    distribution = result["distribution"]
    # 2026-11-14 开发中发现：按 1%/99% 分位截掉两端时，只有两只股票的分布整张为空；改为并入边格。
    assert distribution["n"] == sum(distribution["counts"]) == 2
    assert distribution["quantiles"][2] == pytest.approx(2.0)


def test_field_stats_zero_missing_bool_and_categorical():
    dates = np.array(["20240102", "20240103"])
    listed = np.ones((2, 4), dtype=bool)
    industry = np.array(
        [[0, 801010, 801020, 0], [801010, 801010, 0, 801030]], dtype=np.uint32
    )
    result = datastatus.field_stats(industry, "industry", dates, listed)
    # 行业编码 0 表示缺失；编码不画分布。
    assert result["latest"] == 0.75 and result["distribution"] is None
    flags = np.array([[True, False, False, False], [True, True, False, False]])
    result = datastatus.field_stats(flags, "is_st", dates, listed)
    assert result["latest"] == 1.0 and result["true_share"] == 0.5
    assert result["distribution"] is None
    ages = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.uint16)
    assert datastatus.field_stats(ages, "listing_age", dates, listed)["distribution"]


def test_field_stats_chunk_boundary_matches_direct():
    """跨 256 行分块时逐行计数与一次性计算一致（分块写回的下标若错位会被抓到）。"""
    rng = np.random.default_rng(7)
    rows = 600
    dates = np.array(
        [f"2020{(i // 28) % 12 + 1:02d}{i % 28 + 1:02d}" for i in range(rows)]
    )
    listed = rng.random((rows, 9)) > 0.2
    values = np.where(rng.random((rows, 9)) > 0.3, rng.normal(size=(rows, 9)), np.nan)
    result = datastatus.field_stats(values, "x", dates, listed)
    counts = (np.isfinite(values) & listed).sum(1)
    assert result["latest_count"] == counts[-1]
    assert result["overall"] == pytest.approx(counts.sum() / listed.sum(), abs=1e-4)


def test_spark_downsamples_and_keeps_gaps():
    monthly = [[f"m{i}", None if i < 10 else 1.0] for i in range(240)]
    spark = datastatus._spark(monthly, 60)
    assert len(spark) == 60 and spark[0] is None and spark[-1] == 1.0
    assert datastatus._spark(monthly[:5]) == [None] * 5


def test_group_and_category_of_names():
    assert datastatus.group_of("index_daily") == "index"
    assert datastatus.group_of("fut_daily") == "futures"
    assert datastatus.group_of("futures_settle") == "futures"
    assert datastatus.group_of("cb_daily") == "cb"
    assert datastatus.group_of("fund_adj") == "fund"
    assert datastatus.group_of("daily") == "stock"
    assert datastatus.category("weight_csi300") == "指数成分"
    assert datastatus.category("pe_ttm") == "估值"


def test_industry_moves_weighting_names_and_order():
    codes = np.array([801010, 801010, 801020, 801030, 0])
    close = np.array([11.0, 9.0, 10.3, 10.0, 12.0])
    previous = np.full(5, 10.0)
    quoted = np.array([True, True, True, True, True])
    weights = np.array([3.0, 1.0, 1.0, np.nan, 1.0])
    names = {801010: "农业", 801020: "能源"}
    rows = market.industry_moves(codes, close, previous, quoted, weights, names)
    # 801030 没有名称，编码 0 视为缺失行业，都不显示；农业按市值加权 (10% * 3 - 10%) / 4。
    assert [r["name"] for r in rows] == ["农业", "能源"]
    assert rows[0]["change"] == pytest.approx(0.05)
    equal = market.industry_moves(codes, close, previous, quoted, None, names)
    assert next(r for r in equal if r["name"] == "农业")["change"] == pytest.approx(0.0)


def test_breadth_and_index_quote_require_same_day():
    result = market.breadth(
        np.array([2.0, 1.0, 1.0, np.nan]),
        np.array([1.0, 2.0, 1.0, 1.0]),
        np.array([True, True, True, True]),
    )
    assert result == {"up": 1, "down": 1, "flat": 1}
    frame = pl.DataFrame({"date": ["20240102", "20240103"], "close": [100.0, 101.0]})
    assert market.index_quotes(frame, "20240103") == {"close": 101.0, "change": 0.01}
    # 指数日线没到最新交易日就不显示，不拿旧行情冒充。
    assert market.index_quotes(frame, "20240104") is None
    with_previous = frame.with_columns(previous_close=pl.Series([99.0, 100.0]))
    assert market.index_quotes(with_previous, "20240103")["change"] == pytest.approx(
        0.01
    )


# ---- 接口 ----------------------------------------------------------------


def test_data_endpoints_read_only_and_cached(tmp_path, data, client):
    root, source = data
    before = tree(root)
    overview = client.get("/api/data/overview").json()
    keys = [d["key"] for d in overview["datasets"]]
    assert keys == ["stock", "index"]
    stock = overview["datasets"][0]
    assert stock["panel"]["days"] == len(source.dates)
    assert stock["panel"]["symbols"] == len(source.symbols)
    assert (
        stock["panel"]["end"]
        == f"{source.dates[-1][:4]}-{source.dates[-1][4:6]}-{source.dates[-1][6:]}"
    )
    assert stock["panel"]["size"] > 0
    index = overview["datasets"][1]
    assert [i["symbol"] for i in index["indexes"]] == ["000001.SH", "000300.SH"]
    assert {t["name"] for t in index["tables"]} == {"index_daily", "index_classify"}
    body = wait_fields(client)
    assert body["state"] == "ready"
    names = [f["name"] for f in body["fields"]]
    assert set(names) == set(source._arrays)
    assert (
        names.index("open") < names.index("total_market_cap") < names.index("is_listed")
    )
    close = next(f for f in body["fields"] if f["name"] == "close")
    assert close["label"] == "收盘价" and close["latest"] == 1.0
    assert close["references"] >= 1  # 登记因子里有用收盘价的
    detail = client.get("/api/data/fields/close").json()
    assert detail["latest_count"] == len(source.symbols)
    assert detail["factors"] and detail["distribution"]["n"] == len(source.symbols)
    assert client.get("/api/data/fields/nope").status_code == 404
    cache = list((tmp_path / "platform/cache").glob("datastatus-*.json"))
    assert len(cache) == 1
    # 数据目录一个字节都没动（含清单的 WAL/SHM）。
    assert tree(root) == before


def test_fixture_leaves_no_pending_manifest_writes(data):
    """2026-11-14 全量并行时 test_data_endpoints_read_only_and_cached 偶发失败的根因：
    夹具的清单连接在测试中途被循环垃圾回收关闭、合并 WAL。夹具结束后再回收不应改动任何文件。"""
    before = tree(data[0])
    gc.collect()
    assert tree(data[0]) == before


def test_field_stats_cache_survives_restart_without_recompute(
    tmp_path, data, monkeypatch
):
    root, _ = data
    first = create_app(tmp_path / "platform", data_root=str(root))
    with TestClient(first, base_url="http://127.0.0.1") as client:
        assert wait_fields(client)["state"] == "ready"

    def boom(*args, **kwargs):
        raise AssertionError("命中磁盘缓存时不应重算")

    monkeypatch.setattr(datastatus, "field_stats", boom)
    second = create_app(tmp_path / "platform", data_root=str(root))
    with TestClient(second, base_url="http://127.0.0.1") as client:
        body = client.get("/api/data/fields").json()
        assert body["state"] == "ready" and body["fields"]


def test_overview_rebuilds_when_panel_changes(tmp_path, data, client, monkeypatch):
    root, _ = data
    first = client.get("/api/data/overview").json()["datasets"][0]["panel"]["version"]
    folder = root / "cn/panel"
    (folder / "synthetic-v2").mkdir()
    for path in (folder / "synthetic-v1").iterdir():
        (folder / "synthetic-v2" / path.name).write_bytes(path.read_bytes())
    meta = json.loads((folder / "synthetic-v2/metadata.json").read_text())
    (folder / "synthetic-v2/metadata.json").write_text(
        json.dumps(meta | {"version": "synthetic-v2"})
    )
    (folder / "current.json").write_text(json.dumps({"version": "synthetic-v2"}))
    # 不靠墙钟等待：直接关掉签名检查的间隔。
    monkeypatch.setattr(datastatus.DataStatus, "CHECK_INTERVAL", 0)
    second = client.get("/api/data/overview").json()["datasets"][0]["panel"]["version"]
    assert (first, second) == ("synthetic-v1", "synthetic-v2")
    assert client.get("/api/data/fields").json()["version"] == "synthetic-v2"


def test_failed_field_stats_report_failed_not_pending(tmp_path, data, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("坏文件")

    monkeypatch.setattr(datastatus, "field_stats", boom)
    app = create_app(tmp_path / "platform", data_root=str(data[0]))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert wait_fields(client)["state"] == "failed"
        # 失败不泄漏异常内容。
        assert "坏文件" not in client.get("/api/data/fields").text


def test_missing_panel(tmp_path):
    root = tmp_path / "empty"
    (root / "cn").mkdir(parents=True)
    app = create_app(tmp_path / "platform", data_root=str(root))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/data/overview").json() == {"datasets": []}
        assert client.get("/api/data/fields").json()["state"] == "missing"
        assert client.get("/api/market").json() == {
            "date": None,
            "indexes": [],
            "breadth": None,
            "industries": [],
        }


def test_market_summary_from_synthetic_day(data, client):
    _, source = data
    body = client.get("/api/market").json()
    day = source.dates[-1]
    assert body["date"] == f"{day[:4]}-{day[4:6]}-{day[6:]}"
    assert [i["symbol"] for i in body["indexes"]] == ["000001.SH", "000300.SH"]
    close = np.asarray(source["close"][-1])
    previous = np.asarray(source["previous_close"][-1])
    assert body["breadth"]["up"] == int((close > previous).sum())
    assert body["breadth"]["down"] == int((close < previous).sum())
    assert {r["name"] for r in body["industries"]} == {
        "合成农业",
        "合成能源",
        "合成材料",
    }
    changes = [r["change"] for r in body["industries"]]
    assert changes == sorted(changes, reverse=True)


def test_preferences_apply_to_settings_and_validate(client):
    builtin = client.get("/api/settings").json()["defaults"]
    assert builtin["universe"] == "TOP1800"
    assert (
        client.put("/api/preferences", json={"universe": "TOP800"}).status_code == 403
    )
    saved = client.put(
        "/api/preferences",
        json={
            "universe": "TOP800",
            "rebalance": 5,
            "trade_limitation": False,
            "benchmark": "000300.SH",
        },
        headers=ORIGIN,
    )
    assert saved.status_code == 200
    assert saved.json()["backtest"]["universe"] == "TOP800"
    defaults = client.get("/api/settings").json()["defaults"]
    assert (
        defaults["universe"],
        defaults["rebalance"],
        defaults["trade_limitation"],
    ) == ("TOP800", 5, False)
    assert defaults["benchmark"] == "000300.SH"
    # 中性化保持内置默认（新建 tab 固定以两组扫描开始）。
    assert defaults["group_neutralize"] is None
    for bad in (
        {"universe": "TOP9"},
        {"rebalance": 0},
        {"n_groups": [5, 10]},
        {"group_neutralize": "cap"},
        {"benchmark": "../x"},
    ):
        response = client.put("/api/preferences", json=bad, headers=ORIGIN)
        assert response.status_code == 422, bad
    assert client.get("/api/settings").json()["defaults"]["universe"] == "TOP800"
    reset = client.delete("/api/preferences", headers=ORIGIN).json()
    assert reset["backtest"]["universe"] == "TOP1800"


def test_corrupt_preferences_fall_back_to_builtin(tmp_path, client):
    (tmp_path / "platform/preferences.json").write_text('{"backtest": {"universe": 3}}')
    assert client.get("/api/settings").json()["defaults"]["universe"] == "TOP1800"
    (tmp_path / "platform/preferences.json").write_text("not json")
    assert client.get("/api/preferences").json()["backtest"]["universe"] == "TOP1800"
