"""持仓区间手算、单日快照与扫描身份。"""

import numpy as np
import pytest

from ashare.platform.analysis import run_variant
from ashare.platform.config import DEFAULTS, expand_settings
from ashare.platform.holdings import Holdings, mean_days
from ashare.platform.language import parse
from ashare.platform.storage import Store
from bench.platform.synthetic import write_panel


def test_observed_intervals_hand_calculation():
    snapshot = dict(
        days=np.array([1, 3, 5]),
        offsets=np.array([0, 2, 5, 7]),
        columns=np.array([0, 2, 0, 1, 2, 1, 0]),
        weights=np.array([0.5, -0.5, 0.25, 0.25, -0.5, 0.5, -0.5]),
    )
    # 多头 A 为 4 天、B 为 4 天；空头 C 为 4 天、A 为 2 天，窗口到第 7 行前。
    assert mean_days(snapshot, 7) == (4, 3)


def test_scan_identity_filter_group_and_daily_lookup(tmp_path):
    root = tmp_path / "data"
    source = write_panel(root)
    store = Store(tmp_path / "results")
    variant = parse("rank(close)", factors={})[0]
    configs = expand_settings({"rebalance": [1, 5, 20]})
    job, runs = store.begin("tab", "synthetic", [(variant, c) for c in configs], {})
    assert len({r["entry"] for r in runs}) == 3
    for run, config in zip(runs, configs, strict=True):
        summary, arrays = run_variant(variant.expression, source, config)
        assert summary["is"]["holding_days"] >= 1
        store.save(run["id"], summary, arrays, 0, "synthetic")
    store.finish(job, "complete")
    filtered = store.library(rebalance="5", group="rebalance")
    assert (
        filtered["total"]
        == sum(filtered["counts"].values())
        == filtered["groups"][0]["count"]
        == 1
    )
    assert filtered["items"][0]["holding_days"] >= 1
    reader = Holdings(store, root)
    run = runs[1]["id"]
    days = reader.read(run)["dates"]
    first = reader.read(run, days[0], compare=True)
    second = reader.read(run, days[1], compare=True)
    assert first["snapshot"] == second["snapshot"]
    assert first["items"] == second["items"]
    assert all(
        r["name"].startswith("合成证券") and r["industry"].startswith("合成")
        for r in first["items"]
    )
    assert all(r["status"] == "new" for r in first["items"])
    assert reader.read(run, days[-1], search="不存在")["items"] == []
    with pytest.raises(ValueError):
        reader.read(run, "20260101")
    # 默认周期延续旧条目身份。
    old = dict(DEFAULTS)
    old.pop("rebalance")
    _, previous = store.begin("tab", "synthetic", [(variant, old)], {})
    assert previous[0]["entry"] == runs[0]["entry"]
    with pytest.raises(ValueError, match="8"):
        expand_settings({"rebalance": [1, 5, 20]}, 3)


def test_http_day_query_and_window_guards(tmp_path):
    from fastapi.testclient import TestClient

    from ashare.platform.app import create_app

    source = write_panel(tmp_path / "data")
    app = create_app(tmp_path / "results", data_root=tmp_path / "data")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        store = app.state.store
        variant = parse("rank(returns)", factors={})[0]
        job, runs = store.begin("tab", "synthetic", [(variant, DEFAULTS)], {})
        summary, arrays = run_variant(variant.expression, source, DEFAULTS)
        run = runs[0]["id"]
        store.save(run, summary, arrays, 0, "synthetic")
        store.finish(job, "complete")
        base = f"/api/runs/{run}/holdings"
        dates = client.get(base).json()
        result = client.get(base, params={"date": dates["date"], "compare": True})
        assert result.status_code == 200
        assert result.json()["total"] == len(result.json()["items"]) > 0
        exited = [r for r in result.json()["items"] if r["status"] == "exit"]
        assert exited
        assert all(r["weight"] == 0 and r["previous_weight"] != 0 for r in exited)
        assert client.get(base, params={"date": "20260101"}).status_code == 422
        assert client.get(base, params={"date": "../../secret"}).status_code == 422
        assert client.get("/api/runs/missing/holdings").status_code == 404
        assert not any(
            word in result.text
            for word in ("sqlite", "securities.parquet", "holdings/")
        )


def test_holdings_archive_is_written_in_background(tmp_path, monkeypatch):
    """save 不等持仓档案落盘；读持仓时等后台写完，不会读到半个文件。"""
    import threading

    import ashare.platform.storage as storage

    root = tmp_path / "data"
    source = write_panel(root)
    store = Store(tmp_path / "results")
    variant = parse("rank(close)", factors={})[0]
    job, runs = store.begin("tab", "synthetic", [(variant, dict(DEFAULTS))], {})
    summary, arrays = run_variant(variant.expression, source, dict(DEFAULTS))
    release = threading.Event()
    original = storage.write_archive

    def slow(path, data):
        assert release.wait(10)
        original(path, data)

    monkeypatch.setattr(storage, "write_archive", slow)
    run = runs[0]["id"]
    final = tmp_path / "results" / "holdings" / f"{run}.npz"
    assert store.save(run, summary, arrays, 0, "synthetic")
    assert not final.exists()  # 档案还在后台，save 已经返回
    store.finish(job, "complete")
    threading.Timer(0.2, release.set).start()
    days = Holdings(store, root).read(run)["dates"]
    assert final.exists() and days
    assert not list(final.parent.glob("*.tmp"))
    store.close()
