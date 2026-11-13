"""真实 spawn 工作进程的 HTTP 合同与本机安全边界。"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ashare.platform.app import create_app
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from ashare.platform.runner import PanelCache
from bench.platform.synthetic import write_panel

ORIGIN = {"Origin": "http://127.0.0.1"}


@pytest.fixture
def client(tmp_path):
    data = tmp_path / "data"
    write_panel(data)
    app = create_app(tmp_path / "platform", data_root=str(data))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


def test_host_origin_input_limits_and_no_path_leak(client):
    assert client.get("/api/meta", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/tabs").status_code == 403
    for origin in (
        "null",
        "https://evil.example",
        "http://localhost",
        "http://127.0.0.1:9999",
        "http://127.0.0.1.evil",
    ):
        assert client.post("/api/tabs", headers={"Origin": origin}).status_code == 403
    assert (
        client.post(
            "/api/validate", json={"source": "x" * 40000}, headers=ORIGIN
        ).status_code
        == 413
    )
    response = client.post(
        "/api/validate",
        json={"source": "close", "window": ["20260101", "99991231"]},
        headers=ORIGIN,
    )
    assert response.status_code == 422
    assert "window" not in response.text
    assert client.get("/api/meta").json()["data_date"].startswith("2026")
    assert str(client.app.state.store.root) not in client.get("/api/meta").text
    assert client.get("/api/completions").status_code == 200
    assert client.get("/api/library/correlation").json() == {"ids": [], "matrix": []}
    assert (
        client.get(
            "/api/library/correlation", params={"ids": ",".join(map(str, range(13)))}
        ).status_code
        == 422
    )


def test_validation_error_position_scan_cap_and_test_window_rejection(client):
    response = client.post(
        "/api/validate",
        json={"source": "# test\nalpha = ts_stdd(close, 5)"},
        headers=ORIGIN,
    )
    assert response.status_code == 422
    assert response.json()["line"] == 2
    assert "未知函数 ts_stdd" in response.json()["message"]
    response = client.post(
        "/api/validate",
        json={
            "source": "ts_std(returns,[10,20,30])",
            "settings": {"group_neutralize": [None, "cap", "industry"]},
        },
        headers=ORIGIN,
    )
    assert response.status_code == 422 and "8 组" in response.json()["message"]
    for key in ("window", "start", "end", "TEST", "validation_window"):
        response = client.post(
            "/api/validate",
            json={"source": "close", "settings": {key: "20260101"}},
            headers=ORIGIN,
        )
        assert response.status_code == 422


def wait_job(client, job, deadline=30):
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        result = client.get(f"/api/jobs/{job}").json()
        if result["status"] != "running":
            return result
        time.sleep(0.05)
    pytest.fail("工作进程未在期限内完成")


def test_run_four_variants_sse_cache_and_reuse_worker(client):
    tab = client.post("/api/tabs", headers=ORIGIN).json()["id"]
    request = {
        "name": "synthetic_alpha",
        "source": "vol = ts_std(returns, [20,60])\nalpha = -vol",
        "settings": {"group_neutralize": ["industry", "cap+industry"]},
    }
    response = client.post(f"/api/tabs/{tab}/runs", json=request, headers=ORIGIN)
    assert response.status_code == 202, response.text
    job = response.json()["id"]
    state = client.app.state.manager.tabs[tab]
    pid = state["process"].pid
    result = wait_job(client, job)
    assert result["status"] == "complete"
    assert len(result["runs"]) == 4
    assert all(r["status"] == "complete" for r in result["runs"])
    assert (
        client.post(
            f"/api/tabs/{tab}/name", json={"name": "renamed"}, headers=ORIGIN
        ).status_code
        == 200
    )
    assert {entry["name"] for entry in client.get("/api/library").json()["items"]} == {
        "renamed"
    }
    events = client.get(f"/api/jobs/{job}/events").text
    assert '"kind":"complete"' in events
    ids = [
        int(line.split(": ")[1])
        for line in events.splitlines()
        if line.startswith("id:")
    ]
    resume = client.get(
        f"/api/jobs/{job}/events", headers={"Last-Event-ID": str(ids[-2])}
    ).text
    assert f"id: {ids[-1]}" in resume and f"id: {ids[-2]}\n" not in resume
    baseline = client.get(f"/api/jobs/{job}/benchmark")
    assert baseline.status_code == 200
    assert (
        client.get(
            f"/api/jobs/{job}/benchmark",
            headers={"If-None-Match": baseline.headers["etag"]},
        ).status_code
        == 304
    )
    run = result["runs"][0]["id"]
    for view in ("returns", "long", "groups", "ic", "subs"):
        series = client.get(f"/api/runs/{run}/series?view={view}")
        assert series.status_code == 200
        assert "immutable" in series.headers["cache-control"]
        assert all(d < "20260101" for part in series.json() for d in part["dates"])
        assert (
            client.get(
                f"/api/runs/{run}/series?view={view}",
                headers={"If-None-Match": series.headers["etag"]},
            ).status_code
            == 304
        )
    again = client.post(
        f"/api/tabs/{tab}/runs", json={"source": "-ts_std(returns,20)"}, headers=ORIGIN
    ).json()["id"]
    assert client.app.state.manager.tabs[tab]["process"].pid == pid
    assert wait_job(client, again)["status"] == "complete"
    assert client.get("/api/library").json()["total"] == 5
    client.delete(f"/api/tabs/{tab}", headers=ORIGIN)
    assert not client.app.state.manager.tabs


def test_tabs_parallel_and_cancel_then_rerun(client):
    tabs = [client.post("/api/tabs", headers=ORIGIN).json()["id"] for _ in range(2)]
    jobs = [
        client.post(
            f"/api/tabs/{tab}/runs",
            json={"source": "-ts_std(returns,[10,20,30,40,50,60,70,80])"},
            headers=ORIGIN,
        ).json()["id"]
        for tab in tabs
    ]
    manager = client.app.state.manager
    assert len({manager.tabs[t]["process"].pid for t in tabs}) == 2
    assert client.post(f"/api/jobs/{jobs[0]}/cancel", headers=ORIGIN).json()[
        "cancelled"
    ]
    assert client.get(f"/api/jobs/{jobs[0]}").json()["status"] == "cancelled"
    assert manager.tabs[tabs[0]]["process"] is None
    assert wait_job(client, jobs[1])["status"] == "complete"
    again = client.post(
        f"/api/tabs/{tabs[0]}/runs",
        json={"source": "-ts_std(returns,20)"},
        headers=ORIGIN,
    ).json()["id"]
    assert wait_job(client, again)["status"] == "complete"


def test_mmap_fields_loaded_once_and_cut_before_test(tmp_path):
    write_panel(tmp_path)
    cache = PanelCache(tmp_path)
    assert cache.arrays == {}
    cache.ensure({"close"})
    first = cache["close"]
    assert isinstance(first, np.memmap) and not first.flags.writeable
    cache.ensure({"close", "open"})
    assert cache["close"] is first
    assert set(cache.arrays) == {"close", "open"}
    assert cache.dates[-1] < "20260101"


def test_process_limit_and_no_window_in_public_api(client):
    manager = client.app.state.manager
    for _ in range(manager.limit + 5):
        assert client.post("/api/tabs", headers=ORIGIN).status_code == 200
    assert all(state["process"] is None for state in manager.tabs.values())
    tab = next(iter(manager.tabs))
    count = len(manager.tabs)
    assert (
        client.post("/api/tabs", json={"id": tab}, headers=ORIGIN).json()["id"] == tab
    )
    assert len(manager.tabs) == count
    response = client.post(
        f"/api/tabs/{tab}/runs",
        json={"source": "close", "settings": {"window": ["20260101", "99991231"]}},
        headers=ORIGIN,
    )
    assert response.status_code == 422
    assert manager.tabs[tab]["process"] is None
    assert client.get("/api/test").status_code == 404
    assert (
        client.get("/api/library?sort=sharpe%3BDROP%20TABLE%20entries").status_code
        == 422
    )


def test_framework_imports_without_platform_dependencies():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('fastapi', 'uvicorn'):
            raise ImportError('platform dependencies are unavailable')
sys.meta_path.insert(0, Block())
import ashare
import ashare.data
import ashare.factor
import ashare.factor.groupbt
import ashare.backtest
import ashare.platform
assert 'fastapi' not in sys.modules
assert 'uvicorn' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=30,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )


def test_unbuilt_frontend_gives_one_line_without_server(monkeypatch, tmp_path, capsys):
    from ashare.platform import __main__

    monkeypatch.setattr(__main__, "__file__", str(tmp_path / "__main__.py"))
    monkeypatch.setattr(sys, "argv", ["ashare.platform"])
    __main__.main()
    output = capsys.readouterr()
    assert output.out.count("\n") == 1
    assert "make web" in output.out
    assert output.err == ""


def test_disk_preserves_original_float64(client):
    store = client.app.state.store
    variant = parse("rank(close)", factors={})[0]
    job, runs = store.begin("tab", "alpha", [(variant, DEFAULTS)], {})
    from ashare.platform.analysis import run_variant
    from bench.platform.synthetic import panel

    summary, arrays = run_variant(variant.expression, panel(), DEFAULTS)
    store.save(runs[0]["id"], summary, arrays, 1.0, "synthetic")
    recovered = store.series(runs[0]["id"])
    for key, original in arrays.items():
        if original.dtype.kind == "f":
            assert recovered[key].dtype == np.float64
            np.testing.assert_array_equal(
                recovered[key].view(np.uint64), original.view(np.uint64)
            )
    with store.connect() as db:
        version = db.execute("SELECT version FROM runs WHERE job=?", (job,)).fetchone()[
            0
        ]
        assert json.loads(version) == {}
    assert Path(store.root / "series" / f"{runs[0]['id']}.npz").exists()


def test_worker_budget_queue_and_idle_reaping(tmp_path):
    from ashare.platform.runner import Manager
    from ashare.platform.storage import Store

    data = tmp_path / "data"
    write_panel(data)
    manager = Manager(
        Store(tmp_path / "results"),
        limit=1,
        data_root=str(data),
        mode="processes",
        idle_seconds=0.15,
    )
    try:
        tabs = [manager.create_tab() for _ in range(3)]
        jobs = [
            manager.submit(tab, "rank(close)", {"sub_universes": False}, "synthetic")
            for tab in tabs
        ]
        assert sum(bool(state["process"]) for state in manager.tabs.values()) == 1
        assert sum(bool(state["pending"]) for state in manager.tabs.values()) == 2
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (
            manager.running or any(s["process"] for s in manager.tabs.values())
        ):
            with manager.lock:
                assert sum(s["slots"] for s in manager.tabs.values()) <= 1
            time.sleep(0.05)
        assert not manager.running and all(
            s["process"] is None for s in manager.tabs.values()
        )
        assert all(manager.store.job(job)["status"] == "complete" for job in jobs)
        for job in jobs:
            phases = [event.get("phase") for event in manager.store.events(job, 0)]
            assert all(
                p in phases for p in ("evaluate", "prepared", "is", "validation")
            )
    finally:
        manager.close()
