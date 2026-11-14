"""平台主链路：HTTP → 工作进程（真实 spawn）→ 结果库落盘 → 读回。

落盘的序列、持仓、汇总与在本进程里直接调用 `groupbt.run` / `run_variant` 的结果逐位对照；
提交的配置与库里记下的一致；切基准只重算展示指标、不重跑；重启服务后结果原样读回。
"""

import json
import sqlite3
import time

import numpy as np
import pytest

fastapi = pytest.importorskip(
    "fastapi", reason="平台是可选依赖：uv sync --extra platform"
)
from fastapi.testclient import TestClient  # noqa: E402

from ashare.factor import Evaluator, groupbt  # noqa: E402
from ashare.platform.analysis import CutPanel, run_variant  # noqa: E402
from ashare.platform.app import create_app  # noqa: E402
from ashare.platform.config import expand_settings  # noqa: E402
from ashare.platform.language import parse  # noqa: E402
from ashare.platform.runner import PanelCache  # noqa: E402

ORIGIN = {"Origin": "http://127.0.0.1"}
SOURCE = "vol = ts_std(returns, [5, 10])\nalpha = -vol"
SETTINGS = {"rebalance": [1, 5], "benchmark": "000300.SH"}
WINDOWS = (("is", groupbt.IS), ("validation", groupbt.VALIDATION))


def client_for(datadir, platform_root):
    app = create_app(platform_root, data_root=str(datadir.root))
    return TestClient(app, base_url="http://127.0.0.1")


def wait_job(client, job, deadline=180):
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        result = client.get(f"/api/jobs/{job}").json()
        if result["status"] != "running":
            return result
        time.sleep(0.05)
    pytest.fail(f"任务 {job} 在 {deadline}s 内未结束")


@pytest.fixture(scope="module")
def finished(datadir, tmp_path_factory, monkeypatch_module):
    """提交一次 2 个窗口 × 2 个调仓周期 = 4 组的研究，等它跑完；之后各测试只读。"""
    root = tmp_path_factory.mktemp("platform")
    with client_for(datadir, root) as client:
        tab = client.post("/api/tabs", headers=ORIGIN).json()["id"]
        validated = client.post(
            "/api/validate",
            json={"source": SOURCE, "settings": SETTINGS},
            headers=ORIGIN,
        ).json()
        assert validated["count"] == 4
        response = client.post(
            f"/api/tabs/{tab}/runs",
            json={"name": "lowvol", "source": SOURCE, "settings": SETTINGS},
            headers=ORIGIN,
        )
        assert response.status_code == 202, response.text
        job = response.json()["id"]
        result = wait_job(client, job)
        pid = client.app.state.manager.tabs[tab]["process"].pid
        client.app.state.store.wait_holdings()
        yield dict(client=client, root=root, job=job, result=result, tab=tab, pid=pid)


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    # 共享机器：限制工作进程与算子线程。
    patch.setenv("ASHARE_PLATFORM_WORKERS", "2")
    patch.setenv("ASHARE_PLATFORM_VARIANTS", "2")
    patch.setenv("ASHARE_PLATFORM_OPS_THREADS", "2")
    yield patch
    patch.undo()


@pytest.fixture(scope="module")
def direct(datadir):
    """不经 HTTP、工作进程和落盘，直接在本进程算同样的 4 组。"""
    panel = PanelCache(str(datadir.root))
    variants = parse(SOURCE)
    configs = expand_settings(SETTINGS, len(variants))
    fields = set(groupbt.FIELDS).union(*(v.expression.fields for v in variants))
    panel.ensure(fields)
    out = {}
    for config in configs:
        for variant in variants:
            settings = {k: v for k, v in config.items() if k != "benchmark"}
            summary, arrays = run_variant(variant.expression, panel, settings)
            out[(variant.parameters, config["rebalance"])] = dict(
                variant=variant, settings=settings, summary=summary, arrays=arrays
            )
    return panel, out


def holding_days(snapshot, stop):
    """docs/PLATFORM.md 的定义，逐快照集合运算（与平台的向量化写法独立）：
    每侧总持有天数 ÷ 新进次数；某只股票上一快照同侧不在即算新进；最后一个快照持有到 stop。"""
    days, offsets = list(snapshot["days"]), list(snapshot["offsets"])
    out = []
    for sign in (1, -1):
        previous, total, entries = set(), 0, 0
        for k, day in enumerate(days):
            lo, hi = offsets[k], offsets[k + 1]
            current = {
                int(c)
                for c, w in zip(
                    snapshot["columns"][lo:hi], snapshot["weights"][lo:hi], strict=True
                )
                if w * sign > 0
            }
            duration = (days[k + 1] if k + 1 < len(days) else stop) - day
            total += len(current) * duration
            entries += len(current - previous)
            previous = current
        out.append(total / entries if entries else None)
    return tuple(out)


def key_of(run):
    """runs.label 形如 '5 · 无 · 调仓1天'。"""
    parameter, _, rebalance = run["label"].split(" · ")
    return (parameter,), int(rebalance.removeprefix("调仓").removesuffix("天"))


def test_job_completes_with_submitted_configuration(finished, datadir):
    result = finished["result"]
    assert result["status"] == "complete"
    assert len(result["runs"]) == 4
    assert all(run["status"] == "complete" for run in result["runs"])
    expected = expand_settings(SETTINGS, 2)
    # 调仓周期 1 是默认值，条目身份里省略（docs/PLATFORM.md），读回时补上再比。
    seen = sorted(
        json.dumps({"rebalance": 1} | r["settings"], sort_keys=True)
        for r in result["runs"]
    )
    assert seen == sorted(
        json.dumps(c, sort_keys=True) for c in expected for _ in range(2)
    )
    # 直接读结果库：条目配置（规范化后，基准不进身份）、运行的基准与数据版本。
    db = sqlite3.connect(finished["root"] / "platform.sqlite3")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT r.id, r.status, r.benchmark, r.data_version, e.settings, e.source, e.name "
        "FROM runs r JOIN entries e ON e.id = r.entry WHERE r.job = ?",
        (finished["job"],),
    ).fetchall()
    db.close()
    assert len(rows) == 4
    for row in rows:
        stored = json.loads(row["settings"])
        assert "benchmark" not in stored
        assert stored.get("rebalance", 1) in (1, 5)
        assert row["benchmark"] == "000300.SH"
        assert row["data_version"] == datadir.version
        assert row["name"] == "lowvol"
        assert row["source"].startswith("vol = ts_std(returns,")


def test_saved_series_bitwise_equal_to_direct_calls(finished, direct):
    store = finished["client"].app.state.store
    _, expected = direct
    for run in finished["result"]["runs"]:
        saved = store.series(run["id"])
        reference = expected[key_of(run)]["arrays"]
        series_keys = {k for k in reference if ".holdings." not in k}
        assert series_keys <= set(saved)
        for name in series_keys:
            a, b = np.asarray(saved[name]), np.asarray(reference[name])
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), name


def test_group_returns_equal_plain_groupbt_run(finished, direct):
    """平台走批量共享（run_many）；与单独调用 groupbt.run 逐位一致。"""
    panel, expected = direct
    store = finished["client"].app.state.store
    for run in finished["result"]["runs"]:
        case = expected[key_of(run)]
        signal = CutPanel(panel, groupbt.VALIDATION[1])
        alpha = Evaluator().evaluate(case["variant"].expression, signal)
        saved = store.series(run["id"])
        for part, window in WINDOWS:
            cut = CutPanel(panel, window[1])
            options = dict(case["settings"])
            if part == "validation":
                options["sub_universes"] = False
            result = groupbt.run(
                alpha[: cut.stop], cut, **options, window=window, record_holdings=True
            )
            keep = np.asarray(result.dates) >= window[0]
            assert saved[f"{part}.groups"].tobytes() == (
                result.group_pnl[:, keep].tobytes()
            )
            assert saved[f"{part}.pnl.long_short"].tobytes() == (
                result.pnl["long_short"][keep].tobytes()
            )
            # 平均持仓天数：用这次独立结果的持仓，按定义逐快照集合运算重算。
            summary = run["summary"][part]
            longs, shorts = holding_days(result.holdings, len(result.dates))
            assert summary["holding_days"] == pytest.approx(longs, rel=1e-12)
            assert summary["short_holding_days"] == pytest.approx(shorts, rel=1e-12)


def test_summary_matches_direct_and_rebalance_lengthens_holding(finished, direct):
    _, expected = direct
    for run in finished["result"]["runs"]:
        case = expected[key_of(run)]
        for part in ("is", "validation"):
            saved, reference = run["summary"][part], case["summary"][part]
            for name in (
                "sharpe",
                "return",
                "turnover",
                "ic",
                "coverage",
                "n",
                "t",
                "holding_days",
                "short_holding_days",
            ):
                assert saved[name] == reference[name], (part, name)
    # 调仓 5 天的平均持仓天数长于每日调仓。
    runs = {key_of(r): r for r in finished["result"]["runs"]}
    for parameter in ("5", "10"):
        daily = runs[((parameter,), 1)]["summary"]["is"]["holding_days"]
        weekly = runs[((parameter,), 5)]["summary"]["is"]["holding_days"]
        assert weekly > daily


def test_holdings_endpoint_matches_direct_snapshots(finished, direct):
    client = finished["client"]
    _, expected = direct
    run = finished["result"]["runs"][0]
    case = expected[key_of(run)]
    listing = client.get(f"/api/runs/{run['id']}/holdings").json()
    assert listing["date"] == listing["dates"][-1]
    snapshot_dates = case["arrays"]["validation.holdings.dates"]
    offsets = case["arrays"]["validation.holdings.offsets"]
    last = int(np.searchsorted(snapshot_dates, listing["date"], side="right")) - 1
    lo, hi = offsets[last : last + 2]
    columns = case["arrays"]["validation.holdings.columns"][lo:hi]
    weights = case["arrays"]["validation.holdings.weights"][lo:hi]
    symbols = case["arrays"]["validation.holdings.symbols"]
    day = client.get(
        f"/api/runs/{run['id']}/holdings", params={"date": listing["date"]}
    ).json()
    got = {(i["symbol"], i["side"]): i["weight"] for i in day["items"]}
    want = {
        (str(symbols[c]), "long" if w > 0 else "short"): float(w)
        for c, w in zip(columns, weights, strict=True)
    }
    assert got == want
    longs = sum(w for (_, side), w in got.items() if side == "long")
    assert longs == pytest.approx(0.5, abs=1e-9)
    assert all(i["name"].startswith("合成") for i in day["items"])
    # TEST 期日期、任意格式一律拒绝。
    assert (
        client.get(
            f"/api/runs/{run['id']}/holdings", params={"date": "20260115"}
        ).status_code
        == 422
    )


def test_series_endpoint_payload_decodes_to_saved_arrays(finished):
    import base64

    client = finished["client"]
    store = client.app.state.store
    run = finished["result"]["runs"][0]["id"]
    saved = store.series(run)
    payload = client.get(f"/api/runs/{run}/series", params={"view": "groups"}).json()
    for part in payload:
        assert part["dates"] == saved[f"{part['part']}.dates"].tolist()
        assert all(d < "20260101" for d in part["dates"])
        packed = part["series"]["groups"]
        values = np.frombuffer(base64.b64decode(packed["data"]), dtype="<f4")
        expected = saved[f"{part['part']}.groups"].astype("<f4")
        assert values.reshape(packed["shape"]).tobytes() == expected.tobytes()


def test_benchmark_switch_recomputes_display_only(finished, datadir):
    """切基准：相对指标按指数开盘价重算，不新增任务、不改运行与序列文件、工作进程不变。"""
    client, root, job = finished["client"], finished["root"], finished["job"]
    store = client.app.state.store
    series = sorted((root / "series").glob("*.npz"))
    stamps = {p: p.stat().st_mtime_ns for p in series}
    events = store.events(job, 0)
    before = client.get(f"/api/jobs/{job}").json()
    first = client.get(f"/api/jobs/{job}/benchmark", params={"symbol": "000300.SH"})
    second = client.get(f"/api/jobs/{job}/benchmark", params={"symbol": "000905.SH"})
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert (a["symbol"], b["symbol"]) == ("000300.SH", "000905.SH")
    assert a["relatives"] != b["relatives"]
    # 独立重算一组相对指标：多头第 1 组日收益 − 指数下一开盘到开盘收益。
    import ashare.data

    frame = (
        ashare.data.open(datadir.root)
        .table("index_daily", symbols=["000905.SH"], end="20251231")
        .collect()
        .sort("date")
    )
    prices = dict(zip(frame["date"].to_list(), frame["open"].to_list(), strict=True))
    days = sorted(prices)
    following = {
        d: prices[n] / prices[d] - 1 for d, n in zip(days, days[1:], strict=False)
    }
    run = finished["result"]["runs"][0]
    arrays = store.series(run["id"])
    top = arrays["is.groups"][0]
    base = np.array([following.get(str(d), np.nan) for d in arrays["is.dates"]])
    keep = np.isfinite(top) & np.isfinite(base)
    excess = top[keep] - base[keep]
    got = b["relatives"][run["id"]]["is"]
    assert got["n"] == int(keep.sum())
    assert got["excess"] == pytest.approx(excess.mean() * 240, rel=1e-12)
    assert got["information_ratio"] == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(240), rel=1e-12
    )
    # 没有重跑：任务、运行、事件、序列文件都没变，工作进程还是原来那个。
    assert client.get(f"/api/jobs/{job}").json() == before
    assert store.events(job, 0) == events
    assert {p: p.stat().st_mtime_ns for p in series} == stamps
    assert client.get("/api/library").json()["total"] == 4
    assert (
        client.app.state.manager.tabs[finished["tab"]]["process"].pid == finished["pid"]
    )
    assert client.get("/api/meta").json()["running"] == 0


def test_results_survive_restart(finished, datadir):
    client, root, job = finished["client"], finished["root"], finished["job"]
    before = client.get(f"/api/jobs/{job}").json()
    library = client.get("/api/library").json()
    run = before["runs"][0]["id"]
    holdings = client.get(f"/api/runs/{run}/holdings").json()
    entry = library["items"][0]["id"]
    with client_for(datadir, root) as again:
        assert again.get(f"/api/jobs/{job}").json() == before
        assert again.get("/api/library").json()["total"] == library["total"]
        assert again.get(f"/api/runs/{run}/holdings").json() == holdings
        detail = again.get(f"/api/library/{entry}").json()
        assert detail["id"] == entry
        events = again.get(f"/api/jobs/{job}/events").text
        assert '"kind":"complete"' in events
