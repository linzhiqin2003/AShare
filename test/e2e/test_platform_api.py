"""平台端到端：真实启动 `python -m ashare.platform`（随机端口、临时结果目录、合成数据），
走一遍研究者在浏览器里会做的事，产物与本进程直接调用库函数的结果对照，最后干净关停、确认无遗留进程。
"""

import base64
import sys
import time

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="平台是可选依赖：uv sync --extra platform")

from ashare.factor import groupbt  # noqa: E402
from ashare.platform.analysis import run_variant  # noqa: E402
from ashare.platform.config import DEFAULTS  # noqa: E402
from ashare.platform.language import parse  # noqa: E402
from ashare.platform.runner import PanelCache  # noqa: E402
from test.fixtures.platform_server import TERMINAL, PlatformServer, alive  # noqa: E402

SOURCE = "-ts_std(returns, [5, 10])"


def unpack(packed):
    return np.frombuffer(base64.b64decode(packed["data"]), dtype="<f4").reshape(
        packed["shape"]
    )


@pytest.fixture(scope="module")
def direct(datadir):
    panel = PanelCache(str(datadir.root))
    variants = parse(SOURCE)
    panel.ensure(set(groupbt.FIELDS).union(*(v.expression.fields for v in variants)))
    return {
        v.parameters[0]: run_variant(v.expression, panel, dict(DEFAULTS))
        for v in variants
    }


def test_full_research_over_http(datadir, tmp_path, direct):
    platform = tmp_path / "platform"
    server = PlatformServer(datadir.root, platform)
    server.start()
    try:
        meta = server.get("/api/meta").json()
        last = datadir.dates[-1]
        assert meta["data_date"] == f"{last[:4]}-{last[4:6]}-{last[6:]}"
        assert meta["running"] == 0 and meta["max_workers"] == 2
        if server.official_entry:
            page = server.get("/")
            assert page.status_code == 200 and "<html" in page.text.lower()
        settings = server.get("/api/settings").json()
        assert settings["benchmark"]["symbol"] == "000001.SH"
        assert not settings["benchmark"]["fallback"]
        found = server.get("/api/benchmarks", params={"search": "沪深"}).json()
        assert [i["symbol"] for i in found["items"]] == ["000300.SH"]
        assert server.get("/api/completions").status_code == 200
        # 外来 Origin 的写请求一律拒绝。
        assert (
            server.post(
                "/api/tabs", headers={"Origin": "http://evil.example"}
            ).status_code
            == 403
        )

        tab = server.post("/api/tabs").json()["id"]
        assert server.post("/api/validate", {"source": SOURCE}).json()["count"] == 2
        response = server.post(
            f"/api/tabs/{tab}/runs",
            {"name": "lowvol", "source": SOURCE, "settings": {}},
        )
        assert response.status_code == 202, response.text
        job = response.json()["id"]
        events = server.events(job)
        ids = [e["id"] for e in events]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        kinds = [e["data"]["kind"] for e in events]
        assert kinds[0] in ("queued", "started") and kinds[-1] == "complete"
        assert "progress" in kinds
        result = server.wait_job(job)
        assert result["status"] == "complete"
        assert {r["status"] for r in result["runs"]} == {"complete"}
        server.snapshot_children()  # 工作进程此时空闲常驻，关停后要确认它也退出了

        for run in result["runs"]:
            parameter = run["label"].split(" · ")[0]
            summary, arrays = direct[parameter]
            for part in ("is", "validation"):
                for name in ("sharpe", "return", "turnover", "ic", "holding_days"):
                    assert run["summary"][part][name] == summary[part][name]
            payload = server.get(
                f"/api/runs/{run['id']}/series", params={"view": "returns"}
            ).json()
            for part in payload:
                key = f"{part['part']}.pnl.long_short"
                assert unpack(part["series"]["pnl.long_short"]).tobytes() == (
                    arrays[key].astype("<f4").tobytes()
                )
            listing = server.get(f"/api/runs/{run['id']}/holdings").json()
            day = server.get(
                f"/api/runs/{run['id']}/holdings",
                params={"date": listing["date"], "compare": "true"},
            ).json()
            longs = [i["weight"] for i in day["items"] if i["side"] == "long"]
            assert sum(longs) == pytest.approx(0.5, abs=1e-9)
            assert {i["status"] for i in day["items"]} <= {"new", "held", "exit"}

        library = server.get("/api/library").json()
        assert library["total"] == 2
        entry = library["items"][0]["id"]
        assert server.get(f"/api/library/{entry}").json()["name"] == "lowvol"
        ids = ",".join(item["id"] for item in library["items"])
        matrix = server.get("/api/library/correlation", params={"ids": ids}).json()
        assert len(matrix["matrix"]) == 2 and matrix["matrix"][0][0] == pytest.approx(1)

        # 切基准不重跑：任务与因子库不变。
        before = server.get(f"/api/jobs/{job}").json()
        swaps = [
            server.get(f"/api/jobs/{job}/benchmark", params={"symbol": s}).json()
            for s in ("000300.SH", "000852.SH")
        ]
        assert [s["symbol"] for s in swaps] == ["000300.SH", "000852.SH"]
        assert swaps[0]["relatives"] != swaps[1]["relatives"]
        assert server.get(f"/api/jobs/{job}").json() == before
        assert server.get("/api/library").json()["total"] == 2

        # 改名同步到因子库。
        assert (
            server.post(f"/api/tabs/{tab}/name", {"name": "renamed"}).status_code == 200
        )
        names = {i["name"] for i in server.get("/api/library").json()["items"]}
        assert names == {"renamed"}

        # 另一个 tab：提交后立刻取消，再提交一个小任务能跑完。
        other = server.post("/api/tabs").json()["id"]
        big = server.post(
            f"/api/tabs/{other}/runs",
            {"source": "-ts_std(returns,[10,20,30,40,50,60,70,80])"},
        ).json()["id"]
        assert server.post(f"/api/jobs/{big}/cancel").json()["cancelled"] in (
            True,
            False,
        )
        assert server.wait_job(big)["status"] in TERMINAL
        again = server.post(
            f"/api/tabs/{other}/runs", {"source": "-ts_std(returns, 20)"}
        ).json()["id"]
        assert server.wait_job(again)["status"] == "complete"
        server.snapshot_children()
        assert server.delete(f"/api/tabs/{other}").status_code == 200
    finally:
        stopped = server.close()
    assert stopped["graceful"], server.log_text()
    assert stopped["leftovers"] == {}, stopped
    assert server.seen_children, "没有记到任何工作进程，遗留进程检查形同虚设"

    # 落盘读回：同一结果目录重新起服务，任务与因子库原样可读。
    with PlatformServer(datadir.root, platform) as again_server:
        assert again_server.get(f"/api/jobs/{job}").json()["status"] == "complete"
        assert {
            r["id"] for r in again_server.get(f"/api/jobs/{job}").json()["runs"]
        } == {r["id"] for r in result["runs"]}
        # 条目在提交时就建好（取消的也计入 Bonferroni 的 N）：窗口 {5,10,…,80} 共 9 个。
        assert again_server.get("/api/library").json()["total"] == 9
    log = (platform / "logs/platform.log").read_text(encoding="utf-8")
    assert "任务完成" in log


def test_listens_on_loopback_only(datadir, tmp_path):
    """只监听 127.0.0.1。看监听套接字本身（lsof），不靠从别的地址连：
    2026-11-14 本机 Clash TUN 会在本地替任意地址完成握手，「连得上」不能说明服务对外监听。"""
    import shutil
    import subprocess

    if shutil.which("lsof") is None:
        pytest.skip("没有 lsof")
    with PlatformServer(datadir.root, tmp_path / "platform") as server:
        output = subprocess.run(
            [
                "lsof",
                "-nP",
                "-a",
                "-p",
                str(server.process.pid),
                "-iTCP",
                "-sTCP:LISTEN",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        listening = [line.split()[8] for line in output.splitlines()[1:]]
        assert listening == [f"127.0.0.1:{server.port}"], output


FAKE_SERVER = """
import http.server, subprocess, sys
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *args):
        pass
try:
    http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
except KeyboardInterrupt:
    pass
"""


def test_leftover_check_catches_orphaned_children(tmp_path):
    """助手自测：服务退出后留下的子进程会被报出来并被清掉，检查不是空转。"""
    server = PlatformServer(
        tmp_path,
        tmp_path / "fake",
        command=lambda port: [sys.executable, "-c", FAKE_SERVER, str(port)],
    )
    server.start()
    deadline = time.monotonic() + 20
    while not server.snapshot_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    stopped = server.close()
    assert stopped["graceful"]
    assert len(stopped["leftovers"]) == 1
    (pid,) = stopped["leftovers"]
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(pid)
