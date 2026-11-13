"""通过真实 HTTP 入口测四变体；只输出时间、内存、存储字节，不输出行情指标。"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx


def memory_tree(pid):
    text = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    ).stdout
    rows = [tuple(map(int, line.split())) for line in text.splitlines()]
    selected = {pid}
    for _ in range(5):
        selected.update(p for p, parent, _ in rows if parent in selected)
    return sum(rss * 1024 for p, _, rss in rows if p in selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".cache/platform/speed-r5")
    parser.add_argument("--port", type=int, default=8779)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["threads", "processes"])
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if not root.is_relative_to(Path(".cache").resolve()):
        parser.error("测量结果只允许写入本 worktree 的 .cache")
    root.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    measurements = []
    for mode in args.modes:
        env = dict(
            os.environ,
            ASHARE_PLATFORM_DIR=str(root / mode),
            ASHARE_PLATFORM_EXECUTOR=mode,
            ASHARE_PLATFORM_VARIANTS=str(args.concurrency),
            ASHARE_PLATFORM_WORKERS="4",
            POLARS_MAX_THREADS="2",
            GIT_OPTIONAL_LOCKS="0",
        )
        log = (root / f"{mode}.log").open("w")
        server = subprocess.Popen(
            [sys.executable, "-m", "ashare.platform", "--port", str(args.port)],
            env=env,
            stdout=log,
            stderr=log,
        )
        stop = threading.Event()
        peak = [0]

        def sample(stop=stop, peak=peak, server=server):
            while not stop.wait(0.1):
                try:
                    peak[0] = max(peak[0], memory_tree(server.pid))
                except (PermissionError, subprocess.SubprocessError):
                    break

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            with httpx.Client(
                base_url=url, headers={"Origin": url}, timeout=180
            ) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    try:
                        if client.get("/api/meta").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("服务启动失败")
                tab = client.post("/api/tabs", json={}).json()["id"]
                for attempt in range(args.attempts):
                    peak[0] = 0
                    start = time.perf_counter()
                    response = client.post(
                        f"/api/tabs/{tab}/runs",
                        json=dict(
                            source="vol = ts_std(returns, [20, 60])\nalpha = -vol",
                            settings={"group_neutralize": ["industry", "cap+industry"]},
                            name="speed",
                        ),
                    )
                    response.raise_for_status()
                    job = response.json()["id"]
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        result = client.get(f"/api/jobs/{job}").json()
                        if result["status"] != "running":
                            break
                        time.sleep(0.025)
                    if result["status"] != "complete":
                        raise RuntimeError("运行未完成，请检查本地日志")
                    elapsed = time.perf_counter() - start
                    import sqlite3

                    with sqlite3.connect(root / mode / "platform.sqlite3") as db:
                        events = [
                            json.loads(r[0])
                            for r in db.execute(
                                "SELECT payload FROM events WHERE job=? ORDER BY seq",
                                (job,),
                            )
                        ]
                    timing = next(e for e in events if e["kind"] == "timing")
                    stages = [e["seconds"] for e in events if e["kind"] == "stages"]
                    run = result["runs"][0]["id"]
                    listed = client.get(f"/api/runs/{run}/holdings").json()
                    sizes = []
                    latency = []
                    for _ in range(3):
                        before = time.perf_counter()
                        day = client.get(
                            f"/api/runs/{run}/holdings", params={"date": listed["date"]}
                        )
                        day.raise_for_status()
                        latency.append((time.perf_counter() - before) * 1000)
                        sizes.append(len(day.content))
                    row = dict(
                        mode=mode,
                        attempt="cold" if attempt == 0 else "warm",
                        wall_seconds=elapsed,
                        evaluation_seconds=timing["evaluation_seconds"],
                        worker_seconds=timing["elapsed"],
                        stages=stages,
                        peak_tree_bytes=peak[0] or None,
                        coordinator_peak_bytes=timing["rss"],
                        other_process_peak_bytes={
                            str(e["pid"]): e["rss"]
                            for e in events
                            if e["kind"] == "memory"
                        },
                        holdings_bytes=(root / mode / "holdings" / f"{run}.npz")
                        .stat()
                        .st_size,
                        holdings_http_ms=latency,
                        holdings_json_bytes=sizes[0],
                    )
                    measurements.append(row)
                    print(json.dumps(row), flush=True)
                    (root / "metrics.json").write_text(
                        json.dumps(measurements, indent=2)
                    )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            stop.set()
            monitor.join(timeout=5)
            log.close()


if __name__ == "__main__":
    main()
