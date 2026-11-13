"""经 HTTP API 保存、复核平台落盘数组和原始摘要 JSON；产物只允许在 .cache。"""

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ashare.platform.app import create_app
from bench.groupbt.common import compare

SCENARIOS = {
    "four": ([20, 60], {"group_neutralize": ["industry", "cap+industry"]}),
    "eight": ([5, 20, 60, 120], {"group_neutralize": ["industry", "cap+industry"]}),
    "ledger": ([20], {"rebalance": [1, 3], "weights_portfolio": [False, True]}),
}


def capture(root, scenario, *, attempts=2, data_root=None):
    """使用独立空库，避免历史试验计数、相关性和异步持仓写入影响比较。"""
    if root.exists():
        raise ValueError("运行目录已存在，请选择新的目录")
    windows, settings = SCENARIOS[scenario]
    app = create_app(root, data_root=data_root)
    rows = []
    with TestClient(
        app, base_url="http://localhost", headers={"Origin": "http://localhost"}
    ) as client:
        tab = client.post("/api/tabs", json={}).json()["id"]
        for _ in range(attempts):
            start = time.perf_counter()
            response = client.post(
                f"/api/tabs/{tab}/runs",
                json={
                    "source": f"vol = ts_std(returns, {windows})\nalpha = -vol",
                    "settings": settings,
                },
            )
            response.raise_for_status()
            job = response.json()["id"]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                result = client.get(f"/api/jobs/{job}").json()
                if result["status"] != "running":
                    break
                time.sleep(0.01)
            assert result["status"] == "complete", "平台任务失败，详见本地日志"
            elapsed = time.perf_counter() - start
            app.state.store.wait_holdings(timeout=120)
            with sqlite3.connect(root / "platform.sqlite3") as db:
                events = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM events WHERE job=? ORDER BY seq", (job,)
                    )
                ]
                records = db.execute(
                    "SELECT id,ordinal,summary FROM runs WHERE job=? ORDER BY ordinal",
                    (job,),
                ).fetchall()
            destination = root / "snapshot"
            destination.mkdir(exist_ok=True)
            for run, ordinal, summary in records:
                (destination / f"{ordinal}.json").write_text(summary)
                for kind in ("series", "holdings"):
                    shutil.copyfile(
                        root / kind / f"{run}.npz",
                        destination / f"{ordinal}.{kind}.npz",
                    )
            timing = next(e for e in events if e["kind"] == "timing")
            rows.append(
                {
                    "wall_seconds": elapsed,
                    "timing": timing,
                    "stages": [e["seconds"] for e in events if e["kind"] == "stages"],
                }
            )
    (root / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def compare_snapshots(expected, actual):
    """不比较压缩容器时间戳；逐数组检查类型、形状和全部字节，包括 NaN/-0。"""
    files = sorted(p.name for p in expected.iterdir())
    assert files == sorted(p.name for p in actual.iterdir()), "落盘文件集合不同"
    count = 0
    for name in files:
        if name.endswith(".json"):
            assert (expected / name).read_bytes() == (actual / name).read_bytes(), (
                "摘要 JSON 不同"
            )
        else:
            with (
                np.load(expected / name, allow_pickle=False) as a,
                np.load(actual / name, allow_pickle=False) as b,
            ):
                compare(a, b)
                count += len(a)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "compare"))
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/groupbt-fusion/platform")
    )
    parser.add_argument("--label", default="after")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("-o", type=Path)
    args = parser.parse_args()
    if not args.cache.resolve().is_relative_to(Path(".cache").resolve()):
        parser.error("测量结果只能写入本 worktree 的 .cache")
    report = []
    for scenario in args.scenarios:
        root = (
            args.cache
            / ("baseline" if args.action == "baseline" else args.label)
            / scenario
        )
        rows = capture(root, scenario, attempts=args.attempts)
        count = None
        if args.action == "compare":
            count = compare_snapshots(
                args.cache / "baseline" / scenario / "snapshot", root / "snapshot"
            )
        report.append(
            f"{scenario}：逐位一致数组 {count}；各次耗时 {[round(r['wall_seconds'], 4) for r in rows]} 秒。"
        )
        print(report[-1], flush=True)
    if args.o:
        args.o.write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
