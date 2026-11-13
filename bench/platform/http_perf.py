"""生产数量级的合成接口测量：1000 条目、8 变体、2400 个交易日。"""

import argparse
import copy
import gzip
import json
import statistics
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ashare.platform.analysis import checks, run_variant
from ashare.platform.app import create_app
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from ashare.platform.storage import Store, dumps
from bench.platform.synthetic import panel, write_indexes, write_panel


def seed(root, count=1000):
    store = Store(root)
    if store.trials():
        raise RuntimeError("基准目录必须为空，避免覆盖已有结果")
    source = panel(n=60, full=True)
    sample, _ = run_variant(
        parse("-ts_std(returns,20)")[0].expression, source, DEFAULTS
    )
    days = source.dates[(source.dates >= "20160101") & (source.dates < "20260101")]
    dates = np.concatenate(
        (days[days < "20250101"][-2160:], days[days >= "20250101"][:240])
    )
    rng = np.random.default_rng(7654)
    runs, entries, job = [], [], None
    for i in range(count):
        variant = parse(f"rank(close) + {i} / 10000")[0]
        settings = DEFAULTS | {
            "universe": ("ALL", "TOP800", "TOP1800")[i % 3],
            "group_neutralize": (None, "industry", "cap", "cap+industry")[i % 4],
        }
        jid, items = store.begin(
            "benchmark",
            f"synthetic_{i:04d}",
            [(variant, settings)],
            {"git": "synthetic", "dirty": False},
        )
        run = items[0]["id"]
        entry = items[0]["entry"]
        if job is None:
            job = jid
        arrays = {}
        summary = copy.deepcopy(sample)
        for part, ds in (("is", dates[:2160]), ("validation", dates[2160:])):
            t = len(ds)
            arrays[f"{part}.dates"] = ds
            arrays[f"{part}.pnl.long_short"] = rng.normal(0.0002, 0.01, t)
            arrays[f"{part}.groups"] = rng.normal(0.0002, 0.02, (10, t))
            arrays[f"{part}.ic"] = rng.normal(0.01, 0.08, t)
            arrays[f"{part}.turnover.long_short"] = rng.uniform(0.01, 0.5, t)
            arrays[f"{part}.coverage"] = rng.uniform(0.95, 1, t)
            if part == "is":
                arrays[f"{part}.sub.TOP800"] = rng.normal(0.0002, 0.015, t)
            for name in ("size", "reversal_20", "reversal_60", "momentum"):
                arrays[f"{part}.exposure.{name}"] = rng.normal(0, 0.1, t)
        with (store.root / "series" / f"{run}.npz").open("wb") as file:
            np.savez_compressed(file, **arrays)
        a = summary["is"]
        a["sharpe"] = float(rng.normal(0.6, 1))
        a["fitness"] = float(rng.uniform(0, 1))
        a["turnover"] = float(rng.uniform(0.01, 0.7))
        summary.update(
            checks=checks(summary, count), correlations=[], trial_count=count
        )
        status = "pass" if i % 3 else "fail"
        with store.connect() as db:
            db.execute(
                "UPDATE runs SET job=?,ordinal=?,status='complete',summary=?,data_version='synthetic',elapsed=1 WHERE id=?",
                (job if i < 8 else jid, i % 8, dumps(summary), run),
            )
            db.execute(
                "UPDATE entries SET latest=?,status=?,sharpe=?,validation=?,fitness=?,turnover=?,annual_return=?,drawdown=?,coverage=?,t=?,margin=? WHERE id=?",
                (
                    run,
                    status,
                    a["sharpe"],
                    summary["validation"]["sharpe"],
                    a["fitness"],
                    a["turnover"],
                    a["return"],
                    a["maxdrawdown"],
                    a["coverage"],
                    a["t"],
                    a["margin"],
                    entry,
                ),
            )
            db.execute(
                "UPDATE jobs SET status='complete',total=?,elapsed=1 WHERE id=?",
                (8 if jid == job else 1, jid),
            )
            store._event(
                db,
                job if i < 8 else jid,
                {"kind": "variant", "ordinal": i % 8, "run": run},
            )
        runs.append(run)
        entries.append(entry)
    store.event(job, {"kind": "complete", "elapsed": 1})
    return store, job, runs, entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".cache/platform/http-perf")
    parser.add_argument("--reuse", action="store_true", help="复测已生成的合成基准库")
    args = parser.parse_args()
    root = Path(args.output)
    data = root / "data"
    if args.reuse:
        store = Store(root / "results")
        with store.connect() as db:
            if db.execute(
                "SELECT count(*) FROM runs WHERE data_version IS NOT NULL AND data_version != 'synthetic'"
            ).fetchone()[0]:
                raise RuntimeError("只允许复测合成基准库")
            job = db.execute(
                "SELECT id FROM jobs WHERE total=8 AND status='complete' ORDER BY created LIMIT 1"
            ).fetchone()[0]
            runs = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM runs WHERE job=? ORDER BY ordinal", (job,)
                )
            ]
            entries = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM entries WHERE latest IS NOT NULL ORDER BY id LIMIT 12"
                )
            ]
    else:
        write_panel(data)
        write_indexes(data)
        store, job, runs, entries = seed(root / "results")
    app = create_app(store.root, data_root=str(data))
    results = []
    with TestClient(
        app, base_url="http://127.0.0.1", headers={"Origin": "http://127.0.0.1"}
    ) as client:

        def measure(label, path, method="GET", body=None, times=5):
            samples, raw, compressed = [], 0, 0
            for _ in range(times):
                response = client.request(method, path, json=body)
                response.raise_for_status()
                raw = len(response.content)
                compressed = len(
                    gzip.compress(response.content, compresslevel=6, mtime=0)
                )
                if response.headers.get("content-encoding") != "gzip":
                    compressed = raw
                elif "content-length" in response.headers:
                    compressed = int(response.headers["content-length"])
                samples.append(
                    float(response.headers["server-timing"].split("dur=")[1])
                )
            result = dict(
                endpoint=label, raw=raw, gzip=compressed, ms=statistics.median(samples)
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            return response

        for endpoint in (
            "/api/meta",
            "/api/settings",
            "/api/completions",
            "/api/tabs",
            "/api/benchmarks",
        ):
            measure(endpoint, endpoint)
        directory = app.state.benchmarks
        actual = directory.items
        directory.items = [
            {"symbol": f"{i + 1:06d}.SH", "name": f"合成指数{i:04d}"} for i in range(22)
        ]
        measure("GET /api/benchmarks (22 names, all)", "/api/benchmarks")
        measure(
            "GET /api/benchmarks (22 names, search)", "/api/benchmarks?search=指数000"
        )
        directory.items = actual
        measure(
            "POST /api/validate (8)",
            "/api/validate",
            "POST",
            {"source": "-ts_std(returns,[10,20,30,40,50,60,70,80])"},
        )
        tab = measure("POST /api/tabs", "/api/tabs", "POST", times=1).json()["id"]
        measure(
            "POST /api/tabs/{id}/name",
            f"/api/tabs/{tab}/name",
            "POST",
            {"name": "renamed"},
            times=1,
        )
        measure("GET /api/jobs/{id} (8)", f"/api/jobs/{job}")
        measure("GET /api/jobs/{id}/benchmark (8 × 2400)", f"/api/jobs/{job}/benchmark")
        measure("GET /api/jobs/{id}/events (replay)", f"/api/jobs/{job}/events")
        for view in ("returns", "long", "groups", "ic", "subs"):
            measure(
                f"GET /api/runs/{{id}}/series?view={view}",
                f"/api/runs/{runs[0]}/series?view={view}",
            )
        for group in ("none", "expression", "config"):
            measure(f"GET /api/library?group={group}", f"/api/library?group={group}")
        measure(
            "GET /api/library (filtered)",
            "/api/library?universe=ALL&minimum=1&sort=sharpe",
        )
        measure("GET /api/library/{id}", f"/api/library/{entries[0]}")
        measure(
            "GET /api/library/correlation (12)",
            "/api/library/correlation?ids=" + ",".join(entries[:12]),
        )
        run = measure(
            "POST /api/tabs/{id}/runs (spawn)",
            f"/api/tabs/{tab}/runs",
            "POST",
            {"source": "-ts_std(returns,[10,20,30,40,50,60,70,80])"},
            times=1,
        ).json()["id"]
        measure(
            "POST /api/jobs/{id}/cancel", f"/api/jobs/{run}/cancel", "POST", times=1
        )
        measure("DELETE /api/tabs/{id}", f"/api/tabs/{tab}", "DELETE", times=1)
        # 首次读取与预热缓存分开测；统计八组同时拉取的实际总量。
        started = time.perf_counter()
        responses = [client.get(f"/api/runs/{run}/series") for run in runs[:8]]
        results.append(
            dict(
                endpoint="8 variants × 2400 days (returns)",
                raw=sum(len(r.content) for r in responses),
                gzip=sum(
                    len(gzip.compress(r.content, compresslevel=6, mtime=0))
                    for r in responses
                ),
                ms=(time.perf_counter() - started) * 1000,
            )
        )
    (root / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
