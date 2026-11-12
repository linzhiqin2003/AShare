"""性能轮（P-004…P-008）风险阶段的配对测量与逐日一致性核对。

每次调用只跑一个场景，在全新进程里执行；`--tree` 指向另一份源码目录时用它的
`ashare`（用来跑改动前的提交，导出方式见 docs/perf_log.d）。结果写 JSON：
耗时、每日模型摘要（B、F、D、元数据各自的 SHA-256 前缀）。

    python bench/risk/perf_round.py run --days 800 --scenario rebuild --out a.json
    python bench/risk/perf_round.py pair --days 800 --scenario rebuild \
        --old-tree <旧源码目录> --rounds 5
    python bench/risk/perf_round.py diff a.json b.json
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _digest(array):
    import numpy as np

    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode() + a.tobytes()).hexdigest()[:20]


def summarize(model, *, ignore=()):
    meta = {k: v for k, v in model.metadata.items() if k not in ignore}
    return {
        "date": model.date,
        "B": _digest(model.B),
        "F": _digest(model.F),
        "D": _digest(model.D),
        "meta": hashlib.sha256(
            json.dumps(meta, sort_keys=True, default=str).encode()
        ).hexdigest()[:20],
        "parameter_hash": model.metadata.get("parameter_hash"),
        # 不含 parameter_hash 的元数据：配置新增字段只改它，其余必须逐位相同。
        "meta_core": hashlib.sha256(
            json.dumps(
                {k: v for k, v in meta.items() if k != "parameter_hash"},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:20],
    }


def run(args):
    sys.path.insert(0, str(ROOT / "bench/e2e"))
    if args.tree:
        sys.path.insert(0, str(Path(args.tree) / "src"))
    from common import default_data
    from risk_adapter import load

    import ashare
    from ashare.risk import ModelCache, model_series

    if args.tree:
        assert Path(ashare.__file__).resolve().is_relative_to(Path(args.tree).resolve())
    else:
        assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
    source, config = load(default_data(), args.days)
    if args.config:
        from dataclasses import replace

        config = replace(config, **json.loads(args.config))
    if args.threads is not None:
        os.environ["ASHARE_RISK_THREADS"] = str(args.threads)
    t = len(source.dates)
    (ROOT / ".cache").mkdir(exist_ok=True)
    work = Path(args.work or tempfile.mkdtemp(prefix="riskperf-", dir=ROOT / ".cache"))
    cache_dir = work / "cache"
    days, timings = [], {}
    keep_all = args.digests

    def collect(models):
        out = []
        for model in models:
            out.append(summarize(model) if keep_all else None)
            last = model
        return out, summarize(last)

    if args.scenario in ("rebuild", "nocache"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache = None if args.scenario == "nocache" else ModelCache(cache_dir)
        start = time.perf_counter()
        days, last = collect(model_series(source, config=config, cache=cache))
        if cache is not None and hasattr(cache, "flush"):
            cache.flush()
        timings["seconds"] = time.perf_counter() - start
        if cache is not None:
            timings["cache_bytes"] = sum(
                p.stat().st_size for p in cache_dir.iterdir() if p.is_file()
            )
    elif args.scenario in ("append", "read"):
        # 前缀缓存由同一棵树先建好（不计时），再计时追加最后一天 / 读最后一天。
        if not (cache_dir / "ready").exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache = ModelCache(cache_dir)
            for _ in model_series(source, config=config, cache=cache, end=t - 1):
                pass
            if hasattr(cache, "flush"):
                cache.flush()
            snapshot = work / "prefix"
            shutil.rmtree(snapshot, ignore_errors=True)
            shutil.copytree(cache_dir, snapshot)
            (cache_dir / "ready").write_text("")
        if args.scenario == "append":
            shutil.rmtree(cache_dir)
            shutil.copytree(work / "prefix", cache_dir)
            (cache_dir / "ready").write_text("")
        else:
            cache = ModelCache(cache_dir)
            for _ in model_series(source, config=config, cache=cache, start=t - 1):
                pass
            if hasattr(cache, "flush"):
                cache.flush()
        start = time.perf_counter()
        cache = ModelCache(cache_dir)
        days, last = collect(
            model_series(source, config=config, cache=cache, start=t - 1)
        )
        if hasattr(cache, "flush"):
            cache.flush()
        timings["seconds"] = time.perf_counter() - start
    elif args.scenario == "tolerance":
        # P-006：同一进程里逐日并排跑「分块累加」和「从头算」，记录三个方差相对
        # 从头算的最大误差、改用从头算的列数，以及最终 D 的最大相对差。
        from dataclasses import replace

        import numpy as np

        import ashare.risk.specific as specific

        worst = {"base": 0.0, "corr0": 0.0, "corrnw": 0.0}
        fallback = [0, 0]
        original = specific.SpecificMoments.variances

        def checked(self, day):
            got = original(self, day)
            fallback[:] = [self.fallback_columns, fallback[1] + self.n]
            window = self.window(max(0, day - self.capacity + 1), day)
            for name, a, b in zip(
                worst,
                got[:3],
                specific.direct_variances(window, self.config),
                strict=True,
            ):
                scale = np.where(b != 0, np.abs(b), 1.0)
                error = np.abs(a - b) / scale
                worst[name] = max(worst[name], float(error.max()))
            return got

        specific.SpecificMoments.variances = checked
        fast = model_series(source, config=replace(config, specific_incremental=True))
        slow = model_series(source, config=replace(config, specific_incremental=False))
        start = time.perf_counter()
        d_error, b_same, f_same, total = 0.0, True, True, 0
        for a, b in zip(fast, slow, strict=True):
            d_error = max(d_error, float((np.abs(a.D - b.D) / b.D).max()))
            b_same &= bool(np.array_equal(a.B, b.B))
            f_same &= bool(np.array_equal(a.F, b.F))
            total += 1
            last = summarize(a)
        timings["seconds"] = time.perf_counter() - start
        timings["tolerance"] = {
            "variance_max_relative_error": worst,
            "D_max_relative_error": d_error,
            "B_identical": b_same,
            "F_identical": f_same,
            "days": total,
            "fallback_columns": fallback[0],
            "column_days": fallback[1],
        }
    else:
        raise SystemExit(f"unknown scenario {args.scenario}")
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    result = {
        "scenario": args.scenario,
        "tree": args.tree or "worktree",
        "days": t,
        "symbols": len(source.symbols),
        "load": os.getloadavg()[0],
        **timings,
        "last": last,
        "daily": days if keep_all else None,
    }
    Path(args.out).write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "daily"}))


def pair(args):
    """旧、新交替各跑 rounds 次，每次一个全新进程。"""
    rows = {"old": [], "new": []}
    (ROOT / ".cache").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="riskpair-", dir=ROOT / ".cache"))
    base = [sys.executable, __file__, "run", "--scenario", args.scenario]
    if args.days is not None:
        base += ["--days", str(args.days)]
    try:
        for i in range(args.rounds):
            for side in ("old", "new"):
                out = work / f"{side}-{i}.json"
                command = [*base, "--out", str(out)]
                if args.config and side == "new":
                    command += ["--config", args.config]
                if args.old_config and side == "old":
                    command += ["--config", args.old_config]
                if args.scenario in ("append", "read"):
                    command += ["--work", str(work / side)]
                if side == "old":
                    command += ["--tree", args.old_tree]
                subprocess.run(command, check=True, timeout=args.timeout)
                rows[side].append(json.loads(out.read_text()))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report = {}
    for side, values in rows.items():
        seconds = [r["seconds"] for r in values]
        report[side] = {
            "min": min(seconds),
            "median": statistics.median(seconds),
            "all": seconds,
            "load": [round(r["load"], 1) for r in values],
            "cache_bytes": values[-1].get("cache_bytes"),
            "last": values[-1]["last"],
        }
    report["ratio_min"] = report["new"]["min"] / report["old"]["min"]
    report["ratio_median"] = report["new"]["median"] / report["old"]["median"]
    same = {
        k: report["old"]["last"][k] == report["new"]["last"][k]
        for k in ("B", "F", "D", "meta")
    }
    report["last_identical"] = same
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


def diff(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    da, db = a["daily"], b["daily"]
    assert len(da) == len(db), (len(da), len(db))
    counts = {k: 0 for k in ("B", "F", "D", "meta", "meta_core", "parameter_hash")}
    for x, y in zip(da, db, strict=True):
        assert x["date"] == y["date"]
        for k in counts:
            if k in x and k in y:
                counts[k] += x[k] != y[k]
            else:
                counts[k] = None
    print(json.dumps({"days": len(da), "differing_days": counts}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--days", type=int)
    r.add_argument("--scenario", default="rebuild")
    r.add_argument("--tree")
    r.add_argument("--config")
    r.add_argument("--threads", type=int)
    r.add_argument("--work")
    r.add_argument("--digests", action="store_true")
    r.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--days", type=int)
    p.add_argument("--scenario", default="rebuild")
    p.add_argument("--old-tree", required=True)
    p.add_argument("--config")
    p.add_argument("--old-config")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = parser.parse_args()
    {"run": run, "pair": pair, "diff": diff}[args.action](args)


if __name__ == "__main__":
    main()
