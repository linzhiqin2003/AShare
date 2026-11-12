"""P-004 小基准：只测 ModelCache.save（含落盘）在真实日模型上的耗时与占用。

prepare：用真实面板最近 --days 日建模（不写缓存），把最后 --keep 天的
         (键, 模型, 状态) 存成一个 npz 夹具；
run：在全新进程里用 --tree 的 ModelCache 依次保存这些日模型并等落盘。
pair：旧、新交替。
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    sys.path.insert(0, str(ROOT / "bench/e2e"))
    import numpy as np
    from common import default_data
    from risk_adapter import load

    import ashare.risk.series as series

    source, config = load(default_data(), args.days)
    saved = []

    class Recorder:
        def prepare(self, *a, **k):
            pass

        def load(self, *a, **k):
            return None

        def checkpoint(self, key):
            return None

        def flush(self):
            pass

        def save(self, key, model, state, *, day, latest=True, checkpoint=None):
            if day >= args.days - args.keep:
                saved.append((key, model, state, day))

    for _ in series.model_series(source, config=config, cache=Recorder()):
        pass
    arrays = {}
    for i, (key, model, state, day) in enumerate(saved):
        arrays[f"{i}/key"] = np.asarray(key)
        arrays[f"{i}/day"] = np.asarray(day)
        arrays[f"{i}/B"] = model.B
        arrays[f"{i}/F"] = model.F
        arrays[f"{i}/D"] = model.D
        arrays[f"{i}/factors"] = np.asarray(model.factors)
        arrays[f"{i}/meta"] = np.asarray(
            json.dumps({"date": model.date, "metadata": model.metadata})
        )
        for name, value in state.items():
            arrays[f"{i}/state_{name}"] = value
    arrays["symbols"] = np.asarray(source.symbols)
    arrays["count"] = np.asarray(len(saved))
    np.savez(args.out, **arrays)


def run(args):
    if args.tree:
        sys.path.insert(0, str(Path(args.tree) / "src"))
    import numpy as np

    import ashare
    from ashare.risk import ModelCache, RiskModel

    if args.tree:
        assert Path(ashare.__file__).resolve().is_relative_to(Path(args.tree).resolve())
    data = np.load(args.fixture)
    symbols = tuple(data["symbols"].tolist())
    days = []
    for i in range(int(data["count"])):
        meta = json.loads(str(data[f"{i}/meta"]))
        model = RiskModel(
            symbols,
            data[f"{i}/B"],
            tuple(data[f"{i}/factors"].tolist()),
            data[f"{i}/F"],
            data[f"{i}/D"],
            meta["date"],
            meta["metadata"],
        )
        state = {
            name.split("state_", 1)[1]: data[name]
            for name in data.files
            if name.startswith(f"{i}/state_")
        }
        days.append((str(data[f"{i}/key"]), model, state, int(data[f"{i}/day"])))
    work = Path(tempfile.mkdtemp(prefix="riskcache-"))
    try:
        cache = ModelCache(work)
        if hasattr(cache, "flush"):
            cache.prepare("g" * 64, symbols)
        else:
            cache.prepare("g" * 64)
        start = time.perf_counter()
        for key, model, state, day in days:
            cache.save(key, model, state, day=day, latest=False)
        caller = time.perf_counter() - start
        if hasattr(cache, "flush"):
            cache.flush()
        total = time.perf_counter() - start
        size = sum(p.stat().st_size for p in work.iterdir() if p.is_file())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(
        json.dumps(
            {
                "days": len(days),
                "caller_ms_per_day": caller / len(days) * 1e3,
                "total_ms_per_day": total / len(days) * 1e3,
                "bytes_per_day": size / len(days),
            }
        )
    )


def pair(args):
    rows = {"old": [], "new": []}
    for _ in range(args.rounds):
        for side in ("old", "new"):
            command = [sys.executable, __file__, "run", "--fixture", args.fixture]
            if side == "old":
                command += ["--tree", args.old_tree]
            out = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=600
            ).stdout
            rows[side].append(json.loads(out))
    import os

    report = {"load": os.getloadavg()[0]}
    for side, values in rows.items():
        report[side] = {
            key: {
                "min": min(v[key] for v in values),
                "median": statistics.median(v[key] for v in values),
            }
            for key in ("caller_ms_per_day", "total_ms_per_day", "bytes_per_day")
        }
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--days", type=int, default=300)
    p.add_argument("--keep", type=int, default=100)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--tree")
    q = sub.add_parser("pair")
    q.add_argument("--fixture", required=True)
    q.add_argument("--old-tree", required=True)
    q.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "pair": pair}[args.action](args)


if __name__ == "__main__":
    main()
