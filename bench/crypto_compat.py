"""Old-build versus new-build bitwise properties (no new inputs).

Freeze the original package including its native extension in
.cache/crypto-baseline/panelbt BEFORE editing sources. This runner refuses
another checkout and compares run/batch/step/snapshot, including old blobs.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import astuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(result):
    import numpy as np
    from panelbt.models import COUNTERS, EXTRA_COUNTERS, METRICS

    h = hashlib.sha256()
    for name in (*METRICS[:30], "positions", "contribution"):
        value = getattr(result, name)
        h.update(name.encode())
        if value is None:
            h.update(b"None")
        else:
            a = np.asarray(value)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        value = getattr(result, name)
        h.update(name.encode())
        if value is None:
            h.update(b"None")
        elif value and isinstance(value[0], list):
            for row in value:
                h.update(np.asarray([astuple(x) for x in row], dtype=float).tobytes())
        else:
            h.update(np.asarray([astuple(x) for x in value], dtype=float).tobytes())
    h.update(repr(result.bankruptcy_events).encode())
    for source, names in [
        (result.diagnostics, COUNTERS),
        (result.coverage, EXTRA_COUNTERS[:30]),
    ]:
        for key in names:
            h.update(key.encode())
            h.update(np.asarray(source[key], dtype=np.int64).tobytes())
    return h.hexdigest()


def worker(old, seeds):
    import panelbt as bt

    from test.generators import case as cash
    from test.generators_futures import rich_case as futures
    from test.test_extended import extended_case as margin

    assert Path(bt.__file__).resolve().is_relative_to(ROOT)
    assert Path(bt._core.__file__).resolve().is_relative_to(ROOT)
    paths = ROOT / ".cache/crypto-compat"
    paths.mkdir(parents=True, exist_ok=True)
    checks, touched = {}, dict.fromkeys(("cash", "margin", "held_futures"), 0)
    for category, generate in [
        ("cash", cash),
        ("margin", margin),
        ("held_futures", futures),
    ]:
        for seed in range(seeds):
            m, plan, costs, cfg = generate(seed)
            result = bt.run(m, plan, costs, cfg)
            key = f"{category}/{seed}"
            checks[key + "/run"] = digest(result)
            archive = paths / f"{category}-{seed}.npz"
            columns = paths / f"{category}-{seed}-columns.json"
            if old:
                result.to_npz(archive)
                columns.write_text(json.dumps(list(result.to_frame().columns)))
            else:
                migrated = bt.Result.from_npz(archive)
                assert digest(migrated) == digest(result), key + "/npz"
                for field in ("funding", "clearance_fee", "bankruptcy_deficit"):
                    assert getattr(migrated, field) is None, key + "/" + field
                assert list(migrated.to_frame().columns) == json.loads(
                    columns.read_text()
                )
            checks[key + "/batch"] = digest(
                bt.run_batch(m, [plan, plan], costs, cfg, threads=2)
            )
            dense = plan.to_dense(*m.shape)
            sim = bt.Simulator(m, costs, cfg, kind=plan.kind)
            split = max(1, m.shape[0] // 2)
            for row in dense[:split]:
                sim.step(row)
            blob = sim.snapshot()
            path = paths / f"{category}-{seed}.bin"
            if old:
                path.write_bytes(blob)
            sim = bt.Simulator.restore(blob, m, costs, cfg, kind=plan.kind)
            for row in dense[split:]:
                sim.step(row)
            checks[key + "/snapshot"] = digest(sim.result())
            assert checks[key + "/snapshot"] == checks[key + "/run"]
            step = bt.Simulator(m, costs, cfg, kind=plan.kind)
            for row in dense:
                step.step(row)
            checks[key + "/step"] = digest(step.result())
            assert checks[key + "/step"] == checks[key + "/run"]
            if not old:
                restored = bt.Simulator.restore(
                    path.read_bytes(), m, costs, cfg, kind=plan.kind
                )
                for row in dense[split:]:
                    restored.step(row)
                assert digest(restored.result()) == checks[key + "/run"], key
            touched[category] += int(result.diagnostics["trades"] > 0)
            if category == "held_futures":
                touched[category] += result.coverage["futures_open"]
    return dict(
        checks=checks,
        touched=touched,
        package=str(Path(bt.__file__).relative_to(ROOT)),
        extension=str(Path(bt._core.__file__).relative_to(ROOT)),
        source=bt._core.SOURCE_SHA256,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["old", "new"])
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker == "old", args.seeds)))
        return
    base = ROOT / ".cache/crypto-baseline"
    assert (base / "panelbt/_core.abi3.so").exists(), "freeze the original build first"
    outputs = []
    for which in ("old", "new"):
        env = dict(
            os.environ,
            PYTHONPATH=str(base)
            if which == "old"
            else str(ROOT / "packages/panelbt/python"),
        )
        cp = subprocess.run(
            [sys.executable, __file__, "--worker", which, "--seeds", str(args.seeds)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if cp.returncode:
            raise RuntimeError(cp.stderr)
        outputs.append(json.loads(cp.stdout))
    assert outputs[0]["checks"] == outputs[1]["checks"], [
        k
        for k in outputs[0]["checks"]
        if outputs[0]["checks"][k] != outputs[1]["checks"][k]
    ]
    report = dict(
        seeds_per_account=args.seeds,
        cases=3 * args.seeds,
        compared_api_outputs=len(outputs[0]["checks"]),
        old_snapshots_restored=3 * args.seeds,
        old_npz_migrations_and_frame_schemas=3 * args.seeds,
        bitwise_identical=True,
        builds=[{k: v for k, v in x.items() if k != "checks"} for x in outputs],
    )
    (ROOT / "bench/results/crypto_compat.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
