"""Own-engine regression suite, persisted samples, and generated report refresh."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import panelbt as bt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/compare"))
from common import inputs  # noqa: E402


def main():
    subprocess.run([sys.executable, str(ROOT / "bench/benchmark.py")], check=True)
    report = json.loads((ROOT / ".cache/panelbt/benchmark.json").read_text())
    m, target, c, g = inputs(4000, 5500)
    weights = target.weights.copy()
    weights[::2] *= -1
    target = bt.Targets(target.days, target.indptr, target.cols, weights)
    g = replace(
        g, margin=bt.MarginModel(allow_short=True, short_initial=1, borrow_rate=0.0001)
    )
    bt.run(m, target, c, g)
    samples = []
    for _ in range(7):
        start = perf_counter()
        bt.run(m, target, c, g)
        samples.append(perf_counter() - start)
    report["long_short"] = {"total_s": median(samples), "total_samples": samples}
    output = ROOT / ".cache/panelbt/panelbt-current.json"
    baseline = ROOT / "bench/results/panelbt-baseline.json"
    if not baseline.exists():
        raise FileNotFoundError("frozen panelbt baseline is missing")
    original = json.loads(baseline.read_text())
    regressions = {}
    for name in ("lots", "fractional", "batch", "long_short"):
        before = original[name]["total_s"]
        after = report[name]["total_s"]
        ratio = after / before - 1
        regressions[name] = ratio
        print(
            f"{'REGRESSION >10%' if ratio > 0.1 else 'OK'}: {name}: {after * 1000:.3f} ms, baseline {before * 1000:.3f} ms, {ratio:+.1%}",
            flush=True,
        )
    report["regressions"] = regressions
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
