"""Reproduce constant-window performance cliffs separately from random benchmarks."""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)


def main():
    rng = np.random.default_rng(0)
    panels = {
        "random": rng.normal(size=(1000, 1000)),
        "constant": np.full((1000, 1000), 12.34),
        "piecewise": np.repeat(rng.normal(size=(4, 1000)), 250, axis=0),
    }
    po.set_num_threads(1)
    rows = []
    for op, window in [
        ("std", 20),
        ("std", 250),
        ("corr", 20),
        ("corr", 250),
        ("skew", 250),
        ("slope", 250),
        ("mean", 250),
    ]:
        row = {"op": op, "window": window}
        for name, x in panels.items():
            fn = getattr(po, "ts_" + op)
            args = (x, panels["random"], window) if op == "corr" else (x, window)
            fn(*args)
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                fn(*args)
                elapsed.append((time.perf_counter() - start) * 1000)
            row[name] = statistics.median(elapsed)
        rows.append(row)
        print(row, flush=True)
    output = ROOT / ".cache/panelops/cliff.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"build_id": po.build_id, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
