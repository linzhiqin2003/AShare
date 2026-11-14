"""第 4 题补充：粗筛门槛下放 δ 之后，float64 合格的有没有漏掉、要多复核几个。

    uv run --no-sync python bench/gpu/margin.py --tag real

读 cache/ 里 cpu_profile.py 与 gpu_probe.py 留下的日 IC，只输出计数（不含任何 IC 或 t 值）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

CACHE = Path(__file__).with_name("cache")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    args = ap.parse_args(argv)
    t_cpu = np.load(CACHE / f"cpu_t_{args.tag}.npy")
    out = {"tag": args.tag, "n": int(len(t_cpu)), "backends": {}}
    for path in sorted(CACHE.glob(f"gpu_ic_*_{args.tag}.npy")):
        ic = np.load(path)
        t_gpu = np.array(
            [common.t_stat(x, ic.shape[1], common.SCREEN_H + 1)[0] for x in ic]
        )
        tc = np.where(np.isfinite(t_cpu), t_cpu, -np.inf)
        tg = np.where(np.isfinite(t_gpu), t_gpu, -np.inf)
        rows = {}
        for thr in (0.5, 1.0, 2.0):
            for delta in (0.0, 0.1, 0.25, 0.5):
                gate = tg >= thr - delta
                keep = tc >= thr
                rows[f"thr{thr}_delta{delta}"] = {
                    "missed": int((keep & ~gate).sum()),
                    "rechecked": int(gate.sum()),
                    "cpu_pass": int(keep.sum()),
                    "extra_rechecks": int((gate & ~keep).sum()),
                }
        out["backends"][path.stem.removeprefix("gpu_ic_")] = rows
        print(path.stem, rows["thr1.0_delta0.25"])
    common.write_json(common.RESULTS / f"margin_{args.tag}.json", out)


if __name__ == "__main__":
    main()
