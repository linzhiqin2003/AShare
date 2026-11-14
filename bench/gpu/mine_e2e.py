"""``mine`` 端到端：同一次进化搜索分别用 CPU 粗筛与 GPU 粗筛（含 CPU float64 复算与回测），每代耗时。

    ASHARE_DATA_DIR=<数据根目录> nice -n 15 uv run --no-sync python bench/gpu/mine_e2e.py
    uv run --no-sync python bench/gpu/mine_e2e.py --synthetic

真实面板取 2016-01-01 起的全部数据，切分用项目统一的 `Split.standard()`（训练 2016–2024、验证
2025、留出 2026 起）；合成面板按最近两年留出。策略 Evolution(population, generations, seed)，
适应度默认（多头端信息比率，过粗筛的要回测）。两条路各用一个临时台账。输出
results/mine_e2e_<数据>.json：每代耗时、粗筛与回测个数、GPU 粗筛与 CPU 复算的耗时，以及两条路
的选择是否一致（只有计数和布尔值，不含任何因子指标）。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


class Timed:
    """包一层策略：记录每次 evaluate（一代）的耗时。"""

    def __init__(self, inner):
        self.inner, self.seconds = inner, []

    def describe(self):
        return self.inner.describe()

    def run(self, space, evaluate):
        def timed(batch):
            t0 = time.perf_counter()
            out = evaluate(batch)
            self.seconds.append(time.perf_counter() - t0)
            return out

        return self.inner.run(space, timed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_data_args(ap)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screen-t", type=float, default=1.0)
    args = ap.parse_args(argv)
    if not args.synthetic:
        args.end = (
            args.end if args.end != "20251231" else "20261113"
        )  # 含 2026 年留出段
    import panelops

    from ashare.factor.mining import Evolution, Split, mine

    panelops.set_num_threads(args.threads)
    panel, source, _ = common.load_panel(args)
    split = Split.standard() if not args.synthetic else Split.recent(panel.dates)
    report = {"source": source, "machine": common.machine(), "split": split.describe(),
              "population": args.population, "generations": args.generations,
              "threads": args.threads, "load_before": common.load_avg(), "runs": {}}  # fmt: skip
    results = {}
    for mode in ("cpu", "gpu"):
        strategy = Timed(Evolution(population=args.population, generations=args.generations,
                                   seed=args.seed))  # fmt: skip
        with tempfile.TemporaryDirectory() as root:
            t0 = time.perf_counter()
            r = mine(panel, split=split, strategy=strategy, root=root, screen=mode,
                     screen_t=args.screen_t, threads=args.threads)  # fmt: skip
            total = time.perf_counter() - t0
        s = r.summary
        row = {"total_seconds": total, "generation_seconds": strategy.seconds,
               "proposed": s["proposed"], "unique": s["unique"],
               "screened_out": s["screened_out"], "backtested": s["backtested"],
               "failed": s["failed"]}  # fmt: skip
        if "gpu" in s:
            row["gpu"] = {
                k: v for k, v in s["gpu"].items() if k not in ("device", "kernel")
            }
            row["device"] = s["gpu"]["device"]
        report["runs"][mode] = row
        results[mode] = r
        print(
            mode,
            f"{total:.1f}s",
            [f"{x:.1f}" for x in strategy.seconds],
            {k: row[k] for k in ("screened_out", "backtested")},
            row.get("gpu", ""),
        )
    c, g = results["cpu"], results["gpu"]
    report["same_selection"] = [x["key"] for x in c.selected] == [
        x["key"] for x in g.selected
    ]
    report["same_significant"] = c.summary["significant"] == g.summary["significant"]
    report["same_backtested"] = sorted(
        x["key"] for x in c.candidates if x["stage"] == "backtest"
    ) == sorted(x["key"] for x in g.candidates if x["stage"] == "backtest")
    report["load_after"] = common.load_avg()
    print(
        {
            k: report[k]
            for k in ("same_selection", "same_significant", "same_backtested")
        }
    )
    common.write_json(common.RESULTS / f"mine_e2e_{source['kind']}.json", report)


if __name__ == "__main__":
    main()
