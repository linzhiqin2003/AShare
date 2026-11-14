"""正式模块 `ashare.factor.mining.gpu` 的 GPU 粗筛测量：分阶段耗时、带宽、代规模。

    ASHARE_DATA_DIR=<数据根目录> nice -n 15 uv run --no-sync python bench/gpu/screen_profile.py
    uv run --no-sync python bench/gpu/screen_profile.py --synthetic        # 没有真实数据的机器

输出 results/screen_profile_<设备>_<数据>.json（只有耗时、计数、带宽，不含任何 IC 或 t 值）：

- ``peak_bandwidth``：本机实测的设备内存带宽（大数组 a + 1 的读写字节 / 时间，取最快一次），
  作为「峰值」；
- ``sizes``：一代 256 / 1024 / 4096 个表达式（真实种群 + 默认空间的随机表达式补足），整代
  ``daily_ic`` 的端到端耗时（异步、不打断设备排队；含取回日 IC）与每个表达式的平均耗时；
- ``stages``：profile 模式（每个节点、每个根同步计时）下的分阶段耗时，以及按「每个算子读一遍
  输入、写一遍输出」算的最少字节数与对应带宽；
- ``lower_bound_seconds``：最少字节数 / 峰值带宽，即这一代按带宽算的下限。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def peak_bandwidth(mx, gib=1.0, repeat=7):
    n = int(gib * 2**30 / 4)
    x = mx.zeros((n,), dtype=mx.float32)
    mx.eval(x)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        y = x + 1.0
        mx.eval(y)
        best = min(best, time.perf_counter() - t0)
        del y
    return 2 * 4 * n / best


def population(k, base, seed=0):
    from ashare.factor.mining import Space
    from ashare.factor.mining.strategies import random_candidate

    out, seen = [], set()
    for e in base:
        if e.digest not in seen and len(out) < k:
            seen.add(e.digest)
            out.append(e)
    rng = np.random.default_rng(seed)
    space = Space.default()
    while len(out) < k:
        e = random_candidate(space, rng)
        if e.digest not in seen:
            seen.add(e.digest)
            out.append(e)
    return out


def fused_traffic(sc, exprs, panel_bytes):
    """按融合计划算：每个落显存的节点读一遍它引用的面板、写一遍输出。"""
    from ashare.factor.engine import _schedule

    order, uses = _schedule(exprs)
    want = Counter(e.digest for e in exprs)
    for e in exprs:
        uses[e.digest] += 1
    inline, refs = sc._plan(order, uses, want)
    return sum((len(v) + 1) * panel_bytes for v in refs.values()), len(refs)


def traffic(exprs, panel_bytes):
    """整代去重后每个节点「读一遍输入、写一遍输出」的字节数（字段已在设备上，不算上传），
    以及每个根的 Rank IC（读因子、股票池 1/4、收益的秩一张，写日 IC 可忽略）。"""
    from ashare.factor.engine import _schedule

    order, _ = _schedule(exprs)
    ops, nodes = Counter(), Counter()
    total = 0
    for node in order:
        if node.op in ("field", "constant"):
            continue
        panels = sum(1 for a in node.args if a.op != "constant") + 1
        ops[node.op] += panels * panel_bytes
        nodes[node.op] += 1
        total += panels * panel_bytes
    rank = len({e.digest for e in exprs}) * 2.25 * panel_bytes
    return {"eval_bytes": total, "rank_ic_bytes": rank, "by_op": dict(ops),
            "nodes": dict(nodes), "distinct_nodes": sum(nodes.values())}  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_data_args(ap)
    ap.add_argument("--sizes", default="256,1024,4096")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    import mlx.core as mx

    from ashare.factor.mining.gpu import Screener

    panel, source, read_s = common.load_panel(args)
    tag = source["kind"]
    screen = common.Screen(panel, threads=args.threads)
    base, _ = common.load_population(common.RESULTS / f"population_{tag}.json")
    t, n = screen.fe.shape
    pb = 4 * t * n
    report = {"source": source, "machine": common.machine(), "load_before": common.load_avg(),
              "panel_bytes_f32": pb, "peak_bandwidth": peak_bandwidth(mx), "sizes": {}}  # fmt: skip
    print("peak bandwidth GB/s", report["peak_bandwidth"] / 1e9)

    def make(profile=False):
        return Screener(panel, screen.universe, screen.returns, profile=profile)

    sizes = [int(x) for x in args.sizes.split(",")]
    for k in sizes:
        pop = population(k, base)
        sc = make()
        t0 = time.perf_counter()
        sc.daily_ic(pop)  # 首次：编译内核、上传字段
        first = time.perf_counter() - t0
        runs = []
        for _ in range(args.repeat):
            mx.clear_cache()
            t0 = time.perf_counter()
            outs = sc.daily_ic(pop)
            runs.append(time.perf_counter() - t0)
        med = statistics.median(runs)
        model = traffic(pop, pb)
        model["fused_eval_bytes"], model["materialized_nodes"] = fused_traffic(
            sc, pop, pb
        )
        row = {"exprs": k, "first_seconds": first, "seconds": runs, "median": med,
               "per_expr_ms": 1e3 * med / k, "batches": sc.counts["batches"] // (args.repeat + 1),
               "routed": dict(Counter(o.reason for o in outs if o.ic is None)),
               "peak_device_memory": sc.peak_memory, "model": model,
               "lower_bound_seconds": (model["fused_eval_bytes"] + model["rank_ic_bytes"])
               / report["peak_bandwidth"]}  # fmt: skip
        print(
            k,
            f"{med:.2f}s",
            f"{row['per_expr_ms']:.1f} ms/expr",
            f"lower bound {row['lower_bound_seconds']:.2f}s",
            row["routed"],
        )
        report["sizes"][str(k)] = row
    # 分阶段（同步计时）
    pop = population(sizes[0], base)
    sc = make(profile=True)
    sc.daily_ic(pop)
    sc.seconds.clear()
    t0 = time.perf_counter()
    sc.daily_ic(pop)
    wall = time.perf_counter() - t0
    secs = dict(sc.seconds)
    model = traffic(pop, pb)
    model["fused_eval_bytes"], model["materialized_nodes"] = fused_traffic(sc, pop, pb)
    ops = {k[3:]: v for k, v in secs.items() if k.startswith("op:")}
    eval_s = sum(ops.values())
    report["stages"] = {
        "exprs": len(pop), "wall": wall, "seconds": secs, "eval_seconds": eval_s,
        "eval_bandwidth": model["fused_eval_bytes"] / eval_s if eval_s else None,
        "rank_ic_bandwidth": model["rank_ic_bytes"] / secs["rank_ic"],
        "op_bandwidth": {op: model["by_op"][op] / v for op, v in ops.items()
                         if v > 0 and op in model["by_op"]},
        "model": model,
    }  # fmt: skip
    report["load_after"] = common.load_avg()
    for k, v in sorted(secs.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:24s} {v:7.3f}s")
    st = report["stages"]
    print(
        "eval GB/s",
        st["eval_bandwidth"] / 1e9,
        "rank_ic GB/s",
        st["rank_ic_bandwidth"] / 1e9,
    )
    dev = sc.device.split(":")[0]
    common.write_json(common.RESULTS / f"screen_profile_{dev}_{tag}.json", report)


if __name__ == "__main__":
    main()
