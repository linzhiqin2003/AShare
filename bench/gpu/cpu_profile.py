"""第 1、2 题：现行 CPU 粗筛一代 256 个候选的时间分解，以及整代合并求值、共享子树能快多少。

    nice -n 15 uv run --no-sync python bench/gpu/cpu_profile.py --root <数据根目录> [--threads 4]
    nice -n 15 uv run --no-sync python bench/gpu/cpu_profile.py --synthetic

1. 种群：真实的 `Evolution`（population=256，适应度 = 粗筛 t 值），跑到第 2 代，取第 2 代的
   256 个后代（全部是交叉、变异的产物）存进 results/population_<数据>.json（只有表达式）。
   生成过程本身就是现行做法（`evaluate_many` 默认内存上限 + 逐个 `fe.screen`），每代耗时一并记录。
2. 子树重复率：整代的节点总数、去重后的节点数、按算子分的重复。
3. 三种求值方式交替各跑 ``--repeat`` 次取中位数：逐个 `evaluate`（不共享）、现行 `evaluate_many`
   （默认内存上限，分批共享）、整代一批（内存上限放到能装下整代）；同时记录逐算子耗时和 Rank IC 耗时。
4. CPU float64 参考：每个候选的日 IC 与 t 值存进 cache/（不入库），给第 4 题对照。
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import panelops

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

from ashare.factor import Evaluator  # noqa: E402
from ashare.factor.engine import _compute, _peak, _release, _schedule  # noqa: E402
from ashare.factor.mining.space import size  # noqa: E402

CACHE = Path(__file__).with_name("cache")


def dup_stats(exprs):
    total, ops_total = 0, Counter()
    uniq, ops_uniq = {}, Counter()
    for e in exprs:
        stack = [e]
        while stack:
            n = stack.pop()
            stack.extend(n.args)
            total += 1
            ops_total[n.op] += 1
            if n.digest not in uniq:
                uniq[n.digest] = n
                ops_uniq[n.op] += 1
    inner_total = total - ops_total["field"]
    inner_uniq = len(uniq) - ops_uniq["field"]
    # 求值器本来就在单个表达式内部去重；跨表达式的额外节约按「逐个求值时的节点数」算
    per_expr = sum(len(_schedule([e])[0]) - sum(1 for n in _schedule([e])[0]
                                                 if n.op == "field") for e in exprs)  # fmt: skip
    return {
        "expressions": len(exprs),
        "nodes_total": total,
        "operator_nodes_total": inner_total,
        "operator_nodes_per_expr_eval": per_expr,
        "operator_nodes_unique": inner_uniq,
        "shared_fraction": 1 - inner_uniq / per_expr,
        "mean_size": statistics.mean(size(e) for e in exprs),
        "ops_total": dict(ops_total.most_common()),
        "ops_unique": dict(ops_uniq.most_common()),
    }


def screen_all(screen, it):
    t_eval = t_ic = 0.0
    ics = []
    while True:
        t0 = time.perf_counter()
        try:
            _, values = next(it)
        except StopIteration:
            t_eval += time.perf_counter() - t0
            break
        t1 = time.perf_counter()
        t_eval += t1 - t0
        ics.append(screen.ic(values))
        t_ic += time.perf_counter() - t1
    return t_eval, t_ic, ics


def mode_single(screen, exprs):
    ev = Evaluator()
    return screen_all(
        screen, ((i, ev.evaluate(e, screen.fe.panel)) for i, e in enumerate(exprs))
    )


def mode_current(screen, exprs):
    ev = Evaluator()
    out = screen_all(screen, ev.evaluate_many(dict(enumerate(exprs)), screen.fe.panel))
    return (*out, ev.last_batch.get("batches"))


def mode_whole(screen, exprs, limit):
    ev = Evaluator()
    out = screen_all(screen, ev.evaluate_many(dict(enumerate(exprs)), screen.fe.panel,
                                              memory_limit=limit))  # fmt: skip
    return (*out, ev.last_batch.get("batches"))


def mode_ops(screen, exprs):
    """整代一张图按节点逐个计时（与 evaluate_many 同样的 _compute 和释放规则）。"""
    panel = screen.fe.panel
    shape = screen.fe.shape
    order, uses = _schedule(exprs)
    for e in exprs:
        uses[e.digest] += 1
    want = Counter(e.digest for e in exprs)
    memo, op_t, done = {}, Counter(), set()
    t_ic = 0.0
    roots = {e.digest for e in exprs}
    for node in order:
        t0 = time.perf_counter()
        memo[node.digest] = _compute(
            node, memo, panel, shape, None, node.digest in roots
        )
        op_t[node.op] += time.perf_counter() - t0
        _release(node, uses, memo)
        if node.digest in want and node.digest not in done:
            done.add(node.digest)
            t1 = time.perf_counter()
            screen.ic(memo[node.digest])
            t_ic += time.perf_counter() - t1
            uses[node.digest] -= want[node.digest]
            if uses[node.digest] == 0:
                del memo[node.digest]
    return dict(op_t), t_ic


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_data_args(ap)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--whole-limit-gib", type=float, default=8.0)
    args = ap.parse_args(argv)
    panelops.set_num_threads(args.threads)
    out = {"machine": common.machine(), "threads": args.threads,
           "load_before": common.load_avg()}  # fmt: skip
    panel, source, t_read = common.load_panel(args)
    out["source"], out["read_fields_seconds"] = source, t_read
    tag = source["kind"]
    t0 = time.perf_counter()
    screen = common.Screen(panel, args.threads)
    out["prepare_seconds"] = time.perf_counter() - t0
    pop_path = common.RESULTS / f"population_{tag}.json"
    if pop_path.exists():
        exprs, meta = common.load_population(pop_path)
        out["population_from"] = pop_path.name
    else:
        calls = common.evolve_population(screen, args.population, 2, seed=0)
        exprs = calls[-1]["exprs"]
        meta = {"strategy": f"Evolution(population={args.population}, generations=2, seed=0)",
                "fitness": "screen t (h=5, whole sample)", "generation": len(calls) - 1,
                "source": source}  # fmt: skip
        common.save_population(pop_path, exprs, meta)
        out["evolution_generations_seconds"] = [c["seconds"] for c in calls]
    out["population"] = meta
    out["duplication"] = dup_stats(exprs)
    shape = screen.fe.shape
    panel_bytes = 8 * shape[0] * shape[1]
    peak_panels = _peak(exprs)
    out["whole_batch_simulated_peak_panels"] = peak_panels
    limit = int(min(args.whole_limit_gib * 2**30, (peak_panels + 2) * panel_bytes))
    out["whole_batch_memory_limit_bytes"] = limit
    print(out["duplication"], "peak panels", peak_panels, flush=True)

    runs = {"single": [], "current": [], "whole": [], "ops": []}
    ics = None
    for k in range(args.repeat):
        for name in ("current", "whole", "single", "ops"):
            load = common.load_avg()
            t0 = time.perf_counter()
            if name == "single":
                te, ti, ic = mode_single(screen, exprs)
                rec = {"eval": te, "ic": ti}
            elif name == "current":
                te, ti, ic, b = mode_current(screen, exprs)
                rec = {"eval": te, "ic": ti, "batches": b}
                ics = ic
            elif name == "whole":
                te, ti, ic, b = mode_whole(screen, exprs, limit)
                rec = {"eval": te, "ic": ti, "batches": b}
            else:
                op_t, ti = mode_ops(screen, exprs)
                rec = {"ops": op_t, "ic": ti, "eval": sum(op_t.values())}
            rec["total"] = time.perf_counter() - t0
            rec["load"] = load
            runs[name].append(rec)
            print(k, name, {kk: v for kk, v in rec.items() if kk != "ops"}, flush=True)
    out["runs"] = runs
    out["median"] = {
        name: {key: statistics.median(r[key] for r in rs)
               for key in ("eval", "ic", "total")}
        for name, rs in runs.items()
    }  # fmt: skip
    op_keys = sorted({k for r in runs["ops"] for k in r["ops"]})
    out["median_ops_seconds"] = {
        k: statistics.median(r["ops"].get(k, 0.0) for r in runs["ops"]) for k in op_keys
    }
    out["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    out["load_after"] = common.load_avg()

    # CPU float64 参考（第 4 题）：日 IC 与 t 值，只放 cache/，不入库
    CACHE.mkdir(exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")
    ic_arr = np.asarray(ics, dtype=np.float64)
    ts = np.array([screen.t_of(ic)[0] for ic in ic_arr], dtype=np.float64)
    np.save(CACHE / f"cpu_ic_{tag}.npy", ic_arr)
    np.save(CACHE / f"cpu_t_{tag}.npy", ts)
    out["cpu_t_finite"] = int(np.isfinite(ts).sum())
    out["cpu_t_pass_1.0"] = int((ts >= 1.0).sum())
    common.write_json(common.RESULTS / f"cpu_profile_{tag}.json", out)
    print({k: out[k] for k in ("median", "median_ops_seconds")})


if __name__ == "__main__":
    main()
