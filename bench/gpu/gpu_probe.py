"""第 3、4 题：一个后端（MLX 或 PyTorch）在一个设备上对整代 256 个候选做 float32 粗筛。

    nice -n 15 uv run --no-sync python bench/gpu/gpu_probe.py --backend mlx   --root <数据根目录>
    nice -n 15 uv run --no-sync python bench/gpu/gpu_probe.py --backend torch --device mps --root <数据根目录>
    # NVIDIA 机器（合成数据，同规模）：
    uv run --no-sync python bench/gpu/gpu_probe.py --backend torch --device cuda --synthetic
    uv run --no-sync python bench/gpu/gpu_probe.py --backend mlx --synthetic

种群取 results/population_<数据>.json（由 cpu_profile.py 生成；没有时用同样的 Evolution 现场生成）。
CPU float64 参考取 cache/cpu_t_<数据>.npy、cache/cpu_ic_<数据>.npy（没有时现场算）。

记录：导入后端、把字段/股票池/标签放上设备的一次性成本，首代（含内核编译）与之后每代的端到端耗时
（含取回 T×256 个日 IC 和在 CPU 上算 t 值），逐算子耗时，设备内存峰值；以及 float32 对选择的影响。
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

CACHE = Path(__file__).with_name("cache")
THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 3.0)


def cpu_reference(screen, exprs, tag):
    p_t, p_ic = CACHE / f"cpu_t_{tag}.npy", CACHE / f"cpu_ic_{tag}.npy"
    if p_t.exists() and p_ic.exists():
        return np.load(p_t), np.load(p_ic)
    from ashare.factor import Evaluator

    ics = [screen.ic(v) for _, v in Evaluator().evaluate_many(dict(enumerate(exprs)),
                                                               screen.fe.panel)]  # fmt: skip
    ic = np.asarray(ics, dtype=np.float64)
    t = np.array([screen.t_of(x)[0] for x in ic])
    CACHE.mkdir(exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")
    np.save(p_t, t)
    np.save(p_ic, ic)
    return t, ic


def selection(t_cpu, t_gpu, ic_cpu, ic_gpu):
    out = {}
    both = np.isfinite(t_cpu) & np.isfinite(t_gpu)
    out["t_missing_mismatch"] = int((np.isfinite(t_cpu) != np.isfinite(t_gpu)).sum())
    d = np.abs(t_gpu[both] - t_cpu[both])
    out["t_abs_err"] = {
        q: float(np.quantile(d, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")
    }
    out["t_abs_err_mean"] = float(d.mean())
    ok = np.isfinite(ic_cpu) & np.isfinite(ic_gpu)
    di = np.abs(ic_gpu[ok] - ic_cpu[ok])
    out["daily_ic_abs_err"] = {q: float(np.quantile(di, float(q)))
                               for q in ("0.5", "0.99", "0.9999", "1.0")}  # fmt: skip
    out["daily_ic_missing_mismatch"] = int(
        (np.isfinite(ic_cpu) != np.isfinite(ic_gpu)).sum()
    )
    # 每个候选的「最差的一天」误差：看误差是否集中在个别候选
    per = np.nanmax(np.where(ok, np.abs(ic_gpu - ic_cpu), np.nan), axis=1)
    out["per_candidate_max_daily_ic_err"] = {
        q: float(np.nanquantile(per, float(q))) for q in ("0.5", "0.9", "1.0")
    }
    n = len(t_cpu)
    tc = np.where(np.isfinite(t_cpu), t_cpu, -np.inf)
    tg = np.where(np.isfinite(t_gpu), t_gpu, -np.inf)
    rc, rg = np.argsort(-tc, kind="stable"), np.argsort(-tg, kind="stable")
    for k in (max(1, round(0.1 * n)), 32):
        out[f"top{k}_overlap"] = len(set(rc[:k]) & set(rg[:k])) / k
    out["thresholds"] = {}
    for thr in THRESHOLDS:
        pc, pg = tc >= thr, tg >= thr
        need = float(np.max(np.maximum(thr - tg[pc], 0.0))) if pc.any() else 0.0
        out["thresholds"][str(thr)] = {
            "cpu_pass": int(pc.sum()), "gpu_pass": int(pg.sum()),
            "cpu_pass_gpu_fail": int((pc & ~pg).sum()),
            "gpu_pass_cpu_fail": int((pg & ~pc).sum()),
            "margin_needed": need,
            "near_threshold_cpu_within_0.05": int((np.abs(tc - thr) < 0.05).sum()),
        }  # fmt: skip
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_data_args(ap)
    ap.add_argument(
        "--backend", choices=["mlx", "mlx-fused", "torch", "torch-fused"], required=True
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=1, help="多少个根的 Rank IC 叠成一批")
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    import gpu_ops

    xp = gpu_ops.backend(args.backend, args.device)
    import_seconds = time.perf_counter() - t0
    out = {"machine": common.machine(), "backend": args.backend, "device": xp.device,
           "chunk": args.chunk, "load_before": common.load_avg(),
           "import_seconds": import_seconds}  # fmt: skip
    panel, source, t_read = common.load_panel(args)
    tag = source["kind"]
    out["source"], out["read_fields_seconds"] = source, t_read
    t0 = time.perf_counter()
    screen = common.Screen(panel, args.threads)
    out["prepare_cpu_seconds"] = time.perf_counter() - t0
    pop_path = common.RESULTS / f"population_{tag}.json"
    if pop_path.exists():
        exprs, meta = common.load_population(pop_path)
    else:
        calls = common.evolve_population(screen, args.population, 2, seed=0)
        exprs = calls[-1]["exprs"]
        meta = {"strategy": f"Evolution(population={args.population}, generations=2, seed=0)",
                "generation": len(calls) - 1, "source": source}  # fmt: skip
        common.save_population(pop_path, exprs, meta)
    out["population"] = {"file": pop_path.name, "n": len(exprs)}
    fields = sorted(set().union(*(e.fields for e in exprs)))

    xp.reset_peak()
    gs = gpu_ops.GpuScreen(xp, panel, fields, screen.universe, screen.returns)
    out["upload_seconds"] = gs.upload_seconds
    out["memory_after_upload"] = xp.memory()

    def generation(per_op=False):
        t0 = time.perf_counter()
        ic = gs.run(exprs, per_op_timing=per_op, chunk=args.chunk)
        t1 = time.perf_counter()
        t = np.array([common.t_stat(x, screen.T, screen.span)[0] for x in ic])
        return ic, t, {"device_and_fetch": t1 - t0, "t_stats_cpu": time.perf_counter() - t1,
                       "total": time.perf_counter() - t0, "load": common.load_avg()}  # fmt: skip

    ic_g, t_g, first = generation()
    out["first_generation"] = first  # 含内核编译、分配器预热
    print("first", first, flush=True)
    runs = []
    for k in range(args.repeat):
        _, t_k, rec = generation()
        same = bool(
            np.array_equal(np.nan_to_num(t_k, nan=-99), np.nan_to_num(t_g, nan=-99))
        )
        rec["bitwise_same_as_first"] = same
        runs.append(rec)
        print(k, rec, flush=True)
    out["runs"] = runs
    out["median"] = {k: statistics.median(r[k] for r in runs)
                     for k in ("device_and_fetch", "t_stats_cpu", "total")}  # fmt: skip
    gs.op_seconds.clear()
    generation(per_op=True)
    out["per_op_seconds"] = dict(sorted(gs.op_seconds.items(), key=lambda kv: -kv[1]))
    out["op_counts_per_generation"] = {
        k: v // (args.repeat + 2) for k, v in gs.op_counts.items()
    }
    out["memory_end"] = xp.memory()
    out["device_active_peak_sampled"] = gs.peak_active
    out["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

    t_cpu, ic_cpu = cpu_reference(screen, exprs, tag)
    np.save(CACHE / f"gpu_ic_{args.backend}_{xp.device}_{tag}{args.tag}.npy", ic_g)
    out["float32_vs_float64"] = selection(t_cpu, t_g, ic_cpu, ic_g)
    out["load_after"] = common.load_avg()
    name = f"gpu_{args.backend}_{xp.device}_{tag}{args.tag}.json"
    common.write_json(common.RESULTS / name, out)
    print({k: out[k] for k in ("upload_seconds", "median", "per_op_seconds")})
    print(out["float32_vs_float64"])
    gs = None  # 释放设备上的字段
    xp.release()


if __name__ == "__main__":
    main()
