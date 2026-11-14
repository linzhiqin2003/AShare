"""GPU 探针的公用部分：取数（真实面板只读 / 同规模合成面板）、种群生成与存取、CPU 参考粗筛。

真实面板：``--root`` 指向数据根目录（只读打开，不写任何东西）。
合成面板：``--synthetic``，形状默认与真实面板相同（2430 × 5923），带停牌、退市、上市等摩擦，
供没有真实数据的机器（例如 CUDA 服务器）复现。

粗筛口径与 `ashare.factor.mining.search` 相同：股票池内的因子值对 h=5（screen_horizon，
默认等于调仓周期 5）的标签做逐日 Rank IC，t 值用 `stats.overlap_t`（`ic_summary`）。
探针用整段 (0, T) 当训练段，CPU 与 GPU 两边口径一致。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from ashare.factor.evaluate import FactorEvaluator, ic_summary
from ashare.factor.expression import Expr
from ashare.factor.mining import Space
from ashare.factor.mining import strategies as st
from ashare.factor.mining.space import dedup_key

RESULTS = Path(__file__).with_name("results")
SCREEN_H = 5


def load_avg():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True)
    return out.stdout.strip().strip("{} ")


def machine():
    import platform

    info = {"cpus": os.cpu_count(), "platform": platform.platform()}
    try:
        info["chip"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        info["memory_gib"] = (
            int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                ).stdout
            )
            / 2**30  # fmt: skip
        )
    except (OSError, ValueError):
        pass
    return info


# ---- 取数 -------------------------------------------------------------------------


class Panel:
    """只读字段的内存副本（float64），dates / symbols / version 与 ashare 面板同协议。"""

    def __init__(self, arrays, dates, symbols, version):
        self._arrays = arrays
        self.dates = np.asarray(dates)
        self.symbols = np.asarray(symbols)
        self.version = version

    def __getitem__(self, name):
        return self._arrays[name]


def fields_needed():
    return sorted(set(FactorEvaluator.required_fields()) | set(Space.default().fields))


def load_panel(args):
    """返回 (panel, 来源说明, 读字段秒数)。"""
    t0 = time.perf_counter()
    if args.synthetic:
        from ashare.factor.mining.synthetic import Frictions, market

        p = market(t=args.t, n=args.n, seed=7, frictions=Frictions.typical())
        arrays = {k: np.asarray(p[k]) for k in fields_needed()}
        panel = Panel(arrays, p.dates, p.symbols, p.version)
        source = {"kind": "synthetic", "t": args.t, "n": args.n, "seed": 7}
    else:
        import ashare.data as ad

        data = ad.open(root=args.root)
        raw = data.panel(fields_needed(), start=args.start, end=args.end)
        arrays = {k: np.ascontiguousarray(np.asarray(raw[k])) for k in fields_needed()}
        panel = Panel(arrays, raw.dates, raw.symbols, raw.version)
        source = {"kind": "real", "start": args.start, "end": args.end}
    source["shape"] = [len(panel.dates), len(panel.symbols)]
    return panel, source, time.perf_counter() - t0


def add_data_args(ap):
    ap.add_argument("--root", default=os.environ.get("ASHARE_DATA_DIR"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--t", type=int, default=2430)
    ap.add_argument("--n", type=int, default=5923)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--threads", type=int, default=4)


class Screen:
    """CPU float64 参考粗筛：股票池、h=5 标签、Rank IC 内核都只准备一次。"""

    def __init__(self, panel, threads=4):
        import panelops

        panelops.set_num_threads(threads)
        self.fe = FactorEvaluator(panel, horizons=(SCREEN_H,), screen_horizon=SCREEN_H,
                                  size_buckets=1, threads=threads)  # fmt: skip
        self.universe = self.fe.universe
        self.returns = self.fe.returns[SCREEN_H]
        self.span = self.fe.span[SCREEN_H]
        self.T = self.fe.shape[0]

    def ic(self, values):
        return self.fe.screen(values)[1]

    def t_of(self, ic):
        return t_stat(ic, self.T, self.span)


def t_stat(ic, T, span):
    """与 `screen_metrics` 相同：段 (0, T) 内标签完整落在段内的信号日，`ic_summary` 的 t。"""
    s = ic_summary(np.asarray(ic, dtype=np.float64)[0 : max(0, T - span)], SCREEN_H)
    return s["t"], s["mean"]


# ---- 种群 -------------------------------------------------------------------------


def to_json(e: Expr):
    return e.canonical()


def from_json(node):
    _, op, args, params = node
    return Expr(op, tuple(from_json(a) for a in args), tuple((k, v) for k, v in params))


def evolve_population(screen, population=256, generations=2, seed=0, log=print):
    """跑真实的 Evolution（适应度 = 粗筛 t 值，CPU 现行做法），返回最后一代后代（交叉与变异的产物）
    及每代的耗时。粗筛掉、失败的适应度为 −∞，与 search 的约定一致。"""
    from ashare.factor import Evaluator

    ev = Evaluator()
    calls = []

    def evaluate(exprs):
        t0 = time.perf_counter()
        out = []
        for _, values in ev.evaluate_many({str(i): e for i, e in enumerate(exprs)},
                                          screen.fe.panel):  # fmt: skip
            t, _ = screen.t_of(screen.ic(values))
            out.append(t if t is not None and math.isfinite(t) else -math.inf)
        calls.append({"n": len(exprs), "seconds": time.perf_counter() - t0,
                      "exprs": list(exprs)})  # fmt: skip
        log(
            f"  generation {len(calls) - 1}: {len(exprs)} exprs {calls[-1]['seconds']:.1f}s"
        )
        return out

    st.Evolution(population=population, generations=generations, seed=seed).run(
        Space.default(), evaluate
    )
    return calls


def save_population(path, exprs, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "exprs": [to_json(e) for e in exprs]}))


def load_population(path):
    blob = json.loads(Path(path).read_text())
    return [from_json(x) for x in blob["exprs"]], blob["meta"]


def unique_by_key(exprs):
    seen, out = set(), []
    for e in exprs:
        k = dedup_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
