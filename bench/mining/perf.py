"""性能基线：「评估 100 个因子、全市场、10 年」的耗时拆解，以及搜索每秒评估多少个表达式。

在主目录的真实面板上只读运行；输出只有耗时（不含任何因子值或指标），写到 --out。

    uv run --no-sync python bench/mining/perf.py --root <数据根目录> --out <输出 json>

拆解：准备（读字段、标签、股票池、行情、基准）、表达式求值（`Evaluator.evaluate_many`）、
IC（三个持有期）与分组等便宜部分、多头端回测（`panelbt.run_batch`）；另测 NumPy 直白写法
（逐日 argsort 求秩相关）的 Rank IC 基线和逐个 `panelbt.run` 的回测基线作对比。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import panelbt

from ashare.factor import Evaluator
from ashare.factor.evaluate import FactorEvaluator
from ashare.factor.mining import Space
from ashare.factor.mining import strategies as st


def _load(root, start, end):
    import ashare.data as ad

    data = ad.open(root=root)
    fields = sorted(
        set(FactorEvaluator.required_fields()) | set(Space.default().fields)
    )
    t0 = time.perf_counter()
    panel = data.panel(fields, start=start, end=end)
    for name in fields:  # 真正读进内存（内存映射首次访问）
        np.asarray(panel[name]).sum()
    return data, panel, time.perf_counter() - t0


def _candidates(n, seed):
    space = Space.default()
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        e = st.random_candidate(space, rng)
        if e.digest not in seen:
            seen.add(e.digest)
            out.append(e)
    return out


def _numpy_rank_ic(x, y):
    """直白基线：逐日取成对有效的格子，argsort 求秩（不处理并列）再算相关。"""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        ok = np.isfinite(x[i]) & np.isfinite(y[i])
        if ok.sum() < 3:
            continue
        a = np.argsort(np.argsort(x[i, ok])).astype(float)
        b = np.argsort(np.argsort(y[i, ok])).astype(float)
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def run_once(panel, data, exprs, threads):
    from ashare.backtest import market_from_panel

    timing = {}
    t0 = time.perf_counter()
    bm = market_from_panel(data, panel=panel, validate=False)
    fe = FactorEvaluator(panel, market=bm, threads=threads)
    timing["prepare"] = time.perf_counter() - t0
    tb = time.perf_counter()
    _ = (
        fe.benchmark
    )  # 默认基准：同口径的「前 100%」组合跑一遍回测（第一次用到时算，之后复用）
    timing["benchmark"] = time.perf_counter() - tb
    ev = Evaluator()
    t_expr = t_cheap = 0.0
    targets, partials = [], {}
    it = ev.evaluate_many({str(k): e for k, e in enumerate(exprs)}, panel)
    while True:
        t1 = time.perf_counter()
        try:
            name, values = next(it)
        except StopIteration:
            t_expr += time.perf_counter() - t1
            break
        t2 = time.perf_counter()
        t_expr += t2 - t1
        scores, partial = fe.cheap(values)
        targets.append(fe.targets(scores))
        partials[name] = partial
        t_cheap += time.perf_counter() - t2
    timing["expressions"] = t_expr
    timing["ic_groups_etc"] = t_cheap
    timing["evaluator_batches"] = ev.last_batch.get("batches")
    t3 = time.perf_counter()
    longs = fe.run_long(targets)
    timing["backtest_run_batch"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    for k, name in enumerate(partials):
        fe.finish(name, partials[name], longs[k])
    timing["summaries"] = time.perf_counter() - t4
    # 粗筛（搜索的第一段）：只算股票池内的面板和筛选持有期的 IC
    t5 = time.perf_counter()
    for _, values in Evaluator().evaluate_many(
        {str(k): e for k, e in enumerate(exprs)}, panel
    ):
        fe.screen(values)
    timing["screen_only_with_expressions"] = time.perf_counter() - t5
    timing["total"] = sum(
        v for k, v in timing.items()
        if k not in ("evaluator_batches", "screen_only_with_expressions")
    )  # fmt: skip
    return timing, fe, targets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--factors", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    def load():
        return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()

    out = {"factors": args.factors, "start": args.start, "end": args.end,
           "cpus": os.cpu_count(), "load_before": load()}  # fmt: skip
    data, panel, t_load = _load(args.root, args.start, args.end)
    out["shape"] = [len(panel.dates), len(panel.symbols)]
    out["read_fields"] = t_load
    exprs = _candidates(args.factors, 0)
    runs = []
    for k in range(args.repeat):
        timing, fe, targets = run_once(panel, data, exprs, args.threads)
        runs.append(timing)
        print(k, json.dumps(timing), flush=True)
    out["runs"] = runs
    out["median"] = {key: statistics.median(r[key] for r in runs) for key in runs[0]}

    # 基线 1：NumPy 直白 Rank IC（一个因子、一个持有期），与 panelops 内核对比
    values = Evaluator().evaluate(exprs[0], panel)
    scores = fe._scores(values)
    ret = fe.returns[5]
    base, fast = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        _numpy_rank_ic(scores, ret)
        base.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fe._ic[5](scores)
        fast.append(time.perf_counter() - t0)
    out["rank_ic_one_factor_one_horizon"] = {"numpy_loop": statistics.median(base),
                                             "panelops": statistics.median(fast)}  # fmt: skip
    # cheap 的分步（10 个因子平均，毫秒）
    import panelops

    from ashare.factor.evaluate import _mean

    parts = {}

    def tm(k, f):
        t0 = time.perf_counter()
        f()
        parts[k] = parts.get(k, 0.0) + time.perf_counter() - t0

    for _, values in Evaluator().evaluate_many(
        {str(k): e for k, e in enumerate(exprs[:10])}, panel
    ):
        s = fe._scores(values)
        for h in fe.returns:
            tm(f"ic_h{h}", lambda h=h, s=s: fe._ic[h](s))
        tm("groups", lambda s=s: panelops.group_returns(s, fe.returns[5], 10))
        tm("size_ic", lambda s=s: [f(s) for f in fe._size_ic.values()])
        tm(
            "autocorr",
            lambda s=s: [_mean(panelops.factor_autocorr(s, p)) for p in (1, 5)],
        )
        tm("targets", lambda s=s: fe.targets(s))
    out["cheap_breakdown_ms_per_factor"] = {k: v / 10 * 1000 for k, v in parts.items()}
    # 基线 2：逐个 panelbt.run（10 个）与 run_batch（同样 10 个）
    ten = [t for t in targets[:10] if t is not None]
    t0 = time.perf_counter()
    for t in ten:
        panelbt.run(fe._market, t, fe.costs, fe.config)
    seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    panelbt.run_batch(fe._market, ten, fe.costs, fe.config)
    out["backtest_10"] = {"sequential_run": seq, "run_batch": time.perf_counter() - t0}
    # 搜索吞吐：同一批候选按 mine 的方式（两段式）算每秒评估多少个，见 real_example.py
    out["load_after"] = load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
