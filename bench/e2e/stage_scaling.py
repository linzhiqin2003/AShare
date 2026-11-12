"""线程扩展曲线（1/5/10/15）与只用能效核（taskpolicy -b）的对照。不在默认阶段集合里。

每个点是一个全新子进程，线程数在导入各库之前由环境变量和 set_num_threads 固定。
三种负载：10 个表达式求值（第二遍，页缓存已热）、最近 300 日风险模型重建（不写缓存）、
200 组批量回测。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

POINTS = (("1", 1), ("5", 5), ("10", 10), ("15", 15), ("ecores_default", 0))
WORKLOADS = ("factors", "risk", "backtest")


def worker(kind, threads, data):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import panelops
    import portopt

    panelops.set_num_threads(threads)
    portopt.set_num_threads(threads)
    ctx = {"data": Path(data), "quick": True, "threads": threads}
    if kind == "factors":
        from expressions import expressions
        from stage_factors import open_source

        from ashare.factor import Evaluator

        source, ev = open_source(ctx), Evaluator()
        subset = list(expressions().values())[:10]

        def work():
            for e in subset:
                ev.evaluate(e, source)

        work()
    elif kind == "risk":
        from risk_adapter import load

        from ashare.risk import model_series

        source, config = load(data, 300)

        def work():
            for _ in model_series(source, config=config):
                pass
    else:
        import panelbt
        from stage_backtest import BATCH, config, costs, market_inputs, plans

        import ashare.data
        from ashare.factor import Evaluator, default_registry

        panel = ashare.data.open(data).panel(
            [
                "open",
                "close",
                "high",
                "low",
                "upper_limit",
                "lower_limit",
                "is_suspended",
                "has_quote",
                "volume",
                "split",
                "cash_dividend",
                "delist",
                "adj_factor",
            ]
        )
        market = panelbt.Market(**market_inputs(panel))
        score = Evaluator().evaluate(
            default_registry()["reversal_20"].expression, panel
        )
        batch = plans(score, panel.shape[0], True, BATCH)
        cost, cfg = costs(panel.dates), config(data, panel.symbols)

        def work():
            panelbt.run_batch(market, batch, cost, cfg, threads=threads)

    walls, cpus = [], []
    for _ in range(2):
        cpu, start = time.process_time(), time.perf_counter()
        work()
        walls.append(time.perf_counter() - start)
        cpus.append(time.process_time() - cpu)
    best = min(range(2), key=lambda i: walls[i])
    print(
        json.dumps(
            {
                "wall": walls[best],
                "cpu_ratio": cpus[best] / walls[best],
                "load": os.getloadavg()[0],
            }
        )
    )


def run(ctx):
    cases = []
    for kind in WORKLOADS:
        for label, threads in POINTS:
            env = dict(os.environ)
            if threads:
                for key in (
                    "RAYON_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_NUM_THREADS",
                ):
                    env[key] = str(threads)
            command = [
                sys.executable,
                __file__,
                "--worker",
                kind,
                str(threads),
                str(ctx["data"]),
            ]
            if label.startswith("ecores"):
                command = ["taskpolicy", "-b", *command]
            out = (
                subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[-1]
            )
            point = json.loads(out)
            cases.append(
                {
                    "case": f"{kind}_threads_{label}",
                    "scale": {"workload": kind, "threads": threads},
                    "repeat": 2,
                    "wall_min": point["wall"],
                    "wall_median": point["wall"],
                    "cpu_ratio": point["cpu_ratio"],
                    "load": point["load"],
                }
            )
    return cases


if __name__ == "__main__":
    worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
