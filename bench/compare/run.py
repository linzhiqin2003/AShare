"""Reproducible isolated comparison suite with hard 15-minute process-group caps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from metadata import compact

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench/results"
ENGINES = [
    "panelbt",
    "reference",
    "numpy_full",
    "numpy_sparse",
    "numpy_approx",
    "polars_approx",
    "backtrader_strategy",
    "vectorbt_strategy",
    "vectorbt_rust_strategy",
    "bt_strategy",
]


def render(report):
    lines = [
        "# 引擎对比基线",
        "",
        f"基线开始：{report['timestamp']}；最近更新：{report.get('updated_at', report['timestamp'])}。机器同时有其他任务；数字用于留档，空闲时运行 `make bench-compare && make bench-docs` 会覆盖结果及本文件。",
        "",
        "所有输入使用固定种子，约 100 个目标持仓；成交价等于收盘价；月度间隔为 21。任务一目标暴露 85%（批量 85%–95%），只收比例佣金 0.025%；任务二初始资金 10 亿（其他任务 1000 万），目标暴露 100%，100 股整手及现金缩减；任务三包含全部可选 A 股面板。批量 K=1000 使用 T=1000、N=100、月度调仓。",
        "",
        "运行次数通常为 3，取中位数；超过 60 秒可只测一次。每次独立调用或批量样本设 15 分钟上限（阶段心跳重置计时），超时终止整个进程组，停止该实现更大规模。输入准备及首次调用另列；Numba 首次调用（含可能的编译）不计稳态中位数。无 JIT 的单次首次有效测量若已超过 60 秒，可直接作为唯一样本，并有 sample_includes_first_call 标志。峰值内存包含输入、预热及批量方法试测，不等同胜出方法单独运行的稳态内存；批量子进程内存单列最大单进程值，不冒充聚合值。",
        "",
        "|任务|T×N / 调仓|实现|K|回测本身 ms|含输入准备 ms|首次调用 ms|最大净值差|主进程峰值 MiB|最大子进程峰值 MiB|线程/进程|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in sorted(
        report["records"],
        key=lambda x: (
            x["K"] > 1,
            x["task"],
            x["T"],
            x["N"],
            -x["interval"],
            ENGINES.index(x["engine"]),
        ),
    ):
        status = x["status"]
        ms = f"{x['median_s'] * 1000:.3f}" if status == "ok" else "—"
        inclusive = (
            f"{x['including_preparation_median_s'] * 1000:.3f}"
            if "including_preparation_median_s" in x
            else "—"
        )
        cold = f"{x['first_call_s'] * 1000:.3f}" if "first_call_s" in x else "—"
        child = (
            f"{x['child_peak_rss_bytes'] / 2**20:.1f}"
            if x.get("child_peak_rss_bytes", 0) > 0
            else "—"
        )
        error = f"{x['max_nav_error']:.3g}" if "max_nav_error" in x else "—"
        mem = f"{x.get('peak_rss_bytes', 0) / 2**20:.1f}" if status == "ok" else "—"
        lines.append(
            f"|{x['task']}|{x['T']}×{x['N']} / {x['interval']}|{x['engine']}|{x['K']}|{ms}|{inclusive}|{cold}|{error}|{mem}|{child}|{x.get('threads', '—')}|{status}: {x.get('reason', x.get('semantics', ''))}|"
        )
    lines += [
        "",
        "## 语义与公平性",
        "",
        "- panelbt、reference、NumPy 逐日两版均核对净值，阈值 rtol=atol=1e-9。任务一每个第三方实现也核对逐日净值。",
        "- NumPy/Polars 整块近似版在调仓间隔中保留价格漂移，用无摩擦权重估计换手，扣费后按比例缩放财富；panelbt 按扣费前权益确定股数。bt 在各子资产分配额内扣佣金，同样有扣费口径差异。三者另跑零费率控制，必须逐日一致才允许报告耗时；实际差额列在表中。",
        "- vectorbt Numba 使用只在调仓日触发的 pre_segment/order 回调；其批量和可选 Rust 引擎使用原生 targetpercent、共享现金、退出标的优先的显式调用顺序和固定当行估值（任务一有 5%–15% 现金缓冲，其余目标持仓内调整不会受现金约束；仍逐日核对净值）；backtrader 使用 preload/runonce、cheat-on-open、无 observers、种子 bar 对齐首日，不把 Python 自写简化回测冒充第三方内核。",
        "- 任务二：backtrader 用 next_open 进行篮子预算，bt 用 Algo/原生 transact，vectorbt 用 Numba pre_segment/order 回调实现整手及同比例缩减，成交和账户记账仍由各自引擎完成。vectorbt Rust 的自定义 Numba 回调路径不适用；NumPy/Polars 整块近似版不能表达跨日整手与现金状态。",
        "- 完整规则中，前日持仓、现金预算、取整、阻塞顺延和公司行为共同决定下一日状态；当前 Polars 向量表达式没有这种组合状态的递推接口。用 Python UDF 包装逐日循环不作为 Polars 向量化实现参与比较。",
        "- 近似模式的批量也逐一核对全部 1000 组零费率曲线，并检查两种计时口径的含费结果一致；净值差列取全部组合的最大值。1000 组批量使用各引擎的原生批量（Rayon、vectorbt 分组）或串行/进程试测后的胜者；每组目标内容哈希均不同（N=H=100 时以 1000 档暴露区分）。进程预热不计稳态；批量数据在各工作进程初始化一次。",
        "- 两种耗时均直接重复测量：回测本身使用预先转换的不可变输入；含输入准备从统一 NumPy 行情与 CSR 目标出发，panelbt/reference 计入 Market/Targets 校验，对手计入自己的 DataFrame/稠密矩阵转换。随机数据生成、模块导入和 JIT 预热不计；引擎对象实例化和结果提取计入。批量行情每个工作进程初始化一次，含输入准备口径计逐计划转换；预备输入的独立工作进程固定分配目标，避免缓存遗漏。reference 的独立费用展开仍保留在执行内。",
        "- 两种口径在不同时间窗口取样；忙机上的中位数可能倒置，不能相减后把负差解释成准备耗时。原始样本完整保留，空闲机器重跑后再作细小差异判断。",
        "- 版本、原始样本、首次编译/调用、内存、批量试测、失败日志均在 `bench/results/compare.json` 及 `bench/results/logs/`。",
        "",
        "## 复现",
        "",
        "`make dev && make bench-compare`；沙箱可用 `make dev SANDBOX=1`。`make bench` 只测自家引擎，并与仓库基线比较，退化超过 10% 标红字样 REGRESSION。",
        "",
        "适配器依据：[Backtrader Cheat-On-Open](https://backtrader.com/blog/posts/2017-05-01-cheat-on-open/cheat-on-open/)、[vectorbt Portfolio](https://vectorbt.dev/api/portfolio/base/)、[bt API](https://pmorissette.github.io/bt/bt.html)。",
        "",
        "机器与版本：",
        "```json",
        json.dumps(report["environment"], ensure_ascii=False, indent=2),
        "```",
    ]
    own = ROOT / ".cache/panelbt/panelbt-current.json"
    if not own.exists():
        own = RESULTS / "panelbt-current.json"
    if own.exists():
        current = json.loads(own.read_text())
        lines += [
            "",
            "## 自家引擎与回归检查",
            "",
            "|场景|中位数 ms|相对仓库基线|",
            "|---|---:|---:|",
        ]
        for name in ("lots", "fractional", "batch", "long_short"):
            delta = current.get("regressions", {}).get(name, 0)
            lines.append(
                f"|{name}|{current[name]['total_s'] * 1000:.3f}|{delta:+.1%}{' REGRESSION' if delta > 0.1 else ''}|"
            )
    future_baseline = RESULTS / "step1c-baseline.json"
    if future_baseline.exists():
        future = json.loads(future_baseline.read_text())
        spot = future["spot_full_rules_ms"]
        mixed = future["mixed_full_rules_ms"]
        simple = future["simple_exact_ms"]
        ledger = future["mixed_reference_ledger"]
        lines += [
            "",
            "## 步骤 1c：期货对冲腿",
            "",
            "4000×5500 现货面板加 20 列期货、约 100 只现货、每 21 日调仓；同一已校验市场对象、9 次交替调用、单线程。完整规则的 NumPy 对照不适用：下方 NumPy 基线只实现同义的简化对冲，不含保证金、到期、强平或公司行为。市场构建及输入转换不计入时间。原始采样写入被忽略的 `.cache/panelbt/futures-current.json`；当前机器有并行任务，空闲时用 `make bench && make bench-docs` 重测。",
            "",
            "| 任务 | panelbt 最小 ms | panelbt 中位数 ms | NumPy 中位数 ms | 净值最大误差 |",
            "|---|---:|---:|---:|---:|",
            f"| 完整现货规则 | {spot['min']:.3f} | {spot['median']:.3f} | 不适用 | — |",
            f"| 完整现货+20 列期货 | {mixed['min']:.3f} | {mixed['median']:.3f} | 不适用 | — |",
            f"| 同义简化对冲任务 | {simple['panelbt']['min']:.3f} | {simple['panelbt']['median']:.3f} | {simple['numpy']['median']:.3f} | {future['simple_max_nav_error']:.3g} |",
            "",
            f"简化任务使用活跃证券稀疏 NumPy 运算；该任务 NumPy 比自研快，不能用它外推完整规则。新增混合语料 {ledger['cases']} 组、{ledger['trades']} 笔成交：参考实现 {ledger['divergences']} 分叉，独立账本 {ledger['ledger_failures']} 不符。复现：`uv run --no-sync python bench/validate.py --corpus futures --start 40000 --count 20000`。",
            "",
            f"机器：{future['machine']['platform']}，Python {future['machine']['python']}，NumPy {future['machine']['numpy']}；开始时 1/5/15 分钟负载 {', '.join(f'{x:.2f}' for x in future['machine']['load_start'])}，峰值 RSS {future['machine']['rss_peak_bytes'] / 2**30:.2f} GiB。",
        ]
        if "spot_pair_to_1b" in future:
            pair = future["spot_pair_to_1b"]
            lines += [
                "",
                f"同输入现货路径与 1b `{pair['base_ref']}` 各自 LTO 构建、AB/BA 交替配对（{pair['T']}×{pair['N']}，不含准备）；三场景最小值比均小于 1.05：",
                "",
                "| 场景 | 1b 最小 / 中位 ms | 当前最小 / 中位 ms | 最小值比 |",
                "|---|---:|---:|---:|",
            ]
            for key, label in (
                ("full_rules", "完整规则含退市"),
                ("no_delist_panel", "不传退市面板"),
                ("prices_only", "仅价格"),
            ):
                row = pair["scenarios"][key]
                lines.append(
                    f"| {label} | {row['base_min_ms']:.3f} / {row['base_median_ms']:.3f} | {row['current_min_ms']:.3f} / {row['current_median_ms']:.3f} | {row['minimum_ratio']:.3f} |"
                )
    (ROOT / "docs/BENCHMARKS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engines", nargs="+")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    logs = RESULTS / "logs"
    logs.mkdir(exist_ok=True)
    output = RESULTS / "compare.json"
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "panelbt",
            "polars",
            "pandas",
            "backtrader",
            "vectorbt",
            "vectorbt-rust",
            "bt",
            "numba",
        )
    }
    machine = {"processor": platform.processor()}
    if sys.platform == "darwin":
        try:
            machine["processor"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            machine["memory_bytes"] = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
                )
            )
        except (OSError, subprocess.CalledProcessError):
            machine["hardware_details"] = (
                "sysctl restricted by sandbox; architecture and CPU count remain available"
            )
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "platform": platform.platform(),
            **machine,
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "versions": versions,
            "concurrent_workload": True,
        },
        "records": [],
    }
    if args.resume and output.exists():
        report = json.loads(output.read_text())
        report["environment"]["versions"] = versions
        report["environment"].update(machine)
        report["updated_at"] = datetime.now().astimezone().isoformat()
    sizes = [
        (1000, 100, 21),
        (1000, 500, 21),
        (1000, 1000, 21),
        (4000, 5500, 21),
        (4000, 5500, 1),
    ]
    if args.quick:
        sizes = [(30, 10, 21)]
    jobs = [
        (task, t, n, interval, 1, e)
        for task in (1, 2, 3)
        for t, n, interval in sizes
        for e in ENGINES
    ]
    jobs += [(1, 1000, 100, 21, 1000, e) for e in ENGINES]
    if args.engines:
        jobs = [job for job in jobs if job[-1] in args.engines]
    timed_out = {
        (r["engine"], r["task"]): r["T"] * r["N"]
        for r in report["records"]
        if r["status"] == "timeout" and r["K"] == 1
    }
    for task, t, n, interval, k, engine in jobs:
        key = f"{task}-{t}-{n}-{interval}-{k}-{engine}"
        if any(x.get("key") == key for x in report["records"]):
            continue
        row = dict(key=key, task=task, T=t, N=n, interval=interval, K=k, engine=engine)
        reason = None
        if task == 2 and engine not in (
            "panelbt",
            "reference",
            "numpy_full",
            "numpy_sparse",
            "backtrader_strategy",
            "bt_strategy",
            "vectorbt_strategy",
        ):
            reason = "不支持原生整篮同比例现金缩减与100股申报单位"
        if task == 3 and engine not in ("panelbt", "reference"):
            reason = "完整规则仅测 panelbt 和 reference"
        if k == 1 and t * n >= timed_out.get((engine, task), float("inf")):
            reason = "较小规模已超过15分钟，停止更大规模"
        if reason:
            row.update(status="not_applicable", reason=reason)
        else:
            path = logs / (key + ".json")
            log = logs / (key + ".log")
            env = os.environ.copy()
            env.update(
                MPLCONFIGDIR=str(ROOT / ".cache/mpl"),
                NUMBA_CACHE_DIR=str(ROOT / ".cache/numba"),
                NUMBA_NUM_THREADS="1",
                POLARS_MAX_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                OMP_NUM_THREADS="1",
                RAYON_NUM_THREADS="1",
            )
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("worker.py")),
                "--t",
                str(t),
                "--n",
                str(n),
                "--interval",
                str(interval),
                "--task",
                str(task),
                "--k",
                str(k),
                "--engine",
                engine,
                "--output",
                str(path),
            ]
            print("Running", key, flush=True)
            Path(str(path) + ".running").unlink(missing_ok=True)
            with log.open("w") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    while True:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            heartbeat = Path(str(path) + ".running")
                            started = (
                                json.loads(heartbeat.read_text())["started_at"]
                                if heartbeat.exists()
                                else log.stat().st_mtime
                            )
                            if time.time() - started > 900:
                                raise
                    if code == 0:
                        row.update(json.loads(path.read_text()))
                    else:
                        row.update(
                            status="excluded",
                            reason=f"执行或净值校验失败，见 {log.relative_to(ROOT)}",
                            error=log.read_text()[-6000:],
                        )
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    row.update(status="timeout", reason="超过15分钟")
                    if k == 1:
                        timed_out[engine, task] = t * n
            print(row.get("median_s", row["status"]), flush=True)
        report["records"].append(row)
        output.write_text(
            json.dumps(compact(report), ensure_ascii=False, indent=2) + "\n"
        )
    # The optional open-source Rust engine is explicitly selected and isolated.
    report["vectorbt_rust"] = (
        f"vectorbt-rust {versions['vectorbt-rust']} installed; explicit engine=rust measured separately from engine=numba."
    )
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "bench/compare").glob("*.py"))
    }
    report["source_sha256"].update(
        {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "crates/panelbt-core/src/lib.rs",
                "crates/tradecore/src/lib.rs",
            )
        }
    )
    output.write_text(json.dumps(compact(report), ensure_ascii=False, indent=2) + "\n")
    from collections import Counter

    print(
        "Comparison complete:",
        dict(Counter(r["status"] for r in report["records"])),
        flush=True,
    )
    print("Results: bench/results/compare.json", flush=True)


if __name__ == "__main__":
    main()
