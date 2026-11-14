"""性能回归检查：合成数据上的固定基准，与存档基线比较。

    make perf-check                     # 比较，慢超过 20% 失败（退出码 1）
    make perf-baseline                  # 重录基线 bench/perfcheck/baseline.json
    make perf-check PERFCHECK_ARGS='--only groupbt.single,panelbt.trades'

每项先预热两次，再计时若干次取中位数。比值 = 本次中位数 / 基线中位数：
- 比值 > 1 + 容差：先复测一次，取两次中位数的较小值，仍超出才判失败；
- 比值 < 1 − 容差：通过，提示可以重录基线。
基线记录机器信息（CPU、核数、内存、线程设置）和录制时的负载；机器信息不一致时
比值没有意义，只报告不判失败（`--strict` 时照样判）。结果写 `.cache/testing/perf.json`，
不含主机名和本机路径。线程数固定（默认 4，Rayon 全局池与 panelops 同设），
减少负载起伏对多线程内核的影响。
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("baseline.json")
OUTPUT = ROOT / ".cache/testing/perf.json"
WARMUP = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bench.perfcheck")
    parser.add_argument("--record", action="store_true", help="重录基线")
    parser.add_argument("--only", default="", help="逗号分隔的基准名")
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--strict", action="store_true", help="机器不一致也判失败")
    parser.add_argument(
        "--note", default="", help="写进基线的说明（如录制时的负载情况）"
    )
    return parser.parse_args(argv)


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def memory_gib():
    try:
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1
        )
    except (ValueError, OSError, AttributeError):
        return None


def machine(threads):
    import numpy

    return {
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "memory_gib": memory_gib(),
        "system": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "threads": threads,
    }


def signature(info):
    return {k: info.get(k) for k in ("cpu", "cores", "memory_gib", "system", "threads")}


def load():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def measure(call, repeat):
    for _ in range(WARMUP):
        after = call()
        if callable(after):
            after()
    samples = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        after = call()
        samples.append(time.perf_counter() - start)
        if callable(after):
            after()  # 不计时的收尾（例如等后台写完、关闭结果库）
    return median(samples), samples


def main(argv=None):
    args = parse_args(argv)
    # 必须在导入任何原生扩展之前设定：Rayon 全局池只在第一次使用时读取。
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import panelops

    from . import cases
    from .cases import CASES

    panelops.set_num_threads(args.threads)
    cases.THREADS = args.threads
    names = [n for n in args.only.split(",") if n] or list(CASES)
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise SystemExit(f"没有这些基准：{', '.join(unknown)}")
    baseline = None
    if not args.record and BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
    info = machine(args.threads)
    comparable = baseline is not None and signature(baseline["machine"]) == signature(
        info
    )
    started, load_before = time.perf_counter(), load()
    items = []
    with tempfile.TemporaryDirectory(prefix="perfcheck-") as workdir:
        for name in names:
            setup, repeat = CASES[name]
            call = setup(workdir)
            value, samples = measure(call, repeat)
            item = {"name": name, "median": value, "samples": samples, "repeat": repeat}
            reference = (baseline or {}).get("items", {}).get(name)
            if reference is not None:
                ratio = value / reference["median"]
                if ratio > 1 + args.tolerance:
                    again, more = measure(call, repeat)
                    item["remeasured"] = again
                    value = min(value, again)
                    item["median"], ratio = value, value / reference["median"]
                    samples = samples + more
                slower = ratio > 1 + args.tolerance
                item.update(
                    baseline=reference["median"],
                    ratio=ratio,
                    status="regression"
                    if slower
                    else "faster"
                    if ratio < 1 - args.tolerance
                    else "ok",
                    passed=not (slower and (comparable or args.strict)),
                )
            else:
                item.update(baseline=None, ratio=None, status="new", passed=True)
            items.append(item)
            print(_line(item), flush=True)
            del call
    elapsed = time.perf_counter() - started
    report = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": info,
        "load": {"before": load_before, "after": load()},
        "tolerance": args.tolerance,
        "comparable": comparable,
        # 1 分钟负载超过核数一半时，多线程内核和落盘的计时会大幅抖动（实测 1.2–1.8 倍）。
        "busy": busy(load_before, load()),
        "baseline": None
        if baseline is None
        else {k: baseline.get(k) for k in ("recorded", "machine", "load", "note")},
        "elapsed": elapsed,
        "passed": all(item["passed"] for item in items),
        "regressions": [i["name"] for i in items if not i["passed"]],
        "benchmarks": items,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.record:
        previous = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        kept = previous.get("items", {}) if args.only else {}
        BASELINE.write_text(
            json.dumps(
                {
                    "recorded": report["generated"],
                    "machine": info,
                    "load": report["load"],
                    "note": args.note,
                    "items": kept
                    | {
                        item["name"]: {
                            "median": round(item["median"], 6),
                            "samples": [round(x, 6) for x in item["samples"]],
                        }
                        for item in items
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(f"基线已写入 {BASELINE.relative_to(ROOT)}")
    _summary(report, baseline)
    return 0 if report["passed"] else 1


def _line(item):
    text = f"{item['name']:<24} {item['median'] * 1e3:9.1f} ms"
    if item["ratio"] is not None:
        text += f"   基线 {item['baseline'] * 1e3:9.1f} ms   {item['ratio']:.2f}x   {item['status']}"
    return text


def busy(*loads):
    cores = os.cpu_count() or 1
    return any(value is not None and value[0] > cores / 2 for value in loads)


def _summary(report, baseline):
    print(
        f"共 {len(report['benchmarks'])} 项，用时 {report['elapsed']:.1f} s，"
        f"负载 {report['load']['before']} → {report['load']['after']}"
    )
    if report["busy"]:
        print("机器负载高（1 分钟负载超过核数一半），计时不可靠，请在空闲时复测。")
    if baseline is None:
        return
    if not report["comparable"]:
        print("机器信息与基线不一致，比值仅供参考（--strict 时照样判失败）。")
    failed = report["regressions"]
    faster = [i["name"] for i in report["benchmarks"] if i["status"] == "faster"]
    if failed:
        print(f"变慢超过容差：{', '.join(failed)}")
    if faster:
        print(
            f"变快超过容差，确认后可重录基线（make perf-baseline）：{', '.join(faster)}"
        )


if __name__ == "__main__":
    sys.exit(main())
