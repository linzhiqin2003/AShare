"""Collect the four test metrics into one report.

    python bench/testing/report.py [--date YYYY-MM-DD] [--no-docs]

Reads the outputs of the other tools under ``.cache/testing/``:

- ``coverage.json``  (``make coverage``)      branch coverage
- ``runtime.json``   (``make test``)          test runtime; ``runtime-serial.json``
- ``mutation.json``  (``make mutate-all``)    mutation score
- ``flaky.json``     (``make flaky``)         flaky rate
- ``perf.json``      (``make perf-check``)    performance regressions

A missing input is reported as "not run". Prints the summary, writes
``.cache/testing/report.json`` and adds or replaces the section for the date in
``docs/TESTING_METRICS.md``.
"""

import argparse
import re
import sys
from datetime import date

import budgets
from common import CACHE, ROOT, now_iso, read_json, write_json

METRICS_DOC = ROOT / "docs" / "TESTING_METRICS.md"
DOC_HEADER = """# 测试指标

`make test-report` 生成，每天一节，同一天重跑会覆盖当天那一节。各指标的口径见
`docs/TESTING.md`；机器可读版在 `.cache/testing/report.json`（不入库）。
"""

NOT_RUN = "未运行"


def _age(payload):
    return (payload or {}).get("generated", "")


def coverage_metric(data):
    if not data:
        return {"status": NOT_RUN}
    out = {"status": "ok", "generated": _age(data)}
    py = data.get("python")
    if py:
        t = py["totals"]
        out["python"] = {
            "branch_percent": t["branch_percent"],
            "line_percent": t["line_percent"],
            "branches": t["branches"],
            "covered_branches": t["covered_branches"],
            "target": py.get("target"),
            "files_below_target": len(py.get("below_target", [])),
            "lowest": [
                {"path": r["path"], "branch_percent": r["branch_percent"]}
                for r in py.get("lowest", [])
            ],
            "modules": py.get("modules", {}),
        }
    rust = data.get("rust")
    if rust:
        out["rust"] = {
            "line_percent": rust["totals"].get("line_percent"),
            "branch_percent": rust["totals"].get("branch_percent"),
            "branch_measured": rust.get("branch", False),
            "scope": rust.get("scope"),
            "files_below_target": len(rust.get("below_target", [])),
            "modules": rust.get("modules", {}),
        }
    return out


def runtime_metric(parallel, serial):
    if not parallel and not serial:
        return {"status": NOT_RUN}
    out = {"status": "ok"}
    if parallel:
        out.update(
            generated=_age(parallel),
            workers=parallel.get("workers"),
            wall_seconds=parallel["wall_seconds"],
            test_seconds=parallel["test_seconds"],
            tests=parallel["tests"],
            outcomes=parallel["outcomes"],
            categories=parallel["categories"],
            slowest=parallel["slowest"],
            over_budget=parallel["over_budget"],
        )
    if serial:
        out["serial_wall_seconds"] = serial["wall_seconds"]
        out["serial_over_budget"] = serial["over_budget"]
        out["serial_generated"] = _age(serial)
    return out


def mutation_metric(data):
    if not data:
        return {"status": NOT_RUN}
    total = data.get("total", {})
    modules = sorted(data.get("modules", []), key=lambda m: m.get("score", 0.0))
    return {
        "status": "ok",
        "generated": _age(data),
        "scope": data.get("scope"),
        "score": total.get("score"),
        "killed": total.get("killed"),
        "survived": total.get("survived"),
        "timeout": total.get("timeout"),
        "equivalent": total.get("equivalent"),
        "modules": modules,
    }


def flaky_metric(data):
    if not data:
        return {"status": NOT_RUN}
    return {
        "status": "ok",
        "generated": _age(data),
        "runs": len(data.get("runs", [])),
        "tests": data.get("tests"),
        "flaky": [row["id"] for row in data.get("flaky", [])],
        "failing": [row["id"] for row in data.get("failing", [])],
        "rate": data.get("flaky_rate"),
    }


def perf_metric(data):
    """``perf.json`` comes from the perf-check tool; read only common fields."""
    if not data:
        return {"status": NOT_RUN}
    out = {"status": "ok", "generated": _age(data)}
    for key in ("benchmarks", "results", "cases"):
        if isinstance(data.get(key), list):
            entries = data[key]
            out["benchmarks"] = len(entries)
            out["regressions"] = [
                e.get("name", "?")
                for e in entries
                if isinstance(e, dict)
                and str(e.get("status", "")).lower() in ("fail", "failed", "regression")
            ]
            break
    if isinstance(data.get("regressions"), list):
        out["regressions"] = [
            r.get("name", "?") if isinstance(r, dict) else str(r)
            for r in data["regressions"]
        ]
    if "passed" in data:
        out["passed"] = data["passed"]
    return out


def build_report():
    return {
        "generated": now_iso(),
        "coverage": coverage_metric(read_json(CACHE / "coverage.json")),
        "runtime": runtime_metric(
            read_json(CACHE / "runtime.json"), read_json(CACHE / "runtime-serial.json")
        ),
        "mutation": mutation_metric(read_json(CACHE / "mutation.json")),
        "flaky": flaky_metric(read_json(CACHE / "flaky.json")),
        "perf": perf_metric(read_json(CACHE / "perf.json")),
        "budgets": {
            "python_branch_target": budgets.PYTHON_BRANCH_TARGET,
            "rust_line_target": budgets.RUST_LINE_TARGET,
            "category_seconds": budgets.CATEGORY_BUDGETS,
        },
    }


def _pct(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}%"


def _stamp(metric):
    generated = metric.get("generated", "")
    return f"（{generated[:16].replace('T', ' ')}）" if generated else ""


def render_section(report, day):
    lines = [f"## {day}", ""]
    lines += ["| 指标 | 结果 |", "|---|---|"]
    cov = report["coverage"]
    if cov["status"] == NOT_RUN:
        lines.append(f"| Branch coverage | {NOT_RUN} |")
    else:
        py = cov.get("python")
        if py:
            lines.append(
                f"| Branch coverage（Python） | {_pct(py['branch_percent'])}"
                f"（{py['covered_branches']}/{py['branches']} 个分支；行 "
                f"{_pct(py['line_percent'])}；{py['files_below_target']} 个文件低于 "
                f"{py['target']:.0f}%）{_stamp(cov)} |"
            )
        else:
            lines.append(f"| Branch coverage（Python） | {NOT_RUN} |")
        rust = cov.get("rust")
        if rust:
            if rust["branch_measured"] and rust["branch_percent"] is not None:
                label, value = "Rust 分支覆盖", rust["branch_percent"]
            else:
                label, value = "Rust 行覆盖，stable 无分支数据", rust["line_percent"]
            lines.append(
                f"| Coverage（{label}） | {_pct(value)}，仅 `cargo test` 范围，"
                f"{rust['files_below_target']} 个文件低于目标 |"
            )
    rt = report["runtime"]
    if rt["status"] == NOT_RUN:
        lines.append(f"| Test runtime | {NOT_RUN} |")
    else:
        parts = []
        if "wall_seconds" in rt:
            parts.append(
                f"并行 {rt['wall_seconds']:.0f} s（{rt['workers']} 进程，"
                f"{rt['tests']} 个测试，测试耗时合计 {rt['test_seconds']:.0f} s）"
            )
        if "serial_wall_seconds" in rt:
            parts.append(f"串行 {rt['serial_wall_seconds']:.0f} s")
        # Per-test budgets are judged on the serial run when there is one:
        # parallel workers contend for CPU and inflate single-test times.
        if "serial_over_budget" in rt:
            parts.append(f"{len(rt['serial_over_budget'])} 个超出单测预算（串行计）")
        elif "over_budget" in rt:
            parts.append(f"{len(rt['over_budget'])} 个超出单测预算（并行计，偏高）")
        lines.append(f"| Test runtime | {'；'.join(parts)}{_stamp(rt)} |")
    mut = report["mutation"]
    if mut["status"] == NOT_RUN or mut.get("score") is None:
        lines.append(f"| Mutation score | {NOT_RUN} |")
    else:
        lines.append(
            f"| Mutation score | {_pct(100 * mut['score'])}（杀死 {mut['killed']}、"
            f"存活 {mut['survived']}、超时 {mut['timeout']}、等价 {mut['equivalent']}；"
            f"范围 {mut['scope']}）{_stamp(mut)} |"
        )
    fl = report["flaky"]
    if fl["status"] == NOT_RUN or fl.get("rate") is None:
        lines.append(f"| Flaky rate | {NOT_RUN} |")
    else:
        lines.append(
            f"| Flaky rate | {_pct(100 * fl['rate'], 3)}（{len(fl['flaky'])}/"
            f"{fl['tests']}，{fl['runs']} 轮随机顺序；一直失败 {len(fl['failing'])} 个）"
            f"{_stamp(fl)} |"
        )
    perf = report["perf"]
    if perf["status"] == NOT_RUN:
        lines.append(f"| 性能回归 | {NOT_RUN} |")
    else:
        regs = perf.get("regressions", [])
        count = perf.get("benchmarks")
        head = f"{count} 项基准，" if count is not None else ""
        lines.append(f"| 性能回归 | {head}{len(regs)} 项退步{_stamp(perf)} |")
    lines.append("")

    if cov["status"] != NOT_RUN and cov.get("python"):
        lines += ["分支覆盖最低的文件：", ""]
        for row in cov["python"]["lowest"]:
            lines.append(f"- `{row['path']}` {_pct(row['branch_percent'])}")
        lines.append("")
    if rt["status"] != NOT_RUN and rt.get("categories"):
        cats = "、".join(
            f"{name} {v['tests']} 个 {v['seconds']:.0f} s"
            for name, v in rt["categories"].items()
        )
        lines += [f"各类别（测试耗时合计）：{cats}", ""]
        lines += ["最慢的 5 个：", ""]
        for row in rt["slowest"][:5]:
            lines.append(f"- {row['seconds']:.1f} s `{row['id']}`")
        lines.append("")
    if fl["status"] != NOT_RUN and (fl["flaky"] or fl["failing"]):
        lines += ["时好时坏 / 一直失败：", ""]
        lines += [f"- flaky `{t}`" for t in fl["flaky"]]
        lines += [f"- failing `{t}`" for t in fl["failing"]]
        lines.append("")
    if mut["status"] != NOT_RUN and mut.get("modules"):
        worst = [m for m in mut["modules"] if m.get("score") is not None][:5]
        if worst:
            lines += [
                "变异分数最低的模块："
                + "、".join(f"{m['module']} {_pct(100 * m['score'])}" for m in worst),
                "",
            ]
    return "\n".join(lines)


SECTION = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def update_doc(text, section, day):
    """Replace the section for ``day`` or insert it, newest first."""
    if not text.strip():
        text = DOC_HEADER
    matches = list(SECTION.finditer(text))
    for i, match in enumerate(matches):
        if match.group(1) == day:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[: match.start()] + section.rstrip() + "\n\n" + text[end:]
    for match in matches:
        if match.group(1) < day:
            return (
                text[: match.start()]
                + section.rstrip()
                + "\n\n"
                + text[match.start() :]
            )
    return text.rstrip() + "\n\n" + section.rstrip() + "\n"


def print_report(report):
    print(render_section(report, "当前"))
    for key in ("coverage", "runtime", "mutation", "flaky", "perf"):
        if report[key]["status"] == NOT_RUN:
            print(f"{key}: {NOT_RUN}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    write_json(CACHE / "report.json", report)
    print_report(report)
    if not args.no_docs:
        old = METRICS_DOC.read_text() if METRICS_DOC.exists() else ""
        METRICS_DOC.write_text(
            update_doc(old, render_section(report, args.date), args.date).rstrip()
            + "\n"
        )
        print(f"\nupdated {METRICS_DOC.relative_to(ROOT)} ({args.date})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
