"""Branch coverage for Python and line coverage for Rust.

    python bench/testing/coverage_run.py [--workers N] [--skip-python] [--skip-rust]

Python: coverage.py with ``branch = true`` (config in ``.coveragerc``), run
under pytest-xdist; the workers are measured through coverage's subprocess
patching and combined afterwards. HTML report: ``.cache/testing/htmlcov``.

Rust: ``cargo llvm-cov --workspace`` over ``cargo test``. Branch coverage needs a
nightly toolchain; on stable only line, region and function coverage are
reported. Scope: Rust code reached by ``cargo test`` only. The PyO3 bindings
and the kernels exercised only through the Python tests are not instrumented
in that run, so binding crates read low here even when Python tests cover them.

Writes ``.cache/testing/coverage.json`` (totals, per-file rows, files below the
targets in ``budgets``).
"""

import argparse
import os
import shutil
import subprocess
import sys

import budgets
from common import CACHE, ROOT, now_iso, pytest_command, read_json, run, write_json

PY_JSON = CACHE / "coverage-python.json"
RUST_JSON = CACHE / "coverage-rust.json"


def python_rows(data):
    rows = []
    for path, info in data.get("files", {}).items():
        s = info["summary"]
        rows.append(_python_row(path, s))
    rows.sort(key=lambda r: (r["branch_percent"], r["path"]))
    return rows


def _python_row(path, s):
    branches = s.get("num_branches", 0)
    covered = s.get("covered_branches", 0)
    return {
        "path": path,
        "statements": s.get("num_statements", 0),
        "missing_lines": s.get("missing_lines", 0),
        "branches": branches,
        "covered_branches": covered,
        "line_percent": round(
            100.0 * s.get("covered_lines", 0) / s["num_statements"], 1
        )
        if s.get("num_statements")
        else 100.0,
        "branch_percent": round(100.0 * covered / branches, 1) if branches else 100.0,
        "percent": round(s.get("percent_covered", 100.0), 1),
    }


def python_module(path):
    """Group a file path into a module bucket (``ashare.data``, ``panelbt``, ...)."""
    parts = path.split("/")
    if parts[:2] == ["src", "ashare"]:
        inner = parts[2:]
        return "ashare" if len(inner) <= 1 else f"ashare.{inner[0]}"
    if parts[0] == "packages" and len(parts) > 1:
        return parts[1]
    return parts[0]


def python_modules(data):
    buckets = {}
    for path, info in data.get("files", {}).items():
        s = info["summary"]
        b = buckets.setdefault(
            python_module(path),
            {"statements": 0, "covered_lines": 0, "branches": 0, "covered": 0},
        )
        b["statements"] += s.get("num_statements", 0)
        b["covered_lines"] += s.get("covered_lines", 0)
        b["branches"] += s.get("num_branches", 0)
        b["covered"] += s.get("covered_branches", 0)
    return {
        name: {
            "branch_percent": round(100.0 * b["covered"] / b["branches"], 1)
            if b["branches"]
            else 100.0,
            "line_percent": round(100.0 * b["covered_lines"] / b["statements"], 1)
            if b["statements"]
            else 100.0,
            "branches": b["branches"],
            "statements": b["statements"],
        }
        for name, b in sorted(buckets.items())
    }


def python_totals(data):
    t = data.get("totals", {})
    branches = t.get("num_branches", 0)
    covered = t.get("covered_branches", 0)
    return {
        "branch_percent": round(100.0 * covered / branches, 2) if branches else None,
        "line_percent": round(100.0 * t["covered_lines"] / t["num_statements"], 2)
        if t.get("num_statements")
        else None,
        "combined_percent": round(t.get("percent_covered", 0.0), 2),
        "branches": branches,
        "covered_branches": covered,
        "statements": t.get("num_statements", 0),
    }


def run_python(workers):
    data_dir = CACHE / "coverage"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COVERAGE_RCFILE=str(ROOT / ".coveragerc"))
    cov = [sys.executable, "-m", "coverage"]
    cmd = cov + ["run", "-m"] + pytest_command(workers=workers)[2:]
    code, wall = run(cmd, timeout=budgets.PYTEST_TIMEOUT, env=env)
    for step in (["combine", "--quiet"], ["json", "--quiet"], ["html", "--quiet"]):
        step_code, _ = run(cov + step, timeout=1800, env=env)
        if step_code:
            print(f"coverage {step[0]} failed ({step_code})", file=sys.stderr)
            return None, code or step_code, wall
    return read_json(PY_JSON), code, wall


def rust_rows(data):
    rows = []
    for export in data.get("data", []):
        for f in export.get("files", []):
            lines = f["summary"]["lines"]
            rel_path = os.path.relpath(f["filename"], ROOT)
            rows.append(
                {
                    "path": rel_path,
                    "lines": lines["count"],
                    "covered_lines": lines["covered"],
                    "line_percent": round(lines["percent"], 1),
                }
            )
    rows.sort(key=lambda r: (r["line_percent"], r["path"]))
    return rows


def rust_module(path):
    parts = path.split("/")
    return parts[1] if len(parts) > 1 and parts[0] in ("crates", "packages") else path


def rust_modules(rows):
    buckets = {}
    for row in rows:
        b = buckets.setdefault(rust_module(row["path"]), [0, 0])
        b[0] += row["lines"]
        b[1] += row["covered_lines"]
    return {
        name: {
            "line_percent": round(100.0 * c / n, 1) if n else 100.0,
            "lines": n,
        }
        for name, (n, c) in sorted(buckets.items())
    }


def rust_totals(data):
    totals = data.get("data", [{}])[0].get("totals", {})
    out = {}
    names = {"lines": "line", "regions": "region", "functions": "function"}
    names["branches"] = "branch"
    for key, singular in names.items():
        if key in totals and totals[key].get("count"):
            out[f"{singular}_percent"] = round(totals[key]["percent"], 2)
            out[key] = totals[key]["count"]
    return out


def nightly_available():
    try:
        out = subprocess.run(
            ["rustup", "toolchain", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "nightly" in out


def run_rust():
    if shutil.which("cargo-llvm-cov") is None and not (
        os.path.exists(os.path.expanduser("~/.cargo/bin/cargo-llvm-cov"))
    ):
        print("cargo-llvm-cov not installed; skipping Rust coverage", file=sys.stderr)
        return None, None
    branch = nightly_available()
    cmd = ["cargo"]
    if branch:
        cmd.append("+nightly")
    cmd += [
        "llvm-cov",
        "--workspace",
        "--jobs",
        str(budgets.CARGO_JOBS),
        "--json",
        "--summary-only",
        "--output-path",
        str(RUST_JSON),
    ]
    if branch:
        cmd.append("--branch")
    code, _ = run(cmd, timeout=budgets.CARGO_TIMEOUT)
    if code:
        print(f"cargo llvm-cov failed ({code})", file=sys.stderr)
        return None, branch
    return read_json(RUST_JSON), branch


def print_report(result):
    py = result.get("python")
    if py:
        t = py["totals"]
        print()
        print(
            f"Python branch coverage {t['branch_percent']}% "
            f"({t['covered_branches']}/{t['branches']} branches), "
            f"line {t['line_percent']}%"
        )
        print("by module (branch %, line %):")
        for name, m in py["modules"].items():
            flag = " <" if m["branch_percent"] < budgets.PYTHON_BRANCH_TARGET else ""
            print(
                f"  {name:<24}{m['branch_percent']:>7.1f}{m['line_percent']:>8.1f}"
                f"{flag}"
            )
        print(
            f"files below {budgets.PYTHON_BRANCH_TARGET}% branch: "
            f"{len(py['below_target'])}; lowest {budgets.LOWEST_FILES}:"
        )
        for row in py["lowest"]:
            print(
                f"  {row['branch_percent']:>6.1f}%  "
                f"({row['covered_branches']}/{row['branches']})  {row['path']}"
            )
    rust = result.get("rust")
    if rust:
        t = rust["totals"]
        kind = "branch + line" if rust["branch"] else "line only (no nightly)"
        print()
        print(f"Rust coverage [{kind}, cargo test scope]: {t}")
        for name, m in rust["modules"].items():
            flag = " <" if m["line_percent"] < budgets.RUST_LINE_TARGET else ""
            print(f"  {name:<24}{m['line_percent']:>7.1f}{flag}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=budgets.WORKERS)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    args = parser.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)

    previous = read_json(CACHE / "coverage.json") or {}
    result = {"generated": now_iso()}
    exit_code = 0
    if not args.skip_python:
        data, code, wall = run_python(args.workers)
        exit_code = exit_code or code
        if data:
            rows = python_rows(data)
            result["python"] = {
                "totals": python_totals(data),
                "modules": python_modules(data),
                "target": budgets.PYTHON_BRANCH_TARGET,
                "below_target": [
                    r["path"]
                    for r in rows
                    if r["branch_percent"] < budgets.PYTHON_BRANCH_TARGET
                ],
                "lowest": [r for r in rows if r["branches"]][: budgets.LOWEST_FILES],
                "files": rows,
                "pytest_exit_code": code,
                "wall_seconds": round(wall, 1),
            }
    elif "python" in previous:
        result["python"] = previous["python"]
    if not args.skip_rust:
        data, branch = run_rust()
        if data:
            rows = rust_rows(data)
            result["rust"] = {
                "totals": rust_totals(data),
                "branch": bool(branch),
                "scope": "cargo test --workspace (PyO3 paths exercised only by "
                "Python tests are not counted)",
                "modules": rust_modules(rows),
                "target": budgets.RUST_LINE_TARGET,
                "below_target": [
                    r["path"]
                    for r in rows
                    if r["line_percent"] < budgets.RUST_LINE_TARGET
                ],
                "lowest": rows[: budgets.LOWEST_FILES],
                "files": rows,
            }
        else:
            exit_code = exit_code or 1
    elif "rust" in previous:
        result["rust"] = previous["rust"]
    write_json(CACHE / "coverage.json", result)
    print_report(result)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
