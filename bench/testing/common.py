"""Shared helpers for the test-metric tools (paths, pytest runs, junit parsing)."""

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import budgets

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "testing"

OUTCOMES = ("passed", "failed", "error", "skipped")


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path):
    """Path relative to the repository root, for output that may be published."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def pytest_command(
    *, workers=None, junit=None, randomly_seed=None, marker=None, extra=()
):
    """Build a pytest command line for the project's own interpreter."""
    workers = budgets.WORKERS if workers is None else workers
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += [f"--timeout={budgets.TEST_TIMEOUT}"]
    if workers and int(workers) > 0:
        cmd += ["-n", str(workers)]
    if randomly_seed is not None:
        cmd += ["-p", "randomly", f"--randomly-seed={randomly_seed}"]
    if junit is not None:
        cmd += [f"--junitxml={junit}", "-o", "junit_family=xunit2"]
    if marker:
        cmd += ["-m", marker]
    cmd += list(extra)
    return cmd


def run(cmd, *, timeout, env=None, cwd=ROOT):
    """Run a command with a hard timeout; return (exit code, wall seconds)."""
    print("+", " ".join(str(c) for c in cmd), flush=True)
    start = time.perf_counter()
    try:
        code = subprocess.run(
            cmd, cwd=cwd, env=env, timeout=timeout, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"command timed out after {timeout} s", file=sys.stderr)
        code = 124
    return code, time.perf_counter() - start


def parse_junit(path):
    """Return one record per testcase: id, file, category, time, outcome, message."""
    records = []
    tree = ET.parse(path)
    for case in tree.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        props = {
            p.get("name"): p.get("value")
            for p in case.iter("property")
            if p.get("name") is not None
        }
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = "failed" if tag == "failure" else tag
                message = (node.get("message") or "").strip().splitlines()[:1]
                message = message[0][:200] if message else ""
                break
        records.append(
            {
                "id": f"{classname}::{name}" if classname else name,
                "file": case.get("file") or classname.replace(".", "/"),
                "category": props.get("category", "unknown"),
                "time": float(case.get("time") or 0.0),
                "outcome": outcome,
                "message": message,
            }
        )
    return records


def count_outcomes(records):
    counts = dict.fromkeys(OUTCOMES, 0)
    for record in records:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    return counts
