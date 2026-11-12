"""Mutation check for the exit-rule kernel: each mutation must fail a test.

Usage: python bench/mutations_rules.py   (rebuilds once per mutation; slow).
Every mutation edits the Rust kernel only, rebuilds, runs
test/test_exit_rules.py (the build guard in conftest rejects a stale
extension) and must make at least one test fail. Sources are restored and
rebuilt at the end. Logs and a summary go to .cache/rework/mutations-rules/.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
RULES = ROOT / "crates/panelbt-core/src/rules.rs"
OUT = ROOT / ".cache/rework/mutations-rules"
TESTS = "test/test_exit_rules.py"

CASES = [
    (
        "gap through the stop fills at the stop, not the open",
        RULES,
        "if o <= stop {\n                Some((why, o))",
        "if o <= stop {\n                Some((why, stop))",
    ),
    (
        "both-hit default becomes optimistic",
        RULES,
        "(true, true) if r.both_target =>",
        "(true, true) if !r.both_target =>",
    ),
    (
        "entry row's range extends the trailing peak",
        CORE,
        "if !rb.fresh[i] {\n                    let k = base + i;",
        "if true {\n                    let k = base + i;",
    ),
    (
        "trailing line from cost instead of peak",
        RULES,
        "peak * (1.0 - tr)",
        "cost * (1.0 - tr)",
    ),
    (
        "sell lock ignored",
        RULES,
        "if r.sell_lock && cur > 0.0 {",
        "if false && r.sell_lock && cur > 0.0 {",
    ),
    (
        "unfinished exit is dropped instead of retried",
        RULES,
        "rb.due[i] = reason;",
        "rb.due[i] = 0;",
    ),
    (
        "re-entry block omitted",
        RULES,
        "pub fn blocks(&self, i: usize, day: usize, r: &Rules<'_>) -> bool {",
        "pub fn blocks(&self, i: usize, day: usize, r: &Rules<'_>) -> bool {\n"
        "        if day < usize::MAX { return false; }",
    ),
    (
        "limit buy fills at the limit, not min(open, limit)",
        RULES,
        "(valid && l <= limit, o.min(limit))",
        "(valid && l <= limit, limit)",
    ),
    (
        "max_hold off by one",
        RULES,
        "day as i64 - trk.entry[i] >= r.max_hold",
        "day as i64 - trk.entry[i] > r.max_hold",
    ),
    (
        "dividend option leaves the cost unchanged",
        CORE,
        "trk.cost[i] = (trk.cost[i] - m.dividend[base + i]).max(0.0);",
        "trk.cost[i] = (trk.cost[i] - 0.0 * m.dividend[base + i]).max(0.0);",
    ),
]


def command(args, name, timeout):
    path = OUT / name
    with path.open("w") as stream:
        status = subprocess.run(
            args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
        ).returncode
    return status, path.read_text()


def build(name):
    code, log = command(["make", "dev"], name, 1800)
    assert code == 0, (name, log[-2000:])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    originals = {x: x.read_text() for x in (CORE, RULES)}
    results = []
    try:
        for index, (name, path, old, new) in enumerate(CASES, 1):
            data = originals[path]
            assert old in data, (name, old)
            path.write_text(data.replace(old, new))
            build(f"{index}-build.log")
            code, log = command(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    TESTS,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                f"{index}-test.log",
                1800,
            )
            path.write_text(data)
            failed = sorted(set(re.findall(r"FAILED \S+::(\w+)", log)))
            caught = code == 1 and bool(failed) and "extension is stale" not in log
            results.append({"mutation": name, "caught": caught, "failed": failed})
            print(("caught " if caught else "MISSED ") + name, failed, flush=True)
    finally:
        for path, text in originals.items():
            path.write_text(text)
        build("restore-build.log")
    (OUT / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    assert all(r["caught"] for r in results), [r for r in results if not r["caught"]]


if __name__ == "__main__":
    main()
