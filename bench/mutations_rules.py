"""Mutation check for the exit-rule kernel: each mutation must fail a test.

Usage: python bench/mutations_rules.py   (rebuilds once per mutation; slow).
Every mutation edits the Rust kernel or its binding, rebuilds, runs the rule
and stepping tests (the build guard in conftest rejects a stale extension)
and must make at least one test fail. Sources are restored and
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
BIND = ROOT / "packages/panelbt/src/lib.rs"
OUT = ROOT / ".cache/rework/mutations-rules"
TESTS = ["test/test_exit_rules.py", "test/test_exit_rules_hand.py", "test/test_step.py"]

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
        "o.min(limit),",
        "limit,",
    ),
    (
        "limit buy fills on a touch by default",
        RULES,
        "(l < limit || (r.limit_touch && l == limit))",
        "(l <= limit || (r.limit_touch && l == limit))",
    ),
    (
        "account-limited limit buy not counted as cash_scaled",
        RULES,
        "out.counts[3] += 1;",
        "out.counts[3] += 0;",
    ),
    (
        "snapshot checksum not verified",
        CORE,
        "if codec::crc32(&blob[..blob.len() - 4]).to_le_bytes() != crc {",
        "if false && codec::crc32(&blob[..blob.len() - 4]).to_le_bytes() != crc {",
    ),
    (
        "rule percentage arrays left out of the snapshot fingerprint",
        BIND,
        "for a in self.rule_arrays.iter().skip(3) {",
        "for a in self.rule_arrays.iter().skip(6) {",
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
    originals = {x: x.read_text() for x in (CORE, RULES, BIND)}
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
                    *TESTS,
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
