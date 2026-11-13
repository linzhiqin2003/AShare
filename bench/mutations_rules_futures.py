"""Mutation check for exit rules in mixed spot + futures panels.

Usage: python bench/mutations_rules_futures.py   (rebuilds once per mutation).
Same harness as bench/mutations_rules.py: edit the kernel, binding or the
Python checks, rebuild, run the mixed-rule tests (the conftest build guard
rejects a stale extension), restore and rebuild at the end. Every mutation
must make a test fail. Logs and summary in .cache/rework/mutations-rules-futures/.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
RULES = ROOT / "crates/panelbt-core/src/rules.rs"
API = ROOT / "packages/panelbt/python/panelbt/api.py"
STEP = ROOT / "packages/panelbt/python/panelbt/step.py"
OUT = ROOT / ".cache/rework/mutations-rules-futures"
TESTS = ["test/test_exit_rules_futures.py", "test/test_exit_rules.py"]

CASES = [
    (
        "rule trim ignores the futures collateral",
        RULES,
        "let own = |b: Account| b.own_cash(s.flags[4]) - fm;",
        "let own = |b: Account| b.own_cash(s.flags[4]);",
    ),
    (
        "close/average rule phase sees no collateral",
        CORE,
        "day, shares, *bankrupt, *fm_before, m, s, r, st, trk, rb, out,",
        "day, shares, *bankrupt, 0.0, m, s, r, st, trk, rb, out,",
    ),
    (
        "open rule phase uses the pre-trade collateral",
        CORE,
        "rules::intraday::<CLASSIC>(day, shares, *bankrupt, fm, m, s, r, st, trk, rb, out);",
        "rules::intraday::<CLASSIC>(day, shares, *bankrupt, *fm_before, m, s, r, st, trk, rb, out);",
    ),
    (
        "futures + rules dispatched to the untracked build",
        CORE,
        "(true, true, true) => run_impl::<true, true, true>(m, targets, s),",
        "(true, true, true) => run_impl::<true, true, false>(m, targets, s),",
    ),
    (
        "rule percentages on futures columns accepted",
        API,
        'if np.any(bad):\n                raise ValueError(\n                    f"{k}: rules act',
        'if False:\n                raise ValueError(\n                    f"{k}: rules act',
    ),
    (
        "limit rows accepted by the simulator with futures",
        STEP,
        "and self._market.futures is not None",
        "and False",
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
    originals = {x: x.read_text() for x in (CORE, RULES, API, STEP)}
    results = []
    try:
        for index, (name, path, old, new) in enumerate(CASES, 1):
            data = originals[path]
            assert data.count(old) == 1, (name, old)
            path.write_text(data.replace(old, new))
            if path.suffix == ".rs":
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
            if path.suffix == ".rs":
                build(f"{index}-restore.log")
            failed = sorted(set(re.findall(r"FAILED \S+::(\w+)", log)))
            caught = code == 1 and bool(failed) and "extension is stale" not in log
            results.append({"mutation": name, "caught": caught, "failed": failed})
            print(("caught " if caught else "MISSED ") + name, failed, flush=True)
    finally:
        for path, text in originals.items():
            path.write_text(text)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    assert all(r["caught"] for r in results), [r for r in results if not r["caught"]]


if __name__ == "__main__":
    main()
