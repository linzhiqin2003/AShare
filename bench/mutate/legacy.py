"""Hand-written mutation scripts from earlier review rounds, run and parsed from one place.

They are kept as a supplement to mutmut / cargo-mutants, not folded into the score:
each case is a hand-picked semantic bug (tax omitted, carry touching every holding,
reference and kernel broken together) tied to the test meant to catch it, which the
operator-level tools do not generate. Their counts go to ``mutation_legacy.json``.

Most of them edit the working tree in place and rebuild the extension with ``make dev``,
so they run one at a time, only on a clean tree, and never next to a mutmut or
cargo-mutants run of the same checkout. ``state`` says whether a script can run
unattended today; the reason is recorded instead of silently skipping it.

    python -m bench.mutate.legacy list
    python -m bench.mutate.legacy run NAME [NAME ...]   # runnable ones only
    python -m bench.mutate.legacy parse                 # re-read existing outputs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass

from bench.mutate.common import PYTHON, ROOT, TESTING, be_nice, child_env, git, log, now

RUNNABLE = "runnable"


@dataclass
class Script:
    name: str
    command: list[str]
    output: (
        str | None
    )  # repo-relative JSON written by the script; None = JSON lines on stdout
    state: str  # RUNNABLE or the reason it cannot run unattended
    in_place: bool = True  # edits the working tree
    tracked_output: bool = False  # overwrites a file under version control
    complements: str = ""  # shard it adds hand-picked cases to


PY = str(PYTHON)
SCRIPTS = [
    Script(
        "panelbt-review",
        [PY, "bench/mutations.py"],
        "bench/results/review-mutations.json",
        "stale: 17 of 40 patterns no longer match; needs SANDBOX cargo home (network)",
        tracked_output=True,
        complements="panelbt-engine",
    ),
    Script(
        "panelbt-1c",
        [PY, "bench/mutations_1c.py"],
        "bench/results/step1c-mutations.json",
        "needs SANDBOX=1 cargo home under .cache (downloads crates)",
        tracked_output=True,
        complements="panelbt-engine",
    ),
    Script(
        "panelbt-1c-rework",
        [PY, "bench/mutations_1c_rework.py"],
        None,
        "superseded: needs --root copy; Python-only cases after a Rust case run against "
        "the stale extension and are recorded as caught",
        complements="panelbt-engine",
    ),
    Script(
        "futures-book",
        [PY, "bench/mutations_futures_book.py"],
        "bench/results/mutations_futures_book.json",
        RUNNABLE,
        tracked_output=True,
        complements="panelbt-py",
    ),
    Script(
        "rules",
        [PY, "bench/mutations_rules.py"],
        ".cache/rework/mutations-rules/summary.json",
        RUNNABLE,
        complements="panelbt-rules",
    ),
    Script(
        "rules-futures",
        [PY, "bench/mutations_rules_futures.py"],
        ".cache/rework/mutations-rules-futures/summary.json",
        "restores sources without rebuilding on failure (stale extension risk)",
        complements="panelbt-rules",
    ),
    Script(
        "panelops",
        [PY, "bench/panelops/mutations.py"],
        ".cache/panelops/mutations/results.json",
        "stale: 9 patterns no longer match; rebuild runs `uv sync` with bench groups "
        "(changes the venv, network)",
        complements="panelops-time",
    ),
    Script(
        "panelops-exact",
        [PY, "bench/panelops/exact_mutations.py"],
        None,
        RUNNABLE,
        complements="panelops-time",
    ),
    Script(
        "risk",
        [PY, "bench/risk/mutations.py"],
        ".cache/risk-mutations/mutations.json",
        "stale: 6 of 75 patterns no longer match; Rust cases use a fresh cargo home (network)",
        complements="ashare-risk",
    ),
    Script(
        "risk-perf",
        [PY, "bench/risk/perf_mutations.py"],
        None,
        RUNNABLE,
        complements="ashare-risk",
    ),
    Script(
        "mining",
        [PY, "bench/mining/mutations.py"],
        "bench/mining/results/mutations.json",
        RUNNABLE,
        tracked_output=True,
        complements="ashare-factor",
    ),
    Script(
        "ml",
        [PY, "bench/ml/mutations.py"],
        "bench/ml/results/mutations.json",
        RUNNABLE,
        tracked_output=True,
        complements="ashare-ml",
    ),
    Script(
        "data-round3",
        [PY, "bench/data/round3/mutations.py"],
        None,
        RUNNABLE,
        in_place=False,
        complements="ashare-data",
    ),
    Script(
        "groupbt",
        [PY, "-m", "bench.groupbt.mutate"],
        ".cache/groupbt/mutation.json",
        RUNNABLE,
        complements="panelbt-weights",
    ),
    Script(
        "groupbt-shared",
        [PY, "-m", "bench.groupbt.shared_mutations"],
        ".cache/groupbt-fusion/mutations.json",
        RUNNABLE,
        complements="panelbt-group-weights",
    ),
    Script(
        "groupbt-review",
        [PY, "-m", "bench.groupbt.review_mutations"],
        ".cache/groupbt-review/mutations.json",
        RUNNABLE,
        complements="panelbt-group-weights",
    ),
    Script(
        "paper",
        [PY, "test/paper/mutations.py"],
        ".cache/round3/mutation-results.json",
        "stale: expects one `overflow-checks = true` in Cargo.toml, there are two; "
        "TMPDIR points at a directory it never creates",
        complements="paperbroker-core",
    ),
]
BY_NAME = {s.name: s for s in SCRIPTS}
REPORT = TESTING / "mutation_legacy.json"


def tally(data):
    """(caught, missed) from any of the scripts' result shapes."""
    caught = missed = 0

    def visit(node):
        nonlocal caught, missed
        if isinstance(node, dict):
            if (
                isinstance(node.get("caught"), bool)
                and node.get("name") != "control (no edit)"
            ):
                caught += node["caught"]
                missed += not node["caught"]
            elif node.get("status") in ("KILLED", "SURVIVED"):
                caught += node["status"] == "KILLED"
                missed += node["status"] == "SURVIVED"
            elif isinstance(node.get("python_caught"), bool):  # groupbt review
                for key in ("python_caught", "rust_caught"):
                    caught += bool(node.get(key))
                    missed += node.get(key) is False
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(data)
    if not caught and not missed and isinstance(data, dict) and "mutations" in data:
        # bench/mutations.py writes its JSON only when every case was killed
        caught = len(data["mutations"])
    return caught, missed


def parse_stdout(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def parse(script: Script, stdout: str | None = None):
    if script.output:
        path = ROOT / script.output
        if not path.exists():
            return None
        data = json.loads(path.read_text())
    elif stdout is not None:
        data = parse_stdout(stdout)
    else:
        return None
    caught, missed = tally(data)
    out = {"caught": caught, "missed": missed}
    if script.output:  # may be a committed result of an earlier round, not a fresh run
        out["from"] = script.output
    return out


def run(script: Script, timeout: float):
    if script.state != RUNNABLE:
        log(f"{script.name}: not run ({script.state})")
        return {"state": script.state}
    if script.in_place and git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(
            f"{script.name} edits sources in place: commit or stash changes first"
        )
    log(f"{script.name}: {' '.join(script.command[1:])}")
    started = time.monotonic()
    proc = subprocess.run(
        script.command,
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (ROOT / ".cache/mutate/legacy").mkdir(parents=True, exist_ok=True)
    (ROOT / f".cache/mutate/legacy/{script.name}.log").write_text(
        proc.stdout + proc.stderr
    )
    result = parse(script, proc.stdout) or {}
    result.update(exit=proc.returncode, seconds=round(time.monotonic() - started, 1))
    if script.tracked_output:
        # keep the committed copy; the fresh result lives on in the report and the log
        git("checkout", "--", script.output, check=False)
    if script.in_place and git("diff", "--stat").strip():
        log(f"WARNING {script.name} left the tree modified:\n{git('diff', '--stat')}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+")
    r.add_argument("--timeout", type=float, default=3600)
    sub.add_parser("parse")
    args = p.parse_args(argv)
    be_nice()
    if args.command == "list":
        for s in SCRIPTS:
            print(f"{s.name:16s} {s.complements:22s} {s.state}")
        return
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"scripts": {}}
    if args.command == "parse":
        for s in SCRIPTS:
            got = parse(s)
            report["scripts"][s.name] = {
                "state": s.state,
                "complements": s.complements,
                **(got or {"result": "no output on disk"}),
            }
    else:
        for name in args.names:
            if name not in BY_NAME:
                sys.exit(f"unknown script {name!r}; see `legacy list`")
            s = BY_NAME[name]
            report["scripts"][name] = {
                "state": s.state,
                "complements": s.complements,
                **run(s, args.timeout),
            }
    report["generated"] = now()
    TESTING.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    log(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
