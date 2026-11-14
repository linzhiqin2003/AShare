"""Flaky-test detection: run the full suite N times in random order, in parallel.

    python bench/testing/flaky.py [--runs N] [--seed S] [--workers W]

Run ``i`` uses random seed ``S + i`` (pytest-randomly) and pytest-xdist. A test
is *flaky* when it passed in at least one run and failed or errored in another;
*failing* when it never passed. flaky rate = flaky tests / distinct tests.
Writes ``.cache/testing/flaky.json``; the junit file of every run is kept under
``.cache/testing/flaky/``.
"""

import argparse
import sys
from collections import defaultdict

import budgets
from common import (
    CACHE,
    count_outcomes,
    now_iso,
    parse_junit,
    pytest_command,
    run,
    write_json,
)


def aggregate(runs):
    """``runs``: list of record lists, one per run. Returns the flaky summary."""
    seen = defaultdict(list)
    messages = defaultdict(list)
    for index, records in enumerate(runs):
        for record in records:
            seen[record["id"]].append(record["outcome"])
            if record["outcome"] in ("failed", "error") and record["message"]:
                messages[record["id"]].append(f"run {index}: {record['message']}")
    flaky, failing, partial = [], [], []
    for test_id, outcomes in sorted(seen.items()):
        bad = sum(o in ("failed", "error") for o in outcomes)
        good = sum(o == "passed" for o in outcomes)
        row = {
            "id": test_id,
            "outcomes": outcomes,
            "messages": messages.get(test_id, [])[:5],
        }
        if bad and good:
            flaky.append(row)
        elif bad:
            failing.append(row)
        if len(outcomes) != len(runs):
            partial.append(test_id)
    total = len(seen)
    return {
        "tests": total,
        "flaky": flaky,
        "failing": failing,
        "missing_in_some_runs": partial,
        "flaky_rate": (len(flaky) / total) if total else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=budgets.FLAKY_RUNS)
    parser.add_argument("--seed", type=int, default=budgets.FLAKY_BASE_SEED)
    parser.add_argument("--workers", type=int, default=budgets.WORKERS)
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    directory = CACHE / "flaky"
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("run-*.xml"):
        old.unlink()
    runs, meta = [], []
    for index in range(args.runs):
        seed = args.seed + index
        junit = directory / f"run-{index}.xml"
        cmd = pytest_command(
            workers=args.workers,
            junit=junit,
            randomly_seed=seed,
            extra=args.pytest_args,
        )
        code, wall = run(cmd, timeout=budgets.PYTEST_TIMEOUT)
        records = parse_junit(junit) if junit.exists() else []
        runs.append(records)
        meta.append(
            {
                "run": index,
                "seed": seed,
                "exit_code": code,
                "wall_seconds": round(wall, 1),
                "outcomes": count_outcomes(records),
                "complete": junit.exists(),
            }
        )
        print(f"run {index} seed {seed}: exit {code}, {meta[-1]['outcomes']}")

    summary = aggregate(runs)
    summary.update(
        generated=now_iso(),
        runs=meta,
        workers=args.workers,
        base_seed=args.seed,
        scope="full" if not args.pytest_args else " ".join(args.pytest_args),
    )
    write_json(CACHE / "flaky.json", summary)

    rate = summary["flaky_rate"]
    print()
    print(
        f"flaky: {len(summary['flaky'])} of {summary['tests']} tests (rate {rate:.4%})"
        if rate is not None
        else "flaky: no tests recorded"
    )
    for row in summary["flaky"]:
        print(f"  FLAKY   {row['id']}  {row['outcomes']}")
    for row in summary["failing"]:
        print(f"  FAILING {row['id']}")
    incomplete = [m for m in meta if not m["complete"]]
    if incomplete:
        print(f"incomplete runs: {[m['run'] for m in incomplete]}", file=sys.stderr)
    return 1 if summary["flaky"] or summary["failing"] or incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
