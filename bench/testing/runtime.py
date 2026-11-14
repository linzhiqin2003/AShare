"""Run the test suite (parallel by default) and report test runtime.

    python bench/testing/runtime.py [--workers N] [--category CAT] [-- pytest args]

Writes ``.cache/testing/runtime.json`` for full parallel runs
(``runtime-serial.json`` with ``--workers 0``; nothing for filtered runs):
wall time, summed test time, per-category time, the slowest tests and the
tests over their category budget (``budgets.CATEGORY_BUDGETS``).
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


def summarize(records, wall):
    by_category = defaultdict(lambda: {"tests": 0, "seconds": 0.0})
    for record in records:
        bucket = by_category[record["category"]]
        bucket["tests"] += 1
        bucket["seconds"] += record["time"]
    slowest = sorted(records, key=lambda r: r["time"], reverse=True)
    over = [
        {
            "id": r["id"],
            "category": r["category"],
            "seconds": round(r["time"], 3),
            "budget": budgets.CATEGORY_BUDGETS.get(r["category"]),
        }
        for r in slowest
        if r["category"] in budgets.CATEGORY_BUDGETS
        and r["time"] > budgets.CATEGORY_BUDGETS[r["category"]]
    ]
    return {
        "wall_seconds": round(wall, 1),
        "test_seconds": round(sum(r["time"] for r in records), 1),
        "tests": len(records),
        "outcomes": count_outcomes(records),
        "categories": {
            name: {"tests": v["tests"], "seconds": round(v["seconds"], 1)}
            for name, v in sorted(by_category.items())
        },
        "slowest": [
            {"id": r["id"], "category": r["category"], "seconds": round(r["time"], 3)}
            for r in slowest[: budgets.SLOWEST]
        ],
        "over_budget": over,
        "budgets": budgets.CATEGORY_BUDGETS,
    }


def print_summary(summary):
    print()
    print(
        f"runtime: {summary['wall_seconds']} s wall, "
        f"{summary['test_seconds']} s summed over {summary['tests']} tests "
        f"(workers={summary.get('workers')})"
    )
    print("outcomes:", summary["outcomes"])
    print("by category:")
    for name, value in summary["categories"].items():
        print(f"  {name:<13}{value['tests']:>6} tests {value['seconds']:>9.1f} s")
    print(f"slowest {len(summary['slowest'])}:")
    for row in summary["slowest"]:
        print(f"  {row['seconds']:>8.2f} s  [{row['category']}] {row['id']}")
    over = summary["over_budget"]
    print(f"over budget: {len(over)}")
    for row in over[:50]:
        print(f"  {row['seconds']:>8.2f} s > {row['budget']} s  {row['id']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=budgets.WORKERS)
    parser.add_argument("--category", default="")
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.category}" if args.category else ""
    junit = CACHE / f"junit-runtime{suffix}.xml"
    junit.unlink(missing_ok=True)
    cmd = pytest_command(
        workers=args.workers,
        junit=junit,
        marker=args.category or None,
        extra=args.pytest_args,
    )
    code, wall = run(cmd, timeout=budgets.PYTEST_TIMEOUT)
    if not junit.exists():
        print("no junit output; pytest did not finish", file=sys.stderr)
        return code or 1
    summary = summarize(parse_junit(junit), wall)
    summary.update(
        generated=now_iso(),
        workers=args.workers,
        category=args.category or "all",
        exit_code=code,
    )
    print_summary(summary)
    if not args.category and not args.pytest_args:
        name = "runtime.json" if args.workers > 0 else "runtime-serial.json"
        write_json(CACHE / name, summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
