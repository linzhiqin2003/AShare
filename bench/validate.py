"""Replay configurable synthetic corpora against the reference and independent ledger."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import panelbt as bt  # noqa: E402
from panelbt import reference  # noqa: E402

from test.build_guard import assert_current_build  # noqa: E402
from test.comparison import assert_results_equal  # noqa: E402
from test.generators import case  # noqa: E402
from test.generators_futures import case as futures_case  # noqa: E402
from test.ledger import audit  # noqa: E402
from test.test_extended import extended_case  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", choices=["spot", "extended", "futures"], default="spot"
    )
    parser.add_argument("--start", type=int, default=40000)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/panelbt/validation.json"
    )
    args = parser.parse_args()
    fingerprint = assert_current_build()
    generator = {
        "spot": case,
        "extended": extended_case,
        "futures": futures_case,
    }[args.corpus]
    totals = Counter(
        cases=0, trades=0, divergences=0, ledger_failures=0, opening_checks=0
    )
    failures = []
    for seed in range(args.start, args.start + args.count):
        inputs = generator(seed)
        actual, expected = bt.run(*inputs), reference.run(*inputs)
        try:
            assert_results_equal(actual, expected)
        except AssertionError as exc:
            totals["divergences"] += 1
            failures.append(dict(seed=seed, check="reference", error=str(exc)))
        try:
            totals["opening_checks"] += audit(inputs, actual)
        except AssertionError as exc:
            totals["ledger_failures"] += 1
            failures.append(dict(seed=seed, check="ledger", error=str(exc)))
        totals["cases"] += 1
        totals["trades"] += len(actual.trades)
        totals.update(actual.coverage)
        if totals["cases"] % 2000 == 0:
            print(dict(totals), flush=True)
    result = dict(
        corpus=args.corpus,
        start=args.start,
        counts=dict(totals),
        failures=failures,
        source_sha256=fingerprint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
