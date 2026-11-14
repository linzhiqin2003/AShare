"""Reproducible C1 differential corpus and mandatory branch report."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.models import CRYPTO_METRICS
from panelbt.reference import run as mixed_oracle
from panelbt.reference_crypto import run as oracle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.build_guard import assert_current_build  # noqa: E402
from test.comparison import assert_results_equal  # noqa: E402
from test.generators_crypto import (  # noqa: E402
    BRANCHES,
    COMBINATIONS,
    case,
    combination_counts,
    directed_cases,
    mixed_case,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2000)
    args = parser.parse_args()
    source = assert_current_build()
    cases = [*directed_cases(), *(case(seed) for seed in range(args.seeds))]
    counts = Counter()
    combinations = Counter()
    for index, inputs in enumerate(cases):
        actual, expected = bt.run(*inputs), oracle(*inputs)
        try:
            assert_results_equal(actual, expected)
            for field in CRYPTO_METRICS:
                np.testing.assert_allclose(
                    getattr(actual, field),
                    getattr(expected, field),
                    rtol=1e-12,
                    atol=1e-9,
                    err_msg=field,
                )
        except AssertionError as exc:
            raise AssertionError(f"crypto case {index}: {exc}") from exc
        counts.update(actual.coverage)
        combinations.update(combination_counts(inputs, actual))
    assert all(counts[k] for k in BRANCHES), counts
    assert all(combinations[k] >= 2 for k in COMBINATIONS), combinations
    mixed_counts = Counter()
    for seed in range(200):
        inputs = mixed_case(seed)
        actual, expected = bt.run(*inputs), mixed_oracle(*inputs)
        assert_results_equal(actual, expected)
        np.testing.assert_allclose(
            actual.funding, expected.funding, rtol=1e-12, atol=1e-9
        )
        mixed_counts.update(actual.coverage)
    mixed_branches = (
        "futures_open",
        "funding_paid",
        "funding_received",
        "futures_rounded_to_zero",
    )
    assert all(mixed_counts[k] for k in mixed_branches), mixed_counts
    report = dict(
        source=source,
        random_cases=args.seeds,
        directed_cases=len(cases) - args.seeds,
        branches={k: counts[k] for k in BRANCHES},
        combinations={k: combinations[k] for k in COMBINATIONS},
        mixed_cases=200,
        mixed_branches={k: mixed_counts[k] for k in mixed_branches},
    )
    (ROOT / "bench/results/crypto_corpus.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
