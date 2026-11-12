"""Kernel versus the structurally independent futures reference.

Runs `rich_case` (mixed spot/futures stress corpus) and the older `case`
corpus through `panelbt.run` and `panelbt.reference.run`, compares them with
`test.comparison.assert_results_equal` (rtol=atol=1e-9), groups divergent
seeds by their first differing field and counter set, and tallies how often
each account event was exercised. Raw output goes to `.cache/fix/`.

    uv run --no-sync python bench/futures_reference_corpus.py --rich 20000 --case 5000
"""

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import panelbt as bt  # noqa: E402
from panelbt.models import METRICS  # noqa: E402
from panelbt.reference_futures import MixedEngine  # noqa: E402

from test.build_guard import assert_current_build  # noqa: E402
from test.comparison import assert_results_equal  # noqa: E402
from test.generators_futures import case, rich_case  # noqa: E402

# Event classes that must each occur at least MINIMUM times in the rich corpus.
MINIMUM = 200
COVERAGE_EVENTS = (
    "futures_open",
    "futures_close",
    "margin_call",
    "futures_margin_call",
    "liquidation",
    "futures_liquidation",
    "futures_expiry",
    "expired_target",
    "bankruptcy",
    "futures_rounded_to_zero",
    "futures_margin_scaled",
    "margin_scaled",
    "invalid_quote",
    "reversal",
)


# Reference-side and corpus-structure tallies with the same minimum.
STATS = (
    "reduction_debt",
    "expired_target_forced",
    "roll_day",
    "cash_account_group",
    "cash_account_supervised",
    "financing_group",
    "action_on_expiry_day",
)


def first_divergence(kernel, reference):
    """(field, row) of the earliest differing daily value, or None."""
    best = None
    for name in (*METRICS, "positions"):
        a = np.asarray(getattr(kernel, name), dtype=float)
        b = np.asarray(getattr(reference, name), dtype=float)
        bad = ~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        if bad.any():
            row = int(np.argwhere(bad)[0][0])
            if best is None or row < best[1]:
                best = (name, row)
    return best


def events(market, targets, config, result, engine):
    """Per-group event tallies computed from the reference run."""
    f = market.futures
    fut = np.flatnonzero(f.is_future)
    spot = np.flatnonzero(~f.is_future)
    tally = Counter({k: int(result.coverage[k]) for k in COVERAGE_EVENTS})
    tally.update(engine.stats)
    margin = config.margin
    tally["cash_account_group"] = int(bool(engine.classic))
    tally["financing_group"] = int(margin.allow_financing)
    tally["short_group"] = int(margin.allow_short)
    tally["shares_group"] = int(targets.kind == "shares")
    tally["weights_group"] = int(targets.kind == "weights")
    tally["cash_account_supervised"] = int(
        bool(engine.classic) and engine.book.held_ever
    )
    tally["cash_account_call"] = int(bool(engine.classic)) * int(
        result.coverage["margin_call"]
    )
    pos = result.positions
    previous = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    closed = (previous[:, fut] != 0) & (pos[:, fut] == 0)
    opened = (previous[:, fut] == 0) & (pos[:, fut] != 0)
    tally["roll_day"] = int(np.sum(closed.any(axis=1) & opened.any(axis=1)))
    days = {e.day for e in result.expiry_events}
    action = np.zeros(market.shape[0], bool)
    if market.split is not None:
        action |= (market.split[:, spot] != 1).any(axis=1)
    if market.cash_div is not None:
        action |= (market.cash_div[:, spot] != 0).any(axis=1)
    tally["action_on_expiry_day"] = sum(int(action[d]) for d in days)
    tally["reduction_clipped"] = int(result.coverage["reduction_clipped"])
    tally["delist"] = int(result.diagnostics["delist"])
    tally["blocked"] = int(
        result.diagnostics["blocked_buy"] + result.diagnostics["blocked_sell"]
    )
    tally["negative_cash_close"] = int(np.sum(result.cash < 0))
    return tally


def job(arg):
    corpus, seed = arg
    generator = rich_case if corpus == "rich" else case
    market, targets, costs, config = generator(seed)
    engine = MixedEngine(market, targets, costs, config)
    reference = engine.run()
    kernel = bt.run(market, targets, costs, config)
    tally = events(market, targets, config, reference, engine)
    try:
        assert_results_equal(kernel, reference)
        return corpus, seed, None, tally
    except AssertionError:
        where = first_divergence(kernel, reference)
        counters = sorted(
            k
            for k in (*kernel.coverage, *kernel.diagnostics)
            if kernel.coverage.get(k, kernel.diagnostics.get(k))
            != reference.coverage.get(k, reference.diagnostics.get(k))
        )
        signature = f"{where[0] if where else 'event-lists'}|{','.join(counters)}"
        return corpus, seed, (signature, where[1] if where else None), tally


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rich", type=int, default=20000)
    parser.add_argument("--case", type=int, default=5000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/fix/futures_reference_corpus.json"
    )
    args = parser.parse_args()
    fingerprint = assert_current_build()
    jobs = [("rich", s) for s in range(args.rich)] + [
        ("case", s) for s in range(args.case)
    ]
    start = time.perf_counter()
    with Pool(args.processes) as pool:
        rows = pool.map(job, jobs, chunksize=25)
    elapsed = time.perf_counter() - start
    report = {"build": fingerprint, "seconds": round(elapsed, 1)}
    for corpus in ("rich", "case"):
        mine = [r for r in rows if r[0] == corpus]
        if not mine:
            continue
        totals, groups = Counter(), Counter()
        for _, _, _, tally in mine:
            totals.update(tally)
            groups.update({k: 1 for k, v in tally.items() if v})
        divergent = [(seed, d) for _, seed, d, _ in mine if d is not None]
        by_signature = {}
        for seed, (signature, row) in divergent:
            by_signature.setdefault(signature, []).append([seed, row])
        report[corpus] = {
            "groups": len(mine),
            "divergent": len(divergent),
            "signatures": {
                k: {"count": len(v), "examples": v[:8]}
                for k, v in sorted(by_signature.items(), key=lambda x: -len(x[1]))
            },
            "events": dict(sorted(totals.items())),
            "groups_with_event": dict(sorted(groups.items())),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for corpus in ("rich", "case"):
        if corpus not in report:
            continue
        r = report[corpus]
        print(f"{corpus}: {r['divergent']} divergent of {r['groups']}")
        for k, v in r["signatures"].items():
            print(f"  {v['count']:6d}  {k}  e.g. {v['examples'][:4]}")
        print("  events:", r["events"])
        if corpus == "rich":
            thin = [
                k for k in (*COVERAGE_EVENTS, *STATS) if r["events"].get(k, 0) < MINIMUM
            ]
            print("  below", MINIMUM, ":", thin)
    print(f"{elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
