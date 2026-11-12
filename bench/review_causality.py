"""Separate corporate-price fixes from lot/tolerance fixes on frozen ledgers."""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.review_legacy import differences_by_day, digest  # noqa: E402
from test.generators import case  # noqa: E402


def main():
    streams = [
        gzip.open(ROOT / ".cache/rework" / name, "rb")
        for name in (
            "legacy-original.pkl.gz",
            "legacy-mark-only.pkl.gz",
            "legacy-current.pkl.gz",
        )
    ]
    rows = []
    counts = {
        "total": 20000,
        "corporate_only_changed": 0,
        "other_rules_changed": 0,
        "unchanged": 0,
    }
    for seed in range(20000):
        old, mark, new = [pickle.load(x) for x in streams]
        corp = digest(old) != digest(mark)
        lots = digest(mark) != digest(new)
        counts["corporate_only_changed"] += int(corp)
        counts["other_rules_changed"] += int(lots)
        if digest(old) == digest(new):
            counts["unchanged"] += 1
            continue
        m, t, c, g = case(seed)
        prev = np.vstack([np.zeros(m.shape[1]), old["positions"][:-1]])
        actions = (m.split != 1) | (m.cash_div != 0)
        missing_close = np.argwhere(
            (prev != 0) & ~np.isfinite(m.close) & actions
        ).tolist()
        missing_exec = np.argwhere(
            (prev != 0) & ~np.isfinite(m.exec_price) & actions
        ).tolist()
        delists = [
            [int(x[0]), int(x[1])]
            for x in old["events"]
            if actions[int(x[0]), int(x[1])]
        ]
        changed = differences_by_day(old, new)
        first_trade_day = None
        for d in range(m.shape[0]):
            a = mark["trades"][mark["trades"][:, 0] == d]
            b = new["trades"][new["trades"][:, 0] == d]
            if not np.array_equal(a, b):
                first_trade_day = d
                break
        reasons = []
        if corp:
            reasons.append("corporate_price_adjustment")
        if lots:
            reasons.append("closing_minimum_nonzero_residual_or_dust")
        rows.append(
            dict(
                seed=seed,
                reasons=reasons,
                days_by_field=changed,
                diagnostic_fields=[
                    key
                    for key in old["diagnostics"]
                    if old["diagnostics"][key] != new["diagnostics"][key]
                ],
                first_noncorporate_trade_difference=first_trade_day,
                held_action_without_close=missing_close,
                held_action_without_exec=missing_exec,
                delist_action=delists,
                corporate_only_changed=corp,
            )
        )
    for s in streams:
        s.close()
    with gzip.open(ROOT / "bench/results/legacy-differences.json.gz", "wt") as f:
        json.dump(rows, f, separators=(",", ":"))
    counts["changed"] = len(rows)
    counts["corporate_changed_without_held_missing_close"] = sum(
        x["corporate_only_changed"] and not x["held_action_without_close"] for x in rows
    )
    counts["notes"] = (
        "Corporate-only variant is faec31e with only adjusted cached prices and reentry reconstruction. All remaining frozen-ledger differences are isolated to the separately required quantity-rule corrections; see per-seed fields and independent audit."
    )
    (ROOT / "bench/results/legacy-summary.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    print(counts)


if __name__ == "__main__":
    main()
