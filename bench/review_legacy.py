"""Capture/diff the original generator's 20,000 deterministic portfolios.

Run `dump PATH` under each build's interpreter, from that build's root.
Temporary full ledgers are compressed; the committed fixture contains SHA-256 only.
"""

import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import panelbt as bt

sys.path.insert(0, str(Path.cwd()))
from test.generators import case  # noqa: E402

FIELDS = (
    "nav",
    "cash",
    "turnover",
    "commission",
    "tax",
    "other_fee",
    "slippage_cost",
    "n_blocked_buy",
    "n_blocked_sell",
    "positions",
)


def capture(result):
    old = hasattr(result, "transfer_fee")
    values = {
        key: np.asarray(
            getattr(result, "transfer_fee" if old and key == "other_fee" else key)
        )
        for key in FIELDS
    }
    values["trades"] = np.array(
        [
            [
                x.day,
                x.col,
                x.qty,
                x.price,
                x.commission,
                x.tax,
                x.transfer_fee if old else x.other_fee,
                x.slippage_cost,
            ]
            for x in result.trades
        ],
        dtype=float,
    ).reshape(-1, 8)
    values["events"] = np.array(
        [[x.day, x.col, x.qty, x.price, x.proceeds] for x in result.delist_events],
        dtype=float,
    ).reshape(-1, 5)
    values["diagnostics"] = result.diagnostics
    return values


def digest(values):
    h = hashlib.sha256()
    for key, value in values.items():
        h.update(key.encode())
        if isinstance(value, dict):
            h.update(json.dumps(value, sort_keys=True).encode())
        else:
            h.update(str(value.dtype).encode())
            h.update(value.tobytes())
    return h.digest()


def differences_by_day(left, right):
    """Report bitwise changes, including variable-length fill/event records."""
    changed = {}
    for field in FIELDS:
        mask = left[field].view(np.uint64) != right[field].view(np.uint64)
        days = np.flatnonzero(mask.any(axis=1) if mask.ndim == 2 else mask)
        if len(days):
            changed[field] = days.tolist()
    columns = {
        "trades": (
            "day",
            "col",
            "qty",
            "price",
            "commission",
            "tax",
            "other_fee",
            "slippage_cost",
        ),
        "events": ("day", "col", "qty", "price", "proceeds"),
    }
    for name, names in columns.items():
        a, b = left[name], right[name]
        if a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            continue
        for day in sorted(set(a[:, 0]) | set(b[:, 0])):
            aa, bb = a[a[:, 0] == day], b[b[:, 0] == day]
            if aa.shape != bb.shape:
                changed.setdefault(name + ".rows", []).append(int(day))
                continue
            for col, label in enumerate(names[1:], start=1):
                if not np.array_equal(
                    aa[:, col].view(np.uint64), bb[:, col].view(np.uint64)
                ):
                    changed.setdefault(name + "." + label, []).append(int(day))
    return changed


def dump(path):
    with gzip.open(path, "wb") as stream:
        for seed in range(20000):
            values = capture(bt.run(*case(seed)))
            pickle.dump(values, stream, protocol=5)
    print("captured 20000 portfolios")


def compare(old_path, new_path, output):
    hashes = [[], []]
    changes = []
    with gzip.open(old_path, "rb") as left, gzip.open(new_path, "rb") as right:
        for seed in range(20000):
            a, b = pickle.load(left), pickle.load(right)
            ha, hb = digest(a), digest(b)
            hashes[0].append(np.frombuffer(ha, np.uint8))
            hashes[1].append(np.frombuffer(hb, np.uint8))
            if ha == hb:
                continue
            differences = {}
            for k in FIELDS:
                mask = a[k].view(np.uint64) != b[k].view(np.uint64)
                days = np.flatnonzero(mask.any(axis=1) if mask.ndim == 2 else mask)
                if len(days):
                    differences[k] = days.tolist()
            if not np.array_equal(a["trades"], b["trades"]):
                differences["trades"] = "changed"
            if not np.array_equal(a["events"], b["events"]):
                differences["events"] = "changed"
            if a["diagnostics"] != b["diagnostics"]:
                differences["diagnostics"] = [
                    k
                    for k in a["diagnostics"]
                    if a["diagnostics"][k] != b["diagnostics"][k]
                ]
            m, t, c, g = case(seed)
            previous = np.vstack([np.zeros(m.shape[1]), a["positions"][:-1]])
            actions = (m.split != 1) | (m.cash_div != 0)
            held_missing = np.argwhere(
                (previous != 0) & ~np.isfinite(m.close) & actions
            ).tolist()
            event_actions = [
                [int(x[0]), int(x[1])]
                for x in a["events"]
                if actions[int(x[0]), int(x[1])]
            ]
            small_sells = []
            for x in a["trades"]:
                day, col, q = int(x[0]), int(x[1]), x[2]
                held = previous[day, col] * m.split[day, col]
                if (
                    q < 0
                    and g.qty_step[col] > 0
                    and abs(q) < g.min_qty[col]
                    and abs(q) < held - 1e-9
                ):
                    small_sells.append([day, col])
            changes.append(
                dict(
                    seed=seed,
                    fields=differences,
                    held_missing_action=held_missing,
                    delist_action=event_actions,
                    legacy_small_partial_sells=small_sells,
                )
            )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, original=np.array(hashes[0]), corrected=np.array(hashes[1])
    )
    details = Path("bench/results/legacy-differences.json.gz")
    with gzip.open(details, "wt") as f:
        json.dump(changes, f, separators=(",", ":"))
    print(
        json.dumps(
            dict(total=20000, different=len(changes), identical=20000 - len(changes))
        )
    )


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        compare(*sys.argv[2:])
