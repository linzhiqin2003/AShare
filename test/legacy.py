"""Stable spot-output fingerprints and explicit regression differences."""

import hashlib
import json

import numpy as np

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
