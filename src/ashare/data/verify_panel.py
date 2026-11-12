"""独立从发布矩阵重算事件隐含比；不复用候选裁决结果来证明自己。"""

import json

import numpy as np

from .panel_actions import FACTOR_NOISE, prior_prices, reconciliation_check


def audit_panel(store, dates):
    current = store.root / "panel/current.json"
    if not current.exists():
        return {"status": "not_built"}
    path = current.parent / json.loads(current.read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    ds, symbols = np.array(meta["dates"]), np.array(meta["symbols"])
    start = int(np.searchsorted(ds, dates[0]))
    end = int(np.searchsorted(ds, dates[-1], side="right"))
    if start >= end:
        return {"status": "no_overlap"}
    close = np.load(path / "close.npy", mmap_mode="r")
    factor = np.load(path / "adj_factor.npy", mmap_mode="r")
    listed = np.load(path / "is_listed.npy", mmap_mode="r")[start:end]
    split = np.load(path / "split.npy", mmap_mode="r")[start:end]
    cash = np.load(path / "cash_dividend.npy", mmap_mode="r")[start:end]
    previous = prior_prices(np.asarray(close[start:end]))
    seed = np.full(len(symbols), np.nan)
    needed = np.array(listed[0])
    i = start - 1
    while i >= 0 and needed.any():
        values = close[i]
        fill = needed & np.isfinite(values)
        seed[fill] = values[fill]
        needed[fill] = False
        i -= 1
    previous = np.where(np.isfinite(previous), previous, seed)
    check, bad, ratio, theta = reconciliation_check(
        previous,
        np.asarray(factor[start:end]),
        split,
        cash,
        listed,
        tolerance=meta.get("action_tolerance", 0.005),
        prior_factor=factor[start - 1] if start else None,
    )
    ii, jj = np.nonzero(bad)
    check["examples"] = [
        dict(
            symbol=str(symbols[j]),
            date=str(ds[start + i]),
            ratio=float(ratio[i, j]),
            theta=float(theta[i, j]),
        )
        for i, j in zip(ii[:20], jj[:20], strict=True)
    ]
    changed = listed & np.isfinite(ratio) & (np.abs(ratio - 1) > FACTOR_NOISE)
    missing = changed & ~((split != 1) | (cash != 0))
    ii, jj = np.nonzero(missing)
    return dict(
        status="complete",
        checked=int(changed.sum()),
        failed=int(missing.sum()),
        examples=[
            dict(symbol=str(symbols[j]), date=str(ds[start + i]))
            for i, j in zip(ii[:20], jj[:20], strict=True)
        ],
        reconciliation=check,
    )
