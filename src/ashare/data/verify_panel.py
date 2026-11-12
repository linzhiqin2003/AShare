"""独立从发布矩阵核对公司行为；不调用构建期的任何对账函数。

前收、因子比、隐含比和差异化摊薄比例 k 都在这里另写一遍：构建与校验共用一个函数时，
函数里的错误会同时骗过两者。
"""

import json

import numpy as np

NOISE = 0.001
EPSILON = 1e-10


def _previous_close(close, seed):
    """逐行推进的前收：当天之前最近一个有限收盘价。"""
    output = np.empty(close.shape)
    last = np.array(seed, dtype=float)
    for i in range(close.shape[0]):
        output[i] = last
        row = close[i]
        last = np.where(np.isfinite(row), row, last)
    return output


def _seed(close, start, listed_first):
    seed = np.full(close.shape[1], np.nan)
    needed = np.array(listed_first)
    i = start - 1
    while i >= 0 and needed.any():
        values = np.asarray(close[i])
        fill = needed & np.isfinite(values)
        seed[fill] = values[fill]
        needed[fill] = False
        i -= 1
    return seed


def audit_actions(previous, ratio, split, cash, source, listed, *, tolerance, k_min):
    """按来源逐格核对；返回 (失败掩码, 用于报告的有效隐含比, k)。"""
    with np.errstate(all="ignore"):
        has_cash = cash != 0
        # 公告值隐含比：股数乘数 × 前收 / 除息参考价。
        announced = np.where(has_cash, split / (1 - cash / previous), split)
        # 差异化分红：a = P(1+ks)/(P-kc) 解出 k = (a-1) / (a*c/P + s)。
        s = split - 1
        k = np.where(
            has_cash,
            (ratio - 1) / (ratio * cash / previous + s),
            np.where(s > 0, (ratio - 1) / s, np.nan),
        )
        reference = np.where(has_cash, (previous - k * cash) / (1 + k * s), np.nan)
        diluted = np.where(has_cash, previous / reference, 1 + k * s)
    is_diluted = source == 5
    effective = np.where(is_diluted, diluted, announced)
    with np.errstate(all="ignore"):
        error = np.abs(effective / ratio - 1)
    usable = listed & np.isfinite(ratio) & (ratio > 0)
    moved = usable & (np.abs(ratio - 1) > EPSILON)
    event = listed & ((split != 1) | (cash != 0))
    mismatch = moved & ~(error <= tolerance)
    unbacked = event & ~moved
    wrong_source = listed & (
        (event & ~np.isin(source, (1, 2, 3, 5)))
        | (~event & np.isin(source, (1, 2, 3, 5)))
    )
    k_bad = listed & is_diluted & ~((k >= k_min) & (k < 1))
    return mismatch | unbacked | wrong_source | k_bad, effective, k, k_bad


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
    listed = np.asarray(np.load(path / "is_listed.npy", mmap_mode="r")[start:end])
    split = np.asarray(np.load(path / "split.npy", mmap_mode="r")[start:end])
    cash = np.asarray(np.load(path / "cash_dividend.npy", mmap_mode="r")[start:end])
    source = np.asarray(np.load(path / "action_source.npy", mmap_mode="r")[start:end])
    previous = _previous_close(
        np.asarray(close[start:end]), _seed(close, start, listed[0])
    )
    block = np.asarray(factor[start:end])
    before = np.vstack(
        (
            np.asarray(factor[start - 1 : start])
            if start
            else np.full((1, block.shape[1]), np.nan),
            block[:-1],
        )
    )
    with np.errstate(all="ignore"):
        ratio = block / before
    bad, effective, k, k_bad = audit_actions(
        previous,
        ratio,
        split,
        cash,
        source,
        listed,
        tolerance=meta.get("action_tolerance", 0.005),
        k_min=meta.get("action_k_min", 0.85),
    )
    usable = listed & np.isfinite(ratio) & (ratio > 0)
    event = listed & ((split != 1) | (cash != 0))
    ii, jj = np.nonzero(bad)
    reconciliation = dict(
        checked=int(((usable & (np.abs(ratio - 1) > EPSILON)) | event).sum()),
        failed=int(bad.sum()),
        diluted_checked=int((listed & (source == 5)).sum()),
        diluted_failed=int(k_bad.sum()),
        examples=[
            dict(
                symbol=str(symbols[j]),
                date=str(ds[start + i]),
                ratio=float(ratio[i, j]),
                theta=float(effective[i, j]),
                source=int(source[i, j]),
            )
            for i, j in zip(ii[:20], jj[:20], strict=True)
        ],
    )
    changed = usable & (np.abs(ratio - 1) > NOISE)
    missing = changed & ~event
    ii, jj = np.nonzero(missing)
    return dict(
        status="complete",
        checked=int(changed.sum()),
        failed=int(missing.sum()),
        examples=[
            dict(symbol=str(symbols[j]), date=str(ds[start + i]))
            for i, j in zip(ii[:20], jj[:20], strict=True)
        ],
        reconciliation=reconciliation,
    )


def audit_open_masks(store, dates):
    """用 panelbt 的 tradability 从发布矩阵重算开盘可买、可卖掩码，逐格比对。"""
    from panelbt.markets.ashare import tradability

    current = store.root / "panel/current.json"
    if not current.exists():
        return {"status": "not_built"}
    path = current.parent / json.loads(current.read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    if not {"can_buy_open", "can_sell_open"} <= set(meta["fields"]):
        return {"status": "field_absent"}
    ds, symbols = np.array(meta["dates"]), np.array(meta["symbols"])
    start = int(np.searchsorted(ds, dates[0]))
    end = int(np.searchsorted(ds, dates[-1], side="right"))
    if start >= end:
        return {"status": "no_overlap"}

    def load(name):
        return np.asarray(np.load(path / f"{name}.npy", mmap_mode="r")[start:end])

    buy, sell = tradability(
        load("open"),
        load("high"),
        load("low"),
        load("upper_limit"),
        load("lower_limit"),
        load("is_suspended") | ~load("has_quote"),
    )
    bad = (buy != load("can_buy_open")) | (sell != load("can_sell_open"))
    ii, jj = np.nonzero(bad)
    return dict(
        status="complete",
        checked=int(bad.size),
        failed=int(bad.sum()),
        examples=[
            dict(symbol=str(symbols[j]), date=str(ds[start + i]))
            for i, j in zip(ii[:20], jj[:20], strict=True)
        ],
    )
