"""显式核对发布的真实面板，独立计算逐格对账、20格算式、换码及体积分项。"""

import argparse
import json
from decimal import Decimal

import numpy as np
import polars as pl

from ashare.data.maintenance import disk_usage
from ashare.data.store import Store, atomic_json

CASES = [
    ("688766.SH", "20230614"),
    ("002346.SZ", "20140321"),
    ("601088.SH", "20170710"),
    ("002157.SZ", "20231208"),
    ("689009.SH", "20260807"),
    ("002142.SZ", "20171116"),
    ("600036.SH", "20130905"),
    ("000887.SZ", "20070330"),
    ("301393.SZ", "20261110"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    store = Store(args.root, readonly=True)
    root = store.root
    directory = (
        root
        / "panel"
        / json.loads((root / "panel/current.json").read_text())["version"]
    )
    meta = json.loads((directory / "metadata.json").read_text())
    dates, symbols = np.array(meta["dates"]), np.array(meta["symbols"])
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in meta["fields"]
    }
    source_counts = {str(i): 0 for i in range(6)}
    failed = over_percent = checked = missing_no_suspension = quote_and_suspension = 0
    yearly = {}
    samples = list(CASES)
    for left in range(0, len(symbols), 128):
        right = left + 128
        f, close, split, cash, listed, source = (
            arrays[n][:, left:right]
            for n in (
                "adj_factor",
                "close",
                "split",
                "cash_dividend",
                "is_listed",
                "action_source",
            )
        )
        # 独立逐列维护前收，与构建代码的向量化索引实现不同。
        previous = np.full_like(close, np.nan)
        last = np.full(close.shape[1], np.nan)
        for i in range(len(dates)):
            previous[i] = last
            last = np.where(np.isfinite(close[i]), close[i], last)
        with np.errstate(all="ignore"):
            ratio = f[1:] / f[:-1]
            theta = np.where(
                cash[1:] == 0,
                split[1:],
                previous[1:] * split[1:] / (previous[1:] - cash[1:]),
            )
            error = np.abs(theta / ratio - 1)
        valid = listed[1:] & np.isfinite(ratio) & (ratio > 0)
        changed = valid & (np.abs(ratio - 1) > 1e-10)
        event = listed[1:] & ((split[1:] != 1) | (cash[1:] != 0))
        checked += int((changed | event).sum())
        # 第四轮起来源 5 记公告值、按摊薄比例 k 对账，由 verify 独立核对，这里不计。
        diluted = source[1:] == 5
        failed += int(
            (
                (changed & ~diluted & (~np.isfinite(error) | (error > 0.005)))
                | (event & ~changed)
            ).sum()
        )
        over_percent += int((valid & (error > 0.01)).sum())
        for i in range(6):
            source_counts[str(i)] += int((source == i).sum())
        for year in sorted({d[:4] for d in dates}):
            yearly[year] = yearly.get(year, 0) + int(
                (source[np.char.startswith(dates, year)] == 3).sum()
            )
        quote, suspended = (
            arrays[n][:, left:right] for n in ("has_quote", "is_suspended")
        )
        missing_no_suspension += int((listed & ~quote & ~suspended).sum())
        quote_and_suspension += int((quote & suspended).sum())
    for kind in (1, 2, 3, 4):
        needed = 3 if kind != 4 else 2
        for j, symbol in enumerate(symbols):
            for i in np.flatnonzero(arrays["action_source"][:, j] == kind):
                if (symbol, dates[i]) not in samples:
                    before = arrays["close"][:i, j]
                    if not np.isfinite(before).any():
                        continue
                    samples.append((str(symbol), str(dates[i])))
                    needed -= 1
                    break
            if needed == 0:
                break
    manual = []
    for symbol, day in samples:
        i, j = int(np.searchsorted(dates, day)), int(np.searchsorted(symbols, symbol))
        before = arrays["close"][:i, j]
        p = Decimal(str(float(before[np.flatnonzero(np.isfinite(before))[-1]])))
        old, new, s, c = (
            Decimal(str(float(x)))
            for x in (
                arrays["adj_factor"][i - 1, j],
                arrays["adj_factor"][i, j],
                arrays["split"][i, j],
                arrays["cash_dividend"][i, j],
            )
        )
        a, theta = new / old, p * s / (p - c)
        manual.append(
            dict(
                symbol=symbol,
                date=day,
                previous=str(p),
                old_factor=str(old),
                new_factor=str(new),
                split=str(s),
                cash=str(c),
                ratio=float(a),
                theta=float(theta),
                deviation=float(theta / a - 1),
                source=int(arrays["action_source"][i, j]),
            )
        )
    assert len(manual) == 20 and failed == 0 and over_percent == 0
    seams = []
    for alias in meta["aliases"]:
        j = int(np.searchsorted(symbols, alias["new"]))
        i = int(np.searchsorted(dates, alias["switch"]))
        before = int(np.searchsorted(dates, alias["old_last"]))
        raw = store.scan("daily").filter(pl.col("ts_code") == alias["old"]).collect()
        rows = np.searchsorted(dates, raw["trade_date"].to_numpy())
        equal = {}
        for field, source, scale in (
            ("close", "close", 1),
            ("previous_close", "pre_close", 1),
            ("volume", "vol", 100),
            ("amount", "amount", 1000),
        ):
            equal[field] = int(
                np.count_nonzero(
                    arrays[field][rows, j] != raw[source].to_numpy() * scale
                )
            )
        limits = (
            store.scan("stk_limit").filter(pl.col("ts_code") == alias["old"]).collect()
        )
        lr = np.searchsorted(dates, limits["trade_date"].to_numpy())
        missing = int(np.isnan(arrays["upper_limit"][lr, j]).sum())
        seams.append(
            dict(
                **alias,
                historical_rows=len(raw),
                mismatches=equal,
                old_limit_rows=len(limits),
                missing_limits=missing,
                prior_close=float(arrays["close"][before, j]),
                new_previous=float(arrays["previous_close"][i, j]),
                old_factor=float(arrays["adj_factor"][before, j]),
                new_factor=float(arrays["adj_factor"][i, j]),
                factor_ratio=float(
                    arrays["adj_factor"][i, j] / arrays["adj_factor"][i - 1, j]
                ),
                new_split=float(arrays["split"][i, j]),
                new_cash=float(arrays["cash_dividend"][i, j]),
            )
        )
        assert not any(equal.values()) and missing == 0
    sizes = [
        dict(
            field=name,
            dtype=str(array.dtype),
            itemsize=array.dtype.itemsize,
            array_bytes=array.nbytes,
            file_bytes=(directory / f"{name}.npy").stat().st_size,
        )
        for name, array in arrays.items()
    ]
    assert all(x["itemsize"] == 1 for x in sizes if x["dtype"] == "bool")
    result = dict(
        version=meta["version"],
        shape=meta["shape"],
        fields=len(arrays),
        checked=checked,
        failed=failed,
        over_one_percent=over_percent,
        source_counts=source_counts,
        synthetic_by_year=yearly,
        manual=manual,
        aliases=seams,
        industry_coverage=meta["diagnostics"]["industry_coverage"],
        sizes=sizes,
        missing_quote_without_suspension=missing_no_suspension,
        quote_with_suspension=quote_and_suspension,
        disk=disk_usage(root),
        diagnostics=meta["diagnostics"]["actions"],
    )
    atomic_json(root / "reports/round3/final-real.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("manual", "aliases", "sizes", "diagnostics")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
