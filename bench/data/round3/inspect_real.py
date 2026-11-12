"""显式只读真实库：对账前普查、换码接缝和指定案例取数；结果只写数据目录。"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label", required=True, choices=["before", "after"])
    args = parser.parse_args()
    root = Path(args.root).resolve() / "cn"
    output = root / "reports/round3"
    base = (
        root
        / "panel"
        / json.loads((root / "panel/current.json").read_text())["version"]
    )
    meta = json.loads((base / "metadata.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    dates, symbols = np.array(meta["dates"]), meta["symbols"]
    arrays = {
        n: np.load(base / f"{n}.npy", mmap_mode="r")
        for n in (
            "close",
            "previous_close",
            "adj_factor",
            "split",
            "cash_dividend",
            "is_listed",
        )
    }
    records = []
    for j, symbol in enumerate(symbols):
        close = arrays["close"][:, j]
        idx = np.maximum.accumulate(
            np.where(np.isfinite(close), np.arange(len(close)), -1)
        )
        idx = np.r_[-1, idx[:-1]]
        previous = np.where(idx >= 0, close[np.maximum(idx, 0)], np.nan)
        with np.errstate(all="ignore"):
            a = np.r_[
                np.nan, arrays["adj_factor"][1:, j] / arrays["adj_factor"][:-1, j]
            ]
            event = (arrays["split"][:, j] != 1) | (arrays["cash_dividend"][:, j] != 0)
            theta = np.where(
                event,
                previous
                * arrays["split"][:, j]
                / (previous - arrays["cash_dividend"][:, j]),
                1,
            )
            dev = theta / a - 1
        valid = (
            arrays["is_listed"][:, j]
            & np.isfinite(a)
            & np.isfinite(theta)
            & (event | (np.abs(a - 1) > 1e-10))
        )
        cumulative = np.ones(len(dates))
        for k in range(-3, 4):
            cumulative *= np.roll(
                np.where(np.isfinite(theta), theta, 1) / np.where(np.isfinite(a), a, 1),
                k,
            )
        for i in np.flatnonzero(
            valid & (np.abs(dev) > 0.01) & (np.abs(cumulative - 1) > 0.01)
        ):
            records.append(
                dict(
                    symbol=symbol,
                    date=str(dates[i]),
                    previous=float(previous[i]),
                    ratio=float(a[i]),
                    split=float(arrays["split"][i, j]),
                    cash=float(arrays["cash_dividend"][i, j]),
                    deviation=float(dev[i]),
                )
            )
    result = dict(
        version=meta["version"],
        count=len(records),
        positive=sum(r["deviation"] > 0 for r in records),
        negative=sum(r["deviation"] < 0 for r in records),
        rows=records,
    )
    (output / f"{args.label}-actions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print({k: v for k, v in result.items() if k != "rows"}, flush=True)
    db = sqlite3.connect(
        f"file:{root / 'manifest.sqlite3'}?mode=ro&immutable=1", uri=True
    )
    manifest = [json.loads(row[0]) for row in db.execute("SELECT body FROM partitions")]
    db.close()
    pairs = [
        ("000022.SZ", "001872.SZ"),
        ("000043.SZ", "001914.SZ"),
        ("300114.SZ", "302132.SZ"),
    ]
    codes = [c for pair in pairs for c in pair]
    datasets = {}
    for name in ("daily", "adj_factor"):
        collected = []
        for year in range(2007, 2027):
            files = [
                root / e["file"]
                for e in manifest
                if e["dataset"] == name
                and e["status"] == "complete"
                and e["rows"]
                and e["partition"].startswith(str(year))
            ]
            if files:
                collected.append(
                    pl.scan_parquet(files)
                    .filter(pl.col("ts_code").is_in(codes))
                    .collect()
                )
        datasets[name] = pl.concat(collected, how="diagonal_relaxed").sort("trade_date")
    seams = []
    for old, new in pairs:
        quotes = datasets["daily"]
        a = quotes.filter(pl.col("ts_code") == old)
        b = quotes.filter(
            (pl.col("ts_code") == new) & (pl.col("trade_date") > a["trade_date"].max())
        )
        item = dict(
            old=old,
            new=new,
            old_rows=len(a),
            old_last=a["trade_date"].max(),
            new_first=b["trade_date"].min(),
            quotes=a.tail(2).to_dicts() + b.head(2).to_dicts(),
        )
        factors = datasets["adj_factor"].filter(
            pl.col("trade_date").is_between(
                pl.lit(item["old_last"]), pl.lit(item["new_first"])
            )
            & pl.col("ts_code").is_in([old, new])
        )
        item["factors"] = factors.to_dicts()
        seams.append(item)
    (output / "aliases-observed.json").write_text(
        json.dumps(seams, ensure_ascii=False, indent=2)
    )
    print(
        "seams",
        [{k: v for k, v in x.items() if k not in ("quotes", "factors")} for x in seams],
        flush=True,
    )
    files = [
        root / e["file"]
        for e in manifest
        if e["dataset"] == "dividend" and e["rows"] and e["status"] == "complete"
    ]
    cases = [
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
    raw = pl.concat(
        [
            pl.scan_parquet(f).filter(pl.col("ts_code").is_in([s for s, d in cases]))
            for f in files
        ],
        how="diagonal_relaxed",
    ).collect()
    examples = []
    for s, d in cases:
        i = int(np.searchsorted(dates, d))
        j = symbols.index(s)
        start = max(0, i - 4)
        end = min(len(dates), i + 5)
        examples.append(
            dict(
                symbol=s,
                date=d,
                dividend=raw.filter((pl.col("ts_code") == s) & (pl.col("ex_date") == d))
                .select(
                    "ts_code",
                    "ex_date",
                    "imp_ann_date",
                    "ann_date",
                    "stk_div",
                    "cash_div_tax",
                )
                .to_dicts(),
                timeline=[
                    dict(
                        date=str(dates[k]),
                        **{
                            n: float(arrays[n][k, j])
                            for n in arrays
                            if n != "is_listed"
                        },
                    )
                    for k in range(start, end)
                ],
            )
        )
    (output / "action-cases-observed.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
