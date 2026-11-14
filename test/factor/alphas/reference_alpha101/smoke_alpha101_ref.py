"""Alpha101 参考实现的冒烟：在真实面板上把 101 条（含 D-13 读法 B 的 6 条）各跑一遍。

    python tests/reference_alpha101/smoke_alpha101_ref.py \
        --root /path/to/AShare/data --days 300 --out smoke.json

记录每条的耗时、有效格比例（有限值格 / 上市且有行情的格）、是否全 NaN、是否全常数；
不进 pytest（文件名不以 test_ 开头），需要真实数据。
"""

import argparse
import json
import time

import numpy as np

from ashare.factor.alphas.reference import alpha101_ref as ref


def load_frames(root, days, warmup):
    """取数：优先用 ``inputs.reference_frames``；它不在（因子库正在迁移）时照
    ``REFERENCE_INPUTS.md`` 第 2 节自己换算（不含基准指数，Alpha101 用不到）。"""
    try:
        from ashare.factor.formula.inputs import load_inputs, reference_frames
    except ImportError:
        try:
            from ashare.factor.alphas.inputs import load_inputs, reference_frames
        except ImportError:
            load_inputs = reference_frames = None
    if load_inputs is not None:
        source = load_inputs(root=root, days=days, warmup=warmup, benchmark=None)
        return reference_frames(source), source.version
    import pandas as pd

    import ashare.data as ad

    fields = (
        "open high low close vwap volume amount adj_factor previous_close "
        "total_market_cap is_listed listing_age industry"
    ).split()
    data = ad.open(root=root)
    calendar = np.asarray(data.panel(["close"]).dates)
    panel = data.panel(fields, start=calendar[-(days + warmup)])
    a = {k: np.asarray(panel[k]) for k in fields}

    def frame(values):
        return pd.DataFrame(
            np.array(values, dtype=float), index=panel.dates, columns=panel.symbols
        )

    factor = a["adj_factor"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {k: frame(a[k] * factor) for k in ("open", "high", "low", "close")}
        out["vwap"] = frame(a["vwap"] * factor)
        out["volume"] = frame(a["volume"] / factor)
        out["amount"] = frame(a["amount"])
        first_day = a["listing_age"] == 1
        out["returns"] = frame(
            np.where(first_day, np.nan, a["close"] / a["previous_close"] - 1)
        )
        out["cap"] = frame(a["total_market_cap"])
    out["listed"] = pd.DataFrame(
        a["is_listed"].astype(bool), index=panel.dates, columns=panel.symbols
    )
    industry = a["industry"].astype(float)
    out["industry"] = frame(np.where(industry > 0, industry, np.nan))
    return out, f"{panel.version}+own-loader"


def describe(values, quoted):
    finite = np.isfinite(values)
    row = {
        "valid_ratio": float(finite.sum() / max(quoted.sum(), 1)),
        "valid_cells": int(finite.sum()),
        "all_nan": bool(not finite.any()),
    }
    if finite.any():
        v = values[finite]
        row["all_constant"] = bool(v.min() == v.max())
        row["min"] = float(v.min())
        row["max"] = float(v.max())
        row["distinct_head"] = [float(u) for u in np.unique(v)[:5]]
        # 只看最后 50 行（长窗口在前面都是 NaN）里的有效比例
        tail = finite[-50:]
        row["valid_ratio_last50"] = float(tail.sum() / max(quoted[-50:].sum(), 1))
    else:
        row["all_constant"] = False
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--factors", default="")
    parser.add_argument("--readings", default="days_ago,position")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    frames, version = load_frames(args.root, args.days, args.warmup)
    quoted = np.isfinite(frames["close"].to_numpy())
    load_s = time.perf_counter() - t0
    print(f"loaded {frames['close'].shape} in {load_s:.1f}s", flush=True)
    names = [n for n in args.factors.split(",") if n] or list(ref.NAMES)
    rows = []
    for reading in args.readings.split(","):
        table = ref.references(reading)
        todo = (
            names
            if reading == "days_ago"
            else [n for n in names if n in ref.ARGMAX_DEPENDENT]
        )
        for name in todo:
            start = time.perf_counter()
            try:
                values = np.asarray(table[name](frames), dtype=float)
                row = describe(values, quoted)
                row["error"] = None
            except Exception as exc:  # noqa: BLE001 — 冒烟要把报错记下来继续
                row = {"error": f"{type(exc).__name__}: {exc}"}
            row["seconds"] = round(time.perf_counter() - start, 3)
            row["name"] = name
            row["argmax"] = reading
            rows.append(row)
            print(
                f"{name} [{reading}] {row['seconds']:.2f}s "
                f"valid={row.get('valid_ratio', float('nan')):.3f} "
                f"nan={row.get('all_nan')} const={row.get('all_constant')} "
                f"err={row['error']}",
                flush=True,
            )
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "shape": list(frames["close"].shape),
                        "version": version,
                        "load_seconds": load_s,
                        "rows": rows,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=1,
                )


if __name__ == "__main__":
    main()
