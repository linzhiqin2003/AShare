"""平台测试和测量共用的合成面板，绝不读取真实行情。"""

import json
from pathlib import Path

import numpy as np

from ashare.data.reader import Panel


def panel(n=60, full=False):
    rng = np.random.default_rng(20261113)
    dates = np.arange("2015-01-01", "2026-02-01", dtype="datetime64[D]")
    dates = dates[np.is_busday(dates)]
    if not full:
        dates = np.concatenate(
            (
                dates[:260],
                dates[
                    (dates >= np.datetime64("2024-10-01"))
                    & (dates < np.datetime64("2025-04-01"))
                ],
                dates[-10:],
            )
        )
    dates = np.char.replace(dates.astype(str), "-", "")
    t = len(dates)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, (t, n)), axis=0))
    prev = np.vstack((close[0], close[:-1]))
    quote = np.ones((t, n), dtype=bool)

    def member(lo, hi):
        return np.broadcast_to(
            (np.arange(n) >= lo) & (np.arange(n) < hi), (t, n)
        ).copy()  # noqa: E731

    arrays = dict(
        open=close * np.exp(rng.normal(0, 0.002, (t, n))),
        close=close,
        high=close * 1.03,
        low=close * 0.97,
        vwap=close,
        previous_close=prev,
        adj_factor=np.ones((t, n)),
        upper_limit=close * 1.1,
        lower_limit=close * 0.9,
        total_market_cap=np.exp(rng.normal(23, 1.5, (t, n))),
        industry=np.broadcast_to(801010 + 10 * (np.arange(n) % 3), (t, n)).copy(),
        is_listed=quote.copy(),
        is_suspended=~quote,
        has_quote=quote.copy(),
        can_buy_open=quote.copy(),
        can_sell_open=quote.copy(),
        listing_age=np.full((t, n), 500),
        member_csi300=member(0, n // 3),
        member_csi500=member(n // 3, n * 2 // 3),
        member_csi1000=member(n * 2 // 3, n),
        turnover=rng.uniform(0.001, 0.08, (t, n)),
        amount=rng.uniform(1e6, 1e9, (t, n)),
        volume=rng.uniform(1e4, 1e6, (t, n)),
    )
    for value in arrays.values():
        value.flags.writeable = False
    return Panel(
        dates, np.array([f"S{i:04d}" for i in range(n)]), arrays, "synthetic-v1"
    )


def write_panel(root, source=None):
    source = panel() if source is None else source
    folder = Path(root) / "cn/panel/synthetic-v1"
    folder.mkdir(parents=True)
    (folder.parent / "current.json").write_text(json.dumps({"version": "synthetic-v1"}))
    (folder / "metadata.json").write_text(
        json.dumps(
            dict(
                version="synthetic-v1",
                dates=source.dates.tolist(),
                symbols=source.symbols.tolist(),
                fields={name: {} for name in source._arrays},
            )
        )
    )
    for name, array in source._arrays.items():
        np.save(folder / f"{name}.npy", array)
    return source


def write_indexes(root, source=None, *, codes=None):
    """按真实清单的代码与月份分区写随机日线；价格全为合成数据。"""
    import polars as pl

    from ashare.data.catalog import BENCHMARKS
    from ashare.data.store import Store

    source = panel(full=True) if source is None else source
    store = Store(root)
    rng = np.random.default_rng(332)
    codes = (
        list(codes)
        if codes is not None
        else ["000001.SH", "000300.SH", "000905.SH", "000852.SH", "932000.CSI"]
    )
    if not set(codes) <= BENCHMARKS.keys():
        raise ValueError("合成指数必须来自主流基准名单")
    folder = store.root / "raw/synthetic-index"
    folder.mkdir(parents=True, exist_ok=True)
    for code in codes:
        prices = 1000 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, len(source.dates))))
        frame = pl.DataFrame(
            {
                "ts_code": code,
                "trade_date": source.dates,
                "open": prices,
                "close": prices * 1.001,
            }
        )
        for (month,), rows in (
            frame.with_columns(pl.col("trade_date").str.slice(0, 6).alias("month"))
            .partition_by("month", as_dict=True)
            .items()
        ):
            partition = f"{code}_{month}"
            path = folder / f"{partition}.parquet"
            rows.drop("month").write_parquet(path)
            store.mark(
                f"index_daily/{partition}",
                dict(
                    dataset="index_daily",
                    partition=partition,
                    status="complete",
                    rows=len(rows),
                    file=str(path.relative_to(store.root)),
                    content_hash="synthetic",
                    file_hash="synthetic",
                ),
            )
    return codes
