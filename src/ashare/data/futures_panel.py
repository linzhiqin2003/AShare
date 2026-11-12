"""期货「交易日 × 合约」面板的构建与存储。

只认统一字段（futures_schema）的五张表，来源由适配器决定（默认 Tushare）。

存储：``<root>/cn/futures_panel/<版本>/``，``current.json`` 指向当前版本（与股票面板同一套
代次写法）。期货合约一年左右就到期，全历史一万多个合约若存成一个稠密矩阵，每个字段约
1 GB 且几乎全是空格；按品种分块的稠密矩阵实测仍有 7.1 GB。所以按「交易所_品种」分块，
块内每个合约只存它存续窗口内的那一段（分段列，一维 .npy 可内存映射），读取时按需拼成请求的
「交易日 × 合约」矩阵。
"""

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import polars as pl

from .futures_schema import PANEL_FIELDS, REQUIRED_TABLES
from .futures_tushare import TushareFutures
from .store import atomic_json, lock

FORMAT_VERSION = 2
CODE_FILES = ("futures_schema.py", "futures_tushare.py", "futures_panel.py")
BOOLEAN = {"has_quote", "is_listed"}


def code_version():
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def trading_days(calendar, first, last):
    """各交易所开市日的并集（期货面板的主日历）。"""
    return sorted(
        set(
            calendar.filter(
                pl.col("is_open")
                & pl.col("date").is_between(pl.lit(first), pl.lit(last))
            )["date"].to_list()
        )
    )


def block_key(exchange, product):
    return f"{exchange}_{product}"


def assemble(daily, params, contracts, dates):
    """统一长表 → 按品种分块的「分段列」数组；纯 numpy 散点写入，供构建和测试共用。

    每个合约只存它的存续窗口（挂牌日或首条记录 到 最后交易日或末条记录，与主日历取交集）：
    合约 j 的窗口是主日历第 ``starts[j]`` 行起 ``lengths[j]`` 行，各字段在一维数组里占
    ``offsets[j]`` 起的同样长度。读取时再拼回请求的「交易日 × 合约」矩阵。
    """
    dates = np.asarray(dates)
    long = daily.join(params, on=["date", "symbol"], how="full", coalesce=True)
    long = long.with_columns(
        has_quote=pl.col("open").is_not_null()
        | pl.col("settlement").is_not_null()
        | pl.col("volume").is_not_null()
    )
    long = long.filter(pl.col("date").is_in(list(dates)))
    info = contracts.select(
        "symbol", "exchange", "product", "listing_date", "last_trade_date"
    )
    long = long.join(
        info.select("symbol", "exchange", "product"), on="symbol", how="inner"
    )
    span = long.group_by("symbol").agg(
        first_row=pl.col("date").min(), last_row=pl.col("date").max()
    )
    info = info.join(span, on="symbol", how="left").with_columns(
        window_start=pl.min_horizontal("listing_date", "first_row"),
        window_end=pl.max_horizontal("last_trade_date", "last_row"),
    )
    groups = long.partition_by(["exchange", "product"], as_dict=True)
    blocks = {}
    for (exchange, product), members in info.sort("last_trade_date", "symbol").group_by(
        ["exchange", "product"], maintain_order=True
    ):
        lo = np.searchsorted(
            dates, members["window_start"].fill_null("99999999").to_numpy().astype("U8")
        )
        hi = np.searchsorted(
            dates,
            members["window_end"].fill_null("00000000").to_numpy().astype("U8"),
            side="right",
        )
        keep = hi > lo
        if not keep.any():
            continue
        members = members.filter(pl.Series(keep))
        lo, hi = lo[keep], hi[keep]
        symbols = members["symbol"].to_list()
        lengths = (hi - lo).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        total = int(lengths.sum())
        rows = groups.get((exchange, product), long.clear()).filter(
            pl.col("symbol").is_in(symbols)
        )
        column = {name: j for j, name in enumerate(symbols)}
        c = np.array([column[x] for x in rows["symbol"].to_list()], dtype=np.int64)
        r = np.searchsorted(dates, rows["date"].to_numpy().astype("U8"))
        position = offsets[c] + (r - lo[c]) if len(c) else np.zeros(0, dtype=np.int64)
        arrays = {}
        for name in PANEL_FIELDS:
            if name == "is_listed":
                continue
            if name in BOOLEAN:
                out = np.zeros(total, dtype=bool)
                out[position] = rows[name].fill_null(False).to_numpy()
            else:
                out = np.full(total, np.nan)
                out[position] = rows[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
            arrays[name] = out
        owner = np.repeat(np.arange(len(symbols)), lengths)
        row_of = lo[owner] + (np.arange(total) - offsets[owner])
        day = dates[row_of]
        first = members["listing_date"].fill_null("99999999").to_numpy().astype("U8")
        last = members["last_trade_date"].fill_null("00000000").to_numpy().astype("U8")
        arrays["is_listed"] = (day >= first[owner]) & (day <= last[owner])
        blocks[block_key(exchange, product)] = dict(
            exchange=exchange,
            product=product,
            symbols=symbols,
            starts=lo.astype(int).tolist(),
            lengths=lengths.tolist(),
            offsets=offsets.tolist(),
            arrays=arrays,
        )
    return blocks


def build_futures(store, adapter=None, *, keep=1):
    """由原始库构建期货面板；输入不变时直接复用当前版本。返回元数据。"""
    started = time.perf_counter()
    adapter = adapter or TushareFutures(store)
    contracts = adapter.contracts()
    daily = adapter.daily()
    params = adapter.params()
    calendar = adapter.calendar()
    mapping = adapter.mapping() if hasattr(adapter, "mapping") else None
    for name, table in zip(
        REQUIRED_TABLES, (contracts, daily, params, calendar), strict=False
    ):
        if table.is_empty():
            raise ValueError(f"适配器交出的 {name} 为空")
    first, last = daily["date"].min(), daily["date"].max()
    dates = trading_days(calendar, first, last)
    inputs = {
        k: v["content_hash"]
        for k, v in store.manifest().items()
        if v.get("status") == "complete" and v["dataset"].startswith("futures_")
    }
    version = hashlib.sha256(
        json.dumps(
            dict(
                inputs=inputs,
                code=code_version(),
                format=FORMAT_VERSION,
                source=getattr(adapter, "name", type(adapter).__name__),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    base = store.root / "futures_panel"
    with lock(store.root / ".futures_build.lock"):
        target = base / version
        if (target / "metadata.json").exists():
            atomic_json(base / "current.json", {"version": version})
            return json.loads((target / "metadata.json").read_text())
        building = base / f".building-{uuid.uuid4().hex}"
        building.mkdir(parents=True)
        try:
            blocks = assemble(daily, params, contracts, dates)
            index = {}
            for key, block in blocks.items():
                folder = building / "blocks" / key
                folder.mkdir(parents=True)
                for name, array in block["arrays"].items():
                    np.save(folder / f"{name}.npy", array)
                index[key] = {
                    k: block[k]
                    for k in (
                        "exchange",
                        "product",
                        "symbols",
                        "starts",
                        "lengths",
                        "offsets",
                    )
                }
            contracts.write_parquet(building / "contracts.parquet")
            calendar.write_parquet(building / "calendar.parquet")
            daily.write_parquet(building / "daily.parquet")
            params.write_parquet(building / "params.parquet")
            if mapping is not None:
                mapping.write_parquet(building / "mapping.parquet")
            if hasattr(adapter, "source_continuous"):
                adapter.source_continuous().write_parquet(
                    building / "source_continuous.parquet"
                )
            meta = dict(
                version=version,
                source=getattr(adapter, "name", type(adapter).__name__),
                dates=list(dates),
                blocks=index,
                fields={
                    name: dict(
                        unit=c.unit,
                        dtype="bool" if name in BOOLEAN else "float64",
                        known_at=c.known_at,
                    )
                    for name, c in PANEL_FIELDS.items()
                },
                shape=[len(dates), sum(len(b["symbols"]) for b in index.values())],
                inputs=len(inputs),
                code_version=code_version(),
                format=FORMAT_VERSION,
                elapsed_seconds=time.perf_counter() - started,
            )
            atomic_json(building / "metadata.json", meta)
            if target.exists():
                shutil.rmtree(target)
            building.rename(target)
        finally:
            if building.exists():
                shutil.rmtree(building)
        atomic_json(base / "current.json", {"version": version})
        olds = sorted(
            (
                p
                for p in base.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name != version
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in olds[max(0, keep - 1) :]:
            shutil.rmtree(old)
        return meta


__all__ = ["assemble", "build_futures", "trading_days"]
