"""按年散布到 C 连续矩阵；增量优先使用 APFS 写时复制，不重复写完整历史。"""

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from . import panel_events as events
from .aliases import CanonicalStore
from .catalog import DATASETS, FIELDS, INDEXES, INDUSTRY_CAVEAT
from .maintenance import collect_garbage
from .panel_cache import event_table
from .panel_increment import reuse_prefix
from .panel_io import clone_append
from .panel_quality import code_version, required, securities_table
from .store import atomic_json, lock
from .sync import compact

FORMAT_VERSION = 4
LOG = logging.getLogger(__name__)


def scatter(table, dates, symbols, fields, *, diagnostics=None):
    """一次连接复用坐标；连接丢失的行数和样例必须可审计。"""
    day_table = pl.DataFrame({"trade_date": dates, "_row": np.arange(len(dates))})
    code_table = pl.DataFrame({"ts_code": symbols, "_col": np.arange(len(symbols))})
    coordinates = table.join(day_table, on="trade_date", how="inner").join(
        code_table, on="ts_code", how="inner"
    )
    if diagnostics is not None:
        lost = table.join(day_table, on="trade_date", how="semi").join(
            code_table, on="ts_code", how="anti"
        )
        diagnostics["unknown_symbol_rows"] = diagnostics.get(
            "unknown_symbol_rows", 0
        ) + len(lost)
        diagnostics.setdefault("examples", []).extend(
            lost.select("ts_code", "trade_date")
            .head(max(0, 20 - len(diagnostics.get("examples", []))))
            .to_dicts()
        )
    if coordinates.select(pl.struct("_row", "_col").is_duplicated().any()).item():
        raise ValueError("行情有重复主键；请运行 python -m ashare.data verify")
    rows, cols = coordinates["_row"].to_numpy(), coordinates["_col"].to_numpy()
    for spec in fields:
        array = np.full((len(dates), len(symbols)), np.nan, dtype=spec.dtype)
        array[rows, cols] = (
            coordinates[spec.raw].cast(pl.Float64).to_numpy() * spec.scale
        )
        yield spec.name, array


def build(
    store,
    start=None,
    end=None,
    *,
    incremental=False,
    keep=1,
    strict=False,
    action_tolerance=0.005,
    allow_unreconciled=False,
):
    store = CanonicalStore(store.snapshot())
    started = time.perf_counter()
    with lock(store.root / ".build.lock"):
        calendar_table = store.read("trade_cal").sort("cal_date")
        calendar = np.array(
            sorted(set(calendar_table.filter(pl.col("is_open") == 1)["cal_date"])),
            dtype="U8",
        )
        daily_entries = [
            e for e in store.entries("daily").values() if e["status"] == "complete"
        ]
        if not daily_entries:
            raise FileNotFoundError("缺少 daily；请运行 python -m ashare.data sync")
        start = compact(start) if start else min(e["partition"] for e in daily_entries)
        end = compact(end) if end else max(e["partition"] for e in daily_entries)
        dates = calendar[(calendar >= start) & (calendar <= end)]
        if not len(dates):
            raise ValueError("构建范围没有交易日")
        diagnostics = {
            "input_warnings": required(store, dates, strict=strict),
            "joins": {},
        }
        securities = securities_table(store)
        symbols = securities["ts_code"].to_numpy()
        used = {f.dataset for f in FIELDS.values()} | {"trade_cal"}
        if store.entries("index_member"):
            used.add("index_member")
        inputs = {
            k: v["content_hash"]
            for k, v in store.manifest().items()
            if v["status"] == "complete"
            and v["dataset"] in used
            and (DATASETS[v["dataset"]].partition != "day" or v["partition"] <= end)
        }
        code = code_version()
        version = hashlib.sha256(
            json.dumps(
                dict(
                    inputs=inputs,
                    start=start,
                    end=end,
                    code=code,
                    format=FORMAT_VERSION,
                    action_tolerance=action_tolerance,
                    allow_unreconciled=allow_unreconciled,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        base = store.root / "panel"
        target = base / version
        if target.exists():
            atomic_json(base / "current.json", {"version": version})
            collect_garbage(store, keep, locked=True)
            return json.loads((target / "metadata.json").read_text())
        staging = base / f".building-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        event_cache = {}
        old_path, old, prefix = None, None, 0
        if incremental and (base / "current.json").exists():
            old_path = base / json.loads((base / "current.json").read_text())["version"]
            old = json.loads((old_path / "metadata.json").read_text())
            prefix = reuse_prefix(store, dates, securities, inputs, old_path, old, code)
        tail_dates = dates[prefix:]
        listed, age, delist = events.listing(tail_dates, symbols, securities, calendar)
        shape = (len(dates), len(symbols))
        old_columns = [list(symbols).index(s) for s in old["symbols"]] if prefix else []
        arrays = {}

        def destination(name):
            if name in arrays:
                return arrays[name]
            spec = FIELDS[name]
            if (staging / f"{name}.npy").exists():
                arrays[name] = open_memmap(staging / f"{name}.npy", mode="r+")
                return arrays[name]
            array = (
                clone_append(old_path / f"{name}.npy", staging / f"{name}.npy", shape)
                if prefix and len(old_columns) == shape[1]
                else None
            )
            if array is None:
                array = open_memmap(
                    staging / f"{name}.npy", mode="w+", dtype=spec.dtype, shape=shape
                )
                if prefix:
                    previous = np.load(old_path / f"{name}.npy", mmap_mode="r")
                    if len(old_columns) == shape[1]:
                        array[:prefix] = previous[:prefix]
                    else:
                        array[:prefix] = (
                            np.nan
                            if np.issubdtype(np.dtype(spec.dtype), np.floating)
                            else 0
                        )
                        array[:prefix, old_columns] = previous[:prefix]
            arrays[name] = array
            return array

        def release():
            for name in list(arrays):
                arrays.pop(name).flush()

        def save(name, tail, offset=0):
            spec = FIELDS[name]
            if name not in ("is_listed", "delist"):
                tail[~listed[offset : offset + len(tail)]] = (
                    np.nan if np.issubdtype(np.dtype(spec.dtype), np.floating) else 0
                )
            if np.issubdtype(np.dtype(spec.dtype), np.integer):
                tail = np.nan_to_num(tail, nan=0)
                if np.any(tail > np.iinfo(spec.dtype).max) or np.any(tail < 0):
                    raise ValueError(f"{name} 超出 {spec.dtype} 范围")
            destination(name)[prefix + offset : prefix + offset + len(tail)] = tail

        try:
            for dataset in ("daily", "adj_factor", "daily_basic", "stk_limit"):
                specs = [f for f in FIELDS.values() if f.dataset == dataset and f.raw]
                for spec in specs:
                    destination(spec.name)
                for year in sorted({d[:4] for d in tail_dates}):
                    block = tail_dates[np.char.startswith(tail_dates, year)]
                    offset = int(np.searchsorted(tail_dates, block[0]))
                    table = store.read(dataset, str(block[0]), str(block[-1]))
                    diag = diagnostics["joins"].setdefault(dataset, {})
                    for name, values in scatter(
                        table, block, symbols, specs, diagnostics=diag
                    ):
                        save(name, values, offset)
                release()
            for name, values in (
                ("is_listed", listed),
                ("listing_age", age),
                ("delist", delist),
            ):
                save(name, values)
            release()
            for name, ds, suspension in (
                ("is_st", "stock_st", False),
                ("is_suspended", "suspend_d", True),
            ):
                table = (
                    store.read(ds, str(tail_dates[0]), str(tail_dates[-1]))
                    if len(tail_dates)
                    else store.read(ds, str(dates[0]), str(dates[0])).head(0)
                )
                save(
                    name,
                    events.daily_mask(
                        table, tail_dates, symbols, suspension=suspension
                    ),
                )
            save("has_quote", np.isfinite(destination("close")[prefix:]))
            save(
                "tradable",
                np.isfinite(destination("close")[prefix:])
                & (destination("volume")[prefix:] > 0),
            )
            industry_values, industry_sources, industry_caveat = (
                events.stitched_industry(store, tail_dates, symbols)
            )
            save("industry", industry_values)
            save("industry_source", industry_sources)
            weight_table = (
                event_table(
                    store, "index_weight", staging, old_path, old, event_cache
                ).unique(subset=DATASETS["index_weight"].keys)
                if store.entries("index_weight")
                else pl.DataFrame(
                    schema={
                        "index_code": pl.String,
                        "con_code": pl.String,
                        "trade_date": pl.String,
                        "weight": pl.Float64,
                    }
                )
            )
            diagnostics["index_snapshots"] = {}
            for index, code_value in INDEXES.items():
                available = weight_table.filter(
                    (pl.col("index_code") == code_value)
                    & (pl.col("trade_date") < str(dates[-1]))
                )["trade_date"]
                snapshot = available.max()
                age_days = (
                    (
                        date.fromisoformat(str(dates[-1]))
                        - date.fromisoformat(snapshot)
                    ).days
                    if snapshot
                    else None
                )
                diagnostics["index_snapshots"][index] = dict(
                    snapshot=snapshot, age_days=age_days
                )
                if age_days is not None and age_days > 40:
                    if strict:
                        raise ValueError(
                            f"指数 {index} 快照已 {age_days} 天，超过40天；请补抓 index_weight"
                        )
                    LOG.warning(
                        "指数 %s 快照已 %d 天，沿用 %s", index, age_days, snapshot
                    )
            for index in INDEXES:
                values = events.weights(weight_table, tail_dates, symbols, index)
                members = events.weights(
                    weight_table, tail_dates, symbols, index, membership=True
                )
                save(f"member_{index}", np.isfinite(members) & (members == 1))
                save(f"weight_{index}", values)
                release()
            amount, volume = (
                destination("amount")[prefix:],
                destination("volume")[prefix:],
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                save("vwap", np.where(volume > 0, amount / volume, np.nan))
            from .panel_actions import reconcile

            release()
            action_arrays = {
                n: destination(n)
                for n in (
                    "split",
                    "cash_dividend",
                    "action_source",
                    "close",
                    "adj_factor",
                    "is_listed",
                )
            }
            diagnostics["actions"] = reconcile(
                action_arrays,
                dates,
                symbols,
                event_table(
                    store,
                    "dividend",
                    staging,
                    old_path,
                    old,
                    event_cache,
                    str(dates[0]),
                    end,
                ),
                tolerance=action_tolerance,
                allow_unreconciled=allow_unreconciled,
            )
            action_arrays.clear()
            diagnostics["dividends"] = diagnostics["actions"]["dividends"]
            coverage = []
            for day in (
                "20070104",
                "20130104",
                "20140220",
                "20140221",
                "20190102",
                "20210104",
                "20210730",
                "20210802",
                str(dates[-1]),
            ):
                i = int(np.searchsorted(dates, day))
                if i < len(dates) and dates[i] == day:
                    active = destination("is_listed")[i]
                    missing = active & (destination("industry")[i] == 0)
                    coverage.append(
                        dict(
                            date=day,
                            listed=int(active.sum()),
                            missing=int(missing.sum()),
                            missing_ratio=float(missing.sum() / max(1, active.sum())),
                        )
                    )
            diagnostics["industry_coverage"] = coverage
            for array in arrays.values():
                array.flush()
            arrays.clear()
            metadata = dict(
                version=version,
                inputs=inputs,
                code_version=code,
                format=FORMAT_VERSION,
                built_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                shape=shape,
                dates=list(dates),
                symbols=list(symbols),
                reused_rows=prefix,
                fields={n: asdict(f) for n, f in FIELDS.items()},
                diagnostics=diagnostics,
                event_cache=event_cache,
                event_files={
                    "index_member_all": [
                        v["file"]
                        for v in store.entries("index_member_all").values()
                        if v["status"] == "complete"
                    ]
                },
                aliases=store.aliases,
                action_tolerance=action_tolerance,
                pit_caveat="历史修订不能还原首次发布版本；bool缺失结合is_listed/has_quote；因子合成公司行为仅收盘后可知。",
                industry_caveat=industry_caveat or INDUSTRY_CAVEAT,
            )
            metadata["fields"]["industry"]["known_at"] = metadata["industry_caveat"]
            metadata["fields"]["industry"]["coverage"] = coverage
            securities.write_parquet(staging / "securities.parquet")
            calendar_table.write_parquet(staging / "calendar.parquet")
            atomic_json(staging / "metadata.json", metadata)
            os.replace(staging, target)
            atomic_json(base / "current.json", {"version": version})
            collect_garbage(store, keep, locked=True)
            return metadata
        finally:
            arrays.clear()
            if staging.exists():
                shutil.rmtree(staging)
