"""U 本位五分钟 metrics：归档、对齐、内容版本和按需读取。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from ashare.paths import dataset_dir

from ..store import atomic_json, lock
from . import minute
from .archive import archive_key as kline_key
from .sources import VISION, PublicClient
from .storage import RawStore, digest, encoded
from .timeutil import DAY, iso, millis, months

DEFINITION = "crypto-metrics-v4-registered"
STEP = 300_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")
# 经人工复核登记；每个新合约必须单独登记，证据只能核对，不能移动边界。
REGISTERED_SWITCHES = {"BTCUSDT": 1709510400000, "ETHUSDT": 1709510400000}
SWITCH_WINDOW = DAY
FIELDS = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "top_account_ratio",
    "sum_toptrader_long_short_ratio": "top_position_ratio",
    "count_long_short_ratio": "global_account_ratio",
    "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
}
# 全历史误差分布和阈值依据见 C8_NOTES；相等归入不确定区域。
MATCH_ERROR = 0.01
REJECT_ERROR = 0.1
MISMATCH_ERROR = 0.05
MIN_PAIRED_ROWS = 120
OPPOSITE_RUN_LIMIT = 2
NULL_VALUE = 1
INHERITED_ALIGNMENT = 2
TAKER_MISMATCH = 4
SCHEMA = {
    "create_time": pl.Int64,
    "time": pl.Int64,
    "available_at": pl.Int64,
    "alignment": pl.UInt8,
    "quality": pl.UInt8,
    **dict.fromkeys(FIELDS.values(), pl.Float64),
}


def data_root(root=None):
    return dataset_dir("crypto_metrics", root)


def archive_key(symbol, day):
    if symbol not in SYMBOLS:
        raise ValueError("metrics 只支持 BTCUSDT/ETHUSDT")
    day = iso(millis(day))[:10]
    # 复用远端根路径定义；这里不是本地 data 目录。
    base = kline_key("usdm", "klines", symbol, day, frequency="daily")
    return base.replace(f"klines/{symbol}/1h/", f"metrics/{symbol}/").replace(
        f"{symbol}-1h-", f"{symbol}-metrics-"
    )


def sync(
    *,
    root=None,
    symbols=SYMBOLS,
    start=None,
    end=None,
    workers=4,
    refresh=False,
    client=None,
):
    """列举全部已发布日档；只下缺失对象。每份对象落盘即可断点续传。"""
    if not 1 <= workers <= 4:
        raise ValueError("下载并发必须为 1..4")
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols) or set(symbols) - set(SYMBOLS):
        raise ValueError("交易对非法或重复")
    root = data_root(root)
    began = time.perf_counter()
    own = client is None
    listing_client = client or PublicClient(max_bytes=1_000_000_000)
    try:
        with lock(root / ".metrics.lock"):
            store = RawStore(root)
            clients = []
            try:
                old_path = root / "metrics_plan.json"
                old = json.loads(old_path.read_text()) if old_path.exists() else {}
                archives = dict(old.get("archives", {}))
                keys = []
                for symbol in symbols:
                    prefix = archive_key(symbol, "2020-01-01").rsplit("/", 1)[0] + "/"
                    for key in listing_client.listing(prefix):
                        if not key.endswith(".zip"):
                            continue
                        day = key[-14:-4]
                        if key != archive_key(symbol, day):
                            raise ValueError("上游归档键不匹配")
                        stamp = millis(day)
                        if (start is None or stamp >= millis(start)) and (
                            end is None or stamp < millis(end)
                        ):
                            keys.append(key)
                if not keys:
                    raise ValueError("请求范围没有归档")
                keys = sorted(set(keys))
                pending = set(old.get("pending", [])) | set(keys)
                atomic_json(
                    old_path,
                    dict(
                        definition=DEFINITION,
                        archives=archives,
                        failures=old.get("failures", []),
                        pending=sorted(pending),
                    ),
                )
                failures, fetched, byte_count = [], 0, 0

                local = threading.local()

                def fetch(key):
                    if not refresh and store.complete(key):
                        return key, None, None, None
                    try:
                        # 每个线程复用 HTTP 连接；SQLite 写入只由主线程完成。
                        if client is not None:
                            c = client
                        else:
                            if not hasattr(local, "client"):
                                local.client = PublicClient(max_bytes=1_000_000_000)
                                clients.append(local.client)
                            c = local.client
                        check = c.get(VISION + key + ".CHECKSUM")
                        fields = check.decode().split()
                        if (
                            len(fields) != 2
                            or fields[1].lstrip("*") != key.rsplit("/", 1)[-1]
                        ):
                            raise ValueError("CHECKSUM 文件名不匹配")
                        payload = c.get(VISION + key)
                        if digest(payload) != fields[0]:
                            raise ValueError("CHECKSUM 不匹配")
                        return key, payload, check.decode(), None
                    except Exception as exc:
                        return key, None, None, type(exc).__name__

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for key, payload, checksum, error in pool.map(fetch, keys):
                        if error:
                            failures.append(dict(key=key, error=error))
                            continue
                        if payload is not None:
                            store.put(
                                key,
                                payload,
                                source="archive",
                                checksum=digest(payload),
                                checksum_text=checksum,
                            )
                            fetched += 1
                            byte_count += len(payload)
                        ref = store.manifest.read()[key]
                        archives[key] = {
                            f: ref[f]
                            for f in ("sha256", "bytes", "checksum", "checksum_text")
                        }
                        pending.discard(key)
                # 旧失败不能因本次只重试另一段而遗忘。
                failures += [f for f in old.get("failures", []) if f["key"] not in keys]
                atomic_json(
                    old_path,
                    dict(
                        definition=DEFINITION,
                        archives=archives,
                        failures=failures,
                        pending=sorted(pending),
                    ),
                )
                report = dict(
                    archives=len(archives),
                    downloaded=fetched,
                    downloaded_bytes=byte_count,
                    failures=failures,
                    seconds=time.perf_counter() - began,
                )
                atomic_json(root / "sync_report.json", report)
                if failures:
                    raise ValueError(f"metrics 同步失败 {len(failures)} 项；可续跑")
                return report
            finally:
                for c in clients:
                    c.client.close()
                store.close()
    finally:
        if own:
            listing_client.client.close()


def parse_archive(payload, symbol, day):
    """只允许原字符串完全相同的重复；空串为 null，其余非法数值拒绝。"""
    lo = millis(day)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1 or z.testzip():
            raise ValueError("metrics ZIP 布局或 CRC 损坏")
        reader = csv.DictReader(io.StringIO(z.read(names[0]).decode("utf-8-sig")))
        expected_fields = {"create_time", "symbol", *FIELDS}
        if set(reader.fieldnames or []) != expected_fields or len(
            reader.fieldnames
        ) != len(expected_fields):
            raise ValueError("metrics CSV 表头不匹配")
        seen, rows, duplicates = {}, [], 0
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("metrics CSV 列数不匹配")
            t = millis(row["create_time"])
            if row["symbol"] != symbol or t % STEP or not lo <= t < lo + DAY:
                raise ValueError("metrics 时间网格/日期/交易对不匹配")
            if t in seen:
                if row != seen[t]:
                    raise ValueError("metrics 重复行内容不同")
                duplicates += 1
                continue
            seen[t] = row
            values = [None if row[f] == "" else float(row[f]) for f in FIELDS]
            if any(v is not None and (not np.isfinite(v) or v < 0) for v in values):
                raise ValueError("metrics 数值非法")
            rows.append((t, *values))
    if not rows:
        raise ValueError("metrics 空归档")
    frame = pl.DataFrame(
        rows,
        schema={"create_time": pl.Int64, **dict.fromkeys(FIELDS.values(), pl.Float64)},
        orient="row",
    ).sort("create_time")
    return frame, duplicates


def minute_windows(frame):
    """五根完整分钟线才参与比较；量使用币本位，零卖量为未知。"""
    if (
        frame["open_time"].n_unique() != frame.height
        or (frame["open_time"] % 60_000 != 0).any()
    ):
        raise ValueError("分钟输入有重复或非网格行")
    grouped = (
        frame.with_columns((pl.col("open_time") // STEP * STEP).alias("window"))
        .group_by("window")
        .agg(
            pl.len().alias("n"),
            pl.col("volume").sum(),
            pl.col("taker_buy_volume").sum(),
        )
        .sort("window")
    )
    valid = grouped.filter(
        (pl.col("n") == 5) & (pl.col("volume") > pl.col("taker_buy_volume"))
    )
    return dict(
        zip(
            valid["window"].to_list(),
            (
                valid["taker_buy_volume"]
                / (valid["volume"] - valid["taker_buy_volume"])
            ).to_list(),
            strict=True,
        )
    )


def errors(frame, windows):
    actual = frame["taker_buy_sell_ratio"].to_numpy()
    stamps = frame["create_time"].to_numpy()
    expected = np.array(
        [
            [windows.get(int(t) - STEP, np.nan), windows.get(int(t), np.nan)]
            for t in stamps
        ]
    ).reshape(-1, 2)
    return np.divide(
        np.abs(actual[:, None] - expected),
        np.maximum(np.abs(expected), 1e-12),
        out=np.full_like(expected, np.nan),
        where=np.isfinite(expected),
    )


def alignment_days(
    stamps, error, *, symbol="BTCUSDT", match=MATCH_ERROR, reject=REJECT_ERROR
):
    """登记常量赋值，UTC 日中位数及跨日反向证据仅用于核对。"""
    if symbol not in REGISTERED_SWITCHES:
        raise ValueError("UNREGISTERED_SWITCH: 新合约需要先人工登记时间戳切换时刻")
    switch = REGISTERED_SWITCHES[symbol]
    if switch % DAY:
        raise ValueError("REGISTERED_BOUNDARY: 登记切换时刻必须在 UTC 日界")
    stamps, error = np.asarray(stamps), np.asarray(error, float)
    if (
        stamps.ndim != 1
        or not len(stamps)
        or not np.issubdtype(stamps.dtype, np.integer)
        or error.shape != (len(stamps), 2)
        or not 0 < match < reject
        or np.any(stamps % STEP)
        or np.any(np.diff(stamps) <= 0)
    ):
        raise ValueError("对齐时间/误差/阈值非法")
    day_ids = stamps // DAY * DAY
    days = np.arange(day_ids[0], day_ids[-1] + DAY, DAY)
    indices = np.searchsorted(day_ids, np.append(days, days[-1] + DAY))
    stats = []
    for day, a, b in zip(days, indices[:-1], indices[1:], strict=True):
        paired = error[a:b][np.isfinite(error[a:b]).all(axis=1)]
        medians = np.median(paired, axis=0).tolist() if len(paired) else []
        vote = -1
        if len(paired) >= MIN_PAIRED_ROWS:
            if medians[0] < match and medians[1] > reject:
                vote = 0
            elif medians[1] < match and medians[0] > reject:
                vote = 1
        choice = int(day >= switch)
        if vote >= 0 and vote != choice:
            raise ValueError(
                "REGISTERED_ALIGNMENT: 日证据与登记矛盾，可能出现新切换，请人工复核登记"
            )
        stats.append(
            dict(
                day=int(day),
                rows=int(b - a),
                both_compared=len(paired),
                medians=medians,
                alignment=choice,
                inherited=int(vote < 0),
            )
        )
    run = 0
    for d, a, b in zip(stats, indices[:-1], indices[1:], strict=True):
        choice = d["alignment"]
        block = error[a:b]
        reverse = (
            np.isfinite(block).all(axis=1)
            & (block[:, choice] > reject)
            & (block[:, 1 - choice] < match)
        )
        d["opposite_rows"] = int(reverse.sum())
        d["opposite_max_run"] = 0
        for stamp, opposite in zip(stamps[a:b], reverse, strict=True):
            if opposite and abs(int(stamp) - switch) <= SWITCH_WINDOW:
                raise ValueError(
                    "SWITCH_WINDOW: 切换前后 24 小时存在反向证据，可能出现新切换，请人工复核登记"
                )
            # 连续已观测行累计，午夜和缺口不清零；灰区、缺失值或同向行清零。
            run = run + 1 if opposite else 0
            d["opposite_max_run"] = max(d["opposite_max_run"], run)
            if run >= OPPOSITE_RUN_LIMIT:
                raise ValueError(
                    "SUSTAINED_OPPOSITE: 跨日持续反向证据与登记矛盾，可能出现新切换，请人工复核登记"
                )
    return stats


def resolve_alignment(
    stamps, error, *, symbol="BTCUSDT", match=MATCH_ERROR, reject=REJECT_ERROR
):
    stats = alignment_days(stamps, error, symbol=symbol, match=match, reject=reject)
    indices = (np.asarray(stamps) // DAY * DAY - stats[0]["day"]) // DAY
    return (
        np.array([d["alignment"] for d in stats], dtype=np.uint8)[indices],
        np.array([d["inherited"] for d in stats], dtype=bool)[indices],
    )


def normalize(frame, error, *, symbol="BTCUSDT"):
    stamps = frame["create_time"].to_numpy()
    alignment, inherited = resolve_alignment(stamps, error, symbol=symbol)
    times = stamps - (alignment == 0) * STEP
    chosen = np.asarray(error)[np.arange(len(stamps)), alignment]
    mismatch = np.isfinite(chosen) & (chosen > MISMATCH_ERROR)
    nulls = frame.select(pl.any_horizontal(pl.all().is_null())).to_series().to_numpy()
    return (
        frame.with_columns(
            pl.Series("time", times),
            pl.Series("available_at", times + STEP),
            pl.Series("alignment", alignment),
            pl.Series(
                "quality",
                nulls.astype(np.uint8) * NULL_VALUE
                + inherited.astype(np.uint8) * INHERITED_ALIGNMENT
                + mismatch.astype(np.uint8) * TAKER_MISMATCH,
            ),
        )
        .select(*SCHEMA)
        .cast(SCHEMA)
    )


def alignment_quality(stamps, error, *, symbol="BTCUSDT"):
    """日统计、按 create_time UTC 年统计；比例同时给全部行和可比行分母。"""
    days = alignment_days(stamps, error, symbol=symbol)
    stamps, error = np.asarray(stamps), np.asarray(error)
    years = {}
    for d in days:
        a, b = np.searchsorted(stamps, [d["day"], d["day"] + DAY])
        chosen = error[a:b, d["alignment"]]
        d["compared"] = int(np.isfinite(chosen).sum())
        d["mismatches"] = int(
            np.count_nonzero(np.isfinite(chosen) & (chosen > MISMATCH_ERROR))
        )
        d["mismatch_ratio"] = d["mismatches"] / d["rows"] if d["rows"] else 0.0
        d["comparable_mismatch_ratio"] = (
            d["mismatches"] / d["compared"] if d["compared"] else 0.0
        )
        y = years.setdefault(iso(d["day"])[:4], dict(rows=0, compared=0, mismatches=0))
        for key in ("rows", "compared", "mismatches"):
            y[key] += d[key]
    for y in years.values():
        y["mismatch_ratio"] = y["mismatches"] / y["rows"] if y["rows"] else 0.0
        y["comparable_mismatch_ratio"] = (
            y["mismatches"] / y["compared"] if y["compared"] else 0.0
        )
    return days, years


def gaps(stamps, start, end):
    out, cursor = [], start
    for stamp in stamps:
        if stamp > cursor:
            out.append(dict(start=cursor, end=stamp, rows=(stamp - cursor) // STEP))
        cursor = stamp + STEP
    if cursor < end:
        out.append(dict(start=cursor, end=end, rows=(end - cursor) // STEP))
    return out


def _raw(root, key, ref):
    minute._hash_name(ref["sha256"])
    payload = (root / "objects" / ref["sha256"]).read_bytes()
    check = ref["checksum_text"].split()
    if (
        digest(payload) != ref["sha256"]
        or ref["checksum"] != ref["sha256"]
        or len(payload) != ref["bytes"]
        or len(check) != 2
        or check[0] != ref["sha256"]
        or check[1].lstrip("*") != key.rsplit("/", 1)[-1]
    ):
        raise ValueError("metrics 原始归档/CHECKSUM 损坏")
    return payload


def _load_minutes(symbol, start, end, *, root, version):
    meta = minute.metadata(root=root, version=version)
    frames = []
    for label in months(start, end):
        key = f"usdm/{symbol}/klines/{label}"
        if key in meta["partitions"]:
            frames.append(
                minute.bars(
                    symbol,
                    "usdm",
                    "klines",
                    max(start, millis(label + "-01")),
                    min(end, minute.month_end(label)),
                    root=root,
                    version=version,
                )
            )
    return pl.concat(frames) if frames else pl.DataFrame(schema=minute.SCHEMA)


def build(*, root=None, minute_root=None, minute_version=None):
    """按月复算、核对每个标的已登记的切换常量，冻结分钟源版本和全部质量证据。"""
    root = data_root(root)
    minute_root = minute.data_root(minute_root)
    minute_version = minute.metadata(root=minute_root, version=minute_version)[
        "version"
    ]
    began = time.perf_counter()
    with lock(root / ".metrics.lock"):
        plan = json.loads((root / "metrics_plan.json").read_text())
        if (
            plan.get("failures")
            or plan.get("pending")
            or plan.get("definition")
            not in (
                "crypto-metrics-v1",
                "crypto-metrics-v2-daily",
                "crypto-metrics-v3-daily-guarded",
                DEFINITION,
            )
        ):
            raise ValueError("metrics 同步未完成或定义不匹配")
        by_symbol = {}
        for key, ref in sorted(plan["archives"].items()):
            symbol = key.rsplit("/", 2)[1]
            if symbol not in REGISTERED_SWITCHES:
                raise ValueError(
                    "UNREGISTERED_SWITCH: 新合约需要先人工登记时间戳切换时刻"
                )
            day = key[-14:-4]
            if key != archive_key(symbol, day):
                raise ValueError("metrics 归档键不匹配")
            frame, dup = parse_archive(_raw(root, key, ref), symbol, day)
            by_symbol.setdefault(symbol, []).append((day, frame, dup))
        if not by_symbol:
            raise ValueError("metrics 没有归档")
        partitions, quality = {}, {}
        for symbol, parts in sorted(by_symbol.items()):
            frame = pl.concat([p[1] for p in parts]).sort("create_time")
            if frame["create_time"].n_unique() != frame.height:
                raise ValueError("跨日归档重复")
            err = []
            lo, hi = millis(parts[0][0]), millis(parts[-1][0]) + DAY
            for label in months(lo, hi):
                a, b = millis(label + "-01"), minute.month_end(label)
                chunk = frame.filter(
                    (pl.col("create_time") >= a) & (pl.col("create_time") < b)
                )
                bars = _load_minutes(
                    symbol, a - STEP, b + STEP, root=minute_root, version=minute_version
                )
                err.append(errors(chunk, minute_windows(bars)))
            error = np.concatenate(err)
            normalized = normalize(frame, error, symbol=symbol)
            aligned = normalized["alignment"].to_numpy()
            chosen = error[np.arange(len(error)), aligned]
            mismatches = np.flatnonzero(np.isfinite(chosen) & (chosen > MISMATCH_ERROR))
            daily, yearly = alignment_quality(
                frame["create_time"].to_numpy(), error, symbol=symbol
            )
            switches = [
                b["day"]
                for a, b in zip(daily, daily[1:], strict=False)
                if a["alignment"] != b["alignment"]
            ]
            quality[symbol] = dict(
                rows=frame.height,
                duplicates=sum(p[2] for p in parts),
                null_rows=int(
                    ((normalized["quality"].to_numpy() & NULL_VALUE) != 0).sum()
                ),
                null_cells=int(sum(frame.null_count().row(0))),
                inherited_rows=int(
                    (
                        (normalized["quality"].to_numpy() & INHERITED_ALIGNMENT) != 0
                    ).sum()
                ),
                gaps=gaps(frame["create_time"], lo, hi),
                normalized_gaps=gaps(
                    normalized["time"],
                    int(normalized["time"].min()),
                    int(normalized["available_at"].max()),
                ),
                switch_at=switches,
                days=daily,
                years=yearly,
                undetermined_days=sum(d["inherited"] for d in daily),
                inherited_days=sum(d["inherited"] for d in daily),
                determined_days=sum(not d["inherited"] for d in daily),
                mismatch_ratio=len(mismatches) / frame.height,
                comparable_mismatch_ratio=(
                    len(mismatches) / int(np.isfinite(chosen).sum())
                    if np.isfinite(chosen).any()
                    else 0.0
                ),
                compared=int(np.isfinite(chosen).sum()),
                mismatches=int(len(mismatches)),
                segments=[
                    dict(
                        alignment=k,
                        rows=int((aligned == k).sum()),
                        errors={
                            str(col): np.quantile(
                                error[(aligned == k) & np.isfinite(error[:, col]), col],
                                [0, 0.01, 0.5, 0.9, 0.99, 0.999, 1],
                            ).tolist()
                            if np.any((aligned == k) & np.isfinite(error[:, col]))
                            else []
                            for col in (0, 1)
                        },
                    )
                    for k in (0, 1)
                ],
            )
            for label in months(lo - STEP, hi + STEP):
                a, b = millis(label + "-01"), minute.month_end(label)
                chunk = normalized.filter((pl.col("time") >= a) & (pl.col("time") < b))
                if not chunk.height:
                    continue
                buf = io.BytesIO()
                chunk.write_parquet(buf, compression="zstd")
                payload = buf.getvalue()
                sha = digest(payload)
                part = f"{symbol}/{label}"
                relative = f"metrics/{part}/{sha}.parquet"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(payload)
                tmp.replace(path)
                partitions[part] = dict(
                    file=relative, sha256=sha, bytes=len(payload), rows=chunk.height
                )
        body = dict(
            definition=DEFINITION,
            registered_switches={s: REGISTERED_SWITCHES[s] for s in sorted(by_symbol)},
            thresholds=dict(
                match=MATCH_ERROR,
                reject=REJECT_ERROR,
                mismatch=MISMATCH_ERROR,
                min_paired=MIN_PAIRED_ROWS,
                opposite_run=OPPOSITE_RUN_LIMIT,
                switch_window=SWITCH_WINDOW,
            ),
            minute_version=minute_version,
            archives=plan["archives"],
            partitions=partitions,
            quality=quality,
        )
        version = digest(encoded(body))
        atomic_json(root / "versions" / f"{version}.json", dict(body, version=version))
        atomic_json(root / "latest.json", dict(version=version))
        atomic_json(
            root / "build_report.json",
            dict(version=version, seconds=time.perf_counter() - began),
        )
        return version


def metadata(*, root=None, version=None):
    root = data_root(root)
    version = version or json.loads((root / "latest.json").read_text())["version"]
    minute._hash_name(version)
    meta = json.loads((root / "versions" / f"{version}.json").read_text())
    body = {k: v for k, v in meta.items() if k != "version"}
    if (
        meta.get("version") != version
        or digest(encoded(body)) != version
        or meta.get("definition") != DEFINITION
    ):
        raise ValueError("metrics 内容版本损坏")
    return meta


def read(symbol, start, end, *, root=None, version=None):
    """按规范化窗口开始 time 读取 [start,end)；available_at 为全部字段可知时点。"""
    if symbol not in SYMBOLS:
        raise ValueError("metrics 标的非法")
    start, end = millis(start), millis(end)
    if start > end:
        raise ValueError("读取起点晚于终点")
    root = data_root(root)
    meta = metadata(root=root, version=version)
    frames = []
    for label in months(start, end) if start < end else []:
        part_id = f"{symbol}/{label}"
        part = meta["partitions"].get(part_id)
        if part is None:
            continue
        lo, hi = millis(label + "-01"), minute.month_end(label)
        for key, ref in meta["archives"].items():
            if f"/{symbol}/" in key and lo <= millis(key[-14:-4]) < hi + STEP:
                _raw(root, key, ref)
        minute._hash_name(part["sha256"])
        if part["file"] != f"metrics/{part_id}/{part['sha256']}.parquet":
            raise ValueError("metrics 分区路径损坏")
        payload = (root / part["file"]).read_bytes()
        if digest(payload) != part["sha256"] or len(payload) != part["bytes"]:
            raise ValueError("metrics parquet 损坏")
        chunk = pl.read_parquet(io.BytesIO(payload))
        if chunk.schema != SCHEMA:
            raise ValueError("metrics schema 损坏")
        frames.append(chunk)
    frame = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    return frame.filter((pl.col("time") >= start) & (pl.col("time") < end)).sort("time")


def status(*, root=None, version=None):
    root = data_root(root)
    if not version and not (root / "latest.json").exists():
        plan_path = root / "metrics_plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        return dict(
            built=False,
            raw_archives=len(plan.get("archives", {})),
            pending=len(plan.get("pending", [])),
            failures=len(plan.get("failures", [])),
        )
    meta = metadata(root=root, version=version)
    return dict(
        built=True,
        version=meta["version"],
        partitions=len(meta["partitions"]),
        quality=meta["quality"],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("sync", "build", "verify", "status"))
    p.add_argument("--root")
    p.add_argument("--minute-root")
    p.add_argument("--minute-version")
    p.add_argument("--version")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    if a.command == "sync":
        result = sync(
            root=a.root, start=a.start, end=a.end, workers=a.workers, refresh=a.refresh
        )
    elif a.command == "build":
        result = build(
            root=a.root, minute_root=a.minute_root, minute_version=a.minute_version
        )
    elif a.command == "verify":
        from .verify_metrics import verify

        result = verify(root=a.root, version=a.version, minute_root=a.minute_root)
    else:
        result = status(root=a.root, version=a.version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(isinstance(result, dict) and result.get("ok") is False)


if __name__ == "__main__":
    raise SystemExit(main())
