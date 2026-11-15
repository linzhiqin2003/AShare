"""分钟库独立审计：不导入 minute 的解析、缺口、哈希或聚合判据。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _body_hash(meta):
    return _sha(
        json.dumps(
            {k: v for k, v in meta.items() if k != "version"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def _calendar(label):
    first = datetime.strptime(label, "%Y-%m").replace(tzinfo=UTC)
    following = first.replace(
        year=first.year + (first.month == 12), month=first.month % 12 + 1
    )
    return int(first.timestamp() * 1000), int(following.timestamp() * 1000)


def _csv(payload, kind, lo, hi):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1 or z.testzip():
            raise ValueError("zip layout/CRC")
        reader = csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig")))
        rows = list(reader)
    if rows and rows[0][0].strip().lower().replace(" ", "_") == "open_time":
        rows = rows[1:]
    if not rows:
        raise ValueError("empty archive")
    result, previous, units = {}, None, set()
    for r in rows:
        if len(r) != 12:
            raise ValueError("column count")
        stamp = int(r[0])
        digits = len(str(stamp))
        units.add(digits)
        if digits == 13:
            t = stamp
        elif digits == 16 and stamp % 1000 == 0:
            t = stamp // 1000
        else:
            raise ValueError("timestamp unit")
        if (
            len(units) != 1
            or t % 60000
            or not lo <= t < hi
            or (previous is not None and t <= previous)
        ):
            raise ValueError("timestamp order/grid/range")
        previous = t
        values = tuple(float(r[i]) for i in (1, 2, 3, 4))
        if kind == "klines":
            values += (float(r[5]), float(r[7]), int(r[8]), float(r[9]), float(r[10]))
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("nonfinite/negative")
        result[t] = (t, *values)
    return result


def _missing(times, start, end):
    # 独立地生成整月期望网格，以差集重建缺口，不共享整理器的游标算法。
    absent = sorted(set(range(start, end, 60000)) - set(times))
    groups = []
    for t in absent:
        if groups and groups[-1]["end"] == t:
            groups[-1]["end"] += 60000
            groups[-1]["minutes"] += 1
        else:
            groups.append(dict(start=t, end=t + 60000, minutes=1))
    return groups


def reconcile_hourly(frame, hourly, kind, *, partition=None):
    """逐小时独立聚合；价格精确比，量仅将浮点舍入归为 roundoff。

    返回每个非零差异（不截断）、每字段最大差和缺失/不完整小时。
    量容差为 1e-8 + 1e-12*abs(hourly)，不是价格容差。
    """
    fields = (
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_quote_volume",
        ]
        if kind == "klines"
        else ["open", "high", "low", "close"]
        if kind == "mark"
        else ["close"]
    )
    mapping = {
        f: "mark_" + f if kind == "mark" else "index_close" if kind == "index" else f
        for f in fields
    }
    source = {r["time"]: r for r in hourly.iter_rows(named=True)}
    groups = {}
    for row in frame.iter_rows(named=True):
        groups.setdefault(row["open_time"] // 3_600_000 * 3_600_000, []).append(row)
    maxima = {f: 0.0 for f in fields}
    differences, incomplete, unavailable = [], [], []
    compared = 0
    for hour, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["open_time"])
        complete = [r["open_time"] for r in rows] == list(
            range(hour, hour + 3_600_000, 60000)
        )
        if not complete:
            incomplete.append(dict(time=hour, minutes=len(rows)))
        if hour not in source:
            unavailable.append(hour)
            continue
        ref = source[hour]
        values = dict(
            open=rows[0]["open"],
            close=rows[-1]["close"],
            high=max(r["high"] for r in rows),
            low=min(r["low"] for r in rows),
        )
        for f in fields:
            if f not in values:
                values[f] = math.fsum(r[f] for r in rows)
        compared += 1
        for f in fields:
            expected, actual = ref[mapping[f]], values[f]
            if expected is None or not math.isfinite(expected):
                differences.append(
                    dict(
                        time=hour,
                        field=f,
                        minute=actual,
                        hourly=None,
                        absolute=None,
                        reason="hourly_missing",
                        complete=complete,
                    )
                )
                continue
            delta = abs(actual - expected)
            maxima[f] = max(maxima[f], delta)
            if delta:
                tolerance = (
                    1e-8 + 1e-12 * abs(expected)
                    if f in ("volume", "quote_volume", "taker_buy_quote_volume")
                    else 0
                )
                reason = (
                    "incomplete_minutes"
                    if not complete
                    else "roundoff"
                    if delta <= tolerance
                    else "source_difference"
                )
                differences.append(
                    dict(
                        time=hour,
                        field=f,
                        minute=actual,
                        hourly=expected,
                        absolute=delta,
                        reason=reason,
                        complete=complete,
                    )
                )
    # 整个小时都没有分钟时，同样列出，不能因 group_by 没有组而消失。
    if partition:
        lo, hi = _calendar(partition.rsplit("/", 1)[-1])
        for hour in sorted(t for t in source if lo <= t < hi and t not in groups):
            incomplete.append(dict(time=hour, minutes=0))
    material = sum(d["reason"] != "roundoff" for d in differences)
    return dict(
        partition=partition,
        fields=fields,
        compared_hours=compared,
        max_absolute=maxima,
        differences=differences,
        material_differences=material,
        roundoff_differences=len(differences) - material,
        incomplete_hours=incomplete,
        hourly_unavailable=unavailable,
        # 没有任何小时被比较时是「无法对账」，不是通过（C7 审查 P2-1）。
        matches=compared > 0 and not material and not incomplete and not unavailable,
    )


def verify(*, root=None, version=None, hourly_root=None, hourly_version=None):
    """只读，独立核对全部原始 ZIP、Parquet、缺口与版本；可选严格小时面板对账。"""
    root = Path(root) if root is not None else Path("data/crypto_minute")
    errors, checked, total_rows, gap_segments = [], 0, 0, 0
    result = dict(ok=False, errors=errors)
    try:
        version = version or json.loads((root / "latest.json").read_text())["version"]
        if not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ValueError("version name")
        meta = json.loads((root / "versions" / f"{version}.json").read_text())
        if (
            meta.get("version") != version
            or _body_hash(meta) != version
            or meta.get("definition") != "crypto-minute-v1"
        ):
            raise ValueError("content version")
    except (ValueError, OSError, KeyError, TypeError):
        errors.append(dict(check="version"))
        return result
    hourly = None
    reconciliation = []
    if hourly_root is not None:
        from . import open as open_hourly

        hourly_root = Path(hourly_root)
        hourly_version = (
            hourly_version
            or json.loads((hourly_root / "latest.json").read_text())["version"]
        )
        ds = open_hourly(hourly_root, version=hourly_version)
        if ds.meta.get("allow_inferred_funding"):
            raise ValueError("分钟对账必须使用严格小时版本")
        hourly = ds.read(
            fields=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_quote_volume",
                "mark_open",
                "mark_high",
                "mark_low",
                "mark_close",
                "index_close",
            ]
        )
        result["hourly_version"] = hourly_version
    elif hourly_version is not None:
        raise ValueError("hourly_version 需要 hourly_root")
    for part_id, part in sorted(meta["partitions"].items()):
        try:
            market, symbol, kind, label = part_id.split("/")
            if (
                symbol not in ("BTCUSDT", "ETHUSDT")
                or market not in ("spot", "usdm")
                or kind not in ("klines", "mark", "index")
                or (market == "spot" and kind != "klines")
                or [part[k] for k in ("market", "symbol", "kind", "month")]
                != [market, symbol, kind, label]
            ):
                raise ValueError("partition identity")
            lo, hi = _calendar(label)
            raw_rows, month_days, daily_days = {}, set(), set()
            for key, ref in part["archives"].items():
                frequency = "monthly" if "/monthly/" in key else "daily"
                date = key.rsplit("-1m-", 1)[-1].removesuffix(".zip")
                base = "data/spot" if market == "spot" else "data/futures/um"
                dataset = {
                    "klines": "klines",
                    "mark": "markPriceKlines",
                    "index": "indexPriceKlines",
                }[kind]
                expected = (
                    f"{base}/{frequency}/{dataset}/{symbol}/1m/{symbol}-1m-{date}.zip"
                )
                if (
                    key != expected
                    or date[:7] != label
                    or len(date) != (7 if frequency == "monthly" else 10)
                ):
                    raise ValueError("archive identity")
                if frequency == "monthly":
                    a, b = lo, hi
                else:
                    a = int(
                        datetime.strptime(date, "%Y-%m-%d")
                        .replace(tzinfo=UTC)
                        .timestamp()
                        * 1000
                    )
                    b = a + 86_400_000
                sha = ref["sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", sha):
                    raise ValueError("raw hash name")
                payload = (root / "objects" / sha).read_bytes()
                checksum = ref["checksum_text"].split()
                if (
                    _sha(payload) != sha
                    or ref["checksum"] != sha
                    or len(payload) != ref["bytes"]
                    or len(checksum) != 2
                    or checksum[0] != sha
                    or checksum[1].lstrip("*") != key.rsplit("/", 1)[-1]
                ):
                    raise ValueError("checksum")
                decoded = _csv(payload, kind, a, b)
                if raw_rows.keys() & decoded.keys():
                    raise ValueError("overlap")
                raw_rows.update(decoded)
                (month_days if frequency == "monthly" else daily_days).update(
                    t // 86_400_000 for t in decoded
                )
                checked += 1
            if month_days & daily_days:
                raise ValueError("daily overwrote monthly day")
            sha = part["sha256"]
            if (
                not re.fullmatch(r"[0-9a-f]{64}", sha)
                or part["file"] != f"bars/{part_id}/{sha}.parquet"
            ):
                raise ValueError("parquet identity")
            path = root / part["file"]
            payload = path.read_bytes()
            if _sha(payload) != sha or len(payload) != part["bytes"]:
                raise ValueError("parquet hash")
            frame = pl.read_parquet(io.BytesIO(payload))
            fields = ["open_time", "open", "high", "low", "close"]
            if kind == "klines":
                fields += [
                    "volume",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                ]
            schema = {
                f: pl.Int64 if f in ("open_time", "count") else pl.Float64
                for f in fields
            }
            if frame.schema != schema:
                raise ValueError("parquet schema")
            actual = frame.rows()
            expected = [raw_rows[t] for t in sorted(raw_rows)]
            if actual != expected:
                raise ValueError("raw/parquet values")
            gaps = _missing(raw_rows, lo, hi)
            if part["quality"] != dict(
                expected_rows=(hi - lo) // 60000, rows=len(expected), gaps=gaps
            ):
                raise ValueError("quality gaps/row count")
            total_rows += len(expected)
            gap_segments += len(gaps)
            if hourly is not None:
                instrument = f"{market}:{symbol}"
                subset = hourly.filter(
                    (pl.col("instrument") == instrument)
                    & (pl.col("time") >= lo)
                    & (pl.col("time") < hi)
                )
                reconciliation.append(
                    reconcile_hourly(frame, subset, kind, partition=part_id)
                )
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            zipfile.BadZipFile,
            pl.exceptions.PolarsError,
        ) as exc:
            errors.append(
                dict(
                    check="partition",
                    partition=part_id,
                    reason=str(exc)
                    if isinstance(exc, ValueError)
                    else type(exc).__name__,
                )
            )
    result.update(
        ok=not errors,
        version=version,
        checked_archives=checked,
        partitions=len(meta["partitions"]),
        rows=total_rows,
        gap_segments=gap_segments,
    )
    if hourly is not None:
        result["reconciliation"] = reconciliation
        result["hourly_matches"] = (
            all(r["matches"] for r in reconciliation) and not errors
        )
    return result
