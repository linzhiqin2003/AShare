"""metrics 独立校验：独立解码、分钟聚合、登记对齐核对与缺口计算。"""

import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import UTC, datetime
from statistics import median

import numpy as np
import polars as pl

from ashare.paths import dataset_dir

from ..store import atomic_json
from . import minute


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _decode(payload, symbol, day):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1 or z.testzip():
            raise ValueError("ZIP/CRC")
        reader = csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig")))
        header = next(reader)
        fields = [
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ]
        if sorted(header) != sorted(["create_time", "symbol", *fields]):
            raise ValueError("CSV header")
        positions = [header.index(f) for f in fields]
        records, seen, duplicate = {}, {}, 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError("CSV columns")
            stamp = datetime.fromisoformat(row[header.index("create_time")])
            stamp = (
                stamp.replace(tzinfo=UTC)
                if stamp.tzinfo is None
                else stamp.astimezone(UTC)
            )
            t = int(stamp.timestamp() * 1000)
            if (
                stamp.strftime("%Y-%m-%d") != day
                or t % 300000
                or row[header.index("symbol")] != symbol
            ):
                raise ValueError("CSV date/grid/symbol")
            if t in seen:
                if seen[t] != row:
                    raise ValueError("conflicting duplicate")
                duplicate += 1
                continue
            seen[t] = row
            values = tuple(None if row[i] == "" else float(row[i]) for i in positions)
            if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
                raise ValueError("nonfinite/negative")
            records[t] = values
    if not records:
        raise ValueError("empty archive")
    return records, duplicate


def _alignment(times, error, match, reject, *, symbol="BTCUSDT"):
    """独立使用 Python median 与 UTC 日桶，不调用生产判定代码。"""
    registered = {"BTCUSDT": 1709510400000, "ETHUSDT": 1709510400000}
    if symbol not in registered:
        raise ValueError(
            "UNREGISTERED_SWITCH: register contract switch before building"
        )
    switch = registered[symbol]
    if switch % 86400000:
        raise ValueError("REGISTERED_BOUNDARY: switch must be a UTC day boundary")
    if len(times) == 0 or len(times) != len(error):
        raise ValueError("empty/invalid alignment input")
    grouped = {}
    for t, pair in zip(times, error, strict=True):
        grouped.setdefault(int(t) // 86400000 * 86400000, []).append(pair)
    days = []
    for day in range(min(grouped), max(grouped) + 86400000, 86400000):
        pairs = grouped.get(day, [])
        paired = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
        med = [median(p[i] for p in paired) for i in (0, 1)] if paired else []
        choice = -1
        if len(paired) >= 120:
            if med[0] < match and med[1] > reject:
                choice = 0
            elif med[1] < match and med[0] > reject:
                choice = 1
        assigned = int(day >= switch)
        if choice >= 0 and choice != assigned:
            raise ValueError(
                "REGISTERED_ALIGNMENT: daily evidence contradicts registry; review and register possible new switch"
            )
        days.append(
            dict(
                day=day,
                rows=len(pairs),
                both_compared=len(paired),
                medians=med,
                alignment=assigned,
                inherited=int(choice == -1),
            )
        )
    run = 0
    by_time = dict(zip(map(int, times), error, strict=True))
    for d in days:
        choice = d["alignment"]
        longest = count = 0
        for stamp in range(d["day"], d["day"] + 86400000, 300000):
            if stamp not in by_time:
                continue
            pair = by_time[stamp]
            opposite = (
                all(math.isfinite(v) for v in pair)
                and pair[choice] > reject
                and pair[1 - choice] < match
            )
            if opposite and abs(stamp - switch) <= 86400000:
                raise ValueError(
                    "SWITCH_WINDOW: contrary evidence within switch window; review and register possible new switch"
                )
            if opposite:
                run += 1
                count += 1
                longest = max(longest, run)
            else:
                run = 0
            if run >= 2:
                raise ValueError(
                    "SUSTAINED_OPPOSITE: cross-day contrary evidence; review and register possible new switch"
                )
        d["opposite_rows"] = count
        d["opposite_max_run"] = longest
    lookup = {d["day"]: d for d in days}
    return (
        [lookup[int(t) // 86400000 * 86400000]["alignment"] for t in times],
        [bool(lookup[int(t) // 86400000 * 86400000]["inherited"]) for t in times],
        days,
    )


def _quality(times, error, alignment, days):
    grouped = {d["day"]: [] for d in days}
    for t, pair, a in zip(times, error, alignment, strict=True):
        grouped[int(t) // 86400000 * 86400000].append(pair[a])
    years = {}
    for d in days:
        chosen = grouped[d["day"]]
        d["compared"] = sum(math.isfinite(e) for e in chosen)
        d["mismatches"] = sum(math.isfinite(e) and e > 0.05 for e in chosen)
        d["mismatch_ratio"] = d["mismatches"] / d["rows"] if d["rows"] else 0.0
        d["comparable_mismatch_ratio"] = (
            d["mismatches"] / d["compared"] if d["compared"] else 0.0
        )
        year = datetime.fromtimestamp(d["day"] / 1000, UTC).strftime("%Y")
        y = years.setdefault(year, dict(rows=0, compared=0, mismatches=0))
        for k in ("rows", "compared", "mismatches"):
            y[k] += d[k]
    for y in years.values():
        y["mismatch_ratio"] = y["mismatches"] / y["rows"] if y["rows"] else 0.0
        y["comparable_mismatch_ratio"] = (
            y["mismatches"] / y["compared"] if y["compared"] else 0.0
        )
    return years


def _equal_summary(actual, expected):
    # 独立 math.fsum 聚合的浮点汇总容许 1e-12 差；计数/判定/位精确比较。
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equal_summary(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equal_summary(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def _gaps(times, lo, hi):
    absent = sorted(set(range(lo, hi, 300000)) - set(times))
    out = []
    for t in absent:
        if out and out[-1]["end"] == t:
            out[-1]["end"] = t + 300000
            out[-1]["rows"] += 1
        else:
            out.append(dict(start=t, end=t + 300000, rows=1))
    return out


def _minute_ratios(symbol, lo, hi, *, root, version):
    """与生产聚合独立，完整五分钟组以 math.fsum 累加。"""
    meta = minute.metadata(root=root, version=version)
    ratios = {}
    for part_id, part in meta["partitions"].items():
        if not part_id.startswith(f"usdm/{symbol}/klines/"):
            continue
        month = part["month"]
        mlo = int(
            datetime.fromisoformat(month + "-01").replace(tzinfo=UTC).timestamp() * 1000
        )
        mhi = minute.month_end(month)
        if mhi < lo - 300000 or mlo > hi + 300000:
            continue
        bars = minute.bars(
            symbol, "usdm", "klines", mlo, mhi, root=root, version=version
        )
        groups = {}
        for t, v, b in bars.select(
            "open_time", "volume", "taker_buy_volume"
        ).iter_rows():
            groups.setdefault(t // 300000 * 300000, []).append((t, v, b))
        for t, group in groups.items():
            if sorted(r[0] for r in group) != list(range(t, t + 300000, 60000)):
                continue
            total, buy = math.fsum(r[1] for r in group), math.fsum(r[2] for r in group)
            if total > buy:
                ratios[t] = buy / (total - buy)
    return ratios


def verify(*, root=None, version=None, minute_root=None, symbols=None):
    """差异逐条保存在 metrics 根下；返回值只有汇总，不向仓库输出行情。"""
    root = dataset_dir("crypto_metrics", root)
    minute_root = dataset_dir("crypto_minute", minute_root)
    if version is None and not (root / "latest.json").exists():
        if symbols is not None:
            raise ValueError("scoped verification requires a published version")
        return audit_raw(root=root, minute_root=minute_root)
    version = version or json.loads((root / "latest.json").read_text())["version"]
    if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("invalid version")
    meta = json.loads((root / "versions" / f"{version}.json").read_text())
    body = {k: v for k, v in meta.items() if k != "version"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if (
        _sha(encoded) != version
        or meta.get("version") != version
        or meta.get("definition") != "crypto-metrics-v4-registered"
    ):
        raise ValueError("version hash/definition")
    if meta["thresholds"] != dict(
        match=0.01,
        reject=0.1,
        mismatch=0.05,
        min_paired=120,
        opposite_run=2,
        switch_window=86400000,
    ):
        raise ValueError("unknown thresholds")
    available = {k.rsplit("/", 2)[1] for k in meta["archives"]}
    registered = {"BTCUSDT": 1709510400000, "ETHUSDT": 1709510400000}
    if available - registered.keys() or meta.get("registered_switches") != {
        s: registered[s] for s in available
    }:
        raise ValueError("REGISTERED_METADATA: unregistered or changed switch constant")
    requested = tuple(sorted(available)) if symbols is None else tuple(symbols)
    selected = set(requested)
    if (
        not selected
        or len(selected) != len(requested)
        or selected - available
        or available - {"BTCUSDT", "ETHUSDT"}
        or {k.split("/")[0] for k in meta["partitions"]} != available
    ):
        raise ValueError("invalid verification scope/coverage")
    raw, duplicates, ranges = {}, {}, {}
    for key, ref in sorted(meta["archives"].items()):
        if key.rsplit("/", 2)[1] not in selected:
            continue
        if len(ref["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in ref["sha256"]
        ):
            raise ValueError("object hash")
        payload = (root / "objects" / ref["sha256"]).read_bytes()
        parts = ref["checksum_text"].split()
        if (
            _sha(payload) != ref["sha256"]
            or len(payload) != ref["bytes"]
            or ref["checksum"] != ref["sha256"]
            or len(parts) != 2
            or parts[0] != ref["sha256"]
            or parts[1].lstrip("*") != key.rsplit("/", 1)[-1]
        ):
            raise ValueError("raw CHECKSUM")
        symbol = key.rsplit("/", 2)[1]
        day = key[-14:-4]
        if (
            symbol not in ("BTCUSDT", "ETHUSDT")
            or key.rsplit("/", 1)[-1] != f"{symbol}-metrics-{day}.zip"
            or "/daily/metrics/" not in key
        ):
            raise ValueError("archive identity")
        records, count = _decode(payload, symbol, day)
        target = raw.setdefault(symbol, {})
        if target.keys() & records.keys():
            raise ValueError("inter-archive duplicate")
        target.update(records)
        duplicates[symbol] = duplicates.get(symbol, 0) + count
        lo = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
        ranges.setdefault(symbol, []).append(lo)
    output = {}
    columns = [
        "open_interest",
        "open_interest_value",
        "top_account_ratio",
        "top_position_ratio",
        "global_account_ratio",
        "taker_buy_sell_ratio",
    ]
    for part_id, part in meta["partitions"].items():
        symbol, month = part_id.split("/")
        if symbol not in selected:
            continue
        if part["file"] != f"metrics/{symbol}/{month}/{part['sha256']}.parquet":
            raise ValueError("partition path")
        payload = (root / part["file"]).read_bytes()
        if _sha(payload) != part["sha256"] or len(payload) != part["bytes"]:
            raise ValueError("parquet hash")
        frame = pl.read_parquet(io.BytesIO(payload))
        expected_schema = pl.Schema(
            {
                "create_time": pl.Int64,
                "time": pl.Int64,
                "available_at": pl.Int64,
                "alignment": pl.UInt8,
                "quality": pl.UInt8,
                **dict.fromkeys(columns, pl.Float64),
            }
        )
        if frame.schema != expected_schema:
            raise ValueError("parquet schema")
        if frame.height != part["rows"] or not frame["time"].is_sorted():
            raise ValueError("partition rows/order")
        target = output.setdefault(symbol, {})
        for r in frame.iter_rows(named=True):
            if (
                r["create_time"] in target
                or r["time"] % 300000
                or r["available_at"] != r["time"] + 300000
                or datetime.fromtimestamp(r["time"] / 1000, UTC).strftime("%Y-%m")
                != month
            ):
                raise ValueError("normalized grid/duplicate/partition")
            target[r["create_time"]] = r
    details, summaries, taker_mismatches = [], {}, []
    for symbol, records in raw.items():
        if set(records) != set(output.get(symbol, {})):
            raise ValueError("raw/normalized coverage")
        ratios = _minute_ratios(
            symbol,
            min(records),
            max(records),
            root=minute_root,
            version=meta["minute_version"],
        )
        times = sorted(records)
        error = []
        for t in times:
            actual = records[t][-1]
            error.append(
                [
                    abs(actual - ratios[w]) / max(abs(ratios[w]), 1e-12)
                    if actual is not None and w in ratios
                    else math.nan
                    for w in (t - 300000, t)
                ]
            )
        alignment, inherited, days = _alignment(
            times,
            error,
            meta["thresholds"]["match"],
            meta["thresholds"]["reject"],
            symbol=symbol,
        )
        years = _quality(times, error, alignment, days)
        compared, mismatches, null_cells, null_rows = 0, 0, 0, 0
        for i, t in enumerate(times):
            row = output[symbol][t]
            nulls = sum(v is None for v in records[t])
            null_cells += nulls
            null_rows += bool(nulls)
            expected_time = t if alignment[i] else t - 300000
            delta = error[i][alignment[i]]
            mismatch = math.isfinite(delta) and delta > 0.05
            if (
                row["alignment"] != alignment[i]
                or row["time"] != expected_time
                or row["quality"]
                != int(bool(nulls)) + 2 * int(inherited[i]) + 4 * int(mismatch)
                or tuple(row[f] for f in columns) != records[t]
            ):
                details.append(
                    dict(symbol=symbol, create_time=t, reason="normalized_fields")
                )
            if math.isfinite(delta):
                compared += 1
                if mismatch:
                    mismatches += 1
                    taker_mismatches.append(
                        dict(
                            symbol=symbol,
                            create_time=t,
                            relative_error=delta,
                            reason="taker_ratio",
                        )
                    )
        gaps = _gaps(times, min(ranges[symbol]), max(ranges[symbol]) + 86400000)
        switches = [
            days[i]["day"]
            for i in range(1, len(days))
            if days[i]["alignment"] != days[i - 1]["alignment"]
        ]
        error_array = np.asarray(error)
        labels = np.asarray(alignment)
        segments = []
        for label in (0, 1):
            distributions = {}
            for column in (0, 1):
                values = error_array[
                    (labels == label) & np.isfinite(error_array[:, column]), column
                ]
                distributions[str(column)] = (
                    np.percentile(values, [0, 1, 50, 90, 99, 99.9, 100]).tolist()
                    if len(values)
                    else []
                )
            segments.append(
                dict(
                    alignment=label,
                    rows=int(np.count_nonzero(labels == label)),
                    errors=distributions,
                )
            )
        expected = dict(
            rows=len(times),
            duplicates=duplicates[symbol],
            null_rows=null_rows,
            null_cells=null_cells,
            inherited_rows=sum(inherited),
            gaps=gaps,
            normalized_gaps=_gaps(
                [output[symbol][t]["time"] for t in times],
                min(r["time"] for r in output[symbol].values()),
                max(r["available_at"] for r in output[symbol].values()),
            ),
            switch_at=switches,
            days=days,
            years=years,
            segments=segments,
            undetermined_days=sum(d["inherited"] for d in days),
            inherited_days=sum(d["inherited"] for d in days),
            determined_days=sum(not d["inherited"] for d in days),
            mismatch_ratio=mismatches / len(times),
            comparable_mismatch_ratio=mismatches / compared if compared else 0.0,
            compared=compared,
            mismatches=mismatches,
        )
        if any(
            not _equal_summary(meta["quality"][symbol][k], v)
            for k, v in expected.items()
        ):
            details.append(dict(symbol=symbol, reason="quality_summary"))
        summaries[symbol] = {
            **expected,
            "gap_rows": sum(g["rows"] for g in gaps),
            "gap_segments": len(gaps),
        }
        summaries[symbol].pop("gaps")
        summaries[symbol].pop("normalized_gaps")
    report = dict(
        ok=not details,
        differences=len(details),
        symbols=summaries,
        complete=selected == available,
        version=version,
        minute_version=meta["minute_version"],
    )
    suffix = "-" + "-".join(sorted(selected)) if symbols is not None else ""
    atomic_json(
        root / "verification" / f"{version}{suffix}.json",
        dict(report, details=details, taker_mismatches=taker_mismatches),
    )
    return report


def audit_raw(*, root, minute_root):
    """未发布时独立核查原档和日判定，不能把原档审计冒称发布版本通过。"""
    root, minute_root = (
        dataset_dir("crypto_metrics", root),
        dataset_dir("crypto_minute", minute_root),
    )
    plan = json.loads((root / "metrics_plan.json").read_text())
    if plan.get("failures") or plan.get("pending"):
        raise ValueError("raw sync incomplete")
    mv = minute.metadata(root=minute_root)["version"]
    summaries, details = {}, {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        refs = {k: v for k, v in plan["archives"].items() if f"/{symbol}/" in k}
        if not refs:
            continue
        records, duplicates = {}, 0
        days = []
        for key, ref in sorted(refs.items()):
            sha = ref["sha256"]
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("raw hash identity")
            blob = (root / "objects" / sha).read_bytes()
            parts = ref["checksum_text"].split()
            if (
                _sha(blob) != sha
                or ref["checksum"] != sha
                or len(blob) != ref["bytes"]
                or len(parts) != 2
                or parts[0] != sha
                or parts[1].lstrip("*") != key.rsplit("/", 1)[-1]
            ):
                raise ValueError("raw CHECKSUM")
            day = key[-14:-4]
            values, dup = _decode(blob, symbol, day)
            if records.keys() & values.keys():
                raise ValueError("raw duplicate between archives")
            records.update(values)
            duplicates += dup
            days.append(
                int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)
            )
        ratios = _minute_ratios(
            symbol, min(records), max(records), root=minute_root, version=mv
        )
        times = np.array(sorted(records), dtype=np.int64)
        error = np.array(
            [
                [
                    abs(records[int(t)][-1] - ratios[w]) / max(abs(ratios[w]), 1e-12)
                    if records[int(t)][-1] is not None and w in ratios
                    else math.nan
                    for w in (int(t) - 300000, int(t))
                ]
                for t in times
            ]
        )
        try:
            alignment, inherited, daily = _alignment(
                times, error, 0.01, 0.1, symbol=symbol
            )
            years = _quality(times, error, alignment, daily)
            alignment_ok = True
        except ValueError:
            daily, years, inherited = [], {}, []
            alignment_ok = False
        gaps = _gaps(times.tolist(), min(days), max(days) + 86400000)
        summaries[symbol] = dict(
            rows=len(times),
            duplicates=duplicates,
            daily_alignment_ok=int(alignment_ok),
            days=daily,
            years=years,
            inherited_rows=sum(inherited),
            null_rows=sum(any(v is None for v in row) for row in records.values()),
            null_cells=sum(sum(v is None for v in row) for row in records.values()),
            gap_segments=len(gaps),
            gap_rows=sum(g["rows"] for g in gaps),
        )
        details[symbol] = dict(gaps=gaps)
    report = dict(
        ok=False,
        built=False,
        raw_checksum_ok=True,
        minute_version=mv,
        symbols=summaries,
    )
    atomic_json(root / "verification" / "raw-audit.json", dict(report, details=details))
    return report
