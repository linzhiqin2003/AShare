"""BTC/ETH 官方分钟归档：独立原始库、月分区、内容版本和按需读取。"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..store import atomic_json, lock
from .archive import archive_key as hourly_archive_key
from .sources import PublicClient
from .storage import RawStore, digest, encoded
from .timeutil import DAY, HOUR, iso, millis, month_end, months

MINUTE = 60_000
DEFINITION = "crypto-minute-v1"
KINDS = {"klines": "klines", "mark": "markPriceKlines", "index": "indexPriceKlines"}
PRICES = ("open", "high", "low", "close")
VOLUMES = (
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)
SCHEMA = {
    "open_time": pl.Int64,
    **dict.fromkeys(PRICES, pl.Float64),
    **{f: pl.Int64 if f == "count" else pl.Float64 for f in VOLUMES},
}
SERIES = tuple(
    (symbol, market, kind)
    for symbol in ("BTCUSDT", "ETHUSDT")
    for market, kinds in (("spot", ("klines",)), ("usdm", tuple(KINDS)))
    for kind in kinds
)


def data_root(root=None):
    # D1 合并后改为 ashare.paths.dataset_dir("crypto_minute")。
    return Path(root) if root is not None else Path("data/crypto_minute")


def _validate_series(symbol, market, kind):
    if (symbol, market, kind) not in SERIES:
        raise ValueError("只支持 BTCUSDT/ETHUSDT；spot:klines，usdm:klines/mark/index")


def _schema(kind):
    if kind not in KINDS:
        raise ValueError("未知分钟线种类")
    return (
        SCHEMA if kind == "klines" else {f: SCHEMA[f] for f in ("open_time", *PRICES)}
    )


def archive_key(market, kind, symbol, label, *, frequency="monthly"):
    _validate_series(symbol, market, kind)
    if frequency not in ("monthly", "daily"):
        raise ValueError("归档频率非法")
    pattern = r"\d{4}-\d{2}" if frequency == "monthly" else r"\d{4}-\d{2}-\d{2}"
    if not re.fullmatch(pattern, label):
        raise ValueError("归档日期非法")
    millis(label + "-01" if frequency == "monthly" else label)
    return (
        hourly_archive_key(market, KINDS[kind], symbol, label, frequency=frequency)
        .replace("/1h/", "/1m/")
        .replace("-1h-", "-1m-")
    )


def parse_archive(payload, kind, *, start, end):
    """以数值量级判断单位；不排序修复原档，不吞重复，不按年份猜表头。"""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1 or z.testzip() is not None:
            raise ValueError("ZIP 必须含一份 CRC 正确的 CSV")
        content = z.read(names[0]).decode("utf-8-sig")
    first = content.split("\n", 1)[0].split(",")[0].strip().lower().replace(" ", "_")
    header = first == "open_time"
    # 全列先按文本读，避免浮点数改变 16 位微秒时间戳。
    raw = pl.read_csv(
        io.BytesIO(content.encode()),
        has_header=header,
        infer_schema=False,
        raise_if_empty=False,
    )
    if raw.width != 12 or raw.height == 0:
        raise ValueError("分钟 CSV 必须为非空的 12 列记录")
    raw.columns = [f"c{i}" for i in range(12)]
    try:
        stamp = raw["c0"].cast(pl.Int64)
        ms = (stamp >= 1_000_000_000_000) & (stamp < 10_000_000_000_000)
        us = (stamp >= 1_000_000_000_000_000) & (stamp < 10_000_000_000_000_000)
        if ms.all():
            times = stamp
        elif us.all():
            if (stamp % 1000 != 0).any():
                raise ValueError("微秒开盘时间不能无损转毫秒")
            times = stamp // 1000
        else:
            raise ValueError("无法判断时间戳单位或文件混用单位")
        times = times.rename("open_time")
        if times.n_unique() != len(times):
            raise ValueError("分钟时间戳重复")
        if not times.is_sorted():
            raise ValueError("分钟时间必须严格递增")
        if (times % MINUTE != 0).any() or times.min() < start or times.max() >= end:
            raise ValueError("分钟时间不在网格或归档范围内")
        indices = dict(zip(PRICES, (1, 2, 3, 4), strict=True))
        if kind == "klines":
            indices.update(dict(zip(VOLUMES, (5, 7, 8, 9, 10), strict=True)))
        frame = pl.DataFrame(
            [
                times,
                *[
                    raw[f"c{i}"].cast(_schema(kind)[f]).rename(f)
                    for f, i in indices.items()
                ],
            ]
        )
        if (
            frame.null_count().sum_horizontal().item()
            or frame.select(
                pl.any_horizontal(
                    *[~pl.col(f).is_finite() | (pl.col(f) < 0) for f in indices]
                ).any()
            ).item()
        ):
            raise ValueError("价格或数量含空值、非有限数或负数")
        return frame
    except pl.exceptions.PolarsError as exc:
        raise ValueError("分钟 CSV 字段类型非法") from exc


class SyncError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "分钟归档同步失败: " + ", ".join(r["key"] for r in report["failures"])
        )


def sync(
    *,
    root=None,
    start="2020-01-01",
    end=None,
    windows=None,
    series=SERIES,
    now=None,
    client=None,
    refresh=False,
    max_bytes=2_000_000_000,
    progress=None,
):
    """增量同步；月档中已有任意记录的日子不取日档。404 留缺口，其他失败阻止构建。

    windows 支持不连续的小样本。完成的旧分区保留，新请求按整月重新择档；
    同月旧日档仍保留，但月档已有行的日子淘汰旧日档。refresh 显式复查上游修订。
    """
    root = data_root(root)
    now = millis(now if now is not None else datetime.now(UTC))
    end = millis(end) if end is not None else now // DAY * DAY
    windows = [(millis(a), millis(b)) for a, b in (windows or [(start, end)])]
    if any(a >= b or a % DAY or b % DAY or b > now // DAY * DAY for a, b in windows):
        raise ValueError("同步窗口必须为已完成的 UTC 日，左闭右开")
    series = tuple(series)
    if not series or len(set(series)) != len(series):
        raise ValueError("同步序列不能为空或重复")
    for args in series:
        _validate_series(*args)
    own = client is None
    client = client or PublicClient(max_bytes=max_bytes)
    started, before = time.perf_counter(), client.bytes
    try:
        with lock(root / ".minute.lock"):
            store = RawStore(root)
            try:
                path = root / "minute_plan.json"
                old = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"partitions": {}}
                )
                partitions = dict(old["partitions"])
                labels = sorted({label for a, b in windows for label in months(a, b)})
                retried_prefixes = {
                    archive_key(market, kind, symbol, label).removesuffix(".zip")
                    for symbol, market, kind in series
                    for label in labels
                }

                def retried(key):
                    monthly = key.replace("/daily/", "/monthly/")
                    return any(
                        monthly.startswith(prefix) for prefix in retried_prefixes
                    )

                failures = [f for f in old.get("failures", []) if not retried(f["key"])]
                missing, downloaded = [], []

                def fetch(key):
                    cached = not refresh and store.complete(key)
                    try:
                        payload = client.archive(store, key, refresh=refresh)
                    except Exception as exc:
                        failures.append(dict(key=key, error=type(exc).__name__))
                        return None
                    if payload is None:
                        missing.append(key)
                    elif not cached:
                        downloaded.append(key)
                    return payload

                for symbol, market, kind in series:
                    for label in labels:
                        part_id = f"{market}/{symbol}/{kind}/{label}"
                        if progress:
                            progress(part_id)
                        lo, hi = millis(label + "-01"), month_end(label)
                        requested_days = {
                            day
                            for a, b in windows
                            for day in range(max(a, lo), min(b, hi), DAY)
                        }
                        old_part = partitions.get(part_id, {})
                        requested_days.update(old_part.get("requested_days", []))
                        # 下载失败也保存请求范围，缩短窗口续跑不能遗忘尚未取到的日子。
                        partitions[part_id] = dict(
                            symbol=symbol,
                            market=market,
                            kind=kind,
                            month=label,
                            requested_days=sorted(requested_days),
                            archives=old_part.get("archives", {}),
                        )
                        key = archive_key(market, kind, symbol, label)
                        payload = fetch(key)
                        keys, present_days = [], set()
                        if payload is not None:
                            try:
                                frame = parse_archive(payload, kind, start=lo, end=hi)
                                present_days = set(
                                    (frame["open_time"] // DAY).to_list()
                                )
                                keys.append(key)
                            except Exception as exc:
                                failures.append(dict(key=key, error=type(exc).__name__))
                        # 月归档下载失败时不以日档静默替代；重试后仍以月档为准。
                        if any(f["key"] == key for f in failures):
                            continue
                        for day in sorted(requested_days):
                            if day // DAY in present_days:
                                continue
                            daily = archive_key(
                                market, kind, symbol, iso(day)[:10], frequency="daily"
                            )
                            if fetch(daily) is not None:
                                keys.append(daily)
                        entries = store.manifest.read()
                        partitions[part_id] = dict(
                            symbol=symbol,
                            market=market,
                            kind=kind,
                            month=label,
                            requested_days=sorted(requested_days),
                            archives={
                                k: {
                                    f: entries[k][f]
                                    for f in (
                                        "sha256",
                                        "bytes",
                                        "checksum",
                                        "checksum_text",
                                    )
                                }
                                for k in keys
                            },
                        )
                plan = dict(
                    definition=DEFINITION, partitions=partitions, failures=failures
                )
                atomic_json(path, plan)
                report = dict(
                    failures=failures,
                    missing=missing,
                    downloaded=len(downloaded),
                    downloaded_bytes=client.bytes - before,
                    partitions=len(partitions),
                    seconds=time.perf_counter() - started,
                )
                atomic_json(root / "sync_report.json", report)
                if failures:
                    raise SyncError(report)
                return report
            finally:
                store.close()
    finally:
        if own:
            client.client.close()


def _raw(root, key, ref):
    sha = ref["sha256"]
    _hash_name(sha)
    payload = (root / "objects" / sha).read_bytes()
    parts = ref["checksum_text"].split()
    if (
        digest(payload) != sha
        or len(payload) != ref["bytes"]
        or ref["checksum"] != sha
        or len(parts) != 2
        or parts[0] != sha
        or parts[1].lstrip("*") != key.rsplit("/", 1)[-1]
    ):
        raise ValueError(f"原始归档或 CHECKSUM 损坏: {key}")
    return payload


def _bounds(key):
    label = key.rsplit("-1m-", 1)[-1].removesuffix(".zip")
    start = millis(label + "-01" if len(label) == 7 else label)
    return start, month_end(label) if len(label) == 7 else start + DAY


def _quality(frame, label):
    lo, hi = millis(label + "-01"), month_end(label)
    gaps, cursor = [], lo
    for t in frame["open_time"]:
        if t > cursor:
            gaps.append(dict(start=cursor, end=t, minutes=(t - cursor) // MINUTE))
        cursor = t + MINUTE
    if cursor < hi:
        gaps.append(dict(start=cursor, end=hi, minutes=(hi - cursor) // MINUTE))
    return dict(expected_rows=(hi - lo) // MINUTE, rows=frame.height, gaps=gaps)


def build(*, root=None, quality_notes=()):
    """每次只整理一个月；分块按内容寻址，版本与 latest 在完成后原子发布。"""
    root = data_root(root)
    with lock(root / ".minute.lock"):
        plan = json.loads((root / "minute_plan.json").read_text())
        if plan.get("failures") or plan.get("definition") != DEFINITION:
            raise ValueError("同步尚有失败或定义版本不匹配；先续跑 sync")
        parts = {}
        for part_id, part in sorted(plan["partitions"].items()):
            _validate_series(part["symbol"], part["market"], part["kind"])
            expected_id = (
                f"{part['market']}/{part['symbol']}/{part['kind']}/{part['month']}"
            )
            if part_id != expected_id:
                raise ValueError("分区标识不匹配")
            frames = []
            monthly_days = set()
            for key, ref in sorted(
                part["archives"].items(),
                key=lambda kv: ("/monthly/" not in kv[0], kv[0]),
            ):
                lo, hi = _bounds(key)
                label = iso(lo)[:7] if "/monthly/" in key else iso(lo)[:10]
                expected_key = archive_key(
                    part["market"],
                    part["kind"],
                    part["symbol"],
                    label,
                    frequency="monthly" if "/monthly/" in key else "daily",
                )
                if key != expected_key or iso(lo)[:7] != part["month"]:
                    raise ValueError("归档与分区不匹配")
                frame = parse_archive(
                    _raw(root, key, ref), part["kind"], start=lo, end=hi
                )
                days = set((frame["open_time"] // DAY).to_list())
                if "/monthly/" in key:
                    monthly_days.update(days)
                elif days & monthly_days:
                    raise ValueError("日度归档不能覆盖月度已有记录的日子")
                frames.append(frame)
            frame = (
                pl.concat(frames).sort("open_time")
                if frames
                else pl.DataFrame(schema=_schema(part["kind"]))
            )
            if frame["open_time"].n_unique() != frame.height:
                raise ValueError("分区间分钟时间戳重复")
            quality = _quality(frame, part["month"])
            buf = io.BytesIO()
            frame.write_parquet(buf, compression="zstd", statistics=True)
            payload = buf.getvalue()
            sha = digest(payload)
            relative = f"bars/{part_id}/{sha}.parquet"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or digest(path.read_bytes()) != sha:
                temp = path.with_suffix(".part")
                try:
                    temp.write_bytes(payload)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
            parts[part_id] = dict(
                part, file=relative, sha256=sha, bytes=len(payload), quality=quality
            )
        meta = dict(
            definition=DEFINITION, partitions=parts, quality_notes=list(quality_notes)
        )
        version = digest(encoded(meta))
        atomic_json(root / "versions" / f"{version}.json", dict(meta, version=version))
        atomic_json(root / "latest.json", dict(version=version))
        return version


def _hash_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("非法内容版本或哈希")


def metadata(*, root=None, version=None):
    root = data_root(root)
    version = version or json.loads((root / "latest.json").read_text())["version"]
    _hash_name(version)
    meta = json.loads((root / "versions" / f"{version}.json").read_text())
    body = {k: v for k, v in meta.items() if k != "version"}
    if meta.get("version") != version or digest(encoded(body)) != version:
        raise ValueError("分钟内容版本损坏")
    if meta.get("definition") != DEFINITION:
        raise ValueError("不支持的分钟定义版本")
    return meta


def _read_partition(root, part_id, part):
    _hash_name(part["sha256"])
    expected = f"{part['market']}/{part['symbol']}/{part['kind']}/{part['month']}"
    _validate_series(part["symbol"], part["market"], part["kind"])
    if not re.fullmatch(r"\d{4}-\d{2}", part["month"]):
        raise ValueError("分区月份非法")
    if (
        part_id != expected
        or part["file"] != f"bars/{expected}/{part['sha256']}.parquet"
    ):
        raise ValueError("分钟分区路径损坏")
    for key, ref in part["archives"].items():
        _raw(root, key, ref)
    path = root / part["file"]
    payload = path.read_bytes()
    if digest(payload) != part["sha256"] or len(payload) != part["bytes"]:
        raise ValueError("分钟 Parquet 损坏")
    frame = pl.read_parquet(path)
    if frame.schema != _schema(part["kind"]):
        raise ValueError("分钟 Parquet schema 损坏")
    return frame


def bars(symbol, market, kind, start, end, *, root=None, version=None):
    """左闭右开 UTC 毫秒；只校验和读取涉及的月档，version 可固定旧版本。"""
    _validate_series(symbol, market, kind)
    start, end = millis(start), millis(end)
    if start > end:
        raise ValueError("读取起点晚于终点")
    root = data_root(root)
    meta = metadata(root=root, version=version)
    if start == end:
        return pl.DataFrame(schema=_schema(kind))
    frames = []
    for label in months(start, end):
        part_id = f"{market}/{symbol}/{kind}/{label}"
        if part_id not in meta["partitions"]:
            raise KeyError(f"版本未覆盖月分区: {part_id}")
        frames.append(_read_partition(root, part_id, meta["partitions"][part_id]))
    return (
        pl.concat(frames)
        .filter((pl.col("open_time") >= start) & (pl.col("open_time") < end))
        .sort("open_time")
    )


def aggregate_hourly(frame):
    """保留不足 60 根的小时并标 complete=False；绝不冒充完整小时。"""
    times = frame["open_time"]
    if times.n_unique() != frame.height or (times % MINUTE != 0).any():
        raise ValueError("聚合输入含重复或非分钟网格")
    fields = [f for f in VOLUMES if f in frame.columns]
    return (
        frame.sort("open_time")
        .with_columns(((pl.col("open_time") // HOUR) * HOUR).alias("time"))
        .group_by("time")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            *[pl.col(f).sum() for f in fields],
            pl.len().alias("minutes"),
        )
        .with_columns((pl.col("minutes") == 60).alias("complete"))
        .sort("time")
    )


def status(*, root=None, version=None):
    root = data_root(root)
    out = dict(built=False)
    if (root / "sync_report.json").exists():
        out["sync"] = json.loads((root / "sync_report.json").read_text())
    if version or (root / "latest.json").exists():
        meta = metadata(root=root, version=version)
        out.update(
            built=True,
            version=meta["version"],
            definition=meta["definition"],
            partitions=len(meta["partitions"]),
            rows=sum(p["quality"]["rows"] for p in meta["partitions"].values()),
            gap_segments=sum(
                len(p["quality"]["gaps"]) for p in meta["partitions"].values()
            ),
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "build", "verify", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=argparse.SUPPRESS)
        if name == "sync":
            p.add_argument("--start", default="2020-01-01")
            p.add_argument("--end")
            p.add_argument("--refresh", action="store_true")
            p.add_argument("--max-bytes", type=int, default=2_000_000_000)
        if name in ("verify", "status"):
            p.add_argument("--version")
        if name == "verify":
            p.add_argument("--hourly-root")
            p.add_argument("--hourly-version")
            p.add_argument(
                "--report", help="可选完整 JSON 报告（含逐小时差异，属于行情数据）"
            )
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "verify":
        from .verify_minute import verify

        report_path = args.pop("report")
        result = verify(**args)
        if report_path:
            atomic_json(Path(report_path), result)
    else:
        try:
            result = globals()[command](**args)
        except SyncError as exc:
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
            return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if isinstance(result, dict) and result.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
