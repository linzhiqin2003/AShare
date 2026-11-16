"""第 8 节独立校验入口。只从常量与 check_* 模块获取规则。"""

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .check_content import content, gap_declarations, partition_range, quality_flags
from .check_schema import arrow_schema, declarations, dictionary_units
from .check_structure import descriptor
from .report import Report


def check(root):
    root, report = Path(root), Report()
    manifest = descriptor(root, report)
    if manifest is None or report.fatal:
        return report.result()
    resources = manifest["resources"]
    kinds = {r["kind"] for r in resources}
    if manifest.get("symbology") == "custom" and "instruments" not in kinds:
        report.fail("instruments_schema", value="custom 必须有 instruments")
    if (
        any(
            r["kind"] == "bars"
            and any(f["name"] == "date" for f in r["schema"]["fields"])
            for r in resources
        )
        and "calendar" not in kinds
    ):
        report.fail("calendar_schema", value="日频包必须带 calendar 表")
    gap_declarations(manifest, report)
    quality_flags(manifest, report)
    auxiliary = {}
    # 辅助小表先读取以做引用完整性检查；行情始终逐批处理。
    for kind in ("instruments", "calendar", "corporate_actions"):
        selected = [r for r in resources if r["kind"] == kind]
        if len(selected) > 1:
            report.fail("manifest", value=f"{kind} 表不能有多份互相冲突的定义")
        if selected:
            try:
                auxiliary[kind] = pl.concat(
                    [pl.read_parquet(root / p) for p in selected[0]["path"]]
                )
            except Exception as exc:
                report.fail("columns", value=f"{kind}: {type(exc).__name__}")
    with tempfile.TemporaryDirectory(prefix="ashare-check-") as temp:
        with sqlite3.connect(str(Path(temp) / "keys.sqlite")) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE seen (k TEXT PRIMARY KEY, file TEXT, row INTEGER)")
            db.execute("CREATE TEMP TABLE batch (k TEXT, file TEXT, row INTEGER)")
            for resource in resources:
                db.execute("DELETE FROM seen")
                time_col = declarations(manifest, resource, report)
                referenced = set()
                total_hash, total_bytes, total_rows = hashlib.sha256(), 0, 0
                previous = {}
                for entry in resource["files"]:
                    file = entry["path"]
                    path = root / file
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1 << 20):
                            digest.update(chunk)
                            total_hash.update(chunk)
                    size = path.stat().st_size
                    total_bytes += size
                    if size != entry["bytes"]:
                        report.fail("file_bytes", file, value=size)
                    if (
                        "sha256:" + digest.hexdigest() != entry["hash"]
                    ):  # mutation: hash
                        report.fail("file_hash", file, value="SHA-256 不匹配")
                    try:
                        parquet = pq.ParquetFile(path)
                        rows = parquet.metadata.num_rows
                        total_rows += rows
                        if rows != entry["rows"]:
                            report.fail("file_rows", file, value=rows)
                        for group in range(parquet.metadata.num_row_groups):
                            meta = parquet.metadata.row_group(group)
                            if any(
                                meta.column(i).compression not in {"ZSTD", "SNAPPY"}
                                for i in range(meta.num_columns)
                            ):
                                report.fail(
                                    "compression", file, value="只接受 zstd/snappy"
                                )
                        arrow_schema(resource, parquet.schema_arrow, report, file)
                        last_key, offset = None, 0
                        for batch in parquet.iter_batches(batch_size=65536):
                            frame = pl.from_arrow(batch).with_row_index(
                                "_row", offset=offset + 1
                            )
                            offset += frame.height
                            if (
                                "symbol" in frame.columns
                                and frame.schema["symbol"] == pl.String
                            ):
                                referenced.update(
                                    frame["symbol"].drop_nulls().unique().to_list()
                                )
                            content(
                                frame,
                                resource,
                                manifest,
                                report,
                                file,
                                auxiliary.get("calendar"),
                                auxiliary.get("instruments"),
                            )
                            partition_range(frame, file, time_col, report)
                            key = resource["schema"].get("primaryKey")
                            if (
                                isinstance(key, list)
                                and key
                                and set(key) <= set(frame.columns)
                            ):
                                last_key = _keys(frame, key, file, last_key, db, report)
                            if resource["kind"] == "bars":
                                _consistency(
                                    frame,
                                    time_col,
                                    previous,
                                    auxiliary.get("corporate_actions"),
                                    report,
                                    file,
                                )
                    except Exception as exc:
                        report.fail(
                            "columns",
                            file,
                            value=f"无法检查该文件：{type(exc).__name__}: {exc}",
                        )
                instruments = auxiliary.get("instruments")
                if instruments is not None and "symbol" in instruments.columns:
                    instruments = instruments.filter(
                        pl.col("symbol").is_in(sorted(referenced))
                    )
                dictionary_units(resource, instruments, report)
                if any(
                    resource.get(k) != v
                    for k, v in (
                        ("bytes", total_bytes),
                        ("rows", total_rows),
                        ("hash", "sha256:" + total_hash.hexdigest()),
                    )
                ):
                    report.fail("resource_integrity", value=resource["name"])
    return report.result()


def _keys(frame, key, file, last, db, report):
    values = list(frame.select(key).iter_rows())
    if any(None in row for row in values):
        return None
    for index, value in enumerate(values):
        if last is not None and value < last:
            report.fail("sorted", file, int(frame["_row"][index]), value)
        last = value
    entries = [
        (json.dumps(row, default=str, ensure_ascii=False), file, int(num))
        for row, num in zip(values, frame["_row"], strict=True)
    ]
    db.execute("DELETE FROM batch")
    db.executemany("INSERT INTO batch VALUES (?, ?, ?)", entries)
    for k, filename, row, oldfile in db.execute(
        "SELECT b.k,b.file,b.row,s.file FROM batch b JOIN seen s ON b.k=s.k"
    ):
        report.fail(
            "cross_partition" if file != oldfile else "primary_key",
            filename,
            row,
            json.loads(k),
        )
    db.execute("INSERT OR IGNORE INTO seen SELECT * FROM batch")
    return last


def _consistency(frame, time_col, previous, actions, report, file):
    if not {"symbol", "close", time_col} <= set(frame.columns):
        return
    has_factor = "adj_factor" in frame.columns
    cols = ["symbol", time_col, "close", "_row"] + (
        ["adj_factor"] if has_factor else []
    )
    current = frame.select(cols).with_columns(
        pl.col("close").shift(1).over("symbol").alias("_previous_close"),
        pl.col("symbol").is_first_distinct().alias("_first"),
        *(
            [pl.col("adj_factor").shift(1).over("symbol").alias("_previous_factor")]
            if has_factor
            else []
        ),
    )
    if previous:
        prior = pl.DataFrame(
            {
                "symbol": list(previous),
                "_carry_close": [v[0] for v in previous.values()],
                "_carry_factor": [v[1] for v in previous.values()],
            },
            schema={
                "symbol": pl.String,
                "_carry_close": pl.Float64,
                "_carry_factor": pl.Float64,
            },
        )
        current = current.join(prior, on="symbol", how="left").with_columns(
            pl.when(pl.col("_first"))
            .then(pl.col("_carry_close"))
            .otherwise(pl.col("_previous_close"))
            .alias("_previous_close"),
            *(
                [
                    pl.when(pl.col("_first"))
                    .then(pl.col("_carry_factor"))
                    .otherwise(pl.col("_previous_factor"))
                    .alias("_previous_factor")
                ]
                if has_factor
                else []
            ),
        )
    ratio = pl.col("close") / pl.col("_previous_close")
    report.frame(
        "price_jumps", current, (ratio > 2) | (ratio < 0.5), file, warning=True
    )
    if (
        has_factor
        and actions is not None
        and {"symbol", "ex_date", "split", "cash_dividend"} <= set(actions.columns)
    ):
        joined = current.join(
            actions,
            left_on=["symbol", time_col],
            right_on=["symbol", "ex_date"],
            how="inner",
        )
        if joined.height:
            expected = (
                pl.col("_previous_close")
                * pl.col("split").fill_null(1)
                / (pl.col("_previous_close") - pl.col("cash_dividend").fill_null(0))
            )
            actual = pl.col("adj_factor") / pl.col("_previous_factor")
            report.frame(
                "actions_consistency",
                joined,
                (actual / expected - 1).abs() > 0.005,
                file,
                warning=True,
            )
    last = frame.group_by("symbol").agg(
        pl.col("close").last(), *([pl.col("adj_factor").last()] if has_factor else [])
    )
    for row in last.iter_rows():
        previous[row[0]] = (row[1], row[2] if has_factor else None)
