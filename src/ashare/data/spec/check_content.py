"""逐批独立内容检查；不把完整行情包读入内存。"""

import re
from datetime import UTC, datetime, time

import polars as pl

from .constants import (
    ACTION_KINDS,
    ASSET_UNITS,
    CURRENCIES,
    NONNEGATIVE,
    PRICE_UNITS,
    PRICES,
    SYMBOLOGY,
)


def content(frame, resource, manifest, report, file, calendar, instruments):
    columns = set(frame.columns)
    fields = {f["name"]: f for f in resource["schema"]["fields"]}
    kind = resource["kind"]
    key = resource["schema"].get("primaryKey", [])
    if isinstance(key, list) and key and set(key) <= columns:
        report.frame(
            "primary_key",
            frame,
            pl.any_horizontal(pl.col(k).is_null() for k in key),
            file,
        )
        report.frame(
            "primary_key", frame, frame.select(key).is_duplicated(), file
        )  # mutation: primary_key
    for name, dtype in frame.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            report.frame("nan", frame, pl.col(name).is_nan(), file)  # mutation: nan
            report.frame("finite", frame, pl.col(name).is_infinite(), file)
        if dtype == pl.String:
            report.frame("missing", frame, pl.col(name) == "", file)
    if "symbol" in columns and frame.schema["symbol"] == pl.String:
        pattern = SYMBOLOGY.get(manifest.get("symbology"), r"\S+")
        report.frame(
            "symbols",
            frame,
            pl.col("symbol").is_null()
            | ~pl.col("symbol").str.contains("^(?:" + pattern + ")$"),
            file,
        )
    if kind == "instruments":
        if "asset_type" in columns:
            report.frame(
                "instruments_schema",
                frame,
                pl.col("asset_type").is_null()
                | ~pl.col("asset_type").is_in(list(ASSET_UNITS)),
                file,
            )
        if "currency" in columns:
            report.frame(
                "instruments_schema",
                frame,
                pl.col("currency").is_null()
                | ~pl.col("currency").is_in(list(CURRENCIES)),
                file,
            )
        for field in ("multiplier", "tick_size", "lot_size"):
            if field in columns and frame.schema[field].is_numeric():
                report.frame("instruments_schema", frame, pl.col(field) <= 0, file)
        if {"list_date", "delist_date"} <= columns:
            report.frame(
                "lifecycle", frame, pl.col("list_date") > pl.col("delist_date"), file
            )
        if "successor" in columns and instruments is not None:
            report.frame(
                "symbols",
                frame,
                (pl.col("successor") == pl.col("symbol"))
                | (~pl.col("successor").is_in(instruments["symbol"].implode())),
                file,
            )
    if kind == "corporate_actions" and "kind" in columns:
        report.frame(
            "actions_schema",
            frame,
            pl.col("kind").is_null() | ~pl.col("kind").is_in(list(ACTION_KINDS)),
            file,
        )
    prices = (PRICES if kind == "bars" else set()) | {
        f for f, d in fields.items() if str(d.get("unit")) in PRICE_UNITS
    }
    for name in prices & columns:
        if frame.schema[name].is_numeric():
            bad = (
                pl.col(name) == 0
                if fields.get(name, {}).get("allowNegative") is True
                else pl.col(name) <= 0
            )
            report.frame("prices", frame, bad, file)
    nonnegative = NONNEGATIVE | {
        f
        for f, d in fields.items()
        if str(d.get("unit")) in {"share", "coin", "contract", "bond", "count"}
        or str(d.get("unit")) in CURRENCIES
    }
    for name in columns & nonnegative:
        if frame.schema[name].is_numeric():
            report.frame("nonnegative", frame, pl.col(name) < 0, file)
    if "quality" in columns and frame.schema["quality"].is_integer():
        q = pl.col("quality").cast(pl.UInt32, strict=False)
        allowed = quality_flags(manifest)
        report.frame(
            "quality", frame, q.is_null() | ((q & (0xFFFFFFFF ^ allowed)) != 0), file
        )
        if prices & columns:
            report.frame(
                "quality",
                frame,
                ((q & 4) != 0)
                & pl.any_horizontal(pl.col(f).is_not_null() for f in prices & columns),
                file,
            )
    if kind != "bars":
        _availability(frame, resource, manifest, report, file)
        time_col = next(
            (name for name in ("ts", "date", "ex_date") if name in columns), None
        )
        if time_col and frame.schema[time_col] in (
            pl.Date,
            pl.Datetime("ms", "UTC"),
            pl.Datetime("us", "UTC"),
        ):
            _gaps(frame, resource, manifest, report, file, time_col)
        return
    ohlc = {"open", "high", "low", "close"}
    if ohlc <= columns and all(frame.schema[f].is_numeric() for f in ohlc):
        complete = pl.all_horizontal(pl.col(f).is_not_null() for f in ohlc)
        invalid = (pl.col("low") > pl.min_horizontal("open", "close")) | (
            pl.max_horizontal("open", "close") > pl.col("high")
        )
        report.frame("prices", frame, complete & invalid, file)  # mutation: prices
    time_col = "ts" if "ts" in columns else "date"
    if time_col not in columns or frame.schema[time_col] not in (
        pl.Date,
        pl.Datetime("ms", "UTC"),
        pl.Datetime("us", "UTC"),
    ):
        return
    if time_col == "date" and calendar is not None:
        report.frame(
            "calendar_membership",
            frame,
            ~pl.col("date").is_in(calendar["date"].implode()),
            file,
        )  # mutation: calendar
    if (
        instruments is not None
        and "symbol" in columns
        and "symbol" in instruments.columns
    ):
        life = instruments.select(
            [
                f
                for f in ("symbol", "list_date", "delist_date")
                if f in instruments.columns
            ]
        )
        report.frame(
            "lifecycle", frame, ~pl.col("symbol").is_in(life["symbol"].implode()), file
        )
        joined = frame.join(life, on="symbol", how="left")
        day = (
            pl.col(time_col).dt.convert_time_zone(manifest["calendar"]["tz"]).dt.date()
            if time_col == "ts"
            else pl.col(time_col)
        )
        bad = pl.lit(False)
        if "list_date" in life.columns:
            bad = bad | (day < pl.col("list_date")).fill_null(False)
        if "delist_date" in life.columns:
            bad = bad | (day > pl.col("delist_date")).fill_null(False)
        report.frame("lifecycle", joined, bad, file)  # mutation: lifecycle
    _availability(frame, resource, manifest, report, file)
    _gaps(frame, resource, manifest, report, file, time_col)


def partition_range(frame, file, time_col, report):
    label = file.rsplit("/", 1)[1].removesuffix(".parquet")
    if (
        label == "all"
        or time_col not in frame.columns
        or frame.schema[time_col].base_type() not in {pl.Date, pl.Datetime}
    ):
        return
    fmt = "%Y" if len(label) == 4 else "%Y-%m"
    report.frame(
        "partition_range", frame, pl.col(time_col).dt.strftime(fmt) != label, file
    )


def _start_expr(frame, manifest, time_col):
    if frame.schema[time_col] == pl.Date:
        session = manifest["calendar"].get("sessionOpen", "00:00")
        clock = time.fromisoformat(session)
        return (
            (
                pl.col(time_col).cast(pl.Datetime("us"))
                + pl.duration(
                    hours=clock.hour,
                    minutes=clock.minute,
                    seconds=clock.second,
                    microseconds=clock.microsecond,
                )
            )
            .dt.replace_time_zone(manifest["calendar"]["tz"])
            .dt.convert_time_zone("UTC")
        )
    return pl.col(time_col)


def _availability(frame, resource, manifest, report, file):
    time_col = (
        "ex_date"
        if "ex_date" in frame.columns
        else "ts"
        if "ts" in frame.columns
        else "date"
    )
    targets = {
        f["knownAt"][7:]
        for f in resource["schema"]["fields"]
        if isinstance(f.get("knownAt"), str) and f["knownAt"].startswith("column:")
    }
    if "available_at" in frame.columns:
        targets.add("available_at")
    for target in targets & set(frame.columns):
        if frame.schema[target] in (pl.Datetime("ms", "UTC"), pl.Datetime("us", "UTC")):
            # 静态表也必须有可用的依赖时间；不因没有行情区间而跳过非空检查。
            invalid = pl.col(target).is_null()
            # 事件公告可以早于事件日期，但可知时刻本身仍不可缺失。
            if resource["kind"] == "bars":  # mutation: availability_kind
                if time_col in frame.columns and frame.schema[time_col].base_type() in {
                    pl.Date,
                    pl.Datetime,
                }:
                    # 原始时间值的下界只到区间开始；值可早于收盘，读取侧另取 max。
                    invalid = invalid | (
                        pl.col(target) < _start_expr(frame, manifest, time_col)
                    )
            report.frame(
                "available_at",
                frame,
                invalid,
                file,
            )  # mutation: available_at


def gap_declarations(manifest, report):
    gaps = manifest.get("gaps", [])
    if not isinstance(gaps, list):
        report.fail("gaps", value="gaps 必须是数组")
        return
    names = {r["name"] for r in manifest["resources"]}
    for gap in gaps:
        try:
            start, end = (
                datetime.fromisoformat(gap[k].replace("Z", "+00:00"))
                for k in ("start", "end")
            )
            if (
                start.tzinfo is None
                or end.tzinfo is None
                or start.utcoffset().total_seconds()
                or end.utcoffset().total_seconds()
                or start >= end
                or gap["table"] not in names
                or not isinstance(gap["reason"], str)
                or not gap["reason"].strip()
            ):
                raise ValueError("缺口必须有表、UTC 起止和原因")
            if "symbol" in gap and not re.fullmatch(
                SYMBOLOGY.get(manifest.get("symbology"), r"\S+"), gap["symbol"]
            ):
                raise ValueError("缺口代码非法")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            report.fail("gaps", value=str(exc))


def _gaps(frame, resource, manifest, report, file, time_col):
    for gap in manifest.get("gaps", []):
        if not isinstance(gap, dict) or gap.get("table") != resource["name"]:
            continue
        try:
            start, end = (
                datetime.fromisoformat(gap[k].replace("Z", "+00:00")).astimezone(UTC)
                for k in ("start", "end")
            )
            time_expr = _start_expr(frame, manifest, time_col)
            bad = (time_expr >= start) & (time_expr < end)
            if "symbol" in gap:
                bad = bad & (pl.col("symbol") == gap["symbol"])
            report.frame("gaps", frame, bad, file)
        except (KeyError, ValueError, TypeError):
            pass


def quality_flags(manifest, report=None):
    declared = manifest.get("qualityFlags", {})
    allowed = 15
    if not isinstance(declared, dict):
        if report is not None:
            report.fail("quality", value="qualityFlags 必须是位值到说明的对象")
        return allowed
    for bit, description in declared.items():
        if (
            re.fullmatch(r"[0-9]{1,10}", str(bit))
            and (1 << 16) <= int(bit) <= (1 << 31)
            and int(bit).bit_count() == 1
            and isinstance(description, str)
            and description.strip()
        ):
            allowed |= int(bit)
        elif report is not None:
            report.fail("quality", value=f"非法自定义位 {bit}")
    return allowed
