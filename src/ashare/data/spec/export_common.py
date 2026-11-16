"""导出适配器的描述和 UTC 时间工具（校验器不使用）。"""

from datetime import UTC, datetime

import polars as pl

from .constants import FIELDS


def field(name, dtype="number", unit=None, known="end", description=None):
    result = dict(name=name, type=dtype)
    if unit is not None:
        result["unit"] = unit
    if name not in {"symbol", "date", "ts"}:
        result["knownAt"] = FIELDS.get(name, (None, None, known))[2]
    if description:
        result["description"] = description
    return result


def resource(name, kind, fields, key, frequency=None):
    result = dict(name=name, kind=kind, schema=dict(fields=fields, primaryKey=key))
    if frequency:
        result["frequency"] = frequency
    return result


def manifest(name, source, symbology, resources):
    return dict(
        specVersion="1",
        name=name,
        title=name,
        source=source,
        licenses=[dict(name="proprietary")],
        symbology=symbology,
        calendar=dict(name="24x7", tz="UTC")
        if symbology == "crypto"
        else dict(
            name="SSE", tz="Asia/Shanghai", sessionOpen="09:30", sessionClose="15:00"
        ),
        resources=resources,
    )


def timestamp(value):
    return datetime.fromtimestamp(int(value) / 1000, UTC)


def utc_expr(column):
    return pl.from_epoch(column, time_unit="ms").dt.replace_time_zone("UTC")


def clean(frame):
    return frame.with_columns(pl.col(pl.Float64).fill_nan(None))
