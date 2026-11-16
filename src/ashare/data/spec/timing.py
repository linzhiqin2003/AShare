"""读取方的时点计算；独立校验器不导入本模块。"""

import re
from datetime import time, timedelta

import polars as pl

from .constants import FIELDS
from .dependencies import require_known, validate_time_dependencies


def declarations(resource):
    return {
        f["name"]: {
            **f,
            **(
                {"knownAt": FIELDS[f["name"]][2]}
                if resource["kind"] == "bars"
                and f["name"] in FIELDS
                and "knownAt" not in f
                else {}
            ),
        }
        for f in resource["schema"]["fields"]
    }


def delay(known):
    if not known.startswith("end+"):
        return timedelta(0)
    match = re.fullmatch(r"end\+([1-9][0-9]*)(us|ms|s|m|h|d|w)", known)
    if match is None:
        raise ValueError(f"非法 knownAt: {known}")
    units = dict(
        us="microseconds",
        ms="milliseconds",
        s="seconds",
        m="minutes",
        h="hours",
        d="days",
        w="weeks",
    )
    return timedelta(**{units[match[2]]: int(match[1])})


def interval(resource, manifest, *, end):
    """返回行的 UTC 区间边界；日频使用交易日当地开收盘。"""
    names = {f["name"] for f in resource["schema"]["fields"]}
    if "ts" in names:
        step = "1mo" if resource["frequency"] == "1M" else resource["frequency"]
        return pl.col("ts").dt.offset_by(step) if end else pl.col("ts")
    calendar = manifest["calendar"]
    session = "sessionClose" if end else "sessionOpen"
    if session not in calendar:
        raise ValueError(f"日频时点视图需要 calendar.{session}")
    clock = time.fromisoformat(calendar[session])
    return (
        (
            pl.col("date").cast(pl.Datetime("us"))
            + pl.duration(
                hours=clock.hour,
                minutes=clock.minute,
                seconds=clock.second,
                microseconds=clock.microsecond,
            )
        )
        .dt.replace_time_zone(calendar["tz"])
        .dt.convert_time_zone("UTC")
    )


def knowable(desc, resource, manifest, *, allow_unknown=False):
    known = desc.get("knownAt")
    if known is None:
        return None
    if known == "unknown":
        if allow_unknown:
            return None
        raise ValueError(f"字段 {desc['name']} 的 knownAt=unknown，不能保证时点")
    if known.startswith("column:"):
        return dependency_time(known[7:], resource, manifest)
    if known in {"start", "end"} and resource["kind"] != "bars":
        # 静态表没有行情区间；事件日期也不是公告日期，start/end 均随行可知。
        return None
    return interval(resource, manifest, end=known != "start") + delay(known)


def dependency_time(name, resource, manifest):
    """依赖时间 = max(时间值, 列自身可知时刻)；声明检查已排除多级引用。"""
    value = pl.col(name)
    own = knowable(declarations(resource)[name], resource, manifest)
    if own is None:
        return value
    # max_horizontal 忽略 null，但缺失的发布时间不能凭区间边界变成已知。
    return pl.when(value.is_not_null()).then(pl.max_horizontal(value, own))


def as_of_view(frame, resource, manifest, selected, instant, *, allow_unknown=False):
    """行级 available_at 过滤，字段未到可知时刻置 null，保留主键。"""
    fields = declarations(resource)
    validate_time_dependencies(fields)
    require_known(fields, selected, allow_unknown)
    if "available_at" in fields:
        frame = frame.filter(
            dependency_time("available_at", resource, manifest) <= instant
        )
    if resource["kind"] == "bars":
        frame = frame.filter(interval(resource, manifest, end=False) <= instant)
    expressions = []
    for name in set(selected) - set(resource["schema"]["primaryKey"]):
        at = knowable(fields[name], resource, manifest, allow_unknown=allow_unknown)
        if at is not None:
            expressions.append(
                pl.when(at <= instant).then(pl.col(name)).otherwise(None).alias(name)
            )
    return frame.with_columns(expressions)
