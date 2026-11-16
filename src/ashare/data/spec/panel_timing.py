"""按决策时点流式落格，延迟字段只写入首次可知的行。"""

from datetime import UTC, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .timing import delay, dependency_time, interval


def utc_values(series):
    return (
        series.dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .to_numpy()
        .astype("datetime64[us]")
    )


def decisions(dates, resource, manifest):
    daily = any(f["name"] == "date" for f in resource["schema"]["fields"])
    series = pl.Series("date" if daily else "ts", dates)
    series = series.cast(pl.Date) if daily else series.dt.replace_time_zone("UTC")
    return (
        pl.DataFrame(series).select(interval(resource, manifest, end=True)).to_series()
    )


def lookback(first, previous, fields, resource, manifest):
    lag = max(
        (delay(f.get("knownAt", "end")) for f in fields.values()), default=timedelta(0)
    )
    if not lag or first is None:
        return first
    if previous is None:
        return None
    earliest = previous - lag
    if any(f["name"] == "date" for f in resource["schema"]["fields"]):
        # 收盘之前的日期取整并多读一天，随后按完整日历的前一决策点排除已到期值。
        return earliest.astimezone(
            ZoneInfo(manifest["calendar"]["tz"])
        ).date() - timedelta(days=1)
    step = "1mo" if resource["frequency"] == "1M" else resource["frequency"]
    return pl.Series([earliest]).dt.offset_by("-" + step)[0]


class Placement:
    def __init__(
        self,
        dates,
        previous,
        fields,
        resource,
        manifest,
        arrays,
        *,
        calendar=None,
        history_before=None,
    ):
        self.dates, self.fields, self.arrays = dates, fields, arrays
        self.resource, self.manifest = resource, manifest
        self.calendar = calendar
        self.history_before = (
            np.datetime64(
                history_before.astimezone(UTC).replace(tzinfo=None)
                if hasattr(history_before, "tzinfo")
                else history_before,
                "us",
            )
            if history_before is not None
            else None
        )
        self.decisions = utc_values(decisions(dates, resource, manifest))
        self.previous = (
            np.datetime64(previous.astimezone(UTC).replace(tzinfo=None), "us")
            if previous is not None
            else None
        )
        self.latest = {
            name: np.full(arrays[name].shape, np.iinfo(np.int64).min, dtype=np.int64)
            for name, desc in fields.items()
            if delay(desc.get("knownAt", "end"))
        }

    def fill(self, frame, values, columns):
        dates, arrays = self.dates, self.arrays
        i = np.searchsorted(dates, values)
        inside = (
            (i < len(dates)) & (dates[np.minimum(i, len(dates) - 1)] == values)
            if len(dates)
            else np.zeros(len(values), dtype=bool)
        )
        if self.calendar is not None:
            on_grid = np.isin(values, self.calendar)
        else:
            step = (
                "1mo"
                if self.resource["frequency"] == "1M"
                else self.resource["frequency"]
            )
            on_grid = (frame["ts"].dt.truncate(step) == frame["ts"]).to_numpy()
        historical = (
            values < self.history_before
            if self.history_before is not None
            else np.zeros(len(values), dtype=bool)
        )
        if not np.all(on_grid & (inside | historical)):  # mutation: panel_grid
            raise ValueError("行情时间不在面板日历/网格内")
        if not len(dates):
            return
        arrays["has_quote"][i[inside], columns[inside]] = True
        ends = None
        availability = {}
        for name, desc in self.fields.items():
            known = desc.get("knownAt", "end")
            lag = delay(known)
            dest, valid = i, inside.copy()
            if lag:
                if ends is None:
                    ends = utc_values(
                        frame.select(
                            interval(self.resource, self.manifest, end=True)
                        ).to_series()
                    )
                arrival = ends + np.timedelta64(lag // timedelta(microseconds=1), "us")
                dest = np.searchsorted(
                    self.decisions, arrival
                )  # mutation: delay_placement
                valid = dest < len(dates)
                if self.previous is not None:
                    valid &= arrival > self.previous
            targets = {known[7:]} if known.startswith("column:") else set()
            if "available_at" in frame.columns:
                targets.add("available_at")
            for target in targets:
                if target not in availability:
                    availability[target] = utc_values(
                        frame.select(
                            dependency_time(target, self.resource, self.manifest)
                        ).to_series()
                    )
                at = availability[target]
                valid &= (
                    at <= self.decisions[np.minimum(dest, len(dates) - 1)]
                )  # mutation: panel_available
            if name in self.latest:
                # 多个来源区间在同一决策行到期时，保留来源时间最晚的值（含 null）。
                latest = self.latest[name]
                stamp = ends.astype(np.int64)
                np.maximum.at(latest, (dest[valid], columns[valid]), stamp[valid])
                valid &= stamp == latest[np.minimum(dest, len(dates) - 1), columns]
            arrays[name][dest[valid], columns[valid]] = (
                frame[name].cast(pl.Float64).to_numpy()[valid]
            )
