"""标准层转 T×N；缺行保留 NaN，has_quote 只来自行存在性。"""

from datetime import UTC
from types import MappingProxyType

import numpy as np
import polars as pl

from ..reader import Panel
from .dependencies import marked_metadata, unknown_sources, validate_time_dependencies
from .manifest import normalized
from .panel_timing import Placement, decisions, lookback
from .reader import boundary


class SpecPanel(Panel):
    """保留原 Panel 接口，额外公开只读的字段声明与可知时点。"""

    def __init__(self, dates, symbols, arrays, version, field_metadata):
        super().__init__(dates, symbols, arrays, version)
        self.field_metadata = MappingProxyType(
            {name: MappingProxyType(desc) for name, desc in field_metadata.items()}
        )


def make_panel(package, fields, start, end, symbols, table, *, allow_unknown=False):
    fields = [fields] if isinstance(fields, str) else list(fields)
    resource = package.resources[table]
    if resource["kind"] != "bars":
        raise ValueError("panel 只支持 bars 表")
    names = {f["name"] for f in resource["schema"]["fields"]}
    derived = {"has_quote", "can_buy", "can_sell", "can_buy_open", "can_sell_open"}
    if set(fields) - names - derived:
        raise KeyError(f"未知面板字段：{sorted(set(fields) - names - derived)}")
    time_col = "ts" if "ts" in names else "date"
    daily = time_col == "date"
    first, last = boundary(start, daily), boundary(end, daily)
    chosen = set(fields) - derived
    mask_fields = {"open", "high", "low", "suspended", "limit_up", "limit_down"} & names
    if set(fields) & (derived - {"has_quote"}):
        chosen |= mask_fields
    declarations = {
        field["name"]: field
        for item in normalized(package.manifest)["resources"]
        if item["name"] == table
        for field in item["schema"]["fields"]
    }
    validate_time_dependencies(declarations)
    unknown = unknown_sources(declarations, chosen)
    if unknown and not allow_unknown:  # mutation: unknown_panel_guard
        raise ValueError(
            f"字段 {unknown} 的 knownAt=unknown；显式 allow_unknown=True 才可使用"
        )
    dependencies = {
        declarations[f]["knownAt"][7:]
        for f in chosen
        if str(declarations[f].get("knownAt", "")).startswith("column:")
    }
    dependencies |= {"available_at"} & names
    scan_fields = sorted(chosen | dependencies)
    scan = package._scan(table, start, end, symbols, scan_fields)
    schema = scan.collect_schema()
    if daily:
        calendar = next(
            (r["name"] for r in package.resources.values() if r["kind"] == "calendar"),
            None,
        )
        if calendar is None:
            raise ValueError("日频面板缺少 calendar")
        full_calendar = (
            package.read(calendar, end=end)["date"].sort().unique().to_numpy()
        )
        dates = (
            full_calendar
            if first is None
            else full_calendar[full_calendar >= np.datetime64(first)]
        )
        prior = (
            full_calendar[full_calendar < dates[0]] if len(dates) else full_calendar[:0]
        )
        previous = (
            decisions(prior[-1:], resource, package.manifest)[0] if len(prior) else None
        )
        public_dates = np.char.replace(
            dates.astype("datetime64[D]").astype(str), "-", ""
        )
    else:
        step = "1mo" if resource["frequency"] == "1M" else resource["frequency"]
        bounds = resource.get("bounds", {})
        lo = first or boundary(bounds.get("start"), False)
        hi = last or boundary(bounds.get("end"), False)
        minimum = maximum = None
        if lo is None or hi is None:
            minimum, maximum = (
                scan.select(
                    pl.col(time_col).min().alias("min"),
                    pl.col(time_col).max().alias("max"),
                )
                .collect()
                .row(0)
            )
        lo = lo or minimum
        if hi is None and maximum is not None:
            hi = pl.datetime_range(
                maximum, maximum, interval=step, eager=True, time_zone="UTC"
            ).dt.offset_by(step)[0]
        if lo is None or hi is None or lo >= hi:
            dates = np.array([], dtype="datetime64[us]")
        else:
            # ceil 到规则边界；日内规则按 UTC 对齐，非从请求中的任意分钟开始。
            lower = pl.Series([lo]).dt.truncate(step)
            if lower[0] < lo:
                lower = lower.dt.offset_by(step)
            dates = (
                pl.datetime_range(
                    lower[0],
                    hi,
                    interval=step,
                    closed="left",
                    time_zone="UTC",
                    eager=True,
                )
                .dt.replace_time_zone(None)
                .to_numpy()
                .astype("datetime64[us]")
            )
        public_dates = dates.copy()
        previous = (
            dates[0].astype("datetime64[us]").item().replace(tzinfo=UTC)
            if len(dates)
            else None
        )
    source_fields = {f: declarations[f] for f in chosen}
    source_start = lookback(first, previous, source_fields, resource, package.manifest)
    if source_start != first:
        scan = package._scan(table, source_start, end, symbols, scan_fields)
    if symbols is None:
        instrument = next(
            (
                r["name"]
                for r in package.resources.values()
                if r["kind"] == "instruments"
            ),
            None,
        )
        codes = (
            package.read(instrument, fields=["symbol"])["symbol"].to_list()
            if instrument
            else scan.select(pl.col("symbol").unique().sort())
            .collect()["symbol"]
            .to_list()
        )
    else:
        codes = list(symbols)
    if len(codes) != len(set(codes)):
        raise ValueError("symbols 不允许重复")
    code_index = {code: i for i, code in enumerate(codes)}
    shape = (len(dates), len(codes))
    quote = np.zeros(shape, dtype=bool)
    arrays = {"has_quote": quote}
    for field in chosen:
        if (
            schema[field] in (pl.String, pl.Date)
            or schema[field].base_type() == pl.Datetime
        ):
            raise TypeError(f"面板字段 {field} 必须是数值或布尔")
        arrays[field] = np.full(shape, np.nan, dtype=np.float64)
    placement = Placement(
        dates,
        previous,
        source_fields,
        resource,
        package.manifest,
        arrays,
        calendar=full_calendar if daily else None,
        history_before=first if source_start != first else None,
    )
    # 直接把流式批次填进最终数组，额外内存不随整个长表窗口增长。
    batches = scan.collect_batches(chunk_size=65536, engine="streaming", lazy=True)
    try:
        for frame in batches:
            values = (
                frame[time_col].to_numpy()
                if daily
                else frame[time_col]
                .dt.replace_time_zone(None)
                .to_numpy()
                .astype("datetime64[us]")
            )
            j = np.array([code_index[x] for x in frame["symbol"]], dtype=np.int64)
            placement.fill(frame, values, j)
    finally:
        del batches
    if set(fields) & (derived - {"has_quote"}):
        buy, sell = quote.copy(), quote.copy()
        if "suspended" in arrays:
            valid = arrays["suspended"] == 0
            buy &= valid
            sell &= valid
        opening = arrays.get("open")
        if opening is not None and {"suspended", "limit_up", "limit_down"} & names:
            domain = (
                opening != 0
                if declarations["open"].get("allowNegative") is True
                else opening > 0
            )
            valid = np.isfinite(opening) & domain
            buy &= valid
            sell &= valid
            tolerance = 0.005 if package.manifest["symbology"] == "cn" else 0
            for limit, mask, direction in (
                ("limit_up", buy, 1),
                ("limit_down", sell, -1),
            ):
                if limit in arrays:
                    restricted = np.isfinite(arrays[limit]) & (
                        (opening - arrays[limit]) * direction >= -tolerance
                    )
                    if "high" in arrays and "low" in arrays:
                        restricted |= (
                            np.abs(arrays["high"] - arrays["low"]) <= tolerance
                        ) & ((arrays["high"] - arrays[limit]) * direction >= -tolerance)
                    mask &= ~restricted
        arrays.update(can_buy=buy, can_sell=sell, can_buy_open=buy, can_sell_open=sell)
    arrays = {
        name: value
        for name, value in arrays.items()
        if name in fields or name == "has_quote"
    }
    for value in arrays.values():
        value.flags.writeable = False
    metadata = {
        name: marked_metadata(declarations[name], declarations, [name])
        for name in arrays
        if name in declarations
    }
    for name in arrays.keys() - metadata.keys():
        dependencies = [] if name == "has_quote" else sorted(mask_fields)
        desc = dict(name=name, type="boolean", sourceFields=dependencies)
        metadata[name] = marked_metadata(desc, declarations, dependencies)
    return SpecPanel(public_dates, np.array(codes), arrays, package.version, metadata)
