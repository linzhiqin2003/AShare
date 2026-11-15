"""显式资金费延续推断；保守截断，不把当前快照当作历史规则证据。"""

import math

from .events import asof, validate_events
from .timeutil import HOUR, funding_hour

INFERRED = "inferred_continuation"


def active_until(profile, events, start, end):
    """按实际执行边界前可知的事件确定覆盖需求；退出后的列不限制共同终点。"""
    name = profile["instrument"]
    candidates = sorted(
        {
            e["effective_at"]
            for e in events
            if e["instrument"] == name
            and e["kind"] in ("delist", "settlement", "delivery")
            and e["effective_at"] <= end
            and e["effective_at"] % HOUR == 0
        }
    )
    for stop in candidates:
        at = max(start, stop - HOUR)
        if any(
            e["effective_at"] == stop
            and e["kind"] in ("delist", "settlement", "delivery")
            for e in asof(events, at, instrument=name)
        ):
            return stop
    return end


def prepare_recent(
    profiles,
    data,
    end,
    *,
    start,
    events,
    allow_inferred_funding=False,
    funding_info=None,
):
    """返回构建输入与共同右端；不修改同步原始行。"""
    result = {name: dict(values) for name, values in data.items()}
    events = validate_events(events)
    stops = {p["instrument"]: active_until(p, events, start, end) for p in profiles}
    bounds, notes = [end], {}
    if allow_inferred_funding:
        for p in profiles:
            name = p["instrument"]
            if stops[name] <= max(start, p["listed_at"]):
                continue
            required = (
                ("klines",)
                if p["market"] == "spot"
                else ("klines", "markPriceKlines", "indexPriceKlines")
            )
            for dataset in required:
                archived = [
                    r["time"] + HOUR
                    for r in data[name].get(dataset, [])
                    if r.get("source", "archive") == "archive"
                    and r.get("time_quality", "normal") == "normal"
                ]
                if not archived:
                    raise ValueError(f"{name} {dataset} 没有完整小时归档")
                if max(archived) < stops[name]:
                    bounds.append(max(archived))
    cap = min(bounds)
    for p in profiles:
        name = p["instrument"]
        source = result[name]
        if p["contract_type"] != "PERPETUAL" or stops[name] <= max(
            start, p["listed_at"]
        ):
            continue
        rates = sorted(source.get("fundingRate", []), key=lambda r: r["time"])
        if not rates:
            raise ValueError(f"{name} 没有历史资金费间隔")
        # 已发布归档为费率权威；保留缓存 REST 的原值并显式记录冲突。
        rest_rates = {}
        for raw in source.get("funding_prices", []):
            if int(raw["fundingTime"]) <= rates[-1]["time"] + 1000:
                rest_rates[funding_hour(raw["fundingTime"])] = float(raw["fundingRate"])
        rates = [
            dict(r, archive_rate_override=True)
            if r["time"] in rest_rates and r["rate"] != rest_rates[r["time"]]
            else r
            for r in rates
        ]
        source["fundingRate"] = rates
        last = rates[-1]
        step = last["interval"] * HOUR
        strict_end = last["time"] + step - HOUR
        if not allow_inferred_funding:
            if strict_end < stops[name]:
                bounds.append(strict_end)
            continue
        note = dict(
            anchor=last["time"],
            interval_hours=last["interval"],
            strict_end=strict_end,
            accepted=0,
        )
        notes[name] = note
        entries = (
            []
            if funding_info is None
            else [r for r in funding_info["records"] if r.get("symbol") == p["symbol"]]
        )
        compatible = funding_info is not None and (
            (not entries and last["interval"] == 8)
            or (
                len(entries) == 1
                and entries[0].get("fundingIntervalHours") == last["interval"]
            )
        )
        # 缺笔、缩短间隔、偏离整点、缺价都在第一处停止，不能跳过后继续。
        accepted, previous, failure = [], last["time"], None
        active_cap = min(cap, stops[name])
        if not compatible:
            active_cap = min(active_cap, strict_end)
        # 覆盖豁免仅限退出之后；严格上限也不能证明归档之后没有额外结算。
        candidates = sorted(
            source.get("funding_prices", []), key=lambda r: int(r["fundingTime"])
        )
        for r in candidates:
            raw_time = int(r["fundingTime"])
            if raw_time <= last["time"] + 1000:
                continue
            if raw_time > active_cap + 1000:
                break
            try:
                t = funding_hour(raw_time)
                price = float(r["markPrice"]) if r.get("markPrice") else math.nan
                rate = float(r["fundingRate"])
            except (ValueError, TypeError, KeyError):
                failure = "invalid_rest_record"
                break
            if t - previous != step or t % step:
                failure = "funding_spacing_or_grid"
                break
            if not math.isfinite(price) or price <= 0 or not math.isfinite(rate):
                failure = "funding_price_or_rate_missing"
                break
            accepted.append(
                dict(
                    time=t,
                    raw_time=raw_time,
                    interval=last["interval"],
                    rate=rate,
                    source=INFERRED,
                )
            )
            previous = t
        # 不以相邻检查代替末笔覆盖；funding_schedule 还会在构建中再次检查。
        if failure is None and previous + step <= active_cap:
            failure = "funding_tail_missing"
        bound = min(active_cap, previous) if failure else active_cap
        if bound < stops[name]:
            bounds.append(bound)
        source["fundingRate"] = [*rates, *accepted]
        note.update(
            end=bound,
            accepted=len(accepted),
            reason=failure
            or ("archive_coverage" if compatible else "funding_info_unconfirmed"),
        )
    actual_end = min(bounds)
    # 只给资金费计算器传窗口内的 REST；完整原文留在冻结输入供独立核验。
    context = dict(
        allow_inferred_funding=allow_inferred_funding,
        requested_end=end,
        funding_info=funding_info,
        instruments=notes,
    )
    return result, actual_end, context
