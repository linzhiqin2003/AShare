"""C5 独立核验：从冻结 REST/归档重算许可集合与终点，不调用构建判据。"""

import math

HOUR = 3_600_000


def coverage_stop(inputs, name, start, requested):
    """独立重放退出边界前的事件版本，不读取面板 lifecycle 或构建 helper。"""
    records = [e for e in inputs["events"] if e["instrument"] == name]
    for stop in sorted({e["effective_at"] for e in records}):
        if stop > requested or stop % HOUR:
            continue
        latest = {}
        for event in sorted(records, key=lambda e: e["known_at"]):
            if event["known_at"] <= max(start, stop - HOUR):
                latest[event["event_id"]] = event
        if any(
            e.get("status", "active") == "active"
            and e["effective_at"] == stop
            and e["kind"] in {"delist", "delivery", "settlement"}
            for e in latest.values()
        ):
            return stop
    return requested


def verify_recent(frame, inputs):
    context = inputs.get("recent")
    inferred_rows = frame.filter((frame["quality"] & 4096) != 0)
    if not context:
        return [dict(check="inferred_without_opt_in")] if inferred_rows.height else []
    opted_in = context.get("allow_inferred_funding", False)
    errors = (
        [dict(check="inferred_without_opt_in")]
        if inferred_rows.height and not opted_in
        else []
    )
    end = frame["time"].max() + HOUR
    start = frame["time"].min()
    stops = {
        p["instrument"]: coverage_stop(
            inputs, p["instrument"], start, context["requested_end"]
        )
        for p in inputs["profiles"]
    }
    caps = [context["requested_end"]]
    for p in inputs["profiles"]:
        name = p["instrument"]
        src = inputs["data"][name]
        if not opted_in or stops[name] <= max(start, p["listed_at"]):
            continue
        required = ["klines"] + (
            ["markPriceKlines", "indexPriceKlines"] if p["market"] == "usdm" else []
        )
        for dataset in required:
            times = [
                x["time"]
                for x in src.get(dataset, [])
                if x.get("source", "archive") == "archive"
                and x.get("time_quality", "normal") == "normal"
            ]
            if not times:
                errors.append(dict(check="inferred_archive_coverage", instrument=name))
            elif max(times) + HOUR < stops[name]:
                caps.append(max(times) + HOUR)
    cap = min(caps)
    for p in inputs["profiles"]:
        name = p["instrument"]
        src = inputs["data"][name]
        if p["contract_type"] != "PERPETUAL" or stops[name] <= max(
            start, p["listed_at"]
        ):
            continue
        archives = sorted(
            [
                x
                for x in src["fundingRate"]
                if x.get("source") != "inferred_continuation"
            ],
            key=lambda x: x["time"],
        )
        if not archives:
            errors.append(dict(check="inferred_anchor_missing", instrument=name))
            continue
        anchor = archives[-1]
        delta = anchor["interval"] * HOUR
        info = context.get("funding_info")
        matches = (
            []
            if info is None
            else [x for x in info["records"] if x.get("symbol") == p["symbol"]]
        )
        authorized = opted_in and info is not None
        if matches:
            authorized &= (
                len(matches) == 1
                and matches[0].get("fundingIntervalHours") == anchor["interval"]
            )
        else:
            authorized &= anchor["interval"] == 8
        allowed = {}
        strict_bound = anchor["time"] + delta - HOUR
        active_cap = min(cap, stops[name])
        if not authorized:
            if strict_bound < stops[name]:
                caps.append(strict_bound)
            active_cap = min(active_cap, strict_bound)
        # 推断模式始终审查有效窗口内的 REST；归档间隔上限不是无结算证明。
        if opted_in:
            previous = anchor["time"]
            broken = False
            for raw in sorted(
                src.get("funding_prices", []), key=lambda x: int(x["fundingTime"])
            ):
                observed = int(raw["fundingTime"])
                if observed <= anchor["time"] + 1000:
                    continue
                if observed > active_cap + 1000:
                    break
                t = ((observed + HOUR // 2) // HOUR) * HOUR
                try:
                    price = float(raw.get("markPrice") or "nan")
                    rate = float(raw["fundingRate"])
                except (ValueError, TypeError, KeyError):
                    price, rate = math.nan, math.nan
                valid = (
                    abs(t - observed) <= 1000
                    and t % delta == 0
                    and t == previous + delta
                    and math.isfinite(price)
                    and price > 0
                    and math.isfinite(rate)
                )
                if not valid:
                    broken = True
                    break
                allowed[t] = (anchor["interval"], rate, price)
                previous = t
            if (broken or previous + delta <= active_cap) and previous < stops[name]:
                caps.append(previous)
        actual_records = {
            r["time"]: r
            for r in src["fundingRate"]
            if r.get("source") == "inferred_continuation" and r["time"] <= end
        }
        for t, r in actual_records.items():
            if t not in allowed or (r["interval"], r["rate"]) != allowed[t][:2]:
                errors.append(
                    dict(check="inferred_record_condition", instrument=name, time=t)
                )
        for r in frame.filter(frame["instrument"] == name).iter_rows(named=True):
            t = r["time"] + HOUR
            should_infer = (
                r["lifecycle"] == "active" and anchor["time"] < t and t in allowed
            )
            if bool(r["quality"] & 4096) != should_infer:
                errors.append(dict(check="inferred_quality", instrument=name, time=t))
            if (
                should_infer
                and (r["funding_interval_hours"], r["funding_rate"], r["funding_price"])
                != allowed[t]
            ):
                errors.append(
                    dict(check="inferred_economic_fields", instrument=name, time=t)
                )
            if should_infer and t not in actual_records:
                errors.append(
                    dict(check="inferred_record_missing", instrument=name, time=t)
                )
    caps.extend(
        n["truncated_at"]
        for n in inputs.get("notes", {}).values()
        if "truncated_at" in n
    )
    if end != min(caps):
        errors.append(
            dict(
                check="inferred_terminal" if opted_in else "strict_terminal",
                end=end,
                expected=min(caps),
            )
        )
    return errors
