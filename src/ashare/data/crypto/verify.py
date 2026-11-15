"""独立校验：不导入构建、格式解析、资金费、事件查询与规则生成函数。

只共享文件读取入口；从冻结原始行重新计算连续性、现金流和事件落点。
价格差异仅诊断，绝不覆盖官方记录或要求与小时开盘相等。
"""

import math
from bisect import bisect_right
from collections import defaultdict

from .dataset import CryptoDataset
from .reconcile import reports
from .verify_contract import verify_contract
from .verify_raw import verify_raw
from .verify_recent import verify_recent

HOUR = 3_600_000


def verify_frame(frame, inputs, *, price_threshold=0.01):
    errors, diagnostics = [], []
    rows = frame.to_dicts()
    by_symbol = defaultdict(list)
    for row in rows:
        by_symbol[row["instrument"]].append(row)
    profiles = {p["instrument"]: p for p in inputs["profiles"]}
    times = sorted(set(r["time"] for r in rows))
    if not times or any(b - a != HOUR for a, b in zip(times, times[1:], strict=False)):
        errors.append(dict(check="hour_axis", reason="空轴或小时轴缺口"))

    def fail(check, name, t, **detail):
        errors.append(dict(check=check, instrument=name, time=t, **detail))

    for name, group in by_symbol.items():
        p = profiles[name]
        group.sort(key=lambda r: r["time"])
        if [r["time"] for r in group] != times:
            fail("rectangle", name, None)
        source = inputs["data"][name]
        raw = {r["time"]: r for r in source.get("klines", [])}
        marks = {r["time"]: r for r in source.get("markPriceKlines", [])}
        indices = {r["time"]: r for r in source.get("indexPriceKlines", [])}
        rates = sorted(source.get("fundingRate", []), key=lambda r: r["time"])
        rate_by_time = {r["time"]: r for r in rates}
        official = {
            round(int(r["fundingTime"]) / HOUR) * HOUR: r
            for r in source.get("funding_prices", [])
        }
        delivery = {
            int(r["deliveryTime"]): float(r["deliveryPrice"])
            for r in source.get("deliveries", [])
        }
        if len(rate_by_time) != len(rates):
            fail("funding_duplicate", name, None)
        for r in rates:
            if r["interval"] not in (1, 2, 4, 8) or r["time"] % (HOUR * r["interval"]):
                fail("funding_grid", name, r["time"])
            if abs(r.get("raw_time", r["time"]) - r["time"]) > 1000:
                fail("funding_jitter", name, r["time"])
        active = [r for r in group if r["lifecycle"] == "active"]
        if p["contract_type"] == "PERPETUAL" and active:
            lo, hi = active[0]["time"], active[-1]["time"] + HOUR
            if not rates:
                fail("funding_missing", name, lo)
            else:
                first, last = rates[0], rates[-1]
                first_expected = (
                    (max(lo, p["listed_at"]) // (first["interval"] * HOUR) + 1)
                    * first["interval"]
                    * HOUR
                )
                if first["time"] > first_expected and first_expected <= hi:
                    fail("funding_missing", name, first_expected)
                if last["time"] + last["interval"] * HOUR <= hi:
                    fail(
                        "funding_missing", name, last["time"] + last["interval"] * HOUR
                    )
                for before, after in zip(rates, rates[1:], strict=False):
                    step = after["interval"] * HOUR
                    expected = before["time"] - before["time"] % step + step
                    if (
                        before["time"] < hi
                        and after["time"] > lo
                        and after["time"] != expected
                    ):
                        fail("funding_missing", name, expected)
        previous = None
        previous_close = None
        datasets = [raw, marks, indices] if p["market"] == "usdm" else [raw]
        trade_times = sorted(raw)
        trade_cursor = 0
        normal_mark_times = sorted(
            t
            for t, row in marks.items()
            if row.get("time_quality", "normal") == "normal"
        )
        for r in group:
            t, boundary = r["time"], r["time"] + HOUR
            futures = p["market"] == "usdm"
            if futures and (
                r["settlement"] is None
                or not math.isfinite(r["settlement"])
                or r["settlement"] <= 0
            ):
                fail("settlement_positive", name, t)
            if r["lifecycle"] != "active":
                if (
                    r["tradable"]
                    or r["funding"] != 0
                    or any(
                        not math.isnan(r["mark_" + f])
                        for f in ("open", "high", "low", "close")
                    )
                ):
                    fail("outside_lifecycle", name, t)
                continue
            missing = t not in raw or (futures and (t not in marks or t not in indices))
            # 从原始成交时间轴独立找缺口前的证据，不沿用构建器的状态或辅助线标记。
            while trade_cursor < len(trade_times) and trade_times[trade_cursor] < t:
                trade_cursor += 1
            preceding = raw[trade_times[trade_cursor - 1]] if trade_cursor else None
            halt = (
                t not in raw
                and preceding is not None
                and preceding.get("time_quality") == "partial_halt"
            )
            counterevidence = any(
                t in values and values[t].get("time_quality", "normal") == "normal"
                for values in (raw, marks)
                if futures or values is raw
            )
            mark_cursor = bisect_right(normal_mark_times, t)
            if futures and preceding is not None and mark_cursor:
                counterevidence |= (
                    normal_mark_times[mark_cursor - 1] > preceding["time"]
                )
            if counterevidence:
                halt = False
            time_flags = {
                values[t].get("time_quality", "normal")
                for values in datasets
                if t in values
            }
            for flag, bit in (("partial_halt", 512), ("invalid_close_time", 2048)):
                if bool(r["quality"] & bit) != (flag in time_flags):
                    fail("time_quality", name, t, field=flag)
                if flag in time_flags:
                    diagnostics.append(
                        dict(
                            check=flag,
                            instrument=name,
                            time=t,
                            review=flag == "invalid_close_time",
                        )
                    )
            if bool(r["quality"] & 1024) != halt:
                fail("halt_evidence", name, t)
            if halt:
                diagnostics.append(dict(check="inferred_halt", instrument=name, time=t))
            if missing:
                # 缺口必须逐格有标记与解释，allow_missing 仅是模型选择，不是解释。
                known_suspend = False
                latest = {}
                for e in inputs["events"]:
                    if e["instrument"] == name and e["known_at"] <= t:
                        if (
                            e["event_id"] not in latest
                            or latest[e["event_id"]]["known_at"] < e["known_at"]
                        ):
                            latest[e["event_id"]] = e
                for e in sorted(latest.values(), key=lambda e: e["effective_at"]):
                    if e["effective_at"] <= t and e.get("status", "active") == "active":
                        if e["kind"] == "suspend":
                            known_suspend = True
                        if e["kind"] == "resume":
                            known_suspend = False
                exceptions = inputs.get("exceptions", [])
                explained = (
                    halt
                    or known_suspend
                    or any(
                        e["instrument"] == name
                        and e["start"] <= t < e["end"]
                        and e.get("reason")
                        and e.get("source_url")
                        for e in exceptions
                    )
                )
                if not explained or not (r["quality"] & 1) or r["tradable"]:
                    fail("unexplained_gap", name, t)
                if (
                    any(
                        r[f] != previous_close for f in ("open", "high", "low", "close")
                    )
                    or r["volume_engine"] != 0
                ):
                    fail("gap_valuation", name, t)
            else:
                for f in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "taker_buy_quote_volume",
                ):
                    if r[f] != raw[t][f]:
                        fail("raw_value", name, t, field=f)
                if (
                    r["low"] <= 0
                    or r["low"] > min(r["open"], r["close"])
                    or r["high"] < max(r["open"], r["close"])
                    or r["volume"] < 0
                    or r["quote_volume"] < 0
                ):
                    fail("ohlcv", name, t)
                if r["taker_buy_quote_volume"] < 0 or r["taker_buy_quote_volume"] > r[
                    "quote_volume"
                ] * (1 + 1e-12):
                    fail("taker_volume", name, t)
                if futures:
                    for f in ("open", "high", "low", "close"):
                        if r["mark_" + f] != marks[t][f] or r["mark_" + f] <= 0:
                            fail("mark_ohlc", name, t, field=f)
                    if r["mark_low"] > min(r["mark_open"], r["mark_close"]) or r[
                        "mark_high"
                    ] < max(r["mark_open"], r["mark_close"]):
                        fail("mark_range", name, t)
                    if r["index_close"] != indices[t]["close"]:
                        fail("index_value", name, t)
                    if not math.isclose(
                        r["volume_engine"] * p["multiplier"], r["volume"], rel_tol=1e-12
                    ):
                        fail("volume_unit", name, t)
            latest = {}
            for event in inputs["events"]:
                if event["instrument"] == name and event["known_at"] <= t:
                    key = event["event_id"]
                    if key not in latest or event["known_at"] > latest[key]["known_at"]:
                        latest[key] = event
            expiry = [
                e
                for e in latest.values()
                if e.get("status", "active") == "active"
                and e["effective_at"] == boundary
                and e["kind"] in ("delivery", "settlement", "delist")
            ]
            if futures and bool(expiry) != r["expires"]:
                fail("expiry_boundary", name, t)
            expected_settle = (
                marks[t]["close"] if t in marks and not missing else previous
            )
            if expiry and futures:
                e = expiry[0]
                if r["event_version"] != e["version_id"]:
                    fail("event_version", name, t)
                if e["kind"] == "delivery":
                    expected_settle = delivery.get(boundary)
                    record_time = boundary
                    if expected_settle is None and boundary % (24 * HOUR) == 8 * HOUR:
                        record_time = boundary - 8 * HOUR
                        expected_settle = delivery.get(record_time)
                    if r.get("delivery_record_time") != record_time:
                        fail("delivery_record_time", name, t)
                    if expected_settle is None:
                        fail("delivery_price_missing", name, t)
                    elif t in indices:
                        deviation = expected_settle / indices[t]["close"] - 1
                        if abs(deviation) > price_threshold:
                            diagnostics.append(
                                dict(
                                    check="delivery_index_deviation",
                                    instrument=name,
                                    time=t,
                                    deviation=deviation,
                                )
                            )
                elif e.get("settlement_price") is not None:
                    expected_settle = e["settlement_price"]
                else:
                    expected_settle = marks.get(boundary, {}).get("open")
            if futures and r["settlement"] != expected_settle:
                fail("settlement_value", name, t)
            rate = (
                rate_by_time.get(boundary)
                if p["contract_type"] == "PERPETUAL"
                else None
            )
            if rate is None:
                if r["funding"] != 0 or not math.isnan(r["funding_rate"]):
                    fail("funding_no_event", name, t)
            else:
                recorded = official.get(boundary, {})
                if boundary >= 1698739200000 and not recorded.get("markPrice"):
                    fail("official_funding_price_missing", name, t)
                price = (
                    float(recorded["markPrice"])
                    if recorded.get("markPrice")
                    else marks.get(boundary, {}).get("open")
                )
                if (
                    price is None
                    or not math.isclose(
                        r["funding"],
                        rate["rate"] * price * p["multiplier"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or r["funding_rate"] != rate["rate"]
                    or r["funding_interval_hours"] != rate["interval"]
                ):
                    fail("funding_cash", name, t)
                if r["funding_price"] != price:
                    fail("funding_price_value", name, t)
                if (
                    recorded.get("fundingRate") is not None
                    and float(recorded["fundingRate"]) != rate["rate"]
                ):
                    if (
                        rate.get("archive_rate_override")
                        and rate.get("source") != "inferred_continuation"
                    ):
                        diagnostics.append(
                            dict(
                                check="funding_rate_archive_override",
                                instrument=name,
                                time=t,
                                archive_rate=rate["rate"],
                                rest_rate=float(recorded["fundingRate"]),
                            )
                        )
                    else:
                        fail("funding_rate_sources", name, t)
                if recorded.get("markPrice") and boundary in marks:
                    deviation = (
                        float(recorded["markPrice"]) / marks[boundary]["open"] - 1
                    )
                    diagnostics.append(
                        dict(
                            check="funding_mark_deviation",
                            instrument=name,
                            time=t,
                            deviation=deviation,
                            review=abs(deviation) > price_threshold,
                            outside_range=not marks[boundary]["low"]
                            <= float(recorded["markPrice"])
                            <= marks[boundary]["high"],
                        )
                    )
            previous = r["settlement"]
            previous_close = r["close"]
    # 单币现货/永续按真实币单位比较；成交额不做 1000 倍转换。
    spot = {
        (r["time"], r["coin"]): r["close"] / r["coin_units"]
        for r in rows
        if r["market"] == "spot" and r["lifecycle"] == "active"
    }
    for r in rows:
        other = spot.get((r["time"], r["coin"]))
        if other and r["contract_type"] == "PERPETUAL" and r["lifecycle"] == "active":
            deviation = r["close"] / r["coin_units"] / other - 1
            if abs(deviation) > price_threshold:
                diagnostics.append(
                    dict(
                        check="spot_perpetual_deviation",
                        instrument=r["instrument"],
                        time=r["time"],
                        deviation=deviation,
                    )
                )
    errors.extend(verify_contract(frame, inputs))
    errors.extend(verify_recent(frame, inputs))
    return dict(
        verdict="fail" if errors else "pass",
        rows=len(rows),
        errors=errors,
        diagnostics=diagnostics,
    )


def verify(root, version=None, *, price_threshold=0.01, allow_inferred_funding=False):
    ds = CryptoDataset(root, version, allow_inferred_funding=allow_inferred_funding)
    inputs = ds.inputs()
    report = verify_frame(ds.read(), inputs, price_threshold=price_threshold)
    raw_errors, checked = verify_raw(root, inputs)
    report["errors"].extend(raw_errors)
    report.update(
        version=ds.version,
        archives_checked=checked,
        funding_reconciliations=reports(ds.root, ds.version),
        verdict="fail" if report["errors"] else "pass",
    )
    return report
