"""小时面板构建；时间、生命周期、事件及规则均显式保留。"""

import math

import polars as pl

from .delivery import delivery_record
from .events import asof, validate_events
from .funding import funding_schedule, funding_value, price_map
from .instruments import rule_at, validate_profile
from .timeutil import HOUR, millis

PRICE_FIELDS = ("open", "high", "low", "close")
VOLUME_FIELDS = ("volume", "quote_volume", "taker_buy_quote_volume")
FLOAT_FIELDS = (
    *PRICE_FIELDS,
    *VOLUME_FIELDS,
    *("mark_" + f for f in PRICE_FIELDS),
    "coin_units",
    "multiplier",
    "volume_engine",
    "index_close",
    "settlement",
    "funding",
    "funding_rate",
    "last_funding_rate",
    "funding_price",
    "funding_interval_hours",
    "qty_step",
    "min_qty",
    "min_notional",
)
PANEL_SCHEMA = {
    **dict.fromkeys(FLOAT_FIELDS, pl.Float64),
    "event_version": pl.String,
    "delivery_record_time": pl.Int64,
    "time": pl.Int64,
    "rule_known_at": pl.Int64,
    "rule_effective_at": pl.Int64,
    "quality": pl.Int64,
}
QUALITY = {
    "missing": 1,
    "zero_volume": 2,
    "frozen_zero": 4,
    "suspended": 8,
    "funding_approximation": 16,
    "settlement_approximation": 32,
    "rule_assumption": 64,
    "outside_lifecycle": 128,
    "rest": 256,
    "partial_halt": 512,
    "inferred_halt": 1024,
    "invalid_close_time": 2048,
}


def indexed(rows):
    result = {}
    for r in rows:
        if r["time"] in result:
            raise ValueError("同一数据集时间重复")
        result[r["time"]] = r
    return result


def build_instrument(profile, data, events, start, end, *, allow_missing=False):
    p = validate_profile(profile)
    name = p["instrument"]
    bars = indexed(data.get("klines", []))
    marks = indexed(data.get("markPriceKlines", []))
    indices = indexed(data.get("indexPriceKlines", []))
    prices = price_map(data.get("funding_prices", []))
    deliveries = {
        int(r["deliveryTime"]): float(r["deliveryPrice"])
        for r in data.get("deliveries", [])
    }
    futures = p["market"] == "usdm"
    perpetual = p["contract_type"] == "PERPETUAL"
    if futures and p.get("delivery_at") and p["delivery_at"] <= end:
        if not any(
            e["instrument"] == name
            and e["kind"] in ("settlement", "delivery", "delist")
            and e["known_at"] <= e["effective_at"]
            for e in events
        ):
            raise ValueError(
                f"{name} 档案显示终止，但缺少当时可知的终止事件，不能用尾部推断"
            )
    # 只用在请求终点之前已知的版本验证资金费覆盖，不使用更晚的延期/撤销。
    terminal = [
        e
        for e in asof(events, end, instrument=name)
        if e["kind"] in ("delist", "settlement", "delivery")
    ]
    stop = min((e["effective_at"] for e in terminal), default=None)
    schedule = (
        funding_schedule(
            data.get("fundingRate", []), start, end, p["listed_at"], stop=stop
        )
        if perpetual
        else {}
    )
    first_marks = [r for t, r in sorted(marks.items()) if t >= p["listed_at"]]
    first_price = first_marks[0]["close"] if first_marks else math.nan
    rows, last_settlement, last_close, ended = [], None, None, False
    end_price, suspended, frozen_count = None, False, 0
    history = sorted(data.get("fundingRate", []), key=lambda r: r["time"])
    known_rate, funding_cursor = math.nan, 0
    datasets = [bars, marks, indices] if futures else [bars]
    trade_partial_before = False
    halt_notes = []
    for t in range(start, end, HOUR):
        while funding_cursor < len(history) and history[funding_cursor]["time"] <= t:
            known_rate = history[funding_cursor]["rate"]
            funding_cursor += 1
        known = asof(events, t, instrument=name)
        for event in known:
            if event["effective_at"] % HOUR:
                if event["kind"] in ("settlement", "delivery", "delist"):
                    raise ValueError(f"{name} 终止事件不是整点，需要更细粒度模型")
            if event["effective_at"] <= t:
                if event["kind"] == "suspend":
                    suspended = True
                elif event["kind"] == "resume":
                    suspended = False
                elif event["kind"] in ("settlement", "delivery", "delist"):
                    if not ended and t >= start and event["effective_at"] > start:
                        raise ValueError(f"{name} 迟到的终止事件不能倒填历史")
                    ended = True
                    if end_price is None:
                        end_price = event.get("settlement_price") or deliveries.get(
                            event["effective_at"]
                        )
                        if event["kind"] == "delivery":
                            end_price = float(
                                delivery_record(
                                    data.get("deliveries", []), event["effective_at"]
                                )["deliveryPrice"]
                            )
                        if end_price is None:
                            boundary = marks.get(event["effective_at"])
                            end_price = boundary["open"] if boundary else None
        outside = t < p["listed_at"] or ended
        rule = rule_at(p, t)
        unit = p["multiplier"]
        row = dict(
            time=t,
            instrument=name,
            market=p["market"],
            symbol=p["symbol"],
            coin=p["coin"],
            coin_units=p["coin_units"],
            multiplier=unit,
            contract_type=p["contract_type"],
            lifecycle="pre_listing"
            if t < p["listed_at"]
            else "ended"
            if ended
            else "active",
            tradable=False,
            expires=False,
            quality=0,
            funding=0.0,
            funding_rate=math.nan,
            last_funding_rate=known_rate,
            funding_price=math.nan,
            funding_interval_hours=math.nan,
            funding_price_source="none",
            settlement=math.nan,
            settlement_source="none",
            index_close=math.nan,
            rule_source=rule["source"],
            rule_source_url=rule["source_url"],
            rule_known_at=rule["known_at"],
            rule_effective_at=rule["effective_at"],
            rule_approximation=bool(
                rule.get("old_positions_exempt") or rule["effective_at"] % HOUR
            ),
            qty_step=rule["qty_step"] / unit if futures else rule["qty_step"],
            min_qty=rule["min_qty"] / unit if futures else rule["min_qty"],
            min_notional=rule["min_notional"],
            volume_engine=0.0,
            data_source="none",
            mark_source="none",
            index_source="none",
            event_version=None,
            delivery_record_time=None,
        )
        row.update(
            {f: math.nan for f in (*PRICE_FIELDS, *("mark_" + x for x in PRICE_FIELDS))}
        )
        row.update({f: 0.0 for f in VOLUME_FIELDS})
        if rule["assumed"]:
            row["quality"] |= QUALITY["rule_assumption"]
        if outside:
            row["last_funding_rate"] = math.nan
            row["quality"] |= QUALITY["outside_lifecycle"]
            if futures:
                row["settlement"] = first_price if t < p["listed_at"] else end_price
                row["settlement_source"] = (
                    "pre_listing_placeholder"
                    if t < p["listed_at"]
                    else "terminal_carry"
                )
                if (
                    row["settlement"] is None
                    or not math.isfinite(row["settlement"])
                    or row["settlement"] <= 0
                ):
                    raise ValueError(f"{name} 生命周期外缺少可追溯的 settlement 占位值")
            elif ended:
                if end_price is None and terminal:
                    boundary = min(e["effective_at"] for e in terminal)
                    old_bars = [v for ts, v in bars.items() if ts < boundary]
                    end_price = (
                        max(old_bars, key=lambda x: x["time"])["close"]
                        if old_bars
                        else None
                    )
                if end_price is None:
                    raise ValueError(f"{name} 现货下架后缺最后成交估值")
                row.update({f: end_price for f in PRICE_FIELDS})
                row["settlement_source"] = "unrealized_spot_carry"
            rows.append(row)
            continue
        bar, mark, index = bars.get(t), marks.get(t), indices.get(t)
        missing = bar is None or (futures and (mark is None or index is None))
        # 只有成交线的半截 + 连续缺口能证明停机；正常标记线是明确反证。
        inferred_halt = (
            bar is None
            and trade_partial_before
            and not (
                futures
                and mark is not None
                and mark.get("time_quality", "normal") == "normal"
            )
        )
        qualities = [
            values[t].get("time_quality", "normal")
            for values in datasets
            if t in values
        ]
        for flag in ("partial_halt", "invalid_close_time"):
            if flag in qualities:
                row["quality"] |= QUALITY[flag]
        if inferred_halt:
            row["quality"] |= QUALITY["inferred_halt"]
            if halt_notes and halt_notes[-1]["end"] == t:
                halt_notes[-1]["end"] = t + HOUR
            else:
                halt_notes.append(
                    dict(
                        start=t, end=t + HOUR, evidence="preceding_partial_trade_kline"
                    )
                )
        # 恢复成交后终止推断。标记反证只限缺口内，半截成交线同小时尚不属缺口。
        if bar is not None:
            trade_partial_before = bar.get("time_quality") == "partial_halt"
        if (
            bar is None
            and futures
            and mark is not None
            and mark.get("time_quality", "normal") == "normal"
        ):
            trade_partial_before = False
        if missing:
            if not (allow_missing or suspended or inferred_halt):
                raise ValueError(f"{name} 生命周期内缺少行情/标记/指数: {t}")
            if last_settlement is None:
                raise ValueError(f"{name} 缺失行没有之前的估值，不能用未来价格补")
            row["quality"] |= QUALITY["missing"]
            row.update({f: last_close for f in PRICE_FIELDS})
            if futures:
                row.update({"mark_" + f: last_settlement for f in PRICE_FIELDS})
                row["settlement"] = last_settlement
                row["settlement_source"] = "missing_carry"
        else:
            row.update({f: bar[f] for f in (*PRICE_FIELDS, *VOLUME_FIELDS)})
            row["volume_engine"] = bar["volume"] / unit if futures else bar["volume"]
            row["data_source"] = bar.get("source", "archive")
            if row["data_source"] == "rest":
                row["quality"] |= QUALITY["rest"]
            row["tradable"] = (
                not suspended
                and "invalid_close_time" not in qualities
                and bar["volume"] > 0
            )
            if bar["quote_volume"] == 0:
                row["quality"] |= QUALITY["zero_volume"]
                frozen_count = frozen_count + 1 if bar["close"] == last_close else 1
                if frozen_count >= 3:
                    row["quality"] |= QUALITY["frozen_zero"]
            else:
                frozen_count = 0
            if futures:
                row.update({"mark_" + f: mark[f] for f in PRICE_FIELDS})
                row["mark_source"] = mark.get("source", "archive")
                row["index_source"] = index.get("source", "archive")
                row["index_close"] = index["close"]
                row["settlement"], row["settlement_source"] = (
                    mark["close"],
                    "mark_close",
                )
        if suspended:
            row["quality"] |= QUALITY["suspended"]
        expiry = [
            e
            for e in known
            if e["effective_at"] == t + HOUR
            and e["kind"] in ("delivery", "settlement", "delist")
        ]
        if len(expiry) > 1:
            raise ValueError("同一交易对边界有多个终止事件")
        if expiry:
            e = expiry[0]
            row["event_version"] = e["version_id"]
            if futures:
                if e["kind"] == "delivery":
                    delivery = delivery_record(data.get("deliveries", []), t + HOUR)
                    end_price, source = (
                        float(delivery["deliveryPrice"]),
                        "delivery_rest",
                    )
                    row["delivery_record_time"] = int(delivery["deliveryTime"])
                elif e["settlement_price"] is not None:
                    end_price, source = float(e["settlement_price"]), "announcement"
                elif t + HOUR in marks:
                    end_price, source = (
                        marks[t + HOUR]["open"],
                        "mark_open_approximation",
                    )
                    row["quality"] |= QUALITY["settlement_approximation"]
                else:
                    raise ValueError("下架结算缺官方价及边界标记价")
                row.update(settlement=end_price, settlement_source=source, expires=True)
            else:
                end_price = row["close"]
            ended = True
        if t + HOUR in schedule:
            try:
                row.update(funding_value(schedule[t + HOUR], prices, marks, unit))
            except LookupError:
                if t + HOUR == end:
                    return rows, dict(
                        truncated_at=t,
                        reason="final_funding_price_missing",
                        **({"halts": halt_notes} if halt_notes else {}),
                    )
                raise
            if row["funding_price_source"] == "mark_open_approximation":
                row["quality"] |= QUALITY["funding_approximation"]
        last_settlement = row["settlement"] if futures else row["close"]
        last_close = row["close"]
        rows.append(row)
    return rows, {"halts": halt_notes} if halt_notes else {}


def build_frame(profiles, data, events, start, end, *, allow_missing=False):
    start, end = millis(start), millis(end)
    if start % HOUR or end % HOUR or start >= end:
        raise ValueError("起止必须为递增的 UTC 整点")
    events = validate_events(events)
    rows, notes = [], {}
    if len({p["instrument"] for p in profiles}) != len(profiles):
        raise ValueError("重复交易对档案")
    for p in profiles:
        values, note = build_instrument(
            p, data[p["instrument"]], events, start, end, allow_missing=allow_missing
        )
        rows.extend(values)
        if note:
            notes[p["instrument"]] = note
    if not rows:
        raise ValueError("空面板")
    # 任一列最后缺价就同时截断整幅矩形，不制造不同列的时间轴。
    actual_end = min(
        [end, *(n["truncated_at"] for n in notes.values() if "truncated_at" in n)]
    )
    frame = pl.DataFrame(
        [r for r in rows if r["time"] < actual_end],
        infer_schema_length=None,
        schema_overrides=PANEL_SCHEMA,
    )
    return frame.sort(["time", "instrument"]), notes
