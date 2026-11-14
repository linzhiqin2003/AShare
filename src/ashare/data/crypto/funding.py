"""资金费的记录连续性和每张现金流；不存在结算与缺少结算严格区分。"""

import math

from .timeutil import HOUR, funding_hour, millis

OFFICIAL_PRICE_FROM = millis("2023-10-31T08:00:00Z")


def funding_schedule(records, start, end, listed_at, *, stop=None):
    """验证 (start, end] 中应有的结算；必须传入前后边界记录。

    间隔转换以新记录的字段为准，新的第一个结算取该间隔的下个整点。
    缺第一/最后一笔也报错，不能只核对相邻的现存记录。
    """
    records = sorted(records, key=lambda r: r["time"])
    times = [r["time"] for r in records]
    if len(set(times)) != len(times):
        raise ValueError("资金费记录重复")
    active_start = max(start, listed_at)
    active_end = min(end, stop) if stop is not None else end
    if active_start >= active_end:
        return {}
    if not records:
        raise ValueError("缺少资金费记录，不能按 0")
    for r in records:
        if (
            r["interval"] not in (1, 2, 4, 8)
            or r["time"] % (r["interval"] * HOUR)
            or not math.isfinite(r["rate"])
        ):
            raise ValueError("资金费间隔/整点/费率错误")
    for a, b in zip(records, records[1:], strict=False):
        step = b["interval"] * HOUR
        expected = (a["time"] // step + 1) * step
        if (
            a["time"] < active_end
            and b["time"] > active_start
            and expected != b["time"]
        ):
            raise ValueError(f"缺少资金费记录: {expected}")
    first, last = records[0], records[-1]
    step = first["interval"] * HOUR
    expected_first = (active_start // step + 1) * step
    if first["time"] > expected_first and expected_first <= active_end:
        raise ValueError(f"缺少首笔资金费: {expected_first}")
    step = last["interval"] * HOUR
    expected_last = (last["time"] // step + 1) * step
    if expected_last <= active_end:
        raise ValueError(f"缺少末笔资金费: {expected_last}")
    return {r["time"]: r for r in records if active_start < r["time"] <= active_end}


def price_map(records):
    result = {}
    for r in records:
        t = funding_hour(r["fundingTime"])
        if t in result:
            raise ValueError("REST 资金费整点重复")
        price = float(r["markPrice"]) if r.get("markPrice") else None
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError("资金费关联价格非法")
        result[t] = dict(
            price=price, rate=float(r["fundingRate"]), raw_time=int(r["fundingTime"])
        )
    return result


def funding_value(record, prices, marks, multiplier):
    t = record["time"]
    official = prices.get(t)
    if official and not math.isclose(
        official["rate"], record["rate"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("归档与 REST 资金费率不一致")
    if official and official["price"] is not None:
        price, source = official["price"], "rest"
    elif t < OFFICIAL_PRICE_FROM and t in marks:
        price, source = marks[t]["open"], "mark_open_approximation"
    else:
        raise LookupError(f"资金费有结算但缺价格: {t}")
    return dict(
        funding=record["rate"] * price * multiplier,
        funding_rate=record["rate"],
        funding_price=price,
        funding_interval_hours=record["interval"],
        funding_price_source=source,
    )
