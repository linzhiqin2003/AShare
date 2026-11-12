"""接口响应标准化；成交量统一为股，金额和价格统一为万分之一元。"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

import numpy as np
from paperbroker import COLUMNS, QuoteArray, money

TZ = ZoneInfo("Asia/Shanghai")
RT_FIELDS = "ts_code,name,pre_close,open,high,low,close,vol,amount,num,trade_time,ask_price1,ask_volume1,bid_price1,bid_volume1"


def timestamp(value: str | datetime) -> int:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return int(dt.timestamp() * 1000)


def rows(response: dict) -> list[dict]:
    if response.get("code") != 0:
        raise ValueError(f"Tushare response error code={response.get('code')}")
    data = response.get("data")
    if not data:
        return []
    return [dict(zip(data["fields"], item, strict=True)) for item in data["items"]]


def finite(value: object, default: float = 0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normalize(
    records: list[dict],
    symbols: list[str],
    *,
    limits: dict[str, tuple[float, float]] | None = None,
    suspended: set[str] | None = None,
    auction_kind: int = 0,
    day: str | None = None,
    suspension_intervals: dict[str, list[tuple[str, str]]] | None = None,
) -> QuoteArray:
    ids = {symbol: i for i, symbol in enumerate(symbols)}
    result = []
    for r in records:
        symbol = r.get("ts_code")
        if symbol not in ids:
            continue
        last = finite(r.get("close", r.get("price")))
        if last <= 0 or not r.get("trade_time"):
            continue
        try:
            stamp = timestamp(r["trade_time"])
            local = datetime.fromtimestamp(stamp / 1000, TZ)
        except (ValueError, TypeError, OverflowError):
            continue
        if day and local.strftime("%Y%m%d") != day:
            continue
        upper, lower = (limits or {}).get(symbol, (0, 0))
        unavailable = limits is not None and symbol not in limits
        suspended_now = symbol in (suspended or set()) or any(
            start <= local.strftime("%H:%M:%S") < end
            for start, end in (suspension_intervals or {}).get(symbol, [])
        )

        # -1 表示接口没有提供盘口，0 表示已知该侧无报价。
        def book(key: str, r: dict = r) -> int:
            return -1 if r.get(key) is None else money(finite(r[key]))

        result.append(
            [
                ids[symbol],
                stamp,
                money(last),
                money(finite(r.get("open"), last)),
                money(finite(r.get("high"), last)),
                money(finite(r.get("low"), last)),
                money(finite(r.get("pre_close"), last)),
                shares(r.get("vol")),
                money(finite(r.get("amount"))),
                book("ask_price1"),
                int(finite(r.get("ask_volume1"))),
                book("bid_price1"),
                int(finite(r.get("bid_volume1"))),
                money(upper),
                money(lower),
                int(unavailable or suspended_now),
                shares(r.get("auction_volume")),
                auction_kind,
            ]
        )
    return np.asarray(result, dtype=np.int64).reshape(-1, len(COLUMNS))


def corporate_actions(
    records: list[dict], symbols: list[str], ex_date: str
) -> list[dict]:
    """只采用实施且除权日匹配的记录；按税前现金额建模，税务处理留给 P3。"""
    from fractions import Fraction

    combined: dict[str, tuple[Decimal, Fraction]] = {}
    seen = set()
    for r in records:
        if (
            r.get("ex_date") != ex_date
            or r.get("div_proc") != "实施"
            or r["ts_code"] not in symbols
        ):
            continue
        key = (r["ts_code"], r.get("end_date"), ex_date)
        if key in seen:
            continue
        seen.add(key)
        cash, stock = combined.get(r["ts_code"], (Decimal(0), Fraction(0)))
        combined[r["ts_code"]] = (
            cash + Decimal(str(finite(r.get("cash_div_tax")))),
            stock + Fraction(str(finite(r.get("stk_div")))),
        )
    result = []
    for symbol, (cash, stock) in sorted(combined.items()):
        ratio = (1 + stock).limit_denominator(1000)
        result.append(
            dict(
                symbol=symbols.index(symbol),
                cash_per_share=money(cash),
                split_num=ratio.numerator,
                split_den=ratio.denominator,
                reference_price=None,
            )
        )
    return result


def shares(value: object, multiplier: int = 1) -> int:
    """十进制四舍五入到股，避免浮点手数转换少一股。"""
    return int(
        (Decimal(str(finite(value))) * multiplier).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )


def suspension_rules(
    records: list[dict],
) -> tuple[set[str], dict[str, list[tuple[str, str]]]]:
    """全天停牌与有明确起止时点的临停分开；无复牌时点持续到收盘。"""
    full: set[str] = set()
    intervals: dict[str, list[tuple[str, str]]] = {}
    for r in records:
        if r.get("suspend_type") != "S":
            continue
        value = str(r.get("suspend_timing") or "").strip()
        symbol = r["ts_code"]
        if not value or value in ("全天", "-", "None"):
            full.add(symbol)
            continue
        parts = value.replace("～", "-").replace("~", "-").split("-")
        try:
            from datetime import time

            begin = time.fromisoformat(parts[0].strip()).isoformat()
            end = (
                time.fromisoformat(parts[1].strip()).isoformat()
                if len(parts) > 1 and parts[1].strip()
                else "15:00:00"
            )
        except ValueError:
            # 未识别的停牌信息不得放行交易，但不影响其他证券。
            full.add(symbol)
            continue
        intervals.setdefault(symbol, []).append((begin, end))
    return full, intervals
