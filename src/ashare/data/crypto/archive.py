"""归档路径与月度优先、仅用日度补缺 的分区计划。"""

from .formats import parse_funding, parse_klines, rest_klines
from .timeutil import DAY, iso, millis, month_end, months


def archive_prefix(market, dataset, symbol=None, *, frequency="monthly"):
    base = "data/spot" if market == "spot" else "data/futures/um"
    prefix = f"{base}/{frequency}/{dataset}/"
    if symbol is not None:
        prefix += symbol + "/"
        if dataset != "fundingRate":
            prefix += "1h/"
    return prefix


def archive_key(market, dataset, symbol, label, *, frequency="monthly"):
    middle = "fundingRate" if dataset == "fundingRate" else "1h"
    return (
        archive_prefix(market, dataset, symbol, frequency=frequency)
        + f"{symbol}-{middle}-{label}.zip"
    )


def funding_coverage(key):
    """月归档声明的完整区间，不因 CSV 尾部缺笔缩小。"""
    if "/monthly/fundingRate/" not in key or not key.endswith(".zip"):
        raise ValueError("资金费覆盖必须来自月归档")
    label = key.removesuffix(".zip")[-7:]
    return dict(key=key, start=millis(label + "-01"), end=month_end(label))


def download_series(
    client, store, market, dataset, symbol, start, end, *, now, refresh=False
):
    """月文件已有任何行的日子不混入日文件；只补完整缺失日，不补 REST 尾部。"""
    keys = []
    for label in months(start, end):
        key = archive_key(market, dataset, symbol, label)
        payload = client.archive(store, key, refresh=refresh)
        present_days = set()
        if payload is not None:
            keys.append(key)
            if dataset != "fundingRate":
                monthly = parse_klines(
                    payload, market, start=millis(label + "-01"), end=month_end(label)
                )
                present_days = {r["time"] // DAY for r in monthly}
        if dataset == "fundingRate":
            continue
        lo = max(start, millis(label + "-01"))
        hi = min(end, month_end(label), now // DAY * DAY)
        for day in range(lo // DAY * DAY, hi, DAY):
            if day // DAY in present_days:
                continue
            daily = archive_key(
                market, dataset, symbol, iso(day)[:10], frequency="daily"
            )
            if client.archive(store, daily, refresh=refresh) is not None:
                keys.append(daily)
    return keys


def read_series(store, keys, market, dataset):
    rows = {}
    for key in keys:
        payload = store.get(key)
        if key.startswith("rest/"):
            import json

            values = rest_klines(json.loads(payload), market)
            source = "rest"
        else:
            label = key.rsplit("/", 1)[-1].removesuffix(".zip").split("-", 2)[-1]
            # 合约名不含 '-'；fundingRate 与 1h 路径使用同一文件日期规则。
            lo = millis(label + "-01") if len(label) == 7 else millis(label)
            hi = month_end(label) if len(label) == 7 else lo + DAY
            values = (
                parse_funding(payload, start=lo, end=hi)
                if dataset == "fundingRate"
                else parse_klines(payload, market, start=lo, end=hi)
            )
            source = "archive"
        for value in values:
            row = dict(value, source=source, source_key=key)
            if row["time"] in rows and rows[row["time"]] != row:
                raise ValueError("归档分区重叠；不能静默择一")
            rows[row["time"]] = row
    return sorted(rows.values(), key=lambda r: r["time"])
