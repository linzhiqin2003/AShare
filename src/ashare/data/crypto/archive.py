"""归档路径与月度→日度→最近 REST 的分区计划。"""

from .formats import parse_funding, parse_klines, rest_klines
from .storage import encoded
from .timeutil import DAY, HOUR, iso, millis, month_end, months


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


def download_series(
    client, store, market, dataset, symbol, start, end, *, now, refresh=False
):
    """返回所用的原始对象键。月文件存在则不混入日文件或 REST。"""
    keys = []
    for label in months(start, end):
        key = archive_key(market, dataset, symbol, label)
        if client.archive(store, key, refresh=refresh) is not None:
            keys.append(key)
            continue
        if dataset == "fundingRate":
            continue  # 没有 daily；不能根据 REST 相邻间距猜 interval。
        lo, hi = max(start, millis(label + "-01")), min(end, month_end(label))
        for day in range(lo // DAY * DAY, hi, DAY):
            daily = archive_key(
                market, dataset, symbol, iso(day)[:10], frequency="daily"
            )
            if client.archive(store, daily, refresh=refresh) is not None:
                keys.append(daily)
            elif day >= now - 7 * DAY:
                rkey = f"rest/{market}/{dataset}/{symbol}/{day}"
                if refresh or not store.complete(rkey):
                    rows = client.klines(
                        market,
                        dataset,
                        symbol,
                        max(lo, day),
                        min(hi, day + DAY, now // HOUR * HOUR),
                    )
                    store.put(
                        rkey,
                        encoded(rows),
                        source="rest",
                        market=market,
                        dataset=dataset,
                        symbol=symbol,
                    )
                keys.append(rkey)
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
