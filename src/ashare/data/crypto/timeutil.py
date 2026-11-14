"""时间统一为 UTC 毫秒；区间一律左闭右开。"""

from datetime import UTC, datetime

HOUR = 3_600_000
DAY = 24 * HOUR


def millis(value):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def iso(value):
    return datetime.fromtimestamp(value / 1000, UTC).isoformat()


def month(value):
    return iso(value)[:7]


def months(start, end):
    current = datetime.fromtimestamp(millis(start) / 1000, UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    while millis(current) < millis(end):
        yield current.strftime("%Y-%m")
        current = current.replace(
            year=current.year + (current.month == 12), month=current.month % 12 + 1
        )


def month_end(label):
    d = datetime.fromisoformat(label + "-01").replace(tzinfo=UTC)
    return millis(d.replace(year=d.year + (d.month == 12), month=d.month % 12 + 1))


def funding_hour(value):
    value = int(value)
    rounded = ((value + HOUR // 2) // HOUR) * HOUR
    if abs(value - rounded) > 1000:
        raise ValueError(f"资金费时间偏离整点超过 1 秒: {value}")
    return rounded
