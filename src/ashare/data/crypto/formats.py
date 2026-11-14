"""归档格式解析。根据内容识别表头和现货单位，文件范围另行核对。"""

import csv
import io
import math
import zipfile

from .timeutil import HOUR, funding_hour, millis

KLINE_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
FUNDING_HEADER = ("calc_time", "funding_interval_hours", "last_funding_rate")
MIN_TIME, MAX_TIME = millis("2017-01-01"), millis("2100-01-01")


def csv_bytes(payload):
    if payload[:2] != b"PK":
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("归档必须恰有一个 CSV")
        info = archive.getinfo(names[0])
        if info.file_size > 128 * 1024 * 1024:
            raise ValueError("解压文件超过 128 MiB 安全上限")
        return archive.read(names[0])


def records(payload, header):
    rows = list(csv.reader(io.StringIO(csv_bytes(payload).decode("utf-8-sig"))))
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("空归档")
    if not rows[0][0].isdigit():
        if tuple(x.strip() for x in rows.pop(0)) != header:
            raise ValueError("未知 CSV 表头")
    if not rows or any(len(r) != len(header) for r in rows):
        raise ValueError("CSV 列数错误或没有数据行")
    return rows


def parse_klines(payload, market, *, start=None, end=None):
    if market not in ("spot", "usdm"):
        raise ValueError("未知市场")
    rows = records(payload, KLINE_HEADER)
    # 不用年份推断：现货允许 ms/us，U 本位仅接受已核实的 ms。
    units = {1000 if int(r[0]) >= 100_000_000_000_000 else 1 for r in rows}
    if len(units) != 1 or (market == "usdm" and units != {1}):
        raise ValueError("时间戳单位混杂或 U 本位不是毫秒")
    scale = units.pop()
    out = []
    for r in rows:
        t = int(r[0]) // scale
        close_t = int(r[6]) // scale
        if not MIN_TIME <= t < MAX_TIME or t % HOUR or close_t > t + HOUR - 1 + 1000:
            raise ValueError(f"小时 K 线时间范围/边界错误: {r[0]}")
        time_quality = (
            "invalid_close_time"
            if close_t < t
            else "partial_halt"
            if close_t < t + HOUR - 1 - 1000
            else "normal"
        )
        if (start is not None and t < millis(start)) or (
            end is not None and t >= millis(end)
        ):
            raise ValueError("归档时间不在声明的文件范围内")
        values = {
            name: float(r[i])
            for i, name in enumerate(KLINE_HEADER)
            if i in (1, 2, 3, 4, 5, 7, 9, 10)
        }
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("K 线含非有限数值")
        out.append(
            dict(
                time=t,
                close_time=close_t,
                time_quality=time_quality,
                **values,
                count=int(r[8]),
            )
        )
    if any(a["time"] >= b["time"] for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("K 线重复或乱序")
    return out


def parse_funding(payload, *, start=None, end=None):
    out = []
    for r in records(payload, FUNDING_HEADER):
        raw = int(r[0])
        t = funding_hour(raw)
        interval = int(r[1])
        rate = float(r[2])
        if not MIN_TIME <= t < MAX_TIME or interval not in (1, 2, 4, 8):
            raise ValueError("资金费时间或间隔非法")
        if t % (interval * HOUR) or not math.isfinite(rate):
            raise ValueError("资金费不在当时间隔的整点上")
        if (start is not None and t < millis(start)) or (
            end is not None and t >= millis(end)
        ):
            raise ValueError("资金费不在声明的文件范围内")
        out.append(dict(time=t, raw_time=raw, interval=interval, rate=rate))
    if any(a["time"] >= b["time"] for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("资金费重复或乱序")
    return out


def rest_klines(rows, market):
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return parse_klines(buf.getvalue().encode(), market)
