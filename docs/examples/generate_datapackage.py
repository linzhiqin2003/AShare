"""生成 3 只股票、40 个交易日的纯合成示例；不读取任何真实数据。"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ashare.data.spec import build


class SyntheticAdapter:
    def __init__(self):
        days = []
        day = date(2024, 1, 2)
        while len(days) < 40:
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        self.days = days
        self.codes = ["000001.SZ", "600001.SH", "600002.SH"]

    def describe(self):
        return {
            "specVersion": "1", "name": "synthetic-three-stocks", "title": "纯合成三股票示例",
            "source": "确定性公式生成，无真实行情", "licenses": [{"name": "CC0-1.0"}], "symbology": "cn",
            "calendar": {"name": "SYNTHETIC", "tz": "Asia/Shanghai", "sessionOpen": "09:30", "sessionClose": "15:00"},
            "resources": [
                {"name": "bars", "kind": "bars", "frequency": "1d", "schema": {"fields": [
                    {"name": "symbol", "type": "string"}, {"name": "date", "type": "date"},
                    *[{"name": k, "type": "number"} for k in ("open", "high", "low", "close", "volume")],
                ], "primaryKey": ["symbol", "date"]}},
                {"name": "instruments", "kind": "instruments", "schema": {"fields": [
                    {"name": k, "type": "string"} for k in ("symbol", "asset_type", "currency")
                ], "primaryKey": ["symbol"]}},
                {"name": "calendar", "kind": "calendar", "schema": {"fields": [{"name": "date", "type": "date"}], "primaryKey": ["date"]}},
            ],
        }

    def partitions(self):
        yield "bars", "2024"
        yield "instruments", "all"
        yield "calendar", "all"

    def read(self, table, partition):
        if table == "calendar":
            return pl.DataFrame({"date": self.days})
        if table == "instruments":
            return pl.DataFrame({"symbol": self.codes, "asset_type": ["stock"] * 3, "currency": ["CNY"] * 3})
        rows = []
        for j, symbol in enumerate(self.codes):
            for i, day in enumerate(self.days):
                price = 10 + j * 5 + i / 8
                rows.append(dict(symbol=symbol, date=day, open=price, high=price + 1, low=price - 1, close=price + 0.5, volume=float(1000 + i * 100)))
        return pl.DataFrame(rows)


if __name__ == "__main__":
    print(build(SyntheticAdapter(), Path(__file__).parent / "datapackage"))
