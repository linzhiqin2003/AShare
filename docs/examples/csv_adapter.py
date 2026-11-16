"""将本地 CSV 整理成标准包。CSV 单位须已是股/元，日期写 YYYY-MM-DD。

输入目录：bars/YYYY.csv、instruments/all.csv、calendar/all.csv。
每表列与下述描述对应；symbol 必须按字符串读取以保留前导零。
执行：python docs/examples/csv_adapter.py <csv目录> <输出包目录>
"""

import argparse
from pathlib import Path

import polars as pl

from ashare.data.spec import build
from generate_datapackage import SyntheticAdapter


class CSVAdapter:
    def __init__(self, root):
        self.root = Path(root)

    def describe(self):
        result = SyntheticAdapter().describe()
        result.update(name="csv-stocks", title="本地 CSV 股票", source="用户 CSV，数量股、金额元", licenses=[{"name": "proprietary"}])
        return result

    def partitions(self):
        for table in ("bars", "instruments", "calendar"):
            for path in sorted((self.root / table).glob("*.csv")):
                yield table, path.stem

    def read(self, table, partition):
        types = {"symbol": pl.String, "asset_type": pl.String, "currency": pl.String}
        if table == "bars":
            types.update(dict.fromkeys(("open", "high", "low", "close", "volume"), pl.Float64))
        frame = pl.read_csv(self.root / table / f"{partition}.csv", schema_overrides=types, null_values=["NA", "null", ""], try_parse_dates=True)
        # NaN 只在适配器转换，校验器不替来源修复。
        return frame.with_columns(pl.col(pl.Float64).fill_nan(None)).sort(["date"] if table == "calendar" else ["symbol", "date"] if table == "bars" else ["symbol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    print(build(CSVAdapter(args.source), args.target))
