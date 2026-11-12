"""计算需要重建的最早日期；证券更名不触发历史矩阵重写。"""

import numpy as np
import polars as pl

from .catalog import DATASETS


def reuse_prefix(store, dates, securities, inputs, old_path, old, code):
    if old.get("code_version") != code or old["dates"] != list(
        dates[: len(old["dates"])]
    ):
        return 0
    if old.get("aliases", []) != getattr(store, "aliases", []):
        return 0
    symbols = securities["ts_code"].to_list()
    if not set(old["symbols"]) <= set(symbols):
        return 0
    prefix = len(old["dates"])
    for key in set(inputs) | set(old["inputs"]):
        if inputs.get(key) == old["inputs"].get(key):
            continue
        name, part = key.split("/", 1)
        kind = DATASETS[name].partition
        if kind == "day":
            prefix = min(prefix, int(np.searchsorted(dates, part)))
        elif name == "stock_basic":
            previous = {
                r["ts_code"]: r
                for r in pl.read_parquet(old_path / "securities.parquet").iter_rows(
                    named=True
                )
            }
            for row in securities.iter_rows(named=True):
                before = previous.get(row["ts_code"])
                if before is None:
                    prefix = min(prefix, int(np.searchsorted(dates, row["list_date"])))
                else:
                    for field in ("list_date", "delist_date"):
                        if before.get(field) != row.get(field):
                            for value in (before.get(field), row.get(field)):
                                if value:
                                    prefix = min(
                                        prefix, int(np.searchsorted(dates, value))
                                    )
        elif name == "index_weight":
            month = part.rsplit("_", 1)[-1]
            # 只有「代码_年月」能定位受影响的日期；其他布局整体重建。
            if len(month) != 6 or not month.isdigit():
                return 0
            prefix = min(prefix, int(np.searchsorted(dates, month + "01")))
        elif name == "index_member":
            return 0
        elif name == "index_member_all":
            old_files = old.get("event_files", {}).get(name, [])
            if not old_files:
                return 0
            before = pl.concat(
                [pl.read_parquet(store.root / p) for p in old_files],
                how="diagonal_relaxed",
            )
            after = store.read(name)
            columns = [
                c
                for c in ("ts_code", "l1_code", "in_date", "out_date")
                if c in before.columns
            ]
            # 整列为空的分区会被存成 Null 类型；统一按字符串比较。
            text = [pl.col(c).cast(pl.String) for c in columns]
            a = before.select(text).unique()
            b = after.select(text).unique()
            changed = pl.concat(
                [
                    a.join(b, on=columns, how="anti", nulls_equal=True),
                    b.join(a, on=columns, how="anti", nulls_equal=True),
                ]
            )
            earliest = changed["in_date"].drop_nulls().min()
            if earliest:
                prefix = min(prefix, int(np.searchsorted(dates, earliest)))
        elif name == "trade_cal":
            before = (
                pl.read_parquet(old_path / "calendar.parquet")
                .filter(pl.col("is_open") == 1)["cal_date"]
                .sort()
                .to_list()
            )
            after = (
                store.read(name)
                .filter(pl.col("is_open") == 1)["cal_date"]
                .sort()
                .to_list()
            )
            if [d for d in before if d <= old["dates"][-1]] != [
                d for d in after if d <= old["dates"][-1]
            ]:
                return 0
        else:
            return 0
    return prefix
