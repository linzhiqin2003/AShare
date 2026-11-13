"""实际多空持仓：按调仓日存稀疏快照，单日查询只解压相邻两次快照。"""

from functools import cached_property
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def mean_days(snapshot, stop):
    """只计窗口内观察到的连续同侧区间，窗口两端不外推。

    多头、空头各自：总持仓天数 ÷ 新进次数（上一快照同侧没有这只即算新进）。
    整数运算，向量化；与逐快照集合运算的写法结果完全相同（测试有对照）。
    """
    days = np.asarray(snapshot["days"], dtype=np.int64)
    n = len(days)
    if n == 0:
        return None, None
    offsets = np.asarray(snapshot["offsets"], dtype=np.int64)
    columns = np.asarray(snapshot["columns"], dtype=np.int64)
    weights = np.asarray(snapshot["weights"])
    duration = np.append(days[1:], int(stop)) - days
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
    out = []
    width = int(columns.max()) + 1 if len(columns) else 1
    for mask in (weights > 0, weights < 0):
        # 每个 (快照, 股票) 一个键。快照按日期、股票按编号存放时键本身已升序，不用排序；
        # 否则（或同一快照同侧有重复）退回 np.unique。
        keys = row[mask] * width + columns[mask]
        if len(keys) > 1 and not (keys[1:] > keys[:-1]).all():
            keys = np.unique(keys)
        if not len(keys):
            out.append(None)
            continue
        snapshot_of = keys // width
        total = int(duration[snapshot_of].sum())
        previous = keys - width  # 同一股票在上一个快照的键
        hit = np.searchsorted(keys, previous)
        present = (snapshot_of > 0) & (keys[np.minimum(hit, len(keys) - 1)] == previous)
        count = len(keys) - int(present.sum())
        out.append(total / count if count else None)
    return tuple(out)


def archive_arrays(arrays):
    """一个快照一个 NPZ 成员；只保存实际持仓，因子值来自生成目标的信号日。"""
    result = {}
    dtype = np.dtype(
        [("column", "<u4"), ("weight", "<f8"), ("factor", "<f8"), ("industry", "<i4")]
    )
    for part in ("is", "validation"):
        prefix = f"{part}.holdings."
        snapshot = {
            key[len(prefix) :]: arrays.pop(key)
            for key in list(arrays)
            if key.startswith(prefix)
        }
        if not snapshot:
            continue
        result[f"{part}.dates"] = arrays[f"{part}.dates"]
        for key in ("dates", "signal_dates", "symbols"):
            result[f"{part}.snapshot_{key}"] = snapshot[key]
        offsets = snapshot["offsets"]
        result[f"{part}.nonempty"] = np.diff(offsets) > 0
        for index in range(len(snapshot["days"])):
            lo, hi = offsets[index : index + 2]
            rows = np.empty(int(hi - lo), dtype=dtype)
            for key, source in (
                ("column", "columns"),
                ("weight", "weights"),
                ("factor", "factor"),
                ("industry", "industry"),
            ):
                rows[key] = snapshot[source][lo:hi]
            result[f"{part}.{index}"] = rows
    return result


class Holdings:
    def __init__(self, store, data_root=None):
        self.store, self.data_root = store, data_root

    @cached_property
    def names(self):
        import ashare.data

        data = ashare.data.open(self.data_root)
        securities = dict(data.securities().select("symbol", "name").collect().rows())
        industry = data.table("index_classify").collect()
        industries = {
            int(str(code).split(".")[0]): name
            for code, name in industry.select("industry_code", "industry_name").rows()
            if str(code).split(".")[0].isdigit()
        }
        return securities, industries

    def read(self, run, date=None, *, compare=False, search=""):
        with self.store.connect() as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run,)).fetchone()
        if row is None or row["status"] != "complete":
            return None
        self.store.wait_holdings(run)
        path = self.store.root / "holdings" / f"{run}.npz"
        if not path.exists():
            return {"dates": [], "date": None, "items": [], "total": 0}
        with np.load(path, allow_pickle=False) as archive:
            available = []
            for part in ("is", "validation"):
                if f"{part}.snapshot_dates" not in archive:
                    continue
                snapshots = archive[f"{part}.snapshot_dates"]
                dates = archive[f"{part}.dates"]
                nonempty = archive[f"{part}.nonempty"]
                indices = np.searchsorted(snapshots, dates, side="right") - 1
                if len(nonempty):
                    keep = (indices >= 0) & nonempty[np.maximum(indices, 0)]
                    available.extend(
                        (str(d), part, int(i))
                        for d, i in zip(dates[keep], indices[keep], strict=True)
                    )
            if not available:
                return {"dates": [], "date": None, "items": [], "total": 0}
            if date is None:
                return {"dates": [d for d, _, _ in available], "date": available[-1][0]}
            match = next((item for item in available if item[0] == date), None)
            if match is None:
                raise ValueError("请选择窗口内有持仓的交易日")
            _, part, index = match
            symbols = archive[f"{part}.snapshot_symbols"]
            current = archive[f"{part}.{index}"]
            previous = archive[f"{part}.{index - 1}"] if index else current[:0]
            names, industries = self.names
            old = {
                (int(r["column"]), r["weight"] > 0): float(r["weight"])
                for r in previous
            }
            now = {(int(r["column"]), r["weight"] > 0) for r in current}
            rows = [
                (r, "new" if (int(r["column"]), r["weight"] > 0) not in old else "held")
                for r in current
            ]
            if compare:
                rows += [
                    (r, "exit")
                    for r in previous
                    if (int(r["column"]), r["weight"] > 0) not in now
                ]
            items = []
            for r, status in sorted(
                rows,
                key=lambda item: 0 if item[1] == "exit" else -abs(item[0]["weight"]),
            ):
                symbol = str(symbols[r["column"]])
                name = names.get(symbol, symbol)
                industry = industries.get(int(r["industry"]), str(r["industry"]))
                if search.lower() not in f"{symbol} {name} {industry}".lower():
                    continue
                items.append(
                    dict(
                        symbol=symbol,
                        name=name,
                        industry=industry,
                        factor=float(r["factor"]) if np.isfinite(r["factor"]) else None,
                        weight=0.0 if status == "exit" else float(r["weight"]),
                        side="long" if r["weight"] > 0 else "short",
                        status=status if compare else "held",
                    )
                )
                if compare:
                    items[-1]["previous_weight"] = old.get(
                        (int(r["column"]), r["weight"] > 0), 0.0
                    )
            return dict(
                date=date,
                snapshot=str(archive[f"{part}.snapshot_dates"][index]),
                signal_date=str(archive[f"{part}.snapshot_signal_dates"][index]),
                previous=str(archive[f"{part}.snapshot_dates"][index - 1])
                if index
                else None,
                items=items,
                total=len(items),
            )


def write_archive(path, arrays):
    """低压缩等级减少交互等待；数组仍保留 float64 原值。"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
