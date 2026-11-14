"""已知现货转换映射：固定旧单位接续新报价，不虚构现金或成交。"""

import numpy as np

from ashare.data.crypto.events import asof
from ashare.data.crypto.timeutil import HOUR


def splice(ds, names, times, arrays, profiles, events):
    notes = []
    execution = np.tile(np.asarray(names, dtype=object), (len(times), 1))
    all_profiles = {p["instrument"]: p for p in profiles}
    cache = {}
    for j, origin in enumerate(names):
        current = origin
        ratio = 1.0
        applied = set()
        for i, t in enumerate(times):
            for e in asof(events, int(t), instrument=current):
                if (
                    e["kind"] != "migration"
                    or e["effective_at"] > t
                    or e["version_id"] in applied
                ):
                    continue
                matches = [
                    p
                    for p in profiles
                    if p["market"] == "spot"
                    and e["replacement"] in (p["instrument"], p["symbol"], p["coin"])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{current} 转换缺少唯一的新币现货档案: {e['replacement']}"
                    )
                dest = matches[0]["instrument"]
                if dest in names or dest == origin:
                    raise ValueError("转换链和新币不能同时作为两个独立交易列")
                if dest in applied:
                    raise ValueError("币转换链出现循环")
                if dest not in cache:
                    f = ds.read(
                        start=int(times[0]),
                        end=int(times[-1] + HOUR),
                        instruments=[dest],
                    ).sort("time")
                    if not np.array_equal(f["time"].to_numpy(), times):
                        raise ValueError(f"{dest} 转换后缺完整小时数据")
                    cache[dest] = f
                ratio *= e["ratio"]
                old_rule = min(
                    all_profiles[origin]["rules"], key=lambda r: r["known_at"]
                )
                new_rule = min(matches[0]["rules"], key=lambda r: r["known_at"])
                grid = old_rule["qty_step"] * ratio / new_rule["qty_step"]
                if (
                    not np.isclose(grid, round(grid), rtol=0, atol=1e-10)
                    or old_rule["min_qty"] * ratio < new_rule["min_qty"]
                ):
                    raise ValueError(
                        "转换后数量格点不兼容固定现货快照，需要另建计量单位"
                    )
                applied.add(e["version_id"])
                applied.add(current)
                notes.append(
                    dict(
                        symbol=origin,
                        from_symbol=current,
                        to_symbol=dest,
                        time=int(t),
                        known_at=e["known_at"],
                        ratio=ratio,
                        source_url=e["source_url"],
                        quantity_unit="original_quote_unit",
                    )
                )
                current = dest
            if current != origin:
                execution[i, j] = current
                row = cache[current].row(i, named=True)
                for field in arrays:
                    if field in row:
                        arrays[field][i, j] = row[field]
                for field in ("open", "high", "low", "close"):
                    arrays[field][i, j] *= ratio
                for field in ("volume", "volume_engine", "qty_step", "min_qty"):
                    arrays[field][i, j] /= ratio
                arrays["coin_units"][i, j] = all_profiles[origin]["coin_units"] * ratio
    return notes, execution
