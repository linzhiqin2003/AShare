"""当前面板代内的小型事件长表缓存；按原始分区哈希增补，不留下跨代缓存。"""

from concurrent.futures import ThreadPoolExecutor

import polars as pl

from .aliases import canonical


def event_table(store, name, staging, old_path, old, snapshots, start=None, end=None):
    entries = {
        key: entry
        for key, entry in store.entries(name).items()
        if entry["status"] == "complete"
        and (start is None or entry["partition"] >= start)
        and (end is None or entry["partition"] <= end)
    }
    hashes = {key: entry["content_hash"] for key, entry in entries.items()}
    previous = (old or {}).get("event_cache", {}).get(name, {})
    filename = f"_events_{name}.parquet"
    old_file = old_path / filename if old_path else None
    reusable = (
        {key for key, digest in hashes.items() if previous.get(key) == digest}
        if old_file and old_file.exists()
        else set()
    )
    frames = []
    if reusable:
        frames.append(
            pl.read_parquet(old_file).filter(
                pl.col("_partition_key").is_in(sorted(reusable))
            )
        )

    def read_partition(item):
        key, entry = item
        return pl.read_parquet(store.root / entry["file"]).with_columns(
            pl.lit(key).alias("_partition_key")
        )

    # 数千个带常量列的 LazyFrame 会形成昂贵的优化树；小事件文件并行读后合并。
    pending = [
        (key, entry)
        for key, entry in entries.items()
        if key not in reusable and entry["rows"]
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames.extend(pool.map(read_partition, pending))
    if not frames:
        # 全空分区仍保留源字段及类型，不能把空表当成不存在的数据集。
        entry = next(iter(entries.values()))
        frames = [
            pl.read_parquet(store.root / entry["file"])
            .head(0)
            .with_columns(pl.lit("").alias("_partition_key"))
        ]
    table = pl.concat(frames, how="diagonal_relaxed")
    table.write_parquet(staging / filename)
    snapshots[name] = hashes
    return canonical(table.drop("_partition_key"), name, store.aliases)
