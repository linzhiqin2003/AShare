"""逐分区 lazy scan：冷读解析已核对字节，热读在交付前复查文件身份。"""

import os
from contextlib import contextmanager
from threading import Lock

import polars as pl
from polars.io.plugins import register_io_source


@contextmanager
def verified_source(path, entry, verify, identity):
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        payload = verify(path, entry, stream)
        try:
            # 冷读只使用同一份不可变字节；命中缓存才读取已核对的文件句柄。
            yield payload if payload is not None else stream, payload
        finally:
            if payload is None and identity(os.fstat(stream.fileno())) != before:
                raise ValueError(f"文件在读取中改变：{path.name}")


def verified_scan(entries, empty_schema, verify, identity):
    if not entries:
        return pl.DataFrame(schema=empty_schema).lazy()
    path, entry = entries[0]
    with verified_source(path, entry, verify, identity) as (source, payload):
        schema = pl.scan_parquet(source).collect_schema()
    # 只预取首分区的冷读快照以获得真实物理 schema（包括 timestamp 精度）。
    # 不把整包字节放进 lazy concat；快照消费后释放，后续分区逐个打开。
    prefetched = [payload]
    lock = Lock()

    def project(source, columns, predicate, limit):
        frame = pl.scan_parquet(source)
        if predicate is not None:
            frame = frame.filter(predicate)
        if columns is not None:
            frame = frame.select(columns)
        if limit is not None:
            frame = frame.head(limit)
        return frame.collect(engine="streaming")

    def source_batches(with_columns, predicate, n_rows, batch_size):
        remaining = n_rows
        for index, (path, entry) in enumerate(entries):
            if remaining == 0:
                break
            with lock:
                payload = prefetched.pop() if index == 0 and prefetched else None
            if payload is not None:
                frame = project(payload, with_columns, predicate, remaining)
            else:
                with verified_source(path, entry, verify, identity) as (source, _):
                    frame = project(source, with_columns, predicate, remaining)
                # 此处已完成读取和身份复核，才允许向调用者交付数据。
                del source, _
            del payload
            if remaining is not None:
                remaining -= frame.height
            yield from frame.iter_slices(batch_size or 65536)
            del frame

    return register_io_source(
        source_batches,
        schema=schema,
        validate_schema=True,
        is_pure=False,
        explain_name="Verified Parquet",
    )
