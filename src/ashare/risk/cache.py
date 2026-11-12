"""一代、无损、按需重建暴露的日模型缓存。

格式 5：每日快照用 level 1 deflate 在后台线程里写（`np.load` 可读），证券表每代只存一次（`.symbols.npy`），
快照里存自己的键和所配证券表的 SHA-256；旧格式目录在 `prepare` 时整体判为未命中并清空。

什么时候算落盘（P-004 起写入是异步的）：
- `save` 返回只表示已登记、已排队；同一对象的 `load` 会先等自己排队中的写入。
- `flush()`、`close()`、`with ModelCache(...)` 退出、`model_series` 生成器结束（耗尽或显式 `close()`）
  返回时，此前所有 `save` 都已落盘；有写失败就在这里抛 `RuntimeError("风险模型缓存写入失败")`，原异常在 `__cause__`。
- 另一个进程要读刚写的目录，先在写的一方 `flush()`。
- 写失败之后：排队中的写入一律不再执行；该对象之后的 `save`/`load`/`flush`/`clear`/`stats` 都抛同一个错误
  （换一个新的 `ModelCache` 对象才能继续用这个目录）。生成器被垃圾回收关闭、解释器退出这类无法向调用方抛出的场合，
  错误写到 stderr，对象同样保持失败状态。
- fork 出的子进程不接手父进程排队中的写入（那些由父进程自己写完）；子进程里的写入由子进程新起的写线程负责。
"""

import argparse
import atexit
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import weakref
import zipfile
import zlib
from pathlib import Path
from uuid import uuid4

import numpy as np

from .model import RiskModel

FORMAT = 5
STATE_FIELDS = ("f", "residual", "base_f", "base_d", "bias", "estimation", "cap")
_HIDDEN = (
    ".latest.npy",
    ".latest.json",
    ".checkpoint.cache",
    ".generation.json",
    ".symbols.npy",
    ".prefix-index",
)
# 读快照时这些异常都意味着文件不完整或不是本缓存写的：当作未命中，重算后覆盖。
# zlib.error（压缩流损坏）、NotImplementedError（压缩方法字段损坏）、RuntimeError（加密标志位损坏）
# 都来自 zipfile；捕获它们的 try 块里只有读文件和解码，不含本模块的逻辑。
_UNREADABLE = (
    OSError,
    ValueError,
    KeyError,
    EOFError,
    zipfile.BadZipFile,
    zlib.error,
    NotImplementedError,
    RuntimeError,
)
_QUEUE_LIMIT = 64  # 排队的快照上限：每份约 0.3 MB（全市场），写不过来时 save 等待
_STALE_TEMPORARY = (
    600.0  # 秒；进程被杀留下的临时文件超过这个年龄才清理（别的进程可能正在写）
)
_LIVE = weakref.WeakSet()  # 本进程里起过写线程的缓存对象，解释器退出时统一等待


def _snapshot(path):
    return re.fullmatch(r"(?:[0-9]{6,}-)?[a-f0-9]{64}\.npz", path.name) is not None


def _temporary(path):
    return re.fullmatch(r"\.[a-f0-9]{32}\.tmp", path.name) is not None


def _owned(path):
    return path.is_file() and (
        _snapshot(path) or path.name in _HIDDEN or _temporary(path)
    )


def _symbols_digest(symbols):
    return hashlib.sha256(
        json.dumps(list(symbols), ensure_ascii=False).encode()
    ).hexdigest()


class _Writer:
    """单个后台线程按提交顺序写文件；第一次失败后不再执行排队中的写入，异常交回调用方线程。"""

    def __init__(self):
        self.pid = os.getpid()
        self.jobs = queue.Queue(maxsize=_QUEUE_LIMIT)
        self.error = None
        self.thread = threading.Thread(
            target=self._run, name="risk-cache-writer", daemon=True
        )
        self.thread.start()

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                if self.error is None:
                    job()
            except BaseException as error:  # noqa: BLE001 - 原样交回调用方线程
                self.error = error
            finally:
                self.jobs.task_done()

    def owned(self):
        """fork 出的子进程里，继承来的写线程并不存在，队列的锁也可能停在父进程的状态。"""
        return self.pid == os.getpid()

    def close(self):
        if self.owned() and self.thread.is_alive():
            self.jobs.put(None)
            self.thread.join()


def _savez(stream, arrays, level=1):
    """与 np.savez 同格式（np.load 可读），在后台线程里用最快一级 deflate 压缩。

    zlib 压缩释放 GIL，不占计算线程。全历史实测：不压缩 1.07 GB（旧格式的 1.73 倍），
    level 1 为 0.55 GB（旧格式 0.62 GB，证券表不再每天重复）。
    """
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level
    ) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, value, allow_pickle=False)


def _write_atomic(directory, path, write):
    temporary = directory / f".{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ModelCache:
    """历史保存 F/D 和递推状态，B 只保留最新一天；历史 B 由调用方按需重建。

    写入在后台线程里进行，落盘时点和写失败的规则见模块说明；也可以当上下文管理器用，
    退出时 `close()`（等待落盘，有写失败就抛出）。
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        for path in self.directory.glob("*.npz"):
            parts = path.stem.split("-", 1)
            if _snapshot(path) and len(parts) == 2 and parts[0].isdigit():
                self.entries[int(parts[0])] = parts[1]
        marker = self.directory / ".generation.json"
        self.generation = None
        if marker.exists():
            try:
                content = json.loads(marker.read_text())
                if content.get("format") == FORMAT:
                    self.generation = content["generation"]
            except (ValueError, KeyError, AttributeError):
                self.generation = None
        self._symbols = None
        self._symbols_sha = None
        self._writer = None
        self._pending = set()
        self._failure = None
        self._raised = False  # 写失败是否已经抛给过调用方

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # 后台写入 -------------------------------------------------------------

    def _check(self):
        """写失败后对象保持失败状态：之后的 save/load/flush 都抛同一个错误。"""
        writer = self._writer
        if writer is not None and not writer.owned():
            # fork 出的子进程：父进程排队中的写入归父进程，这里丢掉继承来的写线程和待写集合。
            self._writer, self._pending = None, set()
        elif writer is not None and writer.error is not None and self._failure is None:
            self._failure = writer.error
        if self._failure is not None:
            self._raised = True
            raise RuntimeError("风险模型缓存写入失败") from self._failure

    def _submit(self, path, job):
        self._check()
        if self._writer is None:
            self._writer = _Writer()
            weakref.finalize(self, self._writer.close)
            _LIVE.add(self)
        self._pending.add(path)

        def run():
            job()
            self._pending.discard(path)

        self._writer.jobs.put(run)

    def flush(self):
        """等待后台写入全部落盘；有写失败（包括以前发生过的）则抛出 RuntimeError。"""
        self._check()
        if self._writer is not None:
            self._writer.jobs.join()
            self._check()
            self._pending.clear()

    def close(self):
        """等待落盘并停掉写线程；之后再 save 会重新起一个。"""
        try:
            self.flush()
        finally:
            writer, self._writer = self._writer, None
            if writer is not None:
                writer.close()

    def _wait_for(self, path):
        if path in self._pending:
            self.flush()

    # 代际 -----------------------------------------------------------------

    def _json(self, name, value):
        _write_atomic(
            self.directory,
            self.directory / name,
            lambda stream: stream.write(json.dumps(value, ensure_ascii=False).encode()),
        )

    def prepare(self, generation, symbols=None):
        """symbols 给出时（`model_series` 总会给）证券表以它为准，坏了就重写。"""
        if self.generation != generation:
            self.clear()
            self.generation = generation
            self._json(
                ".generation.json",
                {"generation": generation, "schema": 3, "format": FORMAT},
            )
        if symbols is not None:
            self._symbol_table(symbols)
        self._remove_stale_temporaries()

    def _remove_stale_temporaries(self):
        """进程被杀时 `_write_atomic` 来不及删的临时文件；太新的可能是别的进程正在写，留着。"""
        now = time.time()
        for path in self.directory.iterdir():
            if _temporary(path):
                try:
                    if now - path.stat().st_mtime > _STALE_TEMPORARY:
                        path.unlink()
                except FileNotFoundError:
                    pass

    def clear(self):
        """删除本目录内缓存文件；不递归删除调用方的其他目录。"""
        self.flush()
        removed = 0
        if self.directory.is_dir():
            for path in self.directory.iterdir():
                if _owned(path):
                    removed += path.stat().st_size
                    path.unlink()
        self.entries.clear()
        self.generation = None
        self._symbols = self._symbols_sha = None
        return removed

    def stats(self):
        self.flush()
        files = (
            [p for p in self.directory.iterdir() if _owned(p)]
            if self.directory.is_dir()
            else []
        )
        latest = self.directory / ".checkpoint.cache"
        return {
            "snapshots": len(self.entries),
            "bytes": sum(p.stat().st_size for p in files),
            "latest_exposure_bytes": latest.stat().st_size if latest.exists() else 0,
            "generations": int(self.generation is not None),
        }

    # 证券表 ---------------------------------------------------------------

    def _read_symbols(self):
        """读证券表；读不出时为 None（不当作快照损坏）。"""
        try:
            return tuple(
                np.load(self.directory / ".symbols.npy", allow_pickle=False).tolist()
            )
        except (*_UNREADABLE, FileNotFoundError):
            return None

    def _symbol_table(self, symbols):
        """每代只写一次；内容不同（证券顺序变了或文件被改坏）时整体重写。"""
        if self._symbols is None:
            self._symbols = self._read_symbols()
        if self._symbols != tuple(symbols):
            self._symbols = tuple(symbols)
            array = np.asarray(self._symbols)
            _write_atomic(
                self.directory,
                self.directory / ".symbols.npy",
                lambda stream: np.save(stream, array, allow_pickle=False),
            )
        self._symbols_sha = _symbols_digest(self._symbols)

    def _stored_symbols(self):
        """证券表和它的 SHA-256；读不出时为 None。"""
        if self._symbols is None:
            self._symbols = self._read_symbols()
            if self._symbols is None:
                return None
        if self._symbols_sha is None:
            self._symbols_sha = _symbols_digest(self._symbols)
        return self._symbols, self._symbols_sha

    # 读写 -----------------------------------------------------------------

    def _discard(self, day, path):
        path.unlink(missing_ok=True)
        if self.entries.get(day) == path.stem.split("-", 1)[-1]:
            del self.entries[day]

    def load(
        self, key, *, day=None, with_model=True, exposure_loader=None, fields=None
    ):
        """fields：只读这些状态字段（默认全部）；只读历史窗口的日子用得上。

        参数错误（fields 不是 STATE_FIELDS 的非空子集）在读任何文件之前报 ValueError，不动缓存。
        快照损坏、被换掉或与证券表对不上时返回 None（未命中）；证券表本身坏了不删快照。
        """
        if fields is not None and (
            isinstance(fields, str)
            or not fields
            or not all(isinstance(name, str) for name in fields)
            or not set(fields) <= set(STATE_FIELDS)
        ):
            raise ValueError(f"fields: 需要 {STATE_FIELDS} 的非空子集")
        self._check()
        if day is None:
            day = next((i for i, value in self.entries.items() if value == key), None)
        if day is None:
            return None
        if day in self.entries and self.entries[day] != key:
            # 历史修订使其后所有前缀失效；立即清理旧尾部，不留下第二代。
            self.flush()
            for i in [i for i in self.entries if i >= day]:
                (self.directory / f"{i:06d}-{self.entries.pop(i)}.npz").unlink(
                    missing_ok=True
                )
            (self.directory / ".latest.npy").unlink(missing_ok=True)
            (self.directory / ".latest.json").unlink(missing_ok=True)
            (self.directory / ".checkpoint.cache").unlink(missing_ok=True)
        path = self.directory / f"{day:06d}-{key}.npz"
        self._wait_for(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) != key:
                    raise ValueError("快照键不符")
                state = {
                    name: data[name].copy()
                    for name in (STATE_FIELDS if fields is None else fields)
                }
                if with_model:
                    meta = json.loads(str(data["meta"]))
                    factors = tuple(data["factors"].tolist())
                    covariance, specific = data["F"], data["D"]
                    symbols_sha = str(data["symbols_sha"])
        except _UNREADABLE:
            # 截断、损坏或被换成别的快照：当作未命中，由调用方重算后覆盖。
            self._discard(day, path)
            return None
        model = None
        if with_model:
            stored = self._stored_symbols()
            if stored is None or stored[1] != symbols_sha:
                # 证券表读不出或被改过：只判未命中；快照本身完好，不删（下次 prepare 会重写证券表）。
                return None
            symbols = stored[0]
            latest = self._latest(key)
            if latest is not None:
                b = latest["B"]
            elif exposure_loader is not None:
                b = exposure_loader()
            else:
                raise ValueError(
                    "历史 B 按需重建：请提供 exposure_loader 或使用 model_series"
                )
            model = RiskModel(
                symbols,
                b,
                factors,
                covariance,
                specific,
                meta["date"],
                meta["metadata"],
            )
        return model, state

    def save(self, key, model, state, *, day, latest=True, checkpoint=None):
        """登记后交给后台线程写；数组先复制，调用方之后改动不影响落盘内容。

        返回时只保证已排队（队列满时等待）；何时算落盘见模块说明。
        """
        self._check()
        self._symbol_table(model.symbols)
        path = self.directory / f"{day:06d}-{key}.npz"
        arrays = {
            "key": np.asarray(key),
            "symbols_sha": np.asarray(self._symbols_sha),
            "factors": np.asarray(model.factors),
            "F": np.array(model.F),
            "D": np.array(model.D),
            "meta": np.asarray(
                json.dumps({"date": model.date, "metadata": model.metadata})
            ),
            **{name: np.array(value) for name, value in state.items()},
        }

        def write_day():
            _write_atomic(self.directory, path, lambda stream: _savez(stream, arrays))

        self._submit(path, write_day)
        if latest:
            values = (
                {}
                if checkpoint is None
                else {
                    "rolling_" + name: np.array(value)
                    for name, value in checkpoint.items()
                }
            )
            exposure = {"key": np.asarray(key), "B": np.array(model.B), **values}
            target = self.directory / ".checkpoint.cache"

            def write_latest():
                _write_atomic(
                    self.directory,
                    target,
                    # 只在最后一天写一次，但每次追加都要等它落盘：用最快一级压缩。
                    lambda stream: _savez(stream, exposure),
                )

            self._submit(target, write_latest)
        self.entries[day] = key

    def _latest(self, key):
        path = self.directory / ".checkpoint.cache"
        self._wait_for(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) != key:
                    return None
                return {name: data[name] for name in data.files}
        except _UNREADABLE:
            path.unlink(missing_ok=True)
            return None

    # 输入前缀索引 ---------------------------------------------------------

    def load_index(self):
        """逐日输入摘要和前缀链摘要（各 T×32 字节）；没有、损坏或校验和不符时为 None。"""
        self._check()
        path = self.directory / ".prefix-index"
        self._wait_for(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                rows, prefixes = data["rows"], data["prefixes"]
                checksum = str(data["checksum"])
        except (*_UNREADABLE, FileNotFoundError):
            return None
        if (
            rows.dtype != np.uint8
            or prefixes.dtype != np.uint8
            or rows.ndim != 2
            or rows.shape[1:] != (32,)
            or rows.shape != prefixes.shape
            or hashlib.sha256(rows.tobytes() + prefixes.tobytes()).hexdigest()
            != checksum
        ):
            return None
        return rows, prefixes

    def save_index(self, rows, prefixes):
        rows = np.ascontiguousarray(rows, dtype=np.uint8)
        prefixes = np.ascontiguousarray(prefixes, dtype=np.uint8)
        checksum = hashlib.sha256(rows.tobytes() + prefixes.tobytes()).hexdigest()
        target = self.directory / ".prefix-index"

        def write():
            _write_atomic(
                self.directory,
                target,
                lambda stream: np.savez(
                    stream,
                    rows=rows,
                    prefixes=prefixes,
                    checksum=np.asarray(checksum),
                ),
            )

        self._submit(target, write)

    def checkpoint(self, key):
        latest = self._latest(key)
        if latest is None:
            return None
        values = {
            name.removeprefix("rolling_"): value
            for name, value in latest.items()
            if name.startswith("rolling_")
        }
        return values or None


def report_failure(cache, error, where):
    """无法向调用方抛出时（生成器被回收、解释器退出）把写失败写到 stderr。"""
    sys.stderr.write(
        f"ashare.risk: {where}：风险模型缓存写入失败（{cache.directory}）："
        f"{error.__cause__!r}；排队中的写入没有执行，这个缓存对象之后的 save/load/flush 都会抛出\n"
    )
    sys.stderr.flush()


@atexit.register
def _flush_all():
    """解释器退出前等本进程所有缓存对象落盘（只登记一次，不随实例数增长）。"""
    for cache in list(_LIVE):
        writer = cache._writer
        if writer is None or not writer.owned():
            continue
        if cache._raised:
            continue  # 调用方已经收到过这个错误
        try:
            cache.flush()
        except RuntimeError as error:
            report_failure(cache, error, "解释器退出时")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stats", "clear"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        # 不存在的目录不创建：直接报告空缓存。
        print(
            json.dumps(
                {
                    "snapshots": 0,
                    "bytes": 0,
                    "latest_exposure_bytes": 0,
                    "generations": 0,
                }
                if args.action == "stats"
                else {"removed_bytes": 0}
            )
        )
        return
    cache = ModelCache(args.directory)
    print(
        json.dumps(
            cache.stats()
            if args.action == "stats"
            else {"removed_bytes": cache.clear()}
        )
    )


if __name__ == "__main__":
    main()
