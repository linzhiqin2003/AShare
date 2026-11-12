"""一代、无损、按需重建暴露的日模型缓存。

格式 4：每日快照用 level 1 deflate 在后台线程里写（`np.load` 可读），证券表每代只存一次（`.symbols.npy`），
快照里存自己的键；写入交给一个后台线程，读取、清理和生成器结束前都会等它落盘，
写失败在下一次调用时抛出。旧格式目录在 `prepare` 时整体判为未命中并清空。
"""

import argparse
import atexit
import json
import os
import queue
import re
import threading
import weakref
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np

from .model import RiskModel

FORMAT = 4
STATE_FIELDS = ("f", "residual", "base_f", "base_d", "bias", "estimation", "cap")
_HIDDEN = (
    ".latest.npy",
    ".latest.json",
    ".checkpoint.cache",
    ".generation.json",
    ".symbols.npy",
)
# 读快照时这些异常都意味着文件不完整或不是本缓存写的：当作未命中，重算后覆盖。
_UNREADABLE = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def _snapshot(path):
    return re.fullmatch(r"(?:[0-9]{6,}-)?[a-f0-9]{64}\.npz", path.name) is not None


def _owned(path):
    return path.is_file() and (_snapshot(path) or path.name in _HIDDEN)


class _Writer:
    """单个后台线程按提交顺序写文件；异常保存下来，由调用方线程抛出。"""

    def __init__(self):
        self.jobs = queue.Queue(maxsize=64)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
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

    def submit(self, job):
        self.raise_error()
        self.jobs.put(job)

    def flush(self):
        self.jobs.join()
        self.raise_error()

    def raise_error(self):
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("风险模型缓存写入失败") from error

    def close(self):
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
    """历史保存 F/D 和递推状态，B 只保留最新一天；历史 B 由调用方按需重建。"""

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
        self._writer = None
        self._pending = set()

    # 后台写入 -------------------------------------------------------------

    def _submit(self, path, job):
        if self._writer is None:
            self._writer = _Writer()
            weakref.finalize(self, self._writer.close)
            atexit.register(_flush_quietly, weakref.ref(self))
        self._pending.add(path)

        def run():
            job()
            self._pending.discard(path)

        self._writer.submit(run)

    def flush(self):
        """等待后台写入全部落盘；有写失败则抛出 RuntimeError。"""
        if self._writer is not None:
            self._writer.flush()
            self._pending.clear()

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
        self._symbols = None
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

    def _symbol_table(self, symbols):
        """每代只写一次；内容不同（证券顺序变了）时整体重写。"""
        if self._symbols is None:
            path = self.directory / ".symbols.npy"
            try:
                self._symbols = tuple(np.load(path, allow_pickle=False).tolist())
            except _UNREADABLE:
                self._symbols = None
        if self._symbols != tuple(symbols):
            self._symbols = tuple(symbols)
            array = np.asarray(self._symbols)
            _write_atomic(
                self.directory,
                self.directory / ".symbols.npy",
                lambda stream: np.save(stream, array, allow_pickle=False),
            )

    def _stored_symbols(self):
        if self._symbols is None:
            self._symbols = tuple(
                np.load(self.directory / ".symbols.npy", allow_pickle=False).tolist()
            )
        return self._symbols

    # 读写 -----------------------------------------------------------------

    def _discard(self, day, path):
        path.unlink(missing_ok=True)
        if self.entries.get(day) == path.stem.split("-", 1)[-1]:
            del self.entries[day]

    def load(self, key, *, day=None, with_model=True, exposure_loader=None):
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
                state = {name: data[name].copy() for name in STATE_FIELDS}
                if with_model:
                    meta = json.loads(str(data["meta"]))
                    factors = tuple(data["factors"].tolist())
                    covariance, specific = data["F"], data["D"]
                    symbols = self._stored_symbols()
        except _UNREADABLE:
            # 截断、损坏或被换成别的快照：当作未命中，由调用方重算后覆盖。
            self._discard(day, path)
            return None
        model = None
        if with_model:
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
        """登记后交给后台线程写；数组先复制，调用方之后改动不影响落盘内容。"""
        self._symbol_table(model.symbols)
        path = self.directory / f"{day:06d}-{key}.npz"
        arrays = {
            "key": np.asarray(key),
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
                    # 只在最后一天写一次；递推检查点里大量计数可压缩，保持压缩。
                    lambda stream: np.savez_compressed(stream, **exposure),
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


def _flush_quietly(reference):
    cache = reference()
    if cache is not None:
        try:
            cache.flush()
        except RuntimeError as error:
            print(f"risk model cache: {error} ({error.__cause__!r})")


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
