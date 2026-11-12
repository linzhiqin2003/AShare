"""特征库：只存股票池内的样本，float32，按特征列存，磁盘映射。

样本按（信号日，证券列）排序，第 t 日的样本是 `[day_ptr[t], day_ptr[t+1])` 这一段，
一折取一段日期就是取一段连续的样本，不整体拷贝。每列是一份预处理后的特征值
（`Preprocess` 在建库时做，训练和预测读同一份）。

缓存键：特征表达式（或面板内容）的哈希 + 预处理定义 + 股票池 + 面板版本 + 本模块代码。
新特征只算新的那一列；`incremental=True` 时面板版本不进键，新增日期只追加尾部，追加前
重算已存的最后 `verify_days` 天逐位核对，不一致就整列重算。
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import panelops

from ..factor import Evaluator, Expr
from .preprocess import Preprocess

FORMAT = 1


def _sha(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
        h.update(b"\x00")
    return h.hexdigest()


@functools.cache
def implementation_id():
    """本包求值与预处理代码的哈希：代码改了旧缓存自动失效。导入后第一次调用时取一次，
    之后不再读盘（进程里跑的就是那时的代码，运行中改文件不影响缓存键）。"""
    here = Path(__file__).parent
    return _sha(*(here.joinpath(n).read_bytes() for n in ("store.py", "preprocess.py")))


implementation_id()


def normalize_features(features):
    """{名字: Expr 或 T×N 数组}；也接受 Expr 序列、`Registry`、(名字, 值) 序列。"""
    if hasattr(features, "names") and hasattr(features, "__iter__"):
        items = [(f.name, f.expression) for f in features]
    elif isinstance(features, dict):
        items = list(features.items())
    else:
        items = []
        for item in features:
            if isinstance(item, Expr):
                items.append(
                    (str(item) if len(str(item)) <= 60 else item.digest[:12], item)
                )
            else:
                name, value = item
                items.append((name, value))
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("特征重名")
    if not items:
        raise ValueError("至少需要一个特征")
    for name, value in items:
        if not isinstance(name, str) or not name:
            raise ValueError("特征名必须是非空字符串")
        if not isinstance(value, (Expr, np.ndarray)):
            raise TypeError(f"特征 {name}: 需要 Expr 或 T×N 数组")
    return dict(items)


def feature_digest(value):
    """表达式用它自己的摘要；数组用内容哈希（形状 + 字节）。"""
    if isinstance(value, Expr):
        return "expr:" + value.digest
    a = np.ascontiguousarray(value, dtype=np.float64)
    return "array:" + _sha(a.shape, hashlib.sha256(a.tobytes()).hexdigest())


def default_root():
    """缓存与账本的默认目录：ASHARE_ML_DIR；否则数据根目录（ASHARE_DATA_DIR，默认仓库下的
    data/，与 `ashare.data.open()` 相同）下的 ml/（不入库）。"""
    if os.environ.get("ASHARE_ML_DIR"):
        return Path(os.environ["ASHARE_ML_DIR"])
    data = (
        os.environ.get("ASHARE_DATA_DIR")
        or Path(__file__).resolve().parents[3] / "data"
    )
    return Path(data) / "ml"


class _Rows:
    """面板的一段行（用于增量追加时只重算尾部）。"""

    def __init__(self, source, start):
        self._source, self._start = source, start
        self.dates = np.asarray(source.dates)[start:]
        self.symbols = source.symbols
        self.version = f"{source.version}:rows{start}"

    def __getitem__(self, name):
        return np.asarray(self._source[name])[self._start :]


def _axes_hash(dates, symbols):
    return _sha(json.dumps([list(map(str, dates)), list(map(str, symbols))]))


@dataclass
class _Column:
    name: str
    key: str
    days: int
    samples: int


class FeatureStore:
    """一份特征库。`matrix(样本)` 取 float32 的 (样本 × 特征) 矩阵。

    属性：names（特征名）、dates、symbols、day_ptr（T+1）、rows / cols（每个样本的
    行号和列号，int32）、n_samples、preprocess、path（内存模式为 None）。
    """

    def __init__(
        self,
        *,
        dates,
        symbols,
        day_ptr,
        rows,
        cols,
        columns,
        names,
        digests,
        preprocess,
        path=None,
        meta=None,
    ):
        self.dates, self.symbols = np.asarray(dates), np.asarray(symbols)
        self.day_ptr, self.rows, self.cols = day_ptr, rows, cols
        self._columns = columns  # 名字 → 一维 float32（ndarray 或 memmap）
        self.names = tuple(names)
        self.digests = dict(digests)
        self.preprocess = preprocess
        self.path = path
        self.meta = meta or {}

    @property
    def n_samples(self):
        return int(self.day_ptr[-1])

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def column(self, name):
        return self._columns[name]

    def select(self, names):
        """只保留部分特征（同一份磁盘列，不复制）。"""
        missing = set(names) - set(self.names)
        if missing:
            raise KeyError(f"特征库里没有 {sorted(missing)}")
        return FeatureStore(
            dates=self.dates,
            symbols=self.symbols,
            day_ptr=self.day_ptr,
            rows=self.rows,
            cols=self.cols,
            columns=self._columns,
            names=names,
            digests={n: self.digests[n] for n in names},
            preprocess=self.preprocess,
            path=self.path,
            meta=self.meta,
        )

    def samples(self, first_day, last_day):
        """第 [first_day, last_day) 行的样本范围（slice）。"""
        return slice(int(self.day_ptr[first_day]), int(self.day_ptr[last_day]))

    def matrix(self, selection, names=None, block=1 << 16):
        """按样本选择（slice 或升序整数下标）取 float32 C 连续矩阵（样本 × 特征）。

        列存的数据要转成行主序：按 block 行一段，先把各列这一段取进一块 (特征 × 行)
        的小缓冲，再整块转置写进结果。直接逐列写进行主序矩阵是跨步写，实测慢几倍。"""
        names = self.names if names is None else names
        cols = [
            self._columns[n].view(np.ndarray)
            if isinstance(self._columns[n], np.memmap)
            else self._columns[n]
            for n in names
        ]
        if isinstance(selection, slice):
            start, stop, step = selection.indices(self.n_samples)
            if step != 1:
                selection = np.arange(start, stop, step)
        if isinstance(selection, slice):
            k = stop - start
        else:
            selection = np.asarray(selection)
            k = len(selection)
        out = np.empty((k, len(names)), dtype=np.float32)
        buffer = np.empty((len(names), min(block, max(k, 1))), dtype=np.float32)
        for a in range(0, k, block):
            b = min(a + block, k)
            part = buffer[:, : b - a]
            for j, col in enumerate(cols):
                if isinstance(selection, slice):
                    part[j] = col[start + a : start + b]
                else:
                    np.take(col, selection[a:b], out=part[j])
            out[a:b] = part.T
        return out

    def scatter(self, values, fill=np.nan, selection=None):
        """样本向量 → T×N float64 面板（库外格子为 fill）。"""
        out = np.full(self.shape, fill, dtype=np.float64)
        if selection is None:
            out[self.rows, self.cols] = values
        else:
            out[self.rows[selection], self.cols[selection]] = values
        return out

    def gather(self, panel):
        """T×N 面板 → 样本向量（按库内样本顺序）。"""
        return np.asarray(panel)[self.rows, self.cols]


def _positions(universe):
    t, n = universe.shape
    counts = universe.sum(axis=1)
    day_ptr = np.zeros(t + 1, dtype=np.int64)
    np.cumsum(counts, out=day_ptr[1:])
    rows, cols = np.nonzero(universe)
    return day_ptr, rows.astype(np.int32), cols.astype(np.int32)


def _values(x, universe):
    return np.asarray(x, dtype=np.float64)[universe].astype(np.float32)


def _same(a, b):
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _write_atomic(path, data: bytes):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(name, path)


def build(
    features,
    data,
    universe,
    preprocess=None,
    *,
    root=None,
    universe_id=None,
    incremental=False,
    verify_days=20,
    threads=None,
    memory=False,
    log=None,
):
    """建（或复用、追加）特征库。

    features：见 `normalize_features`；data：面板（`ashare.data.Panel` 或任何
    PanelSource）；universe：T×N 布尔股票池。memory=True 时不落盘（小数据、测试）。
    universe_id：增量模式下股票池定义的名字；同名但已存日期上的掩码不同会整库重建。
    """
    preprocess = preprocess or Preprocess()
    features = normalize_features(features)
    dates, symbols = np.asarray(data.dates), np.asarray(data.symbols)
    t, n = len(dates), len(symbols)
    universe = np.ascontiguousarray(universe, dtype=bool)
    if universe.shape != (t, n):
        raise ValueError("universe: 形状必须与面板一致")
    for name, value in features.items():
        if isinstance(value, np.ndarray) and value.shape != (t, n):
            raise ValueError(f"特征 {name}: 形状必须与面板一致")
    digests = {k: feature_digest(v) for k, v in features.items()}
    log = log or (lambda *_: None)
    old_threads = panelops.get_num_threads()
    if threads is not None:
        panelops.set_num_threads(threads)
    try:
        if memory:
            return _build_memory(features, data, universe, preprocess, digests)
        root = Path(root) if root is not None else default_root()
        version = "incremental" if incremental else data.version
        uid = universe_id or (
            "mask:" + hashlib.sha256(np.packbits(universe).tobytes()).hexdigest()
        )
        key = _sha(
            FORMAT,
            implementation_id(),
            preprocess.digest,
            version,
            uid,
            _sha(json.dumps(list(map(str, symbols)))),
            "" if incremental else _axes_hash(dates, symbols),
        )
        path = root / "stores" / key
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return _build_disk(
                path,
                features,
                data,
                universe,
                preprocess,
                digests,
                incremental,
                verify_days,
                log,
            )
    finally:
        panelops.set_num_threads(old_threads)


def _evaluate(features, source, names):
    """逐个给出 (名字, T×N float64)：表达式批量求值共享公共子式，数组原样。"""
    exprs = {k: features[k] for k in names if isinstance(features[k], Expr)}
    it = Evaluator().evaluate_many(exprs, source) if exprs else iter(())
    for name in names:
        if isinstance(features[name], Expr):
            got, value = next(it)
            assert got == name
            yield name, value
        else:
            yield (
                name,
                np.asarray(features[name])[len(features[name]) - len(source.dates) :],
            )


def _build_memory(features, data, universe, preprocess, digests):
    day_ptr, rows, cols = _positions(universe)
    ctx = preprocess.context(data)
    columns = {}
    for name, value in _evaluate(features, data, list(features)):
        columns[name] = _values(preprocess.apply(value, universe, ctx), universe)
    return FeatureStore(
        dates=data.dates,
        symbols=data.symbols,
        day_ptr=day_ptr,
        rows=rows,
        cols=cols,
        columns=columns,
        names=list(features),
        digests=digests,
        preprocess=preprocess,
    )


def _build_disk(
    path, features, data, universe, preprocess, digests, incremental, verify_days, log
):
    dates = np.asarray(data.dates).astype(str)
    t = len(dates)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    day_ptr, rows, cols = _positions(universe)
    if meta is not None:
        old_t = len(meta["dates"])
        ok = old_t <= t and meta["dates"] == dates[:old_t].tolist()
        if ok:
            old = np.load(path / "index.npz")
            s_old = int(old["day_ptr"][-1])
            ok = (
                np.array_equal(old["day_ptr"], day_ptr[: old_t + 1])
                and np.array_equal(old["rows"], rows[:s_old])
                and np.array_equal(old["cols"], cols[:s_old])
            )
        if not ok:
            log("特征库：已存日期或股票池与当前面板不一致，整库重建")
            for child in path.iterdir():
                if child.name != "lock":
                    child.unlink() if child.is_file() else shutil.rmtree(child)
            meta = None
    if meta is None:
        meta = {"format": FORMAT, "dates": [], "columns": {}, "versions": []}
    if meta["dates"] != dates.tolist():
        buffer = tempfile.SpooledTemporaryFile()
        np.savez(buffer, day_ptr=day_ptr, rows=rows, cols=cols)
        buffer.seek(0)
        _write_atomic(path / "index.npz", buffer.read())
    meta["dates"] = dates.tolist()
    meta["symbols"] = list(map(str, data.symbols))
    meta["preprocess"] = preprocess.describe()
    if data.version not in meta["versions"]:
        meta["versions"].append(data.version)
    s = int(day_ptr[-1])
    ctx = None
    todo_full, todo_tail = [], {}
    for name, digest in digests.items():
        col = meta["columns"].get(digest)
        if col is not None and col["days"] == t:
            continue
        value = features[name]
        lookback = value.lookback if isinstance(value, Expr) else 0
        if col is not None and incremental and lookback is not None and col["days"] < t:
            todo_tail[name] = (col, lookback)
        else:
            todo_full.append(name)
    started = time.perf_counter()
    if todo_full or todo_tail:
        ctx = preprocess.context(data)
    for name, values in _evaluate(features, data, todo_full):
        x = _values(preprocess.apply(values, universe, ctx), universe)
        file = path / (digests[name].replace(":", "_") + ".f32")
        _write_atomic(file, x.tobytes())
        meta["columns"][digests[name]] = {"name": name, "days": t, "samples": s}
        log(f"特征库：{name} 建好（{time.perf_counter() - started:.1f}s）")
    for name, (col, lookback) in todo_tail.items():
        _append(
            path,
            name,
            digests[name],
            col,
            lookback,
            features,
            data,
            universe,
            preprocess,
            ctx,
            day_ptr,
            verify_days,
            meta,
            log,
        )
    _write_atomic(meta_path, json.dumps(meta).encode())
    columns = {
        name: np.memmap(
            path / (d.replace(":", "_") + ".f32"),
            dtype=np.float32,
            mode="r",
            shape=(s,),
        )
        if s
        else np.empty(0, np.float32)
        for name, d in digests.items()
    }
    return FeatureStore(
        dates=data.dates,
        symbols=data.symbols,
        day_ptr=day_ptr,
        rows=rows,
        cols=cols,
        columns=columns,
        names=list(features),
        digests=digests,
        preprocess=preprocess,
        path=path,
        meta=meta,
    )


def _append(
    path,
    name,
    digest,
    col,
    lookback,
    features,
    data,
    universe,
    preprocess,
    ctx,
    day_ptr,
    verify_days,
    meta,
    log,
):
    t = len(data.dates)
    old_days = col["days"]
    check = min(verify_days, old_days)
    start = max(0, old_days - check - lookback)
    source = _Rows(data, start)
    sub = {name: features[name]}
    ((_, values),) = list(_evaluate(sub, source, [name]))
    sub_ctx = {k: v[start:] for k, v in ctx.items()}
    pre = preprocess.apply(values, universe[start:], sub_ctx)
    file = path / (digest.replace(":", "_") + ".f32")
    stored = np.memmap(file, dtype=np.float32, mode="r", shape=(col["samples"],))
    a, b = int(day_ptr[old_days - check]), int(day_ptr[old_days])
    fresh = _values(
        pre[old_days - check - start : old_days - start],
        universe[old_days - check : old_days],
    )
    if not _same(np.asarray(stored[a:b]), fresh):
        del stored
        log(f"特征库：{name} 尾部核对不一致，整列重算")
        ((_, full),) = list(_evaluate(sub, data, [name]))
        x = _values(preprocess.apply(full, universe, ctx), universe)
        _write_atomic(file, x.tobytes())
    else:
        del stored
        tail = _values(pre[old_days - start :], universe[old_days:])
        with open(file, "r+b") as f:
            f.truncate(col["samples"] * 4)
            f.seek(col["samples"] * 4)
            f.write(tail.tobytes())
            f.flush()
            os.fsync(f.fileno())
        log(f"特征库：{name} 追加 {t - old_days} 天")
    meta["columns"][digest] = {"name": name, "days": t, "samples": int(day_ptr[-1])}
