"""取数协议、按依赖释放中间值的求值器和可校验磁盘缓存。"""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
import panelops

from .expression import Expr


class PanelSource(Protocol):
    """数据层负责可知时点和版本；普通字段 T×N，市场字段 T 或 T×1。"""

    dates: object
    symbols: object
    version: str

    def __getitem__(self, field: str) -> np.ndarray: ...


def _checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _implementation_id():
    """求值规则变动同样使旧缓存失效，不依赖手工版本号。"""
    h = hashlib.sha256()
    for name in ("engine.py", "expression.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


class Evaluator:
    """每次调用复用公共节点，最后一个消费者完成后释放中间数组。"""

    def __init__(self, cache_dir=None):
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.last_counts = {}
        self.last_peak_arrays = 0
        self.last_batch = {}
        self.cache_hit = False
        if self.cache_dir is not None and self.cache_dir.exists():
            for path in self.cache_dir.glob("*.tmp"):
                parts = path.name.split(".")
                if len(parts) == 4 and parts[1].isdigit():
                    try:
                        os.kill(int(parts[1]), 0)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        continue
                    else:
                        continue
                path.unlink(missing_ok=True)

    def evaluate(
        self, expression: Expr, source: PanelSource, *, mask=None, universe_id="all"
    ):
        if not isinstance(expression, Expr):
            raise TypeError("expression 必须是 Expr")
        shape = self._check(source, mask, universe_id)
        path = self._cache_path(expression, source, mask, universe_id)
        self.cache_hit = False
        self.last_counts = {}
        self.last_peak_arrays = 0
        cached = self._load(path, shape)
        if cached is not None:
            self.cache_hit = True
            return cached
        order, uses = _schedule([expression])
        memo = {}
        for node in order:
            self.last_counts[node.digest] = 1
            memo[node.digest] = _compute(
                node, memo, source, shape, mask, node is expression
            )
            self.last_peak_arrays = max(
                self.last_peak_arrays,
                sum(isinstance(v, np.ndarray) for v in memo.values()),
            )
            _release(node, uses, memo)
        result = _finish(memo[expression.digest], shape, mask)
        self._store(path, result)
        return result

    def evaluate_many(
        self,
        expressions,
        source: PanelSource,
        *,
        mask=None,
        universe_id="all",
        memory_limit=None,
    ):
        """批量求值：一批表达式合成一张图，公共子式与字段叶子只算一次。

        ``expressions`` 是 ``{名字: Expr}`` 或 ``Expr`` 序列；返回迭代器，按输入顺序
        逐个给出 ``(名字或下标, 结果)``，每个结果与单独 ``evaluate`` 逐位相同。
        中间结果在最后一个消费者算完后释放，已给出的结果不再由求值器持有，
        调用方用完即丢就不会累积。``memory_limit`` 是预计同时存活的面板字节上限
        （默认物理内存的 1/12）；按图模拟的峰值超限时自动拆成几批，批与批之间不共享
        中间结果（结果仍逐位相同）。缓存目录的读写规则与 ``evaluate`` 相同。
        """
        items = (
            list(expressions.items())
            if isinstance(expressions, dict)
            else list(enumerate(expressions))
        )
        if any(not isinstance(e, Expr) for _, e in items):
            raise TypeError("expression 必须是 Expr")
        shape = self._check(source, mask, universe_id)
        if memory_limit is None:
            memory_limit = default_memory_limit()
        elif (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit <= 0
        ):
            raise ValueError("memory_limit 必须是正整数字节数")
        paths = [self._cache_path(e, source, mask, universe_id) for _, e in items]
        return self._many(items, paths, source, shape, mask, memory_limit)

    def _many(self, items, paths, source, shape, mask, memory_limit):
        self.cache_hit = False
        self.last_counts = {}
        self.last_peak_arrays = 0
        self.last_batch = {"expressions": len(items), "cache_hits": 0, "batches": 0}
        panel_bytes = 8 * shape[0] * shape[1]
        pending, hits = [], {}  # 未命中的位置；命中的在原位置直接给出
        for position, path in enumerate(paths):
            cached = self._load(path, shape)
            if cached is None:
                pending.append(position)
            else:
                hits[position] = cached
                self.last_batch["cache_hits"] += 1
        batches = _split(
            [items[i][1] for i in pending], max(1, memory_limit // panel_bytes)
        )
        self.last_batch["batches"] = len(batches)
        cursor = 0
        for batch in batches:
            roots = [items[pending[cursor + k]][1] for k in range(len(batch))]
            positions = [pending[cursor + k] for k in range(len(batch))]
            cursor += len(batch)
            order, uses = _schedule(roots)
            # 每个根在批内每出现一次就多一个消费者，给出后才减掉。
            for root in roots:
                uses[root.digest] += 1
            root_digests = {root.digest for root in roots}
            memo, emitted = {}, 0
            for node in order:  # _schedule 已去重，每个节点只算一次
                self.last_counts[node.digest] = 1
                memo[node.digest] = _compute(
                    node, memo, source, shape, mask, node.digest in root_digests
                )
                self.last_peak_arrays = max(
                    self.last_peak_arrays,
                    sum(isinstance(v, np.ndarray) for v in memo.values()),
                )
                _release(node, uses, memo)
                while emitted < len(roots) and roots[emitted].digest in memo:
                    position = positions[emitted]
                    while hits and min(hits) < position:
                        first = min(hits)
                        yield items[first][0], hits.pop(first)
                    root = roots[emitted]
                    result = _finish(memo[root.digest], shape, mask)
                    self._store(paths[position], result)
                    uses[root.digest] -= 1
                    if uses[root.digest] == 0:
                        del memo[root.digest]
                    emitted += 1
                    yield items[position][0], result
                    del result
        for first in sorted(hits):
            yield items[first][0], hits[first]

    def _check(self, source, mask, universe_id):
        shape = (len(source.dates), len(source.symbols))
        if mask is not None:
            panelops.api.panel(mask, "mask", bool, shape)
        if not isinstance(source.version, str) or not source.version:
            raise ValueError("数据 version 必须是非空字符串")
        if not isinstance(universe_id, str) or not universe_id:
            raise ValueError("universe_id 必须是非空字符串")
        return shape

    def _cache_path(self, expression, source, mask, universe_id):
        if self.cache_dir is None:
            return None  # 没有缓存目录时缓存键用不上，不必对日期和代码表做哈希
        axes = hashlib.sha256(
            json.dumps(
                [list(map(str, source.dates)), list(map(str, source.symbols))],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        key = [
            expression.digest,
            source.version,
            universe_id,
            panelops.build_id,
            _implementation_id(),
            axes,
            None if mask is None else hashlib.sha256(mask.tobytes()).hexdigest(),
        ]
        digest = hashlib.sha256(
            json.dumps(key, separators=(",", ":")).encode()
        ).hexdigest()
        return self.cache_dir / (digest + ".npy")

    @staticmethod
    def _load(path, shape):
        if path is None or not path.exists():
            return None
        try:
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["sha256"] != _checksum(path):
                raise ValueError("缓存校验和不符")
            result = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                result.shape != shape
                or result.dtype != np.float64
                or not result.flags.c_contiguous
            ):
                raise ValueError("缓存面板协议不符")
            return result
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            # 半截、损坏及旧格式缓存均按未命中重算，后续原子替换。
            return None

    @staticmethod
    def _store(path, result):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = []
        try:
            fd, name = tempfile.mkstemp(
                prefix=f"{path.stem}.{os.getpid()}.", suffix=".tmp", dir=path.parent
            )
            temporary.append(Path(name))
            with os.fdopen(fd, "wb") as output:
                np.save(output, result, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            checksum = _checksum(temporary[0])
            fd, name = tempfile.mkstemp(
                prefix=f"{path.stem}.{os.getpid()}.", suffix=".tmp", dir=path.parent
            )
            temporary.append(Path(name))
            with os.fdopen(fd, "w") as output:
                json.dump({"sha256": checksum}, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary[0], path)
            os.replace(temporary[1], path.with_suffix(".json"))
        finally:
            for temp in temporary:
                temp.unlink(missing_ok=True)


def default_memory_limit():
    """批量求值的默认面板内存上限：物理内存的 1/12（24 GB 机器上 2 GB，约 9 张全尺寸面板）。

    实测 1/4（6 GB）在别的任务同时占内存时出现过 2 倍的抖动，1/12 稳定且只比逐个求值
    多约 0.4 GB 驻留。"""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 12
    except (ValueError, OSError):
        return 4 * 2**30


def _schedule(roots):
    """依次对每个根做后序遍历，跳过已出现的节点；返回顺序和每个节点的消费者数。"""
    order, seen, uses = [], set(), Counter()
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, ready = stack.pop()
            if ready:
                order.append(node)
            elif node.digest not in seen:
                seen.add(node.digest)
                stack.append((node, True))
                for child in node.args:
                    uses[child.digest] += 1
                stack.extend((child, False) for child in reversed(node.args))
    return order, uses


def _peak(roots):
    """按与 _many 相同的顺序和释放规则模拟同时存活的面板数（含正在写的输出）。"""
    order, uses = _schedule(roots)
    for root in roots:
        uses[root.digest] += 1
    live, done, peak, emitted = set(), set(), 0, 0
    for node in order:
        done.add(node.digest)
        if node.op != "constant":
            peak = max(peak, len(live) + 1)
            live.add(node.digest)
        for child in node.args:
            uses[child.digest] -= 1
            if uses[child.digest] == 0:
                live.discard(child.digest)
        while emitted < len(roots) and roots[emitted].digest in done:
            uses[roots[emitted].digest] -= 1
            if uses[roots[emitted].digest] == 0:
                live.discard(roots[emitted].digest)
            emitted += 1
    # 调用方手里还拿着刚给出的一个结果。
    return peak + 1


def _split(roots, limit):
    """按输入顺序贪心分批，使每批模拟峰值不超过 limit 个面板；单个超限的自成一批。"""
    batches, current = [], []
    for root in roots:
        if current and _peak([*current, root]) > limit:
            batches.append(current)
            current = []
        current.append(root)
    if current:
        batches.append(current)
    return batches


def _release(node, uses, memo):
    for child in node.args:
        uses[child.digest] -= 1
        if uses[child.digest] == 0:
            del memo[child.digest]


def _panel(value, shape):
    if np.isscalar(value):
        return np.full(shape, value, dtype=np.float64)
    return np.ascontiguousarray(value)


def _finish(value, shape, mask):
    result = np.ascontiguousarray(_panel(value, shape), dtype=np.float64)
    if mask is not None and np.isscalar(value):
        result[~mask] = np.nan
    # 创建只读视图，不修改数据源原数组的 writeable 标志。
    result = result.view()
    result.flags.writeable = False
    return result


def _compute(node, memo, source, shape, mask, root):
    """单个节点的求值；单个与批量入口共用，保证逐位相同。

    浮点字段叶子只在作为结果直接给出时才把非有限值和掩码外的格子改写成 NaN。
    作为算子输入时省掉这一遍 isfinite 和 where 复制：Rust 内核对每个输入格子都按
    「非有限即缺失」处理，掩码也随每次调用传入，结果逐位相同（见 PERF_LOG P-010）。
    配置了 NumPy 分派（``panelops.api.BACKENDS`` 非空）时仍照旧改写。"""
    params = dict(node.params)
    if node.op == "field":
        raw = source[params["value"]]
        dtype = np.int64 if params["dtype"] == "int64" else np.float64
        if not isinstance(raw, np.ndarray) or raw.dtype.kind not in (
            "b",
            "i",
            "u",
            "f",
        ):
            raise TypeError("数据字段必须是数值数组")
        if dtype == np.int64 and raw.dtype.kind not in ("i", "u"):
            raise TypeError("行业字段必须为整数")
        raw = np.ascontiguousarray(raw, dtype=dtype)
        if params["broadcast"]:
            # 数值或布尔序列已在上面统一转换为 float64，这里只核对形状。
            if raw.shape not in ((shape[0],), (shape[0], 1)):
                raise ValueError("市场字段必须是长度 T 或 T×1 的数值序列")
            raw = np.ascontiguousarray(np.broadcast_to(raw.reshape(-1, 1), shape))
        else:
            panelops.api.panel(raw, params["value"], dtype=dtype, shape=shape)
        if dtype == np.float64 and not root and not panelops.api.BACKENDS:
            return raw
        valid = np.isfinite(raw) if dtype == np.float64 else np.ones(shape, dtype=bool)
        if mask is not None:
            valid &= mask
        return (
            raw
            if valid.all()
            else np.where(valid, raw, -1 if dtype == np.int64 else np.nan)
        )
    if node.op == "constant":
        return params["value"]
    args = [memo[a.digest] for a in node.args]
    if node.op.startswith(("ts_", "cs_")):
        args = [_panel(a, shape) for a in args]
        if node.op == "cs_group_demean":
            return panelops.cs_group_demean(
                args[0], groups=args[1], mask=mask, **params
            )
        if node.op == "cs_neutralize":
            count = params.pop("exposure_count")
            groups = args[1 + count] if params.pop("has_groups") else None
            weights = args[-1] if params.pop("has_weights") else None
            return panelops.cs_neutralize(
                args[0],
                exposures=args[1 : 1 + count],
                groups=groups,
                weights=weights,
                mask=mask,
                **params,
            )
        return getattr(panelops, node.op)(*args, mask=mask, **params)
    return panelops.elementwise(
        node.op,
        *args,
        mask=mask if any(isinstance(a, np.ndarray) for a in args) else None,
        **params,
    )
