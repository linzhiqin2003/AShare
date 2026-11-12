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
        shape = (len(source.dates), len(source.symbols))
        if mask is not None:
            panelops.api.panel(mask, "mask", bool, shape)
        if not isinstance(source.version, str) or not source.version:
            raise ValueError("数据 version 必须是非空字符串")
        if not isinstance(universe_id, str) or not universe_id:
            raise ValueError("universe_id 必须是非空字符串")
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
        path = None if self.cache_dir is None else self.cache_dir / (digest + ".npy")
        self.cache_hit = False
        self.last_counts = {}
        self.last_peak_arrays = 0
        if path is not None and path.exists():
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
                self.cache_hit = True
                return result
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                # 半截、损坏及旧格式缓存均按未命中重算，后续原子替换。
                pass

        order, seen, stack = [], set(), [(expression, False)]
        uses = Counter()
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
        memo = {}

        def panel(value):
            if np.isscalar(value):
                return np.full(shape, value, dtype=np.float64)
            return np.ascontiguousarray(value)

        for node in order:
            params = dict(node.params)
            self.last_counts[node.digest] = 1
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
                    raw = np.ascontiguousarray(
                        np.broadcast_to(raw.reshape(-1, 1), shape)
                    )
                else:
                    panelops.api.panel(raw, params["value"], dtype=dtype, shape=shape)
                valid = (
                    np.isfinite(raw)
                    if dtype == np.float64
                    else np.ones(shape, dtype=bool)
                )
                if mask is not None:
                    valid &= mask
                result = (
                    raw
                    if valid.all()
                    else np.where(valid, raw, -1 if dtype == np.int64 else np.nan)
                )
            elif node.op == "constant":
                result = params["value"]
            else:
                args = [memo[a.digest] for a in node.args]
                if node.op.startswith(("ts_", "cs_")):
                    args = [panel(a) for a in args]
                    if node.op == "cs_group_demean":
                        result = panelops.cs_group_demean(
                            args[0], groups=args[1], mask=mask, **params
                        )
                    elif node.op == "cs_neutralize":
                        count = params.pop("exposure_count")
                        groups = args[1 + count] if params.pop("has_groups") else None
                        weights = args[-1] if params.pop("has_weights") else None
                        result = panelops.cs_neutralize(
                            args[0],
                            exposures=args[1 : 1 + count],
                            groups=groups,
                            weights=weights,
                            mask=mask,
                            **params,
                        )
                    else:
                        result = getattr(panelops, node.op)(*args, mask=mask, **params)
                else:
                    result = panelops.elementwise(
                        node.op,
                        *args,
                        mask=mask
                        if any(isinstance(a, np.ndarray) for a in args)
                        else None,
                        **params,
                    )
                del args
            memo[node.digest] = result
            self.last_peak_arrays = max(
                self.last_peak_arrays,
                sum(isinstance(v, np.ndarray) for v in memo.values()),
            )
            for child in node.args:
                uses[child.digest] -= 1
                if uses[child.digest] == 0:
                    del memo[child.digest]
        result = np.ascontiguousarray(panel(memo[expression.digest]), dtype=np.float64)
        if mask is not None and np.isscalar(memo[expression.digest]):
            result[~mask] = np.nan
        # 创建只读视图，不修改数据源原数组的 writeable 标志。
        result = result.view()
        result.flags.writeable = False
        if path is not None:
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
        return result
