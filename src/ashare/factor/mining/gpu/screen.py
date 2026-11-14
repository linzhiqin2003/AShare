"""整代表达式的 GPU 粗筛：字段只上传一次、共享子树只算一次、按设备内存分批、每个根只取回
T 个 float32 日 IC，以及「GPU 粗筛 → CPU float64 复算」的块间流水线。

GPU 只决定一个候选送不送 CPU 复算，不产生任何进台账、进统计检验或对外显示的数字。送 CPU 的：

- GPU 日 IC 算出的训练段 t 值不低于门槛 ``gate_t``（`gate_threshold`）；
- t 值不是有限数（全缺失、样本不足等，宁可多算也不漏）；
- GPU 算不了的表达式（算子或参数不支持、字段缺失、常量根）、某个节点的值超出 float32
  安全范围（`backend.RANGE_HIGH`、`RANGE_LOW`）、排序主要由舍入决定（`DEGENERATE_FRACTION`）、
  GPU 上出错。

求值顺序与释放规则照搬 `ashare.factor.engine._schedule` 与 `_split`（整批共享子树、最后一个
消费者用完即释放；批的大小按每张面板 4·T·N 字节、设备内存预算换算）。
"""

from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ...engine import _schedule, _split
from .backend import supported_node
from .kernels import ELEM_INPUTS, ELEM_OPS, ELEM_STACK

_ELEM = frozenset(ELEM_OPS) - {"load", "const"}  # 能并进逐元素程序的算子

# GPU float32 粗筛 t 值相对 CPU float64 的保险边际。出处 docs/GPU_PROBE.md 第四节：真实面板一代
# 256 个表达式上，三种 GPU 写法（含数值退化的表达式）t 值误差最大 0.25，MLX 自定义内核 0.062；
# 0.3 大于全部实测最大误差，真实面板上只多复核约 1 个。换后端、改内核后要在同一份种群上重测
# 最大误差，边际至少取它的 1.2 倍（test/factor/gpu/test_gpu_mine.py 守住小面板上的不漏）。
GPU_SCREEN_T_MARGIN = 0.3

# 舍入并列比例（`Backend.rank_ic`）超过它的表达式交给 CPU。排序主要由舍入决定时，float32 与
# float64 的 t 值可以差得远超边际。实测：两块合成小面板、1800 个随机表达式里，数学上恒为常数
# 的 ts_corr(low, -low) 比例 1.0、t 差 1.24（小截面上噪声排序的 t 值很不稳），ts_std(ts_rank(x))
# 这类大量数学上相等的值比例 0.03–0.19、t 差最大 0.09；真实面板一代 256 个里比例 1.0 的两个
# 退化表达式 t 差 0.06、0.04，其余比例 ≤ 0.2 的 t 差最大 0.015。取 0.2：真实面板只多送 CPU
# 5 个（2%），留下来的 t 差都远小于边际（test_gpu_screen.py 守住）。
DEGENERATE_FRACTION = 0.2

# 每块表达式个数：GPU 粗筛第 k+1 块的同时 CPU 复算第 k 块（docs/GPU_PROBE.md 第六节）。
DEFAULT_CHUNK = 64

# 设备上除中间面板以外常驻的面板数（字段另计）：收益与其排序状态约 5 张、一次 Rank IC 的
# 临时面板约 9 张、范围检查 2 张。
RESERVED_PANELS = 16


@dataclass(frozen=True)
class GpuScreen:
    """`mine(..., screen=GpuScreen(...))` 的选项；``screen="gpu"`` 等于全用默认值。

    margin：门槛下放的 t 值（默认 `GPU_SCREEN_T_MARGIN`）；chunk：流水线每块的表达式个数；
    memory_limit：设备内存预算（字节），默认取后端给出的一半可用内存。"""

    margin: float = GPU_SCREEN_T_MARGIN
    chunk: int = DEFAULT_CHUNK
    memory_limit: int | None = None

    def __post_init__(self):
        if not (isinstance(self.margin, (int, float)) and math.isfinite(self.margin)
                and self.margin >= 0):  # fmt: skip
            raise ValueError("margin 必须是非负有限数")
        if (
            isinstance(self.chunk, bool)
            or not isinstance(self.chunk, int)
            or self.chunk < 1
        ):
            raise ValueError("chunk 必须是正整数")
        if self.memory_limit is not None and (
            isinstance(self.memory_limit, bool)
            or not isinstance(self.memory_limit, int)
            or self.memory_limit <= 0
        ):
            raise ValueError("memory_limit 必须是正整数字节数")


def gate_threshold(screen_t, alpha, margin):
    """GPU t 值不低于它的送 CPU 复算。

    取 min(screen_t, z_{1−α}) − margin：screen_t 决定谁过粗筛；z_{1−α} 保证训练段 IC 单侧
    p ≤ α 的候选也全部复算——粗筛掉的候选的 p 值照样进多重检验的计数（`_summarize`），
    Student t 的单侧 p 不小于正态的，所以 t < z_{1−α} 的 p 一定大于 α，不复算不改变任何计数。
    α ≥ 0.5 时 t 为负也可能显著，全部复算（返回 −∞）。"""
    if not (0 < alpha < 0.5):
        return -math.inf
    return min(float(screen_t), NormalDist().inv_cdf(1 - alpha)) - float(margin)


@dataclass
class Outcome:
    """一个表达式的 GPU 粗筛结果：ic（float64 化的 float32 日 IC）或 reason（为什么交给 CPU）。"""

    ic: np.ndarray | None = None
    reason: str | None = None
    noise: float | None = None  # 舍入并列比例（诊断用）


_FAILED = object()


class Screener:
    """一份面板视图上的 GPU 粗筛：股票池、收益只上传一次，字段第一次用到时上传一次。

    source：按字段名取 (T, N) 数组的面板（`mine` 里是截掉留出段的视图）；universe：
    (T, N) 布尔股票池；returns：筛选持有期的收益面板（与 `FactorEvaluator.returns[h]` 同一份）。"""

    def __init__(self, source, universe, returns, *, backend=None, memory_limit=None,
                 rows=None, profile=False):  # fmt: skip
        from .backend import default_backend

        t0 = time.perf_counter()
        self.backend = backend or default_backend()
        universe = np.asarray(universe, dtype=bool)
        if np.asarray(returns).shape != universe.shape:
            raise ValueError("returns 与 universe 形状不同")
        # rows：只用前 rows 行（mine 里取验证段起点：粗筛只看训练段的 t 值，验证段的
        # 行不必算）；取回的日 IC 在 rows 之后补 NaN，长度仍是面板的行数。
        self.total_rows = universe.shape[0]
        self.rows = self.total_rows if rows is None else int(rows)
        if not 0 < self.rows <= self.total_rows:
            raise ValueError("rows 超出面板行数")
        self.shape = (self.rows, universe.shape[1])
        self.source = source
        self.universe = self.backend.upload(universe[: self.rows])
        self.returns = self.backend.upload(
            np.asarray(returns, dtype=np.float64)[: self.rows]
        )
        self.backend.evaluate(self.universe, self.returns)
        self.memory_limit = memory_limit or self.backend.memory_budget()
        self._fields = {}  # 字段名 → (设备数组, 范围标记) 或 _FAILED
        self.seconds = Counter(upload=time.perf_counter() - t0)
        self.counts = Counter()
        self.peak_memory = 0
        # profile=True：每个节点、每个根的 Rank IC 和取回都单独同步计时（seconds 里的
        # "op:<算子>"、"rank_ic"、"readback"），会打断设备上的异步排队，只用于测量。
        self.profile = profile

    @property
    def device(self):
        return self.backend.device

    @property
    def kernel_id(self):
        return self.backend.kernel_id

    # -- 支持判定 --

    def unsupported(self, expr):
        """GPU 算不了的原因（None = 能算）。"""
        stack, seen = [expr], set()
        if expr.op == "constant":
            return "constant root"
        while stack:
            node = stack.pop()
            if node.digest in seen:
                continue
            seen.add(node.digest)
            params = dict(node.params)
            if node.op == "field":
                if params.get("dtype") != "float64" or params.get("broadcast"):
                    return f"field {params.get('value')}: dtype/broadcast"
                if self._field(params["value"]) is _FAILED:
                    return f"field {params['value']}: unavailable"
                continue
            if node.op == "constant":
                continue
            if not supported_node(node.op, params):
                return f"op {node.op}"
            if all(a.op == "constant" for a in node.args):
                return f"op {node.op}: constant operands"
            if (
                node.op.startswith(("ts_", "cs_")) and node.args[0].op == "constant"
            ) or (node.op == "ts_corr" and node.args[1].op == "constant"):
                return f"op {node.op}: constant operand"
            stack.extend(node.args)
        return None

    def _field(self, name):
        if name not in self._fields:
            t0 = time.perf_counter()
            try:
                raw = np.asarray(self.source[name])
                if (
                    raw.shape != (self.total_rows, self.shape[1])
                    or raw.dtype.kind not in "biuf"
                ):
                    raise ValueError(name)
                raw = raw[: self.rows]
                arr = self.backend.upload(np.asarray(raw, dtype=np.float64))
                flag = self.backend.range_flag(arr)
                self.backend.evaluate(arr, flag)
                self._fields[name] = (arr, flag)
                self.counts["fields"] += 1
            except Exception:  # noqa: BLE001 —— 取不到的字段：表达式交给 CPU，由 CPU 报错
                self._fields[name] = _FAILED
            self.seconds["upload"] += time.perf_counter() - t0
        return self._fields[name]

    # -- 求值 --

    def daily_ic(self, exprs):
        """每个表达式一个 `Outcome`（与输入同序）。"""
        t0 = time.perf_counter()
        out = [None] * len(exprs)
        todo = []
        for i, e in enumerate(exprs):
            reason = self.unsupported(e)
            if reason is None:
                todo.append(i)
            else:
                out[i] = Outcome(reason=f"unsupported: {reason}")
                self.counts["unsupported"] += 1
        t, n = self.shape
        panel = 4 * t * n
        limit = max(1, self.memory_limit // panel - RESERVED_PANELS - len(self._fields))
        roots = [exprs[i] for i in todo]
        start = 0
        for batch in _split(roots, limit):
            results = self._batch(batch)
            for k, res in enumerate(results):
                out[todo[start + k]] = res
            start += len(batch)
            self.counts["batches"] += 1
        self.seconds["gpu"] += time.perf_counter() - t0
        self.counts["screened"] += len(todo)
        return out

    def _plan(self, order, uses, want):
        """算子链融合的计划：哪些逐元素节点并进消费者的程序里（不单独落显存）。

        能并进去的：逐元素、不是根、只有一个消费者、消费者也是逐元素。一个程序引用的面板
        超过 `ELEM_INPUTS` 张时，把它的子节点退回单独计算。返回 (inline 集合, 每个要算的
        节点引用的叶子摘要列表)。共享子树（多个消费者）照旧只算一次。"""
        consumer = {}
        for node in order:
            for a in node.args:
                consumer[a.digest] = node
        inline = set()
        for node in order:
            d = node.digest
            if (node.op in _ELEM and d not in want and uses[d] == 1
                    and consumer.get(d) is not None and consumer[d].op in _ELEM):  # fmt: skip
                inline.add(d)
        panels = {}  # 逐元素节点 → 作为程序时引用的面板叶子（有序、去重）
        depth = {}  # 逐元素节点 → 后缀求值要的栈深
        for node in order:
            if node.op not in _ELEM:
                continue
            leaves = self._leaves(node, inline, panels)
            need = self._depth(node, inline, depth)
            if len(leaves) > ELEM_INPUTS or need > ELEM_STACK:
                for a in node.args:
                    inline.discard(a.digest)
                leaves = self._leaves(node, inline, panels)
                need = self._depth(node, inline, depth)
            panels[node.digest] = leaves
            depth[node.digest] = need
        refs = {}
        for node in order:
            if node.digest in inline or node.op in ("field", "constant"):
                continue
            if node.op in _ELEM:
                refs[node.digest] = panels[node.digest]
            else:
                refs[node.digest] = list(dict.fromkeys(a.digest for a in node.args))
        return inline, refs

    @staticmethod
    def _depth(node, inline, depth):
        """后缀程序的栈深：叶子 1；二元算子 max(左, 右 + 1)；一元同操作数。"""
        ds = [depth[a.digest] if a.digest in inline else 1 for a in node.args]
        return ds[0] if len(ds) == 1 else max(ds[0], ds[1] + 1)

    @staticmethod
    def _leaves(node, inline, panels):
        out = []
        for a in node.args:
            if a.op == "constant":
                continue
            out.extend(panels[a.digest] if a.digest in inline else [a.digest])
        return list(dict.fromkeys(out))

    def _program(self, node, inline, leaves):
        """node 的后缀程序（子节点里并进来的递归展开）：[(操作码, 参数)]、常量表。"""
        prog, consts = [], []
        slot = {d: k for k, d in enumerate(leaves)}

        def emit(n):
            if n.op == "constant":
                consts.append(float(dict(n.params)["value"]))
                prog.append((ELEM_OPS["const"], len(consts) - 1))
            elif n is not node and n.digest not in inline:
                prog.append((ELEM_OPS["load"], slot[n.digest]))
            else:
                for a in n.args:
                    emit(a)
                prog.append((ELEM_OPS[n.op], 0))

        emit(node)
        return prog, consts

    def _batch(self, roots):
        order, uses = _schedule(roots)
        for r in roots:
            uses[r.digest] += 1
        want = Counter(r.digest for r in roots)
        inline, refs = self._plan(order, uses, want)
        live = Counter(want)  # 每个落显存的节点还剩几个消费者（程序引用 + 作为根）
        for leaves in refs.values():
            for d in leaves:
                live[d] += 1
        memo, done = {}, {}
        for node in order:
            d = node.digest
            if d in inline:
                continue
            params = dict(node.params)
            if node.op == "field":
                val = self._fields[params["value"]]
            elif node.op == "constant":
                continue  # 只出现在逐元素程序里（常量根、常量操作数在支持判定里已交给 CPU）
            else:
                leaves = refs[d]
                args = [memo[x] for x in leaves]
                if any(a is _FAILED for a in args):
                    val = _FAILED
                else:
                    try:
                        val = self._compute(node, params, inline, leaves, args)
                    except Exception:  # noqa: BLE001 —— 这棵表达式交给 CPU
                        val = _FAILED
                for x in leaves:
                    live[x] -= 1
                    if live[x] == 0:
                        del memo[x]
            memo[d] = val
            if d in want and d not in done:
                done[d] = self._root(val)
                live[d] -= want[d]
                if live[d] == 0:
                    del memo[d]
        return [done[r.digest] for r in roots]

    def _compute(self, node, params, inline, leaves, args):
        be = self.backend
        t0 = time.perf_counter()
        if node.op in _ELEM:
            prog, consts = self._program(node, inline, leaves)
            value, flag = be.fuse(prog, [a[0] for a in args], consts, self.shape)
            name = "fused" if len([p for p in prog if p[0] > 1]) > 1 else node.op
            flag = be.any_flag([flag, *(a[1] for a in args)])
        else:
            by = {x: a for x, a in zip(leaves, args, strict=True)}
            value = be.compute(node.op, params, [by[a.digest][0] for a in node.args],
                               self.shape)  # fmt: skip
            name = node.op
            flag = be.any_flag([a[1] for a in args])
        if self.profile:
            be.evaluate(value, flag)
            self.seconds[f"op:{name}"] += time.perf_counter() - t0
            self.counts[f"op:{name}"] += 1
        return value, flag

    def _root(self, val):
        be = self.backend
        if val is _FAILED:
            self.counts["error"] += 1
            return Outcome(reason="gpu error")
        try:
            t0 = time.perf_counter()
            ic, noise = be.rank_ic(val[0], self.universe, self.returns)
            be.evaluate(ic, noise, val[1])
            if self.profile:
                self.seconds["rank_ic"] += time.perf_counter() - t0
            self.peak_memory = max(self.peak_memory, be.active_memory())
            if bool(val[1].item()):
                self.counts["range"] += 1
                return Outcome(reason="float32 range")
            share = float(noise.item())
            if share > DEGENERATE_FRACTION:
                self.counts["degenerate"] += 1
                return Outcome(reason="rounding ties", noise=share)
            t0 = time.perf_counter()
            host = be.to_host(ic)
            if self.rows < self.total_rows:
                host = np.concatenate(
                    [host, np.full(self.total_rows - self.rows, np.nan)]
                )
            if self.profile:
                self.seconds["readback"] += time.perf_counter() - t0
            return Outcome(ic=host, noise=share)
        except Exception:  # noqa: BLE001
            self.counts["error"] += 1
            return Outcome(reason="gpu error")

    def describe(self):
        return {"device": self.device, "kernel": self.kernel_id,
                "memory_limit": int(self.memory_limit)}  # fmt: skip


def pipeline(screener, items, decide, recheck, chunk=DEFAULT_CHUNK):
    """分块流水线：主线程在 GPU 上粗筛第 k+1 块时，后台一个线程在 CPU 上复算第 k 块。

    items：[(键, Expr)]；decide(键, Outcome) → True 表示送 CPU 复算；recheck(键列表) → {键: 结果}
    （在后台线程里跑，结果原样合并返回）。返回 (gpu_dropped, computed)：前者是没送复算的键
    列表（按输入顺序），后者是全部复算结果。复算出错照常抛出。"""
    dropped, futures = [], []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-recheck") as pool:
        for s in range(0, len(items), chunk):
            block = items[s : s + chunk]
            outcomes = screener.daily_ic([e for _, e in block])
            send = []
            for (key, _), oc in zip(block, outcomes, strict=True):
                (send if decide(key, oc) else dropped).append(key)
            if send:
                futures.append(pool.submit(recheck, send))
        computed = {}
        for f in futures:
            computed.update(f.result())
    return dropped, computed
