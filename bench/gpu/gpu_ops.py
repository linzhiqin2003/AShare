"""float32 GPU 算子与整代粗筛（探针，不是正式实现）。两个后端同一套算子代码：

- ``mlx``：Apple MLX（本机 Metal；Linux 上 ``mlx[cuda12]`` / ``mlx[cuda13]`` 走 CUDA，设备都叫 gpu）；
- ``torch``：PyTorch，设备 ``mps``（Mac）或 ``cuda``（NVIDIA），``cpu`` 只用来调试。

语义以 panelops（SPEC.md）为准，限于默认搜索空间用到的情形：
- 非有限值一律视为缺失；每个算子输出后把非有限值改成 NaN（与「作为下一个算子的输入时非有限即
  缺失」等价，elementwise 本来就把非有限结果写成 NaN）。
- 滑动窗口含今天，``min_periods`` 等于窗口（搜索空间不改它）：窗口内任何一格缺失，结果就缺失，
  所以直接让 NaN 在窗口求和里传播即可。窗口用 W 个行切片逐项相加（不用前缀和：float32 的
  前缀和在 2430 行上误差太大）。
- 常数窗口按「窗口内每个值与今天的值逐位相等」判定：std 恰为 0，zscore、corr 为缺失。
- 截面秩：平均秩（并列取平均），与 panelops/_numpy.py 的 rank 同一算法（排序 + 前后累积极值）。
- Rank IC：每天在因子与收益都有效的交集上分别求秩再算 Pearson；≥2 对、方差非零才有值。
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

# ---- 后端垫片 -----------------------------------------------------------------------


class MLX:
    name = "mlx"

    def __init__(self, device="gpu"):
        import mlx.core as mx

        self.mx = mx
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.device = device
        self.f32 = mx.float32

    def asarray(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.float32(np.nan))
        return self.mx.array(x)

    def asbool(self, x):
        return self.mx.array(np.asarray(x, dtype=bool))

    def numpy(self, a):
        return np.array(a, copy=True)

    def sync(self, *arrays):
        if arrays:
            self.mx.eval(*arrays)
        self.mx.synchronize()

    def nan_rows(self, k, n):
        return self.mx.full((k, n), np.nan, dtype=self.f32)

    def concat(self, parts):
        return self.mx.concatenate(parts, axis=0)

    def where(self, c, a, b):
        return self.mx.where(c, a, b)

    def isfinite(self, a):
        return self.mx.isfinite(a)

    def clean(self, a):
        return self.mx.where(self.mx.isfinite(a), a, np.nan)

    def f(self, b):
        return b.astype(self.f32)

    def sqrt(self, a):
        return self.mx.sqrt(a)

    def log(self, a):
        return self.mx.log(a)

    def abs(self, a):
        return self.mx.abs(a)

    def maximum(self, a, b):
        return self.mx.maximum(a, b)

    def minimum(self, a, b):
        return self.mx.minimum(a, b)

    def rowsum(self, a):
        return self.mx.sum(a, axis=-1, keepdims=True)

    def argsort(self, a):
        return self.mx.argsort(a, axis=-1)

    def take(self, a, idx):
        return self.mx.take_along_axis(a, idx, axis=-1)

    def scatter(self, idx, values):
        out = self.mx.zeros_like(values)
        return self.mx.put_along_axis(out, idx, values, axis=-1)

    def cummax(self, a):
        return self.mx.cummax(a, axis=-1)

    def rcummin(self, a):
        return self.mx.cummin(a, axis=-1, reverse=True)

    def arange_like(self, a):
        return self.mx.broadcast_to(
            self.mx.arange(a.shape[-1], dtype=self.f32), a.shape
        )

    def ones_col(self, a, value):
        return self.mx.full((*a.shape[:-1], 1), value, dtype=a.dtype)

    def cat_cols(self, parts):
        return self.mx.concatenate(parts, axis=-1)

    def stack(self, parts):
        return self.mx.stack(parts)

    def memory(self):
        return {
            "active": self.mx.get_active_memory(),
            "peak": self.mx.get_peak_memory(),
        }

    def reset_peak(self):
        self.mx.reset_peak_memory()

    def release(self):
        self.mx.clear_cache()


class Torch:
    name = "torch"

    def __init__(self, device="mps"):
        import torch

        self.t = torch
        self.dev = torch.device(device)
        self.device = device
        self.f32 = torch.float32
        torch.set_grad_enabled(False)

    def asarray(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.float32(np.nan))
        return self.t.from_numpy(np.ascontiguousarray(x)).to(self.dev)

    def asbool(self, x):
        return self.t.from_numpy(np.ascontiguousarray(x, dtype=bool)).to(self.dev)

    def numpy(self, a):
        return a.detach().to("cpu").numpy().copy()

    def sync(self, *arrays):
        if self.device == "mps":
            self.t.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.t.cuda.synchronize()

    def nan_rows(self, k, n):
        return self.t.full((k, n), float("nan"), dtype=self.f32, device=self.dev)

    def concat(self, parts):
        return self.t.cat(parts, dim=0)

    def where(self, c, a, b):
        t = self.t
        if not isinstance(a, t.Tensor):
            a = t.tensor(a, dtype=self.f32, device=self.dev)
        if not isinstance(b, t.Tensor):
            b = t.tensor(b, dtype=self.f32, device=self.dev)
        return t.where(c, a, b)

    def isfinite(self, a):
        return self.t.isfinite(a)

    def clean(self, a):
        return self.t.nan_to_num(a, nan=float("nan"), posinf=float("nan"),
                                 neginf=float("nan"))  # fmt: skip

    def f(self, b):
        return b.to(self.f32)

    def sqrt(self, a):
        return self.t.sqrt(a)

    def log(self, a):
        return self.t.log(a)

    def abs(self, a):
        return self.t.abs(a)

    def maximum(self, a, b):
        return self.t.maximum(a, b)

    def minimum(self, a, b):
        return self.t.minimum(a, b)

    def rowsum(self, a):
        return a.sum(dim=-1, keepdim=True)

    def argsort(self, a):
        return self.t.argsort(a, dim=-1)

    def take(self, a, idx):
        return self.t.gather(a, -1, idx)

    def scatter(self, idx, values):
        return self.t.empty_like(values).scatter_(-1, idx, values)

    def cummax(self, a):
        return self.t.cummax(a, dim=-1).values

    def rcummin(self, a):
        t = self.t
        return t.flip(t.cummin(t.flip(a, [-1]), dim=-1).values, [-1])

    def arange_like(self, a):
        return self.t.arange(a.shape[-1], dtype=self.f32, device=self.dev).expand(
            a.shape
        )

    def ones_col(self, a, value):
        return self.t.full((*a.shape[:-1], 1), value, dtype=a.dtype, device=self.dev)

    def cat_cols(self, parts):
        return self.t.cat(parts, dim=-1)

    def stack(self, parts):
        return self.t.stack(parts)

    def memory(self):
        t = self.t
        if self.device == "mps":
            return {"active": t.mps.current_allocated_memory(),
                    "driver": t.mps.driver_allocated_memory()}  # fmt: skip
        if self.device.startswith("cuda"):
            return {"active": t.cuda.memory_allocated(),
                    "peak": t.cuda.max_memory_allocated()}  # fmt: skip
        return {}

    def reset_peak(self):
        if self.device.startswith("cuda"):
            self.t.cuda.reset_peak_memory_stats()

    def release(self):
        if self.device == "mps":
            self.t.mps.empty_cache()
        elif self.device.startswith("cuda"):
            self.t.cuda.empty_cache()


def backend(name, device=None):
    if name == "mlx":
        return MLX(device or "gpu")
    if name == "torch":
        return Torch(device or "mps")
    if name == "mlx-fused":
        from fused import MLXFused

        return MLXFused(device or "gpu")
    if name == "torch-fused":
        from fused import TorchFused

        return TorchFused(device or "mps")
    raise ValueError(name)


# ---- 算子 -------------------------------------------------------------------------


def _views(x, w):
    """窗口内第 k 天前（k = 0 是今天）的值，只覆盖 t >= w-1 的行；形状 (T-w+1, N)。"""
    t = x.shape[0]
    return [x[w - 1 - k : t - k] for k in range(w)]


def _pad(xp, body, t):
    k = t - body.shape[0]
    return body if k == 0 else xp.concat([xp.nan_rows(k, body.shape[1]), body])


def _sum(vs):
    acc = vs[0]
    for v in vs[1:]:
        acc = acc + v
    return acc


def _flat(xp, vs):
    """窗口内每个值都等于今天的值（逐位）；有缺失时为 False。"""
    d = _sum([xp.abs(v - vs[0]) for v in vs[1:]]) if len(vs) > 1 else vs[0] * 0
    return d == 0


def ts(xp, op, x, w):
    t, n = x.shape
    if t < w:
        return xp.nan_rows(t, n)
    vs = _views(x, w)
    if op == "mean":
        out = _sum(vs) / w
    elif op == "sum":
        out = _sum(vs)
    elif op in ("std", "zscore"):
        m = _sum(vs) / w
        var = _sum([(v - m) * (v - m) for v in vs]) / (w - 1)
        sd = xp.sqrt(var)
        flat = _flat(xp, vs)
        if op == "std":
            out = xp.where(flat, vs[0] * 0, sd)
        else:
            out = xp.where(flat, np.nan, (vs[0] - m) / sd)
    elif op in ("max", "min"):
        f = xp.maximum if op == "max" else xp.minimum
        acc = vs[0]
        for v in vs[1:]:
            acc = f(acc, v)
        out = acc
    elif op == "rank":
        today = vs[0]
        less = _sum([xp.f(v < today) for v in vs])
        eq = _sum([xp.f(v == today) for v in vs])
        poison = _sum([v * 0 for v in vs])
        out = 1 + less + (eq - 1) / 2 + poison
    elif op == "decay_linear":
        out = _sum([v * float(w - k) for k, v in enumerate(vs)]) / (w * (w + 1) / 2)
    else:
        raise ValueError(op)
    return xp.clean(_pad(xp, out, t))


def ts_corr(xp, x, y, w):
    t, n = x.shape
    if t < w:
        return xp.nan_rows(t, n)
    vx, vy = _views(x, w), _views(y, w)
    poison = _sum([a * 0 + b * 0 for a, b in zip(vx, vy, strict=True)])
    vx = [a + poison for a in vx]  # 成对有效：任何一边缺失，两边都按缺失
    vy = [b + poison for b in vy]
    mx_, my = _sum(vx) / w, _sum(vy) / w
    dx = [a - mx_ for a in vx]
    dy = [b - my for b in vy]
    cov = _sum([a * b for a, b in zip(dx, dy, strict=True)])
    sx = xp.sqrt(_sum([a * a for a in dx]))
    sy = xp.sqrt(_sum([b * b for b in dy]))
    out = cov / sx / sy
    flat = _flat(xp, vx) | _flat(xp, vy)
    out = xp.where(flat, np.nan, out)
    return xp.clean(_pad(xp, out, t))


def ts_delay(xp, x, p):
    t, n = x.shape
    if p == 0:
        return x
    if p >= t:
        return xp.nan_rows(t, n)
    return xp.concat([xp.nan_rows(p, n), x[: t - p]])


def rank(xp, x):
    """逐行平均秩（一起算），非有限为缺失。x 可以是 (..., N)。"""
    valid = xp.isfinite(x)
    a = xp.where(valid, x, np.inf)
    order = xp.argsort(a)
    s = xp.take(a, order)
    pos = xp.arange_like(s)
    n = s.shape[-1]
    edge = xp.ones_col(s, 1.0) > 0
    new = xp.cat_cols([edge, s[..., 1:] != s[..., :-1]])  # 并列组的第一个
    end = xp.cat_cols([s[..., :-1] != s[..., 1:], edge])  # 并列组的最后一个
    start = xp.cummax(xp.where(new, pos, 0.0))
    stop = xp.rcummin(xp.where(end, pos, float(n)))
    r = xp.scatter(order, 1 + (start + stop) / 2)
    return xp.where(valid, r, np.nan)


def cs_zscore(xp, x):
    valid = xp.isfinite(x)
    cnt = xp.rowsum(xp.f(valid))
    z = xp.where(valid, x, 0.0)
    mean = xp.rowsum(z) / cnt
    d = x - mean
    ss = xp.rowsum(xp.where(valid, d * d, 0.0))
    return xp.clean(d / xp.sqrt(ss / cnt))


def rank_ic(xp, score, ret):
    """逐日 Rank IC，长度 T（score、ret 形状 (..., T, N)）。"""
    v = xp.isfinite(score) & xp.isfinite(ret)
    rs = rank(xp, xp.where(v, score, np.nan))
    rr = rank(xp, xp.where(v, ret, np.nan))
    n = xp.rowsum(xp.f(v))
    c = (n + 1) / 2  # 平均秩的和恒为 n(n+1)/2，均值恰为 (n+1)/2
    dx = xp.where(v, rs - c, 0.0)
    dy = xp.where(v, rr - c, 0.0)
    num = xp.rowsum(dx * dy)
    den = xp.sqrt(xp.rowsum(dx * dx)) * xp.sqrt(xp.rowsum(dy * dy))
    out = num / den
    out = xp.where(n >= 2, out, np.nan)
    return xp.clean(out)[..., 0]


ELEM = {"add", "subtract", "multiply", "divide", "negate", "abs", "log",
        "maximum", "minimum"}  # fmt: skip
TS1 = {"ts_mean": "mean", "ts_sum": "sum", "ts_std": "std", "ts_zscore": "zscore",
       "ts_max": "max", "ts_min": "min", "ts_rank": "rank",
       "ts_decay_linear": "decay_linear"}  # fmt: skip


def compute(xp, op, params, args):
    """一个节点；args 已经是设备上的 float32 面板（或标量）。"""
    if op in ELEM:
        a = args
        if op == "add":
            r = a[0] + a[1]
        elif op == "subtract":
            r = a[0] - a[1]
        elif op == "multiply":
            r = a[0] * a[1]
        elif op == "divide":
            r = a[0] / a[1]
        elif op == "negate":
            r = -a[0]
        elif op == "abs":
            r = xp.abs(a[0])
        elif op == "log":
            r = xp.log(a[0])
        elif op == "maximum":
            r = xp.maximum(a[0], a[1])
        else:
            r = xp.minimum(a[0], a[1])
        return xp.clean(r)
    if op in TS1:
        if params.get("min_periods") not in (None, params["window"]):
            raise NotImplementedError("min_periods != window")
        if hasattr(xp, "ts_fused"):
            return xp.ts_fused(TS1[op], args[0], params["window"])
        return ts(xp, TS1[op], args[0], params["window"])
    if op == "ts_corr":
        if hasattr(xp, "corr_fused"):
            return xp.corr_fused(args[0], args[1], params["window"])
        return ts_corr(xp, args[0], args[1], params["window"])
    if op == "ts_delay":
        return ts_delay(xp, args[0], params["periods"])
    if op == "ts_delta":
        return xp.clean(args[0] - ts_delay(xp, args[0], params["periods"]))
    if op == "cs_rank":
        if params.get("normalize"):
            raise NotImplementedError("normalize")
        return rank(xp, args[0])
    if op == "cs_zscore":
        return cs_zscore(xp, args[0])
    raise NotImplementedError(op)


# ---- 整代粗筛 ---------------------------------------------------------------------


class GpuScreen:
    """一次性把字段、股票池、h=5 标签放上设备；之后每代只传表达式、取回 T 个日 IC。"""

    def __init__(self, xp, panel, fields, universe, returns):
        t0 = time.perf_counter()
        self.xp = xp
        self.fields = {f: xp.asarray(panel[f]) for f in fields}
        self.universe = xp.asbool(universe)
        self.returns = xp.asarray(returns)
        xp.sync(*self.fields.values(), self.universe, self.returns)
        self.upload_seconds = time.perf_counter() - t0
        self.op_seconds = Counter()
        self.op_counts = Counter()
        self.peak_active = 0

    def run(self, exprs, *, per_op_timing=False, chunk=1):
        """按 `ashare.factor.engine._schedule` 的顺序逐节点求值（相同子树只算一次、最后一个消费者
        用完即释放），每个根直接在设备上做股票池遮蔽和 Rank IC；返回 (len(exprs), T) 的 float64。
        chunk > 1 时把这么多个根的 Rank IC 叠成一批一起排序。"""
        from ashare.factor.engine import _schedule

        xp = self.xp
        order, uses = _schedule(exprs)
        for e in exprs:
            uses[e.digest] += 1
        want = Counter(e.digest for e in exprs)
        memo, ics, pending = {}, {}, []

        def flush():
            if not pending:
                return
            t0 = time.perf_counter()
            if hasattr(xp, "rank_ic_fused"):
                out = [xp.rank_ic_fused(p[1], self.returns) for p in pending]
            elif len(pending) == 1:
                out = [rank_ic(xp, pending[0][1], self.returns)]
            else:
                stacked = rank_ic(xp, xp.stack([p[1] for p in pending]), self.returns)
                out = [stacked[i] for i in range(len(pending))]
            xp.sync(*out)
            for (d, _), ic in zip(pending, out, strict=True):
                ics[d] = ic
            pending.clear()
            if per_op_timing:
                self.op_seconds["rank_ic"] += time.perf_counter() - t0

        for node in order:
            p = dict(node.params)
            t0 = time.perf_counter()
            if node.op == "field":
                val = self.fields[p["value"]]
            elif node.op == "constant":
                val = float(p["value"])
            else:
                val = compute(xp, node.op, p, [memo[a.digest] for a in node.args])
                xp.sync(val)
                self.op_counts[node.op] += 1
                if per_op_timing:
                    self.op_seconds[node.op] += time.perf_counter() - t0
            memo[node.digest] = val
            for child in node.args:
                uses[child.digest] -= 1
                if uses[child.digest] == 0:
                    del memo[child.digest]
            if node.digest in want and node.digest not in ics:
                t1 = time.perf_counter()
                score = xp.where(self.universe & xp.isfinite(val), val, np.nan)
                pending.append((node.digest, score))
                if per_op_timing:
                    self.op_seconds["universe_mask"] += time.perf_counter() - t1
                if len(pending) >= chunk:
                    flush()
                uses[node.digest] -= want[node.digest]
                if uses[node.digest] == 0:
                    del memo[node.digest]
            mem = xp.memory().get("active", 0)
            self.peak_active = max(self.peak_active, mem)
        flush()
        rows = [ics[e.digest] for e in exprs]
        return np.asarray(xp.numpy(xp.stack(rows)), dtype=np.float64)
