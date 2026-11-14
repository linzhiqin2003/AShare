"""GPU 后端：接口与 MLX 实现（macOS 走 Metal，Linux 走 CUDA，内核正文同一份）。

后端只提供 float32 的设备数组和逐节点算子；整代求值、共享子树、按显存分批在 `screen` 里，
与具体框架无关。`probe()` 在运行时确认设备：没装 MLX、MLX 看不到 GPU、内核编译不过，都抛
`GpuUnavailable` 并说明原因，**不会静默退回 CPU**——用 CPU 冒充 GPU 粗筛既慢又让人误读
性能数字。

数值约定（与 panelops 的 float64 语义对齐，差分测试见 test/factor/gpu/）：

- 上传时非有限值一律写成 NaN；每个算子的输出也把非有限值写成 NaN（下一个算子的输入非有限即
  缺失，二者等价）；
- 取大、取小显式传播缺失（Metal 的 ``fmax`` 遇 NaN 返回另一边）；
- `range_flag` 给出「这个节点的值超出 float32 安全范围」的标记（有限值的绝对值大于
  `RANGE_HIGH` 或在 (0, `RANGE_LOW`) 内）。节点值都在安全范围内时，后续的乘除、平方和不会
  上溢或下溢，float32 与 float64 只差舍入；超出的整棵表达式交给 CPU。
"""

from __future__ import annotations

import abc
import sys

import numpy as np

from .kernels import (
    BODIES,
    CUDA_HEADER,
    ELEM_INPUTS,
    KERNEL_ID,
    NOISE_GAP,
    RANGE_HIGH,
    RANGE_LOW,
    RANK_IC_MAX_N,
    SIMD_WIDTH,
    TS_CODES,
    source,
)


class GpuUnavailable(RuntimeError):
    """要求用 GPU 粗筛，但这台机器上不可用（原因在消息里）。"""


class Backend(abc.ABC):
    """一个 GPU 后端：设备数组的上传与取回、逐节点算子、逐日 Rank IC。

    数组都是 (T, N) 的 float32（股票池是 bool）；算子的输入输出约定见模块说明。"""

    device: str  # 例如 "metal:Apple M5 Pro"、"cuda:NVIDIA H100"
    kernel_id: str = KERNEL_ID

    @abc.abstractmethod
    def upload(self, array):
        """float64 / bool 主机数组 → 设备数组（浮点非有限写成 NaN）。"""

    @abc.abstractmethod
    def to_host(self, array):
        """设备数组 → float64 numpy。"""

    @abc.abstractmethod
    def compute(self, op, params, args, shape):
        """一个算子节点；args 是设备数组或 Python 标量。不支持时抛 NotImplementedError。"""

    @abc.abstractmethod
    def rank_ic(self, values, universe, returns):
        """(逐日 Rank IC, 舍入并列比例)，都是设备数组。股票池外和非有限的格子不参与（与
        `FactorEvaluator._scores` 同口径）；同一份 returns 的排序只做一次。

        舍入并列比例：每天把与收益成对有效的因子值排序，相邻两个值之差非零、却不超过
        `NOISE_GAP` 倍其绝对值的（float32 舍入量级的差），占全部非零差的比例（全面板）。
        比例高说明排序主要由舍入决定（数学上恒为常数或大量相等的表达式，例如
        ``ts_corr(low, -low)``、``ts_std(ts_rank(x))``），这时 float32 与 float64 的排序都是
        噪声、谁也代表不了谁，交给 CPU。"""

    @abc.abstractmethod
    def range_flag(self, values):
        """0 维布尔设备数组：values 里有没有超出 float32 安全范围的有限值。"""

    @abc.abstractmethod
    def any_flag(self, flags):
        """若干个 `range_flag` 的「或」（惰性）。"""

    @abc.abstractmethod
    def evaluate(self, *arrays):
        """把这些数组算出来（等待设备完成）。"""

    @abc.abstractmethod
    def memory_budget(self):
        """默认可用于粗筛的设备内存字节数。"""

    @abc.abstractmethod
    def active_memory(self):
        """当前设备内存占用（字节）。"""

    def release(self):  # noqa: B027 —— 可选钩子，默认什么都不做
        """释放缓存的设备内存（可选）。"""


# ---- MLX ---------------------------------------------------------------------------

_ELEMENTWISE = {"add", "subtract", "multiply", "divide", "negate", "abs", "sign", "log",
                "maximum", "minimum"}  # fmt: skip
_TS = {"ts_mean": "mean", "ts_sum": "sum", "ts_std": "std", "ts_zscore": "zscore",
       "ts_max": "max", "ts_min": "min", "ts_rank": "rank",
       "ts_decay_linear": "decay_linear", "ts_skew": "skew", "ts_slope": "slope",
       "ts_argmax": "argmax", "ts_argmin": "argmin"}  # fmt: skip
_CS_PARAMS = frozenset({"normalize", "ddof", "lower", "upper", "scale", "rcond"})
# 每个算子允许的参数（规范化以后的参数名）；min_periods 必须等于 window（ts_ema 除外）。
SUPPORTED = {
    **{op: frozenset() for op in _ELEMENTWISE},
    "ts_mean": frozenset({"window", "min_periods", "exact"}),
    "ts_sum": frozenset({"window", "min_periods", "exact"}),
    "ts_std": frozenset({"window", "min_periods", "ddof"}),
    "ts_zscore": frozenset({"window", "min_periods", "ddof"}),
    "ts_max": frozenset({"window", "min_periods"}),
    "ts_min": frozenset({"window", "min_periods"}),
    "ts_rank": frozenset({"window", "min_periods", "normalize"}),
    "ts_decay_linear": frozenset({"window", "min_periods"}),
    "ts_skew": frozenset({"window", "min_periods"}),
    "ts_slope": frozenset({"window", "min_periods"}),
    "ts_argmax": frozenset({"window", "min_periods", "ties"}),
    "ts_argmin": frozenset({"window", "min_periods", "ties"}),
    "ts_corr": frozenset({"window", "min_periods"}),
    "ts_ema": frozenset({"window", "min_periods", "alpha"}),
    "ts_delay": frozenset({"periods"}),
    "ts_delta": frozenset({"periods"}),
    # panelops 的截面算子共用一个签名，规范化后的参数里带着与本算子无关的 lower、upper、
    # scale、rcond（rank 不看 ddof，zscore 不看 normalize）；只有 normalize（rank）和
    # ddof（zscore）改变结果。
    "cs_rank": _CS_PARAMS,
    "cs_zscore": _CS_PARAMS,
}


def supported_node(op, params):
    """这个节点 GPU 能不能算（只看算子和参数；字段、常量另判）。"""
    allowed = SUPPORTED.get(op)
    if allowed is None or not set(params) <= allowed:
        return False
    if op.startswith("ts_") and "window" in params and op != "ts_ema":
        if params.get("min_periods", params["window"]) != params["window"]:
            return False
    return True


def probe():
    """确认 GPU 可用，返回 (mlx.core 模块, 平台 "metal" 或 "cuda", 设备名)；否则抛 GpuUnavailable。"""
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise GpuUnavailable(
            "没装 MLX：GPU 粗筛要可选依赖 gpu（uv sync --extra gpu；Linux 上装 mlx[cuda12]）"
        ) from exc
    if sys.platform == "darwin":
        platform = "metal"
        ok = mx.metal.is_available()
    elif sys.platform.startswith("linux"):
        platform = "cuda"
        ok = hasattr(mx, "cuda") and mx.cuda.is_available()
    else:
        raise GpuUnavailable(
            f"不支持的平台 {sys.platform}：GPU 粗筛只支持 macOS（Metal）和 Linux（CUDA）"
        )
    if not ok or not mx.is_available(mx.gpu):
        raise GpuUnavailable(
            f"MLX {mx.__version__} 在这台机器上看不到 GPU（{platform}）："
            + ("需要 Apple 芯片" if platform == "metal" else
               "需要 NVIDIA 驱动 >= 550、算力 >= SM 7.5，并安装 mlx[cuda12]")
        )  # fmt: skip
    try:
        info = mx.device_info()
    except Exception:  # noqa: BLE001 —— 设备名只是展示用
        info = {}
    return mx, platform, str(info.get("device_name", "gpu"))


class MLXBackend(Backend):
    """MLX + 自定义内核。所有计算显式放在 GPU 流上，不依赖进程的默认设备。"""

    name = "mlx"

    def __init__(self):
        mx, platform, device_name = probe()
        self.mx = mx
        self.platform = platform
        self.device = f"{platform}:{device_name}"
        self.stream = mx.gpu
        make = mx.fast.cuda_kernel if platform == "cuda" else mx.fast.metal_kernel
        self._kernels = {}
        try:
            for name, (body, inputs, outputs) in BODIES.items():
                extra = {"header": CUDA_HEADER} if platform == "cuda" else {}
                self._kernels[name] = make(name=name, input_names=inputs,
                                           output_names=outputs,
                                           source=source(body, platform), **extra)  # fmt: skip
            self._smoke()
        except GpuUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 编译或首次运行失败
            raise GpuUnavailable(
                f"GPU 内核在 {self.device} 上编译或运行失败：{type(exc).__name__}: {exc}"
            ) from exc
        self._ret = None  # (returns 数组, 排序状态)

    # -- 基础 --

    def _smoke(self):
        x = self.upload(np.arange(12, dtype=np.float64).reshape(4, 3))
        out = self.to_host(
            self.compute("ts_mean", {"window": 2, "min_periods": 2}, [x], (4, 3))
        )
        want = np.vstack([np.full(3, np.nan), (np.arange(9).reshape(3, 3) * 2 + 3) / 2])
        if not np.array_equal(out, want, equal_nan=True):
            raise GpuUnavailable(f"GPU 内核在 {self.device} 上的自检结果不对")

    def upload(self, array):
        mx = self.mx
        a = np.asarray(array)
        if a.dtype == bool:
            return mx.array(np.ascontiguousarray(a), dtype=mx.bool_)
        a = np.asarray(a, dtype=np.float32)
        a = np.where(np.isfinite(a), a, np.float32(np.nan))
        return mx.array(np.ascontiguousarray(a))

    def to_host(self, array):
        self.mx.eval(array)
        return np.array(array, dtype=np.float64, copy=True)

    def evaluate(self, *arrays):
        if arrays:
            self.mx.eval(*arrays)

    def memory_budget(self):
        try:
            info = self.mx.device_info()
        except Exception:  # noqa: BLE001
            info = {}
        total = info.get("max_recommended_working_set_size") or info.get("memory_size")
        return int(total) // 2 if total else 4 * 2**30

    def active_memory(self):
        return int(self.mx.get_active_memory())

    def release(self):
        self.mx.clear_cache()

    def _clean(self, a):
        mx = self.mx
        return mx.where(mx.isfinite(a), a, np.nan, stream=self.stream)

    def _nan(self, shape):
        return self.mx.full(shape, np.nan, dtype=self.mx.float32, stream=self.stream)

    def _launch(self, name, inputs, width, template, shape, grid=None, group=256):
        g = grid or ((width + group - 1) // group * group, 1, 1)
        return self._kernels[name](inputs=inputs, template=template, grid=g,
                                   threadgroup=(group, 1, 1), output_shapes=[shape],
                                   output_dtypes=[self.mx.float32],
                                   stream=self.stream)[0]  # fmt: skip

    # -- 算子 --

    def compute(self, op, params, args, shape):
        mx, s = self.mx, self.stream
        if not supported_node(op, params):
            raise NotImplementedError(op)
        if op in _ELEMENTWISE:
            a = args
            if op == "add":
                r = mx.add(a[0], a[1], stream=s)
            elif op == "subtract":
                r = mx.subtract(a[0], a[1], stream=s)
            elif op == "multiply":
                r = mx.multiply(a[0], a[1], stream=s)
            elif op == "divide":
                r = mx.divide(a[0], a[1], stream=s)
            elif op == "negate":
                r = mx.negative(a[0], stream=s)
            elif op == "abs":
                r = mx.abs(a[0], stream=s)
            elif op == "sign":  # MLX 的 sign(NaN) 是 0，显式保留缺失
                r = mx.where(mx.isnan(a[0], stream=s), np.nan, mx.sign(a[0], stream=s),
                             stream=s)  # fmt: skip
            elif op == "log":
                r = mx.log(a[0], stream=s)
            else:
                f = mx.maximum if op == "maximum" else mx.minimum
                r = f(a[0], a[1], stream=s)
                for x in a:
                    if isinstance(x, mx.array):
                        r = mx.where(mx.isnan(x, stream=s), np.nan, r, stream=s)
            return self._clean(r)
        x = args[0]
        if not isinstance(x, self.mx.array):
            raise NotImplementedError(f"{op} 的输入是常量")
        t, n = shape
        if op in _TS:
            w = int(params["window"])
            if t < w:
                return self._nan(shape)
            tmpl = [("W", w), ("T", t), ("N", n), ("OP", TS_CODES[_TS[op]]),
                    ("DDOF", int(params.get("ddof", 1))),
                    ("NORM", int(bool(params.get("normalize", False)))),
                    ("TIES", 1 if params.get("ties") == "earliest" else 0)]  # fmt: skip
            return self._launch("ash_ts", [x], t * n, tmpl, shape)
        if op == "ts_corr":
            y = args[1]
            if not isinstance(y, mx.array):
                raise NotImplementedError("ts_corr 的输入是常量")
            w = int(params["window"])
            if t < w:
                return self._nan(shape)
            return self._launch("ash_corr", [x, y], t * n, [("W", w), ("T", t), ("N", n)],
                                shape)  # fmt: skip
        if op == "ts_ema":
            w = int(params["window"])
            alpha = params.get("alpha")
            alpha = 2.0 / (w + 1) if alpha is None else float(alpha)
            mp = int(params.get("min_periods") or w)
            al = mx.array(np.array([alpha], dtype=np.float32))
            return self._launch(
                "ash_ema", [x, al], n, [("T", t), ("N", n), ("MP", mp)], shape
            )
        if op in ("ts_delay", "ts_delta"):
            p = int(params["periods"])
            if p >= t:
                return self._nan(shape)
            tmpl = [("W", p + 1), ("T", t), ("N", n), ("OP", TS_CODES[op[3:]]), ("DDOF", 0),
                    ("NORM", 0), ("TIES", 0)]  # fmt: skip
            return self._launch("ash_ts", [x], t * n, tmpl, shape)
        if op == "cs_rank":
            r = self._rank_rows(x)
            if params.get("normalize"):
                cnt = mx.sum(mx.isfinite(x, stream=s), axis=-1, keepdims=True, stream=s)
                cnt = cnt.astype(mx.float32)
                r = mx.where(cnt == 1, 0.5, (r - 1) / (cnt - 1), stream=s)
                r = mx.where(mx.isfinite(x, stream=s), r, np.nan, stream=s)
            return r
        if op == "cs_zscore":
            ddof = int(params.get("ddof", 0))
            valid = mx.isfinite(x, stream=s)
            cnt = mx.sum(valid, axis=-1, keepdims=True, stream=s).astype(mx.float32)
            z = mx.where(valid, x, 0.0, stream=s)
            mean = mx.sum(z, axis=-1, keepdims=True, stream=s) / cnt
            d = x - mean
            ss = mx.sum(mx.where(valid, d * d, 0.0, stream=s), axis=-1, keepdims=True,
                        stream=s)  # fmt: skip
            sd = mx.sqrt(ss / (cnt - ddof), stream=s)
            # 常数行按逐位相等判定（float32 的均值不一定逐位等于那个值，不能靠 ss == 0）
            hi = mx.max(
                mx.where(valid, x, -np.inf, stream=s), axis=-1, keepdims=True, stream=s
            )
            lo = mx.min(
                mx.where(valid, x, np.inf, stream=s), axis=-1, keepdims=True, stream=s
            )
            sd = mx.where((cnt > ddof) & (hi != lo), sd, np.nan, stream=s)
            return self._clean(d / sd)
        raise NotImplementedError(op)

    def fuse(self, program, panels, consts, shape):
        """逐元素算子链（`kernels.ELEM_OPS` 的后缀程序）一个内核算完：(值, 范围标记)。

        panels 至多 `ELEM_INPUTS` 张设备面板，consts 为程序里引用的常量。"""
        mx, s = self.mx, self.stream
        if not panels or len(panels) > ELEM_INPUTS:
            raise NotImplementedError("逐元素链的面板输入个数")
        size = int(np.prod(shape))
        xs = list(panels) + [panels[0]] * (ELEM_INPUTS - len(panels))
        prog = mx.array(np.asarray(program, dtype=np.int32).reshape(-1))
        cs = mx.array(np.asarray(list(consts) or [0.0], dtype=np.float32))
        flags = (size + 31) // 32
        group = 256
        grid = (size + group - 1) // group * group
        out, fl = self._kernels["ash_elem"](
            inputs=[*xs, prog, cs], template=[("SIZE", size), ("L", len(program)),
                                              ("FLAGS", flags)],
            grid=(grid, 1, 1), threadgroup=(group, 1, 1),
            output_shapes=[shape, (flags,)], output_dtypes=[mx.float32, mx.float32],
            stream=s)  # fmt: skip
        return out, mx.max(fl, stream=s) > 0

    def _rank_rows(self, x):
        """逐行平均秩：一次 argsort + 每格二分求并列组（自定义内核）。"""
        mx, s = self.mx, self.stream
        t, n = x.shape
        key = mx.where(mx.isfinite(x, stream=s), x, np.inf, stream=s)
        order = mx.argsort(key, axis=-1, stream=s)
        srt = mx.take_along_axis(key, order, axis=-1, stream=s)
        return self._launch("ash_rank", [srt, order.astype(mx.uint32, stream=s)], t * n,
                            [("T", t), ("N", n)], (t, n))  # fmt: skip

    def mask(self, values, universe):
        mx, s = self.mx, self.stream
        if not isinstance(values, mx.array):
            raise NotImplementedError("根是常量")
        return mx.where(
            universe & mx.isfinite(values, stream=s), values, np.nan, stream=s
        )

    def range_flag(self, values):
        mx, s = self.mx, self.stream
        if not isinstance(values, mx.array):
            v = float(values)
            bad = np.isfinite(v) and (abs(v) > RANGE_HIGH or 0 < abs(v) < RANGE_LOW)
            return mx.array(bool(bad))
        a = mx.abs(values, stream=s)
        bad = (a > RANGE_HIGH) | ((a < RANGE_LOW) & (a > 0))
        return mx.any(mx.isfinite(values, stream=s) & bad, stream=s)

    def any_flag(self, flags):
        mx = self.mx
        out = None
        for f in flags:
            out = f if out is None else mx.logical_or(out, f, stream=self.stream)
        return out if out is not None else mx.array(False)

    # -- Rank IC --

    def _returns(self, ret):
        """收益的逐日排序、并列组起止、有效标记：每份收益面板只算一次。"""
        if self._ret is not None and self._ret[0] is ret:
            return self._ret[1]
        mx, s = self.mx, self.stream
        vr = mx.isfinite(ret, stream=s)
        key = mx.where(vr, ret, np.inf, stream=s)
        order = mx.argsort(key, axis=-1, stream=s)
        srt = mx.take_along_axis(key, order, axis=-1, stream=s)
        t, n = srt.shape
        pos = mx.broadcast_to(
            mx.arange(n, dtype=mx.float32, stream=s), srt.shape, stream=s
        )
        edge = mx.full((t, 1), True, stream=s)
        new = mx.concatenate([edge, srt[:, 1:] != srt[:, :-1]], axis=-1, stream=s)
        end = mx.concatenate([srt[:, :-1] != srt[:, 1:], edge], axis=-1, stream=s)
        gs = mx.cummax(mx.where(new, pos, 0.0, stream=s), axis=-1, stream=s)
        ge = mx.cummin(mx.where(end, pos, float(n), stream=s), axis=-1, reverse=True,
                       stream=s)  # fmt: skip
        state = {
            "vr": vr,
            "order": order.astype(mx.uint32, stream=s),
            "order_i": order.astype(mx.int32, stream=s),
            "gs": gs.astype(mx.uint32, stream=s),
            "ge": ge.astype(mx.uint32, stream=s),
            "valid_sorted": mx.take_along_axis(vr, order, axis=-1, stream=s),
        }
        if n <= RANK_IC_MAX_N:
            # 单线程组内核用：每格在当天收益排序里的位置（逆排列，收益无效记全 1）与所在
            # 并列组起止（高 16 位起、低 16 位止）
            posr = mx.argsort(order, axis=-1, stream=s)
            packed = (state["gs"] << 16) | state["ge"]
            state["gsge"] = mx.take_along_axis(packed, posr.astype(mx.int32, stream=s),
                                               axis=-1, stream=s)  # fmt: skip
            state["posr"] = mx.where(vr, posr.astype(mx.uint32, stream=s),
                                     mx.array(0xFFFFFFFF, dtype=mx.uint32), stream=s)  # fmt: skip
            del state["order"], state["order_i"], state["gs"], state["ge"]
            del state["valid_sorted"], state["vr"]
        mx.eval(*state.values())
        self._ret = (ret, state)
        return state

    def rank_ic(self, values, universe, returns):
        mx, s = self.mx, self.stream
        if not isinstance(values, mx.array):
            raise NotImplementedError("根是常量")
        st = self._returns(returns)
        t, n = values.shape
        if n <= RANK_IC_MAX_N:
            np2 = max(32, 1 << (n - 1).bit_length())
            tg = min(np2, 1024)
            tmpl = [("T", t), ("N", n), ("NP", np2), ("TG", tg), ("E", np2 // tg),
                    ("NSG", tg // 32)]  # fmt: skip
            ic, noise = self._kernels["ash_rank_ic"](
                inputs=[values, universe, st["posr"], st["gsge"]], template=tmpl,
                grid=(t * tg, 1, 1), threadgroup=(tg, 1, 1),
                output_shapes=[(t,), (t, 2)], output_dtypes=[mx.float32, mx.float32],
                stream=s)  # fmt: skip
            tot = mx.sum(noise, axis=0, stream=s)
            return ic, tot[0] / mx.maximum(tot[1], 1.0, stream=s)
        score = self.mask(values, universe)
        vs = mx.isfinite(score, stream=s)
        v = vs & st["vr"]
        key = mx.where(v, score, np.inf, stream=s)
        order_s = mx.argsort(key, axis=-1, stream=s)
        srt = mx.take_along_axis(key, order_s, axis=-1, stream=s)
        rs = self._launch("ash_rank", [srt, order_s.astype(mx.uint32, stream=s)], t * n,
                          [("T", t), ("N", n)], (t, n))  # fmt: skip
        lo, hi = srt[:, :-1], srt[:, 1:]
        gap = mx.subtract(hi, lo, stream=s)
        both = mx.isfinite(lo, stream=s) & mx.isfinite(hi, stream=s)
        nonzero = both & (gap > 0)
        scale = mx.maximum(mx.abs(lo, stream=s), mx.abs(hi, stream=s), stream=s)
        tiny = nonzero & (gap <= NOISE_GAP * scale)
        noise = mx.sum(tiny, stream=s).astype(mx.float32) / mx.maximum(
            mx.sum(nonzero, stream=s).astype(mx.float32), 1.0, stream=s
        )
        m = (
            mx.take_along_axis(vs, st["order_i"], axis=-1, stream=s)
            & st["valid_sorted"]
        )
        m = m.astype(mx.float32, stream=s)
        cum = mx.cumsum(m, axis=-1, stream=s)
        rr = self._launch("ash_ret_rank", [st["order"], m, st["gs"], st["ge"], cum], t * n,
                          [("T", t), ("N", n)], (t, n))  # fmt: skip
        ic = self._launch("ash_pearson", [rs, rr], t, [("T", t), ("N", n)], (t,),
                          grid=(t * SIMD_WIDTH, 1, 1), group=SIMD_WIDTH)  # fmt: skip
        return ic, noise


def default_backend():
    """这台机器上的 GPU 后端（目前只有 MLX）；不可用时抛 GpuUnavailable。"""
    return MLXBackend()
