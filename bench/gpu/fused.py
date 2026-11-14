"""融合内核版本：滑动窗口一族和逐日 Rank IC 各写成一个自定义内核（一趟读写），其余算子沿用 gpu_ops。

同一份内核正文同时给 Metal（``mx.fast.metal_kernel``）和 CUDA（``mx.fast.cuda_kernel``）用：
两者对这种「每个线程一条序列、循环读写」的写法语法几乎一样，差别只在线程下标和 NaN/isfinite 的写法，
由 `_prelude` 补上。本机只验证了 Metal；CUDA 版本的正文由 `--backend mlx-fused` 在 Linux 上自动选用，
需要在 NVIDIA 机器上跑 check_ops / gpu_probe 验证。

语义与 gpu_ops 相同（窗口内任一缺失即缺失、常数窗口逐位判定、平均秩），求和顺序也相同
（今天、昨天……依次相加），所以与切片写法逐位一致；差分检查见 check_ops.py。
"""

from __future__ import annotations

import sys

import gpu_ops
import numpy as np

TS_CODES = {"mean": 0, "sum": 1, "std": 2, "zscore": 3, "max": 4, "min": 5,
            "rank": 6, "decay_linear": 7}  # fmt: skip

_TS_BODY = """
    int col = COL_INDEX;
    if (col >= N) return;
    for (int t = 0; t < T; ++t) {
        float res = NAN_F;
        if (t >= W - 1) {
            float v0 = x[t * N + col];
            float acc = 0.0f;
            for (int k = 0; k < W; ++k) acc += x[(t - k) * N + col];
            if (IS_FINITE(acc)) {
                if (OP == 0) res = acc / W;
                else if (OP == 1) res = acc;
                else if (OP == 2 || OP == 3) {
                    float m = acc / W;
                    float ss = 0.0f;
                    bool flat = true;
                    for (int k = 0; k < W; ++k) {
                        float v = x[(t - k) * N + col];
                        float d = v - m;
                        ss += d * d;
                        flat = flat && (v == v0);
                    }
                    float sd = sqrt(ss / (W - 1));
                    if (OP == 2) res = flat ? 0.0f : sd;
                    else res = flat ? NAN_F : (v0 - m) / sd;
                } else if (OP == 4 || OP == 5) {
                    float e = v0;
                    for (int k = 1; k < W; ++k) {
                        float v = x[(t - k) * N + col];
                        e = (OP == 4) ? (v > e ? v : e) : (v < e ? v : e);
                    }
                    res = e;
                } else if (OP == 6) {
                    float less = 0.0f, eq = 0.0f;
                    for (int k = 0; k < W; ++k) {
                        float v = x[(t - k) * N + col];
                        less += (v < v0) ? 1.0f : 0.0f;
                        eq += (v == v0) ? 1.0f : 0.0f;
                    }
                    res = 1.0f + less + (eq - 1.0f) / 2.0f;
                } else if (OP == 7) {
                    float s = 0.0f;
                    for (int k = 0; k < W; ++k) s += x[(t - k) * N + col] * (float)(W - k);
                    res = s / (W * (W + 1) / 2.0f);
                }
            }
        }
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
    }
"""

_CORR_BODY = """
    int col = COL_INDEX;
    if (col >= N) return;
    for (int t = 0; t < T; ++t) {
        float res = NAN_F;
        if (t >= W - 1) {
            float sx = 0.0f, sy = 0.0f;
            for (int k = 0; k < W; ++k) {
                float a = x[(t - k) * N + col], b = y[(t - k) * N + col];
                sx += a; sy += b;
            }
            if (IS_FINITE(sx) && IS_FINITE(sy)) {
                float mx_ = sx / W, my = sy / W;
                float cxy = 0.0f, cxx = 0.0f, cyy = 0.0f;
                float x0 = x[t * N + col], y0 = y[t * N + col];
                bool fx = true, fy = true;
                for (int k = 0; k < W; ++k) {
                    float a = x[(t - k) * N + col], b = y[(t - k) * N + col];
                    float dx = a - mx_, dy = b - my;
                    cxy += dx * dy; cxx += dx * dx; cyy += dy * dy;
                    fx = fx && (a == x0); fy = fy && (b == y0);
                }
                res = (fx || fy) ? NAN_F : cxy / sqrt(cxx) / sqrt(cyy);
            }
        }
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
    }
"""

# 排好序以后求平均秩：每个格子一个线程，在本行的排序里二分找并列组的起止（无效值当 +inf 排在最后），
# 把秩写回原来的列。读写都按排序位置并行，代价与并列组大小无关。
_RANK_BODY = """
    int g = COL_INDEX;
    if (g >= T * N) return;
    int row = (g / N) * N;
    int j = (int)order[g];
    float key = srt[g];
    if (!IS_FINITE(key)) { out[row + j] = NAN_F; return; }
    int lo = 0, hi = N;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (srt[row + m] < key) lo = m + 1; else hi = m;
    }
    int a = lo;
    hi = N;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (srt[row + m] <= key) lo = m + 1; else hi = m;
    }
    out[row + j] = (float)(a + lo + 1) / 2.0f;
"""

# 收益只排序一次：并列组起止（gs、ge，按排序位置）也只算一次。每个因子只需「交集有效」在收益排序里的
# 累计个数 cum，平均秩 = cum[gs-1] + (组内有效数 + 1) / 2，与在交集上重新排序的结果相同。
_RET_RANK_BODY = """
    int g = COL_INDEX;
    if (g >= T * N) return;
    int row = (g / N) * N;
    int j = (int)order[g];
    if (m[g] == 0.0f) { out[row + j] = NAN_F; return; }
    int a = (int)gs[g], b = (int)ge[g];
    float before = (a > 0) ? cum[row + a - 1] : 0.0f;
    float inside = cum[row + b] - before;
    out[row + j] = before + (inside + 1.0f) / 2.0f;
"""


# 逐日 Pearson：一个 SIMD 组（32 个线程）一天，先数交集个数，再按 (n+1)/2 居中累加，组内规约。
_PEARSON_BODY = """
    int t = ROW_ID;
    int lane = LANE_ID;
    if (t >= T) return;
    int row = t * N;
    float cnt = 0.0f;
    for (int j = lane; j < N; j += 32)
        cnt += (IS_FINITE(rs[row + j]) && IS_FINITE(rr[row + j])) ? 1.0f : 0.0f;
    cnt = SIMD_SUM(cnt);
    float c = (cnt + 1.0f) / 2.0f;
    float sxy = 0.0f, sxx = 0.0f, syy = 0.0f;
    for (int j = lane; j < N; j += 32) {
        float a = rs[row + j], b = rr[row + j];
        if (IS_FINITE(a) && IS_FINITE(b)) {
            float dx = a - c, dy = b - c;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
    }
    sxy = SIMD_SUM(sxy); sxx = SIMD_SUM(sxx); syy = SIMD_SUM(syy);
    if (lane == 0) {
        float res = (cnt >= 2.0f) ? sxy / sqrt(sxx) / sqrt(syy) : NAN_F;
        ic[t] = IS_FINITE(res) ? res : NAN_F;
    }
"""

_CUDA_HEADER = """
__device__ inline float ash_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return __shfl_sync(0xffffffff, v, 0);
}
"""


def _prelude(body, cuda):
    if cuda:
        rep = {"COL_INDEX": "(int)(blockIdx.x * blockDim.x + threadIdx.x)",
               "ROW_ID": "(int)blockIdx.x", "LANE_ID": "(int)threadIdx.x",
               "SIMD_SUM": "ash_warp_sum", "IS_FINITE": "isfinite",
               "NAN_F": "__int_as_float(0x7fc00000)"}  # fmt: skip
    else:
        rep = {"COL_INDEX": "(int)thread_position_in_grid.x",
               "ROW_ID": "(int)threadgroup_position_in_grid.x",
               "LANE_ID": "(int)thread_index_in_simdgroup", "SIMD_SUM": "metal::simd_sum",
               "IS_FINITE": "metal::isfinite", "NAN_F": "NAN"}  # fmt: skip
    for k, v in rep.items():
        body = body.replace(k, v)
    return body


class MLXFused(gpu_ops.MLX):
    """MLX + 自定义内核（Metal 或 CUDA，按平台选）。"""

    name = "mlx-fused"

    def __init__(self, device="gpu"):
        super().__init__(device)
        mx = self.mx
        self.cuda = sys.platform.startswith("linux")
        make = mx.fast.cuda_kernel if self.cuda else mx.fast.metal_kernel
        self._ts = make(name="ash_ts", input_names=["x"], output_names=["out"],
                        source=_prelude(_TS_BODY, self.cuda))  # fmt: skip
        self._corr = make(name="ash_corr", input_names=["x", "y"], output_names=["out"],
                          source=_prelude(_CORR_BODY, self.cuda))  # fmt: skip
        self._rank = make(name="ash_rank", input_names=["srt", "order"],
                          output_names=["out"], source=_prelude(_RANK_BODY, self.cuda))  # fmt: skip
        self._ret_rank = make(name="ash_ret_rank",
                              input_names=["order", "m", "gs", "ge", "cum"],
                              output_names=["out"],
                              source=_prelude(_RET_RANK_BODY, self.cuda))  # fmt: skip
        extra = {"header": _CUDA_HEADER} if self.cuda else {}
        self._pearson = make(name="ash_pearson", input_names=["rs", "rr"],
                             output_names=["ic"], source=_prelude(_PEARSON_BODY, self.cuda),
                             **extra)  # fmt: skip
        self._ret = None

    def _launch(self, kernel, inputs, width, template, outputs):
        tg = 256
        grid = ((width + tg - 1) // tg * tg, 1, 1)
        return kernel(inputs=inputs, template=template, grid=grid, threadgroup=(tg, 1, 1),
                      output_shapes=[o[0] for o in outputs],
                      output_dtypes=[o[1] for o in outputs])  # fmt: skip

    def ts_fused(self, op, x, w):
        t, n = x.shape
        return self._launch(self._ts, [x], n,
                            [("W", w), ("T", t), ("N", n), ("OP", TS_CODES[op])],
                            [((t, n), self.f32)])[0]  # fmt: skip

    def corr_fused(self, x, y, w):
        t, n = x.shape
        return self._launch(self._corr, [x, y], n, [("W", w), ("T", t), ("N", n)],
                            [((t, n), self.f32)])[0]  # fmt: skip

    def _returns(self, ret):
        """收益的逐日排序、并列组起止、有效标记：每份收益面板只算一次。"""
        if self._ret is not None and self._ret[0] is ret:
            return self._ret[1]
        mx, xp = self.mx, self
        vr = mx.isfinite(ret)
        key = mx.where(vr, ret, np.inf)
        order = mx.argsort(key, axis=-1)
        srt = mx.take_along_axis(key, order, axis=-1)
        pos = xp.arange_like(srt)
        n = srt.shape[-1]
        edge = xp.ones_col(srt, 1.0) > 0
        new = xp.cat_cols([edge, srt[..., 1:] != srt[..., :-1]])
        end = xp.cat_cols([srt[..., :-1] != srt[..., 1:], edge])
        gs = xp.cummax(xp.where(new, pos, 0.0)).astype(mx.uint32)
        ge = xp.rcummin(xp.where(end, pos, float(n))).astype(mx.uint32)
        valid_sorted = mx.take_along_axis(vr, order, axis=-1)
        state = {"vr": vr, "order": order.astype(mx.uint32), "gs": gs, "ge": ge,
                 "valid_sorted": valid_sorted}  # fmt: skip
        mx.eval(*state.values())
        self._ret = (ret, state)
        return state

    def rank_ic_fused(self, score, ret):
        mx = self.mx
        st = self._returns(ret)
        t, n = score.shape
        total = t * n
        vs = mx.isfinite(score)
        v = vs & st["vr"]
        key = mx.where(v, score, np.inf)
        order_s = mx.argsort(key, axis=-1)
        srt = mx.take_along_axis(key, order_s, axis=-1)
        shape = [((t, n), self.f32)]
        rs = self._launch(self._rank, [srt, order_s.astype(mx.uint32)], total,
                          [("T", t), ("N", n)], shape)[0]  # fmt: skip
        m = (mx.take_along_axis(vs, st["order"].astype(mx.int32), axis=-1)
             & st["valid_sorted"]).astype(self.f32)  # fmt: skip
        cum = mx.cumsum(m, axis=-1)
        rr = self._launch(self._ret_rank, [st["order"], m, st["gs"], st["ge"], cum], total,
                          [("T", t), ("N", n)], shape)[0]  # fmt: skip
        return self._pearson(inputs=[rs, rr], template=[("T", t), ("N", n)],
                             grid=(t * 32, 1, 1), threadgroup=(32, 1, 1),
                             output_shapes=[(t,)], output_dtypes=[self.f32])[0]  # fmt: skip


# ---- PyTorch：Metal 上用 torch.mps.compile_shader 跑同一份内核正文 ---------------------------


def _metal_full(name, inputs, outputs, scalars, body, grouped=False):
    """把 MLX 风格的内核正文包成完整的 Metal 函数（参数按 输入、输出、标量 的顺序）。"""
    args = [f"device const {t}* {n}" for n, t in inputs]
    args += [f"device float* {n}" for n in outputs]
    args += [f"constant int& {n}" for n in scalars]
    if grouped:
        args += ["uint ash_tg [[threadgroup_position_in_grid]]",
                 "uint ash_lane [[thread_index_in_simdgroup]]"]  # fmt: skip
        body = (body.replace("ROW_ID", "(int)ash_tg").replace("LANE_ID", "(int)ash_lane")
                .replace("SIMD_SUM", "metal::simd_sum"))  # fmt: skip
    else:
        args += ["uint ash_tid [[thread_position_in_grid]]"]
        body = body.replace("COL_INDEX", "(int)ash_tid")
    body = body.replace("IS_FINITE", "metal::isfinite").replace("NAN_F", "NAN")
    return f"kernel void {name}({', '.join(args)}) {{\n{body}\n}}\n"


class TorchFused(gpu_ops.Torch):
    """PyTorch MPS + torch.mps.compile_shader（与 MLXFused 同一份内核正文）。CUDA 上 PyTorch 的对应做法是
    torch.compile（Triton）或 cpp_extension.load_inline，这里没有实现，见 docs/GPU_PROBE.md。"""

    name = "torch-fused"

    def __init__(self, device="mps"):
        super().__init__(device)
        if device != "mps":
            raise NotImplementedError("torch-fused 只实现了 MPS（compile_shader）")
        src = "#include <metal_stdlib>\nusing namespace metal;\n"
        src += _metal_full(
            "ash_ts", [("x", "float")], ["out"], ["W", "T", "N", "OP"], _TS_BODY
        )
        src += _metal_full("ash_corr", [("x", "float"), ("y", "float")], ["out"],
                           ["W", "T", "N"], _CORR_BODY)  # fmt: skip
        src += _metal_full("ash_rank", [("srt", "float"), ("order", "int")], ["out"],
                           ["T", "N"], _RANK_BODY)  # fmt: skip
        src += _metal_full("ash_ret_rank", [("order", "int"), ("m", "float"), ("gs", "int"),
                           ("ge", "int"), ("cum", "float")], ["out"], ["T", "N"],
                           _RET_RANK_BODY)  # fmt: skip
        src += _metal_full("ash_pearson", [("rs", "float"), ("rr", "float")], ["ic"],
                           ["T", "N"], _PEARSON_BODY, grouped=True)  # fmt: skip
        self.lib = self.t.mps.compile_shader(src)
        self._ret = None

    def _empty(self, *shape):
        return self.t.empty(shape, dtype=self.f32, device=self.dev)

    def ts_fused(self, op, x, w):
        t, n = x.shape
        out = self._empty(t, n)
        self.lib.ash_ts(
            x.contiguous(), out, w, t, n, TS_CODES[op], threads=n, group_size=256
        )
        return out

    def corr_fused(self, x, y, w):
        t, n = x.shape
        out = self._empty(t, n)
        self.lib.ash_corr(x.contiguous(), y.contiguous(), out, w, t, n, threads=n,
                          group_size=256)  # fmt: skip
        return out

    def _returns(self, ret):
        if self._ret is not None and self._ret[0] is ret:
            return self._ret[1]
        torch = self.t
        vr = torch.isfinite(ret)
        key = torch.where(vr, ret, float("inf"))
        srt, order = torch.sort(key, dim=-1)
        pos = self.arange_like(srt)
        n = srt.shape[-1]
        edge = self.ones_col(srt, 1.0) > 0
        new = self.cat_cols([edge, srt[..., 1:] != srt[..., :-1]])
        end = self.cat_cols([srt[..., :-1] != srt[..., 1:], edge])
        gs = self.cummax(self.where(new, pos, 0.0)).to(torch.int32)
        ge = self.rcummin(self.where(end, pos, float(n))).to(torch.int32)
        state = {"vr": vr, "order": order, "order32": order.to(torch.int32).contiguous(),
                 "gs": gs.contiguous(), "ge": ge.contiguous(),
                 "valid_sorted": torch.gather(vr, -1, order)}  # fmt: skip
        self._ret = (ret, state)
        return state

    def rank_ic_fused(self, score, ret):
        torch = self.t
        st = self._returns(ret)
        t, n = score.shape
        vs = torch.isfinite(score)
        v = vs & st["vr"]
        srt, order_s = torch.sort(torch.where(v, score, float("inf")), dim=-1)
        rs = self._empty(t, n)
        self.lib.ash_rank(srt.contiguous(), order_s.to(torch.int32).contiguous(), rs, t, n,
                          threads=t * n, group_size=256)  # fmt: skip
        m = (
            (torch.gather(vs, -1, st["order"]) & st["valid_sorted"])
            .to(self.f32)
            .contiguous()
        )
        cum = torch.cumsum(m, dim=-1).contiguous()
        rr = self._empty(t, n)
        self.lib.ash_ret_rank(st["order32"], m, st["gs"], st["ge"], cum, rr, t, n,
                              threads=t * n, group_size=256)  # fmt: skip
        ic = self._empty(t)
        self.lib.ash_pearson(rs, rr, ic, t, n, threads=t * 32, group_size=32)
        return ic
