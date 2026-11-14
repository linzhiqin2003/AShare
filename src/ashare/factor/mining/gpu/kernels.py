"""GPU 粗筛的自定义内核：正文只有一份，Metal 与 CUDA 共用。

内核正文按「每个线程一格（窗口算子）或一条序列（EMA）」写成，Metal（``mx.fast.metal_kernel``）
和 CUDA（``mx.fast.cuda_kernel``）的语法在这种写法下几乎一样，差别只在线程下标、
``isfinite``、NaN、平方根和 SIMD 规约，由 `source` 按设备换成各自的写法。正文出自
``bench/gpu/fused.py``（docs/GPU_PROBE.md 第三节），这里补齐了 ddof、归一化排名、偏度、
斜率、极值位置和 EMA。

语义以 panelops（packages/panelops/SPEC.md）为准，限于 ``min_periods == window`` 的情形
（搜索空间不改它；别的取值整棵表达式交给 CPU）：

- 非有限即缺失；窗口含今天，窗口内任何一格缺失结果就缺失（窗口和不是有限数即有缺失）；
- 常数窗口按「窗口内每个值与今天的值逐位相等」判定：标准差恰为 0，z 分、相关、偏度为缺失；
- 秩为平均秩（并列取平均）；Rank IC 在因子与收益都有效的交集上分别求秩再算 Pearson。

`KERNEL_ID` 是全部内核正文的摘要，进台账，换内核就换编号。
"""

from __future__ import annotations

import hashlib

# ts 内核的算子编号（模板常量 OP）
TS_CODES = {
    "mean": 0, "sum": 1, "std": 2, "zscore": 3, "max": 4, "min": 5, "rank": 6,
    "decay_linear": 7, "skew": 8, "slope": 9, "argmax": 10, "argmin": 11, "delay": 12,
    "delta": 13,
}  # fmt: skip

# 模板常量：W 窗口、T 行数、N 列数、OP 算子、DDOF（std、zscore）、NORM（rank 归一化）、
# TIES（argmax/argmin：0 = latest、1 = earliest）
TS_BODY = """
    int g = COL_INDEX;
    if (g >= T * N) return;
    int t = g / N, col = g % N;
    if (OP >= 12) {  // 延迟、差分：W = 期数 + 1
        float res = NAN_F;
        if (t >= W - 1) {
            float lag = x[(t - (W - 1)) * N + col];
            res = (OP == 12) ? lag : x[t * N + col] - lag;
        }
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
        return;
    }
    {
        float res = NAN_F;
        if (t >= W - 1) {
            float v0 = x[t * N + col];
            float acc = 0.0f;
            for (int k = 0; k < W; ++k) acc += x[(t - k) * N + col];
            if (IS_FINITE(acc)) {
                float m = acc / W;
                if (OP == 0) res = m;
                else if (OP == 1) res = acc;
                else if (OP == 2 || OP == 3 || OP == 8) {
                    float ss = 0.0f;
                    bool flat = true;
                    for (int k = 0; k < W; ++k) {
                        float v = x[(t - k) * N + col];
                        float d = v - m;
                        ss += d * d;
                        flat = flat && (v == v0);
                    }
                    if (OP == 8) {
                        // 先除以标准差再立方：离差的立方在 1e13 量级的输入上就会溢出
                        float sd = SQRT(ss / W);
                        float s3 = 0.0f;
                        for (int k = 0; k < W; ++k) {
                            float z = (x[(t - k) * N + col] - m) / sd;
                            s3 += z * z * z;
                        }
                        res = (flat || W < 3) ? NAN_F : s3 / W;
                    } else {
                        float sd = (W > DDOF) ? SQRT(ss / (W - DDOF)) : NAN_F;
                        if (OP == 2) res = (flat && W > DDOF) ? 0.0f : sd;
                        else res = flat ? NAN_F : (v0 - m) / sd;
                    }
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
                    if (NORM) res = (W == 1) ? 0.5f : (res - 1.0f) / (W - 1);
                } else if (OP == 7) {
                    float s = 0.0f;
                    for (int k = 0; k < W; ++k) s += x[(t - k) * N + col] * (float)(W - k);
                    res = s / (W * (W + 1) / 2.0f);
                } else if (OP == 9) {
                    float cross = 0.0f;
                    for (int k = 0; k < W; ++k) {
                        float dp = (W - 1) / 2.0f - (float)k;
                        cross += dp * (x[(t - k) * N + col] - m);
                    }
                    float pv = (float)W * ((float)W * W - 1.0f) / 12.0f;
                    res = (W >= 2) ? cross / pv : NAN_F;
                } else if (OP == 10 || OP == 11) {
                    float e = v0;
                    int at = 0;
                    for (int k = 1; k < W; ++k) {
                        float v = x[(t - k) * N + col];
                        bool better = (OP == 10) ? (v > e) : (v < e);
                        if (better || (TIES == 1 && v == e)) { e = v; at = k; }
                    }
                    res = (float)at;
                }
            }
        }
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
    }
"""

CORR_BODY = """
    int g = COL_INDEX;
    if (g >= T * N) return;
    int t = g / N, col = g % N;
    {
        float res = NAN_F;
        if (t >= W - 1) {
            float sx = 0.0f, sy = 0.0f;
            for (int k = 0; k < W; ++k) {
                sx += x[(t - k) * N + col];
                sy += y[(t - k) * N + col];
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
                res = (fx || fy || W < 2) ? NAN_F : cxy / SQRT(cxx) / SQRT(cyy);
            }
        }
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
    }
"""

# EMA（adjust=False）：第一个有效值作初值，缺失行保持状态，累计有效个数 >= MP 才输出；
# alpha 由调用方算好（float32）作为输入。
EMA_BODY = """
    int col = COL_INDEX;
    if (col >= N) return;
    float al = alpha[0];
    float state = 0.0f;
    int seen = 0;
    for (int t = 0; t < T; ++t) {
        float v = x[t * N + col];
        if (IS_FINITE(v)) {
            state = (seen == 0) ? v : al * v + (1.0f - al) * state;
            seen += 1;
        }
        float res = (seen >= MP) ? state : NAN_F;
        out[t * N + col] = IS_FINITE(res) ? res : NAN_F;
    }
"""

# 排好序以后求平均秩：每个格子一个线程，在本行的排序里二分找并列组的起止（无效值当 +inf
# 排在最后），把秩写回原来的列。读写都按排序位置并行，代价与并列组大小无关。
RANK_BODY = """
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

# 收益只排序一次：并列组起止（gs、ge，按排序位置）也只算一次。每个因子只需「交集有效」在
# 收益排序里的累计个数 cum，平均秩 = cum[gs-1] + (组内有效数 + 1) / 2，与在交集上重新排序
# 的结果相同。
RET_RANK_BODY = """
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
PEARSON_BODY = """
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
        float res = (cnt >= 2.0f) ? sxy / SQRT(sxx) / SQRT(syy) : NAN_F;
        ic[t] = IS_FINITE(res) ? res : NAN_F;
    }
"""

# 一天一个线程组的 Rank IC：分数与收益的交集上分别求平均秩、再算 Pearson，全部在线程组
# 共享内存里完成，不写任何 (T, N) 中间面板，也不需要按因子 argsort。步骤：
#   1. 读入当天的因子值（股票池外、非有限、收益无效的记 +inf），在共享内存里双调排序；
#   2. 相邻差：非零差的个数、非零差不超过 GAP 倍绝对值的个数（舍入并列比例，见 backend）；
#   3. 每个有效格子在排好的序列里二分出并列组的起止，平均秩 = (起 + 止 + 1) / 2；
#   4. 收益只在整面板排序一次（posr：每格在当天收益排序里的位置，无效为 0xFFFFFFFF；
#      gsge：所在并列组的起止位置，高 16 位起、低 16 位止）。把交集有效的位置记 1、前缀和，
#      平均秩 = cum[起 − 1] + (组内有效数 + 1) / 2，与在交集上重新排序相同；
#   5. 按 (n + 1) / 2 居中求 Pearson，线程组内规约（固定顺序，结果可复现）。
# 模板常量：T、N；NP = 不小于 N 的 2 的幂（≤ 8192，共享内存 32 KB）；TG 线程数；E = NP / TG；
# NSG = TG / 32。
RANK_IC_BODY = """
    TG_SHARED float buf[NP];
    int row = GROUP_ID;
    int tid = TID;
    int lane = tid % 32;
    int sg = tid / 32;
    int base = row * N;
    // 按列（合并访存）读入：key 为交集有效的因子值，否则 +inf
    float key[E];
    unsigned int pos[E];
    unsigned int gg[E];
    bool ok[E];
    for (int e = 0; e < E; ++e) {
        int j = tid + e * TG;
        bool v = false;
        float s = INF_F;
        unsigned int p = 0xFFFFFFFFu, g = 0u;
        if (j < N) {
            p = posr[base + j];
            float sc = score[base + j];
            v = univ[base + j] && IS_FINITE(sc) && (p != 0xFFFFFFFFu);
            if (v) { s = sc; g = gsge[base + j]; }
        }
        ok[e] = v; key[e] = s; pos[e] = p; gg[e] = g;
        buf[j] = s;
    }
    BARRIER;
    // 双调排序：每个线程持有连续 E 个（v[e] = 第 tid*E+e 个）。步长 < E 在寄存器里换，
    // E ≤ 步长 < 32E 在 SIMD 组内用 shuffle 换，步长 ≥ 32E 才经共享内存（每步一个栅栏）。
    float v[E];
    for (int e = 0; e < E; ++e) v[e] = buf[tid * E + e];
    for (int k = 2; k <= NP; k <<= 1) {
        if ((k >> 1) >= 32 * E) {
            BARRIER;
            for (int e = 0; e < E; ++e) buf[tid * E + e] = v[e];
            BARRIER;
            for (int jj = k >> 1; jj >= 32 * E; jj >>= 1) {
                for (int q = tid; q < NP / 2; q += TG) {
                    int i = ((q & ~(jj - 1)) << 1) | (q & (jj - 1));
                    float a = buf[i], b = buf[i + jj];
                    bool up = ((i & k) == 0);
                    if (up ? (a > b) : (a < b)) { buf[i] = b; buf[i + jj] = a; }
                }
                BARRIER;
            }
            for (int e = 0; e < E; ++e) v[e] = buf[tid * E + e];
        }
        int top = (k >> 1) < 32 * E ? (k >> 1) : 16 * E;
        for (int jj = top; jj >= E; jj >>= 1) {
            int m = jj / E;
            for (int e = 0; e < E; ++e) {
                float a = v[e];
                float b = SHFL_XOR(a, m);
                int i = tid * E + e;
                bool up = ((i & k) == 0);
                bool lower = (i & jj) == 0;
                v[e] = (up == lower) ? (a < b ? a : b) : (a < b ? b : a);
            }
        }
        for (int jj = E >> 1; jj > 0; jj >>= 1) {  // 上下界是常量：展开后寄存器下标都是常量
            if (jj >= k) continue;
            for (int e = 0; e < E; ++e) {
                if ((e & jj) == 0) {
                    int i = tid * E + e;
                    bool up = ((i & k) == 0);
                    float a = v[e], b = v[e + jj];
                    if (up ? (a > b) : (a < b)) { v[e] = b; v[e + jj] = a; }
                }
            }
        }
    }
    BARRIER;
    for (int e = 0; e < E; ++e) buf[tid * E + e] = v[e];
    BARRIER;
    // 舍入并列计数与因子的平均秩（在排好的序列里二分并列组起止）
    float tiny = 0.0f, nz = 0.0f;
    float rs[E];
    for (int e = 0; e < E; ++e) {
        int i = tid * E + e;
        if (i + 1 < NP) {
            float a = v[e];
            float b = (e + 1 < E) ? v[e + 1] : buf[i + 1];
            if (IS_FINITE(a) && IS_FINITE(b) && b > a) {
                nz += 1.0f;
                float aa = a < 0.0f ? -a : a, bb = b < 0.0f ? -b : b;
                if (b - a <= GAP * (aa > bb ? aa : bb)) tiny += 1.0f;
            }
        }
        rs[e] = 0.0f;
        if (ok[e]) {
            int lo = 0, hi = NP;
            while (lo < hi) { int m = (lo + hi) >> 1; if (buf[m] < key[e]) lo = m + 1; else hi = m; }
            int a = lo;
            lo += 1;  // 并列组的止：多数格子没有并列，先看下一个
            if (lo < NP && buf[lo] <= key[e]) {
                hi = NP;
                while (lo < hi) { int m = (lo + hi) >> 1; if (buf[m] <= key[e]) lo = m + 1; else hi = m; }
            }
            rs[e] = (float)(a + lo + 1) * 0.5f;
        }
    }
    BARRIER;
    // 收益一侧：交集有效的位置记 1，按块前缀和（线程内 → SIMD 组内 → 组间）
    for (int e = 0; e < E; ++e) buf[tid + e * TG] = 0.0f;
    BARRIER;
    for (int e = 0; e < E; ++e) if (ok[e]) buf[pos[e]] = 1.0f;
    BARRIER;
    float c[E];
    float run = 0.0f;
    for (int e = 0; e < E; ++e) { run += buf[tid * E + e]; c[e] = run; }
    float incl = SIMD_PREFIX(run);
    BARRIER;
    if (lane == 31) buf[sg] = incl;
    BARRIER;
    float off = incl - run;
    for (int q = 0; q < sg; ++q) off += buf[q];
    BARRIER;
    for (int e = 0; e < E; ++e) buf[tid * E + e] = c[e] + off;
    BARRIER;
    float rr[E];
    float cnt = 0.0f;
    for (int e = 0; e < E; ++e) {
        rr[e] = 0.0f;
        if (ok[e]) {
            int a = (int)(gg[e] >> 16), b = (int)(gg[e] & 0xFFFFu);
            float before = (a > 0) ? buf[a - 1] : 0.0f;
            rr[e] = before + (buf[b] - before + 1.0f) * 0.5f;
            cnt += 1.0f;
        }
    }
    BARRIER;
    cnt = SIMD_SUM(cnt); tiny = SIMD_SUM(tiny); nz = SIMD_SUM(nz);
    if (lane == 0) { buf[sg] = cnt; buf[NSG + sg] = tiny; buf[2 * NSG + sg] = nz; }
    BARRIER;
    float n = 0.0f, tt = 0.0f, zz = 0.0f;
    for (int q = 0; q < NSG; ++q) { n += buf[q]; tt += buf[NSG + q]; zz += buf[2 * NSG + q]; }
    BARRIER;
    float cc = (n + 1.0f) * 0.5f;
    float sxy = 0.0f, sxx = 0.0f, syy = 0.0f;
    for (int e = 0; e < E; ++e) {
        if (ok[e]) {
            float dx = rs[e] - cc, dy = rr[e] - cc;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
    }
    sxy = SIMD_SUM(sxy); sxx = SIMD_SUM(sxx); syy = SIMD_SUM(syy);
    if (lane == 0) { buf[sg] = sxy; buf[NSG + sg] = sxx; buf[2 * NSG + sg] = syy; }
    BARRIER;
    if (tid == 0) {
        float a = 0.0f, b = 0.0f, d = 0.0f;
        for (int q = 0; q < NSG; ++q) { a += buf[q]; b += buf[NSG + q]; d += buf[2 * NSG + q]; }
        float res = (n >= 2.0f) ? a / SQRT(b) / SQRT(d) : NAN_F;
        ic[row] = IS_FINITE(res) ? res : NAN_F;
        noise[2 * row] = tt;
        noise[2 * row + 1] = zz;
    }
"""

# 逐元素算子链：一个线程一格，按后缀程序在寄存器栈上求值（一次读输入、一次写输出，中间
# 结果不落显存），同时做 float32 安全范围检查（乘除的结果超出范围、或非零乘除下溢成 0、上溢
# 成无穷），每个 SIMD 组写一个标记。程序是数据（prog：每条两个整数，操作码与参数），所以
# 任何算子链都用同一个编译好的内核，不会每种表达式结构编译一次。
# 操作码见 ELEM_OPS；模板常量 SIZE 格子数、L 指令条数。输入 x0..x3（不用的传同一份）。
ELEM_OPS = {"load": 0, "const": 1, "add": 2, "subtract": 3, "multiply": 4, "divide": 5,
            "negate": 6, "abs": 7, "sign": 8, "log": 9, "maximum": 10, "minimum": 11}  # fmt: skip
ELEM_INPUTS = 4
ELEM_STACK = 8
ELEM_BODY = """
    int g = COL_INDEX;
    bool bad = false;
    if (g < SIZE) {
        float st[STACK];
        int sp = 0;
        for (int pc = 0; pc < L; ++pc) {
            int op = prog[2 * pc], arg = prog[2 * pc + 1];
            if (op == 0) {
                st[sp++] = (arg == 0) ? x0[g] : (arg == 1) ? x1[g] : (arg == 2) ? x2[g] : x3[g];
                continue;
            }
            if (op == 1) { st[sp++] = consts[arg]; continue; }
            float r;
            if (op >= 6 && op <= 9) {
                float a = st[sp - 1];
                if (op == 6) r = -a;
                else if (op == 7) r = a < 0.0f ? -a : a;
                else if (op == 8) r = (a != a) ? NAN_F : (a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f));
                else r = LOGF(a);
            } else {
                float b = st[--sp];
                float a = st[sp - 1];
                if (op == 2) r = a + b;
                else if (op == 3) r = a - b;
                else if (op == 4) r = a * b;
                else if (op == 5) r = a / b;
                else if (op == 10) r = (a != a || b != b) ? NAN_F : (a > b ? a : b);
                else r = (a != a || b != b) ? NAN_F : (a < b ? a : b);
                if ((op == 4 || op == 5) && IS_FINITE(a) && IS_FINITE(b)) {
                    float m = r < 0.0f ? -r : r;
                    bool zero_in = (a == 0.0f) || (op == 4 && b == 0.0f);
                    if (IS_FINITE(r)) {
                        if (m > HIGH || (m > 0.0f && m < LOW) || (r == 0.0f && !zero_in)) bad = true;
                    } else if (!(op == 5 && b == 0.0f)) {
                        bad = true;
                    }
                }
            }
            st[sp - 1] = IS_FINITE(r) ? r : NAN_F;
        }
        out[g] = st[0];
    }
    bool hit = SIMD_ANY(bad);
    if ((COL_INDEX % 32) == 0 && COL_INDEX / 32 < FLAGS) flags[COL_INDEX / 32] = hit ? 1.0f : 0.0f;
"""

CUDA_HEADER = """
__device__ inline float ash_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return __shfl_sync(0xffffffff, v, 0);
}
__device__ inline float ash_warp_prefix(float v) {
    int lane = threadIdx.x & 31;
    for (int o = 1; o < 32; o <<= 1) {
        float u = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += u;
    }
    return v;
}
__device__ inline bool ash_warp_any(bool v) {
    return __any_sync(0xffffffff, v);
}
__device__ inline float ash_shfl_xor(float v, int m) {
    return __shfl_xor_sync(0xffffffff, v, m);
}
"""

_REPLACE = {
    "cuda": {
        "COL_INDEX": "(int)(blockIdx.x * blockDim.x + threadIdx.x)",
        "ROW_ID": "(int)blockIdx.x",
        "LANE_ID": "(int)threadIdx.x",
        "SIMD_SUM": "ash_warp_sum",
        "SIMD_PREFIX": "ash_warp_prefix",
        "SHFL_XOR": "ash_shfl_xor",
        "SIMD_ANY": "ash_warp_any",
        "LOGF": "logf",
        "IS_FINITE": "isfinite",
        "SQRT": "sqrtf",
        "NAN_F": "__int_as_float(0x7fc00000)",
        "INF_F": "__int_as_float(0x7f800000)",
        "TG_SHARED": "__shared__",
        "BARRIER": "__syncthreads()",
        "GROUP_ID": "(int)blockIdx.x",
        "TID": "(int)threadIdx.x",
    },
    "metal": {
        "COL_INDEX": "(int)thread_position_in_grid.x",
        "ROW_ID": "(int)threadgroup_position_in_grid.x",
        "LANE_ID": "(int)thread_index_in_simdgroup",
        "SIMD_SUM": "metal::simd_sum",
        "SIMD_PREFIX": "metal::simd_prefix_inclusive_sum",
        "SHFL_XOR": "metal::simd_shuffle_xor",
        "SIMD_ANY": "metal::simd_any",
        "LOGF": "metal::log",
        "IS_FINITE": "metal::isfinite",
        "SQRT": "metal::sqrt",
        "NAN_F": "NAN",
        "INF_F": "INFINITY",
        "TG_SHARED": "threadgroup",
        "BARRIER": "threadgroup_barrier(metal::mem_flags::mem_threadgroup)",
        "GROUP_ID": "(int)threadgroup_position_in_grid.x",
        "TID": "(int)thread_position_in_threadgroup.x",
    },
}

BODIES = {
    "ash_ts": (TS_BODY, ["x"], ["out"]),
    "ash_corr": (CORR_BODY, ["x", "y"], ["out"]),
    "ash_ema": (EMA_BODY, ["x", "alpha"], ["out"]),
    "ash_rank": (RANK_BODY, ["srt", "order"], ["out"]),
    "ash_ret_rank": (RET_RANK_BODY, ["order", "m", "gs", "ge", "cum"], ["out"]),
    "ash_pearson": (PEARSON_BODY, ["rs", "rr"], ["ic"]),
    "ash_rank_ic": (RANK_IC_BODY, ["score", "univ", "posr", "gsge"], ["ic", "noise"]),
    "ash_elem": (
        ELEM_BODY,
        ["x0", "x1", "x2", "x3", "prog", "consts"],
        ["out", "flags"],
    ),
}

# 单线程组 Rank IC 能处理的最大截面（共享内存 32 KB / 4 字节）；更宽的截面走 argsort 写法。
RANK_IC_MAX_N = 8192

# float32 安全范围：节点值的绝对值不超过 1e15、非零值不小于 1e-15 时，两数相乘除在
# [1e-30, 1e30] 内，窗口内离差平方和（≤ 4e30·W）也远离 float32 的上限 3.4e38 与最小正规数
# 1.2e-38。真实面板的字段（成交额约 1e11、成交量约 1e10、换手率约 1e-4）都在范围内。
RANGE_HIGH = 1e15
RANGE_LOW = 1e-15

# 舍入量级的相邻差：float32 的单位舍入约 6e-8，几步运算后约十几个单位；相对差不超过 1e-6
# 的视为舍入造成的（见 `Backend.rank_ic`）。取 1e-5 时，均值在 1 附近、离散度小的正常比值
# 在 5923 只的截面上相邻差就到这个量级，会被误判。
NOISE_GAP = 1e-6

# SIMD 组宽度：Pearson 内核按 32 写死（Apple GPU 的 SIMD 组与 NVIDIA 的 warp 都是 32）。
SIMD_WIDTH = 32


def source(body, platform):
    """把正文里的占位符换成 ``platform``（"metal" 或 "cuda"）的写法。"""
    for k, v in _REPLACE[platform].items():
        body = body.replace(k, v)
    body = body.replace("GAP", f"{NOISE_GAP!r}f").replace("STACK", str(ELEM_STACK))
    return body.replace("HIGH", f"{RANGE_HIGH!r}f").replace("LOW", f"{RANGE_LOW!r}f")


def _kernel_id():
    h = hashlib.sha256()
    for name in sorted(BODIES):
        h.update(name.encode())
        h.update(BODIES[name][0].encode())
    h.update(CUDA_HEADER.encode())
    h.update(repr((NOISE_GAP, RANGE_HIGH, RANGE_LOW, ELEM_STACK)).encode())
    h.update(repr(sorted(_REPLACE["metal"].items())).encode())
    h.update(repr(sorted(_REPLACE["cuda"].items())).encode())
    return h.hexdigest()[:16]


KERNEL_ID = _kernel_id()
