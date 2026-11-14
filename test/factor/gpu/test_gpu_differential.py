"""GPU float32 算子对 panelops float64 的差分测试（docs/GPU_MINING.md「精确性约定」）。

来源：docs/GPU_PROBE.md 第四节（算子级误差实测）与第六节（差分测试要求）。判据：

- 缺失位置逐格一致（停牌、上市前、退市后、常数窗口、正负无穷、窗口不满）；
- 数值误差 = |gpu − cpu| / max(|cpu|, 该输出有限值绝对值的中位数)，不超过下面的容差。
  容差取探针实测最大值放宽一个量级：均值类实测 ≤ 1e-6 → 1e-5；标准差、z 分、相关 ≤ 1e-4 →
  1e-3；差分 ≤ 3e-4（相消）→ 3e-3。探针之后补的偏度、斜率、截面 z 分按标准差一类、EMA
  按均值一类（本机实测偏度 2e-4、斜率 3e-6、截面 z 分 1e-5、EMA 3e-7）；
- 排名类（时序、截面排名，极值位置）：float64 下不相等、float32 下相等的格子会并列取平均，
  与 CPU 不同的格子比例 ≤ 0.1%（探针实测 0–0.016%）；
- Rank IC 日值绝对误差 ≤ 1e-3（探针实测 4e-4）。

没装 MLX 或没有 GPU 时整个文件 skip。
"""

import numpy as np
import panelops
import pytest

from ashare.factor.mining.gpu import available

OK, WHY = available()
pytestmark = pytest.mark.skipif(not OK, reason=f"GPU 不可用：{WHY}")

MEAN_TOL = 1e-5
SPREAD_TOL = 1e-3
DELTA_TOL = 3e-3
RANK_CELLS = 1e-3
RANK_IC_TOL = 1e-3

T, N = 160, 67  # N 不是 32、256 的倍数：线程组边界


def adversarial(seed=0):
    """带停牌、上市前、退市后、常数窗口、大量并列、正负无穷的一对面板（x 为正，y 有正负）。"""
    rng = np.random.default_rng(seed)
    x = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    y = rng.normal(0, 1, (T, N))
    x[rng.random((T, N)) < 0.03] = np.nan  # 零星缺失
    x[40:52, 5] = np.nan  # 停牌
    x[:30, 6] = np.nan  # 上市前
    x[120:, 7] = np.nan  # 退市后
    x[60:90, 8] = 13.25  # 常数窗口
    y[60:90, 8] = -0.5
    x[:, 9:15] = np.round(x[:, 9:15])  # 大量并列
    y[:, 9:15] = np.round(y[:, 9:15])
    x[100, 16] = np.inf
    x[101, 17] = -np.inf
    y[102, 18] = np.inf
    y[rng.random((T, N)) < 0.02] = np.nan
    y[70:75, 20] = 0.0  # 零：log、除法的定义域边界
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


@pytest.fixture(scope="module")
def be():
    from ashare.factor.mining.gpu.backend import MLXBackend

    return MLXBackend()


@pytest.fixture(scope="module")
def data(be):
    x, y = adversarial()
    return x, y, be.upload(x), be.upload(y)


def compare(gpu, cpu):
    g, c = np.asarray(gpu, np.float64), np.asarray(cpu, np.float64)
    gf, cf = np.isfinite(g), np.isfinite(c)
    out = {"missing": int((gf != cf).sum()), "rel": 0.0, "differ": 0.0}
    both = gf & cf
    if both.any():
        d = np.abs(g[both] - c[both])
        scale = float(np.median(np.abs(c[both]))) or 1.0
        out["rel"] = float((d / np.maximum(np.abs(c[both]), scale)).max())
        out["differ"] = float((d > 1e-3 * max(scale, 1.0)).mean())
    return out


def _full(op, params):
    p = dict(params)
    if "window" in p and op != "ts_ema":
        p.setdefault("min_periods", p["window"])
    if op == "ts_ema":
        p.setdefault("alpha", 2 / (p["window"] + 1))
        p.setdefault("min_periods", p["window"])
    return p


UNARY = [
    ("ts_mean", {"window": 1}, MEAN_TOL), ("ts_mean", {"window": 5}, MEAN_TOL),
    ("ts_mean", {"window": 60}, MEAN_TOL), ("ts_sum", {"window": 10}, MEAN_TOL),
    ("ts_std", {"window": 2}, SPREAD_TOL), ("ts_std", {"window": 20}, SPREAD_TOL),
    ("ts_std", {"window": 5, "ddof": 0}, SPREAD_TOL),
    ("ts_std", {"window": 1}, SPREAD_TOL),  # count <= ddof：全缺失
    ("ts_zscore", {"window": 10}, SPREAD_TOL), ("ts_zscore", {"window": 20, "ddof": 0}, SPREAD_TOL),
    ("ts_max", {"window": 5}, MEAN_TOL), ("ts_min", {"window": 20}, MEAN_TOL),
    ("ts_decay_linear", {"window": 10}, MEAN_TOL),
    ("ts_skew", {"window": 10}, SPREAD_TOL), ("ts_skew", {"window": 2}, SPREAD_TOL),
    ("ts_slope", {"window": 10}, SPREAD_TOL), ("ts_slope", {"window": 1}, SPREAD_TOL),
    ("ts_ema", {"window": 10}, MEAN_TOL), ("ts_ema", {"window": 5, "min_periods": 2}, MEAN_TOL),
    ("ts_delay", {"periods": 1}, MEAN_TOL), ("ts_delay", {"periods": 0}, MEAN_TOL),
    ("ts_delay", {"periods": 200}, MEAN_TOL),
    ("ts_delta", {"periods": 5}, DELTA_TOL),
    ("cs_zscore", {}, SPREAD_TOL), ("cs_zscore", {"ddof": 1}, SPREAD_TOL),
    ("negate", {}, MEAN_TOL), ("abs", {}, MEAN_TOL), ("log", {}, MEAN_TOL),
    ("sign", {}, MEAN_TOL),
    ("ts_mean", {"window": 200}, MEAN_TOL),  # 窗口比面板长
]  # fmt: skip
RANKS = [
    ("ts_rank", {"window": 5}), ("ts_rank", {"window": 20}),
    ("ts_rank", {"window": 5, "normalize": True}), ("ts_rank", {"window": 1, "normalize": True}),
    ("ts_argmax", {"window": 10}), ("ts_argmin", {"window": 10}),
    ("ts_argmax", {"window": 10, "ties": "earliest"}),
    ("ts_argmin", {"window": 5, "ties": "earliest"}),
    ("cs_rank", {}), ("cs_rank", {"normalize": True}),
]  # fmt: skip
BINARY = [
    ("add", MEAN_TOL), ("subtract", DELTA_TOL), ("multiply", MEAN_TOL), ("divide", MEAN_TOL),
    ("maximum", MEAN_TOL), ("minimum", MEAN_TOL),
]  # fmt: skip
ELEM = {"negate", "abs", "log", "sign", "add", "subtract", "multiply", "divide", "maximum",
        "minimum"}  # fmt: skip


def _cpu(op, args, params):
    if op in ELEM:
        return panelops.elementwise(op, *args)
    return getattr(panelops, op)(*args, **params)


@pytest.mark.parametrize("which", ["x", "y"])
@pytest.mark.parametrize(("op", "params", "tol"), UNARY, ids=lambda v: str(v))
def test_unary_matches_panelops(be, data, op, params, tol, which):
    x, y, gx, gy = data
    a, ga = (x, gx) if which == "x" else (y, gy)
    got = be.to_host(be.compute(op, _full(op, params), [ga], (T, N)))
    r = compare(got, _cpu(op, [a], params))
    assert r["missing"] == 0, r
    assert r["rel"] <= tol, r


@pytest.mark.parametrize(("op", "params"), RANKS, ids=lambda v: str(v))
def test_ranks_match_panelops(be, data, op, params):
    x, y, gx, gy = data
    for a, ga in ((x, gx), (y, gy)):
        got = be.to_host(be.compute(op, _full(op, params), [ga], (T, N)))
        r = compare(got, _cpu(op, [a], params))
        assert r["missing"] == 0, r
        assert r["differ"] <= RANK_CELLS, r


@pytest.mark.parametrize(("op", "tol"), BINARY)
def test_binary_matches_panelops(be, data, op, tol):
    x, y, gx, gy = data
    for args, gargs in (((x, y), (gx, gy)), ((y, x), (gy, gx)), ((x, 2.5), (gx, 2.5))):
        got = be.to_host(be.compute(op, {}, list(gargs), (T, N)))
        r = compare(got, panelops.elementwise(op, *args))
        assert r["missing"] == 0, (args[1] if np.isscalar(args[1]) else "panel", r)
        assert r["rel"] <= tol, r


@pytest.mark.parametrize("window", [2, 10, 20])
def test_corr_matches_panelops(be, data, window):
    x, y, gx, gy = data
    p = {"window": window, "min_periods": window}
    for (a, b), (ga, gb) in (((x, y), (gx, gy)), ((x, x), (gx, gx))):
        r = compare(be.to_host(be.compute("ts_corr", p, [ga, gb], (T, N))),
                    panelops.ts_corr(a, b, window=window))  # fmt: skip
        assert r["missing"] == 0, r
        assert r["rel"] <= SPREAD_TOL, r


def _all(be, a):
    return be.upload(np.ones(np.shape(a), dtype=bool))


def _returns(x):
    r = np.vstack([x[5:] / x[:-5] - 1, np.full((5, x.shape[1]), np.nan)])
    return np.ascontiguousarray(r)


@pytest.mark.parametrize("case", ["plain", "ties", "sparse"])
def test_rank_ic_matches_panelops(be, data, case):
    x, y, _, _ = data
    ret = _returns(x)
    if case == "plain":
        score = y
    elif case == "ties":  # 时序排名只有 1..5 几个值，收益保留两位小数
        score, ret = panelops.ts_rank(x, window=5), np.round(ret, 2)
    else:  # 某些天只剩 0、1、2 个有效值，某天因子全相同（方差为 0）
        score = y.copy()
        score[10, 2:] = np.nan
        score[11, 1:] = np.nan
        score[12, :] = np.nan
        score[13, :] = 1.0
    score, ret = np.ascontiguousarray(score), np.ascontiguousarray(ret)
    got = be.to_host(be.rank_ic(be.upload(score), _all(be, score), be.upload(ret))[0])
    want = panelops.rank_ic(score, ret)
    assert np.array_equal(np.isfinite(got), np.isfinite(want))
    ok = np.isfinite(want)
    assert np.abs(got[ok] - want[ok]).max() <= RANK_IC_TOL


@pytest.mark.parametrize("n", [1, 31, 33, 1025, 8192, 8193])
def test_rank_ic_widths_and_universe(be, n):
    """截面宽度跨过线程组边界（32、1024）、单线程组上限 8192（更宽的走 argsort 写法）；
    股票池外的格子不参与。"""
    rng = np.random.default_rng(n)
    t = 12
    score = rng.normal(size=(t, n))
    ret = np.round(rng.normal(size=(t, n)), 1)
    score[rng.random((t, n)) < 0.05] = np.nan
    ret[rng.random((t, n)) < 0.05] = np.nan
    univ = rng.random((t, n)) < 0.9
    got = be.to_host(be.rank_ic(be.upload(score), be.upload(univ), be.upload(ret))[0])
    want = panelops.rank_ic(np.ascontiguousarray(np.where(univ, score, np.nan)), ret)
    assert np.array_equal(np.isfinite(got), np.isfinite(want))
    ok = np.isfinite(want)
    if ok.any():
        assert np.abs(got[ok] - want[ok]).max() <= RANK_IC_TOL


def test_returns_order_is_reused_per_panel(be, data):
    """收益排序按同一份设备数组缓存；换一份收益必须重排（缓存键是数组本身，不是形状）。"""
    x, y, _, _ = data
    r1, r2 = _returns(x), _returns(x[::-1].copy())
    s = be.upload(y)
    a = be.to_host(be.rank_ic(s, _all(be, y), be.upload(r1))[0])
    b = be.to_host(be.rank_ic(s, _all(be, y), be.upload(r2))[0])
    assert np.allclose(a, panelops.rank_ic(y, r1), atol=RANK_IC_TOL, equal_nan=True)
    assert np.allclose(b, panelops.rank_ic(y, r2), atol=RANK_IC_TOL, equal_nan=True)


def test_range_flag(be):
    ok = be.upload(np.array([[1.0, -3e14, np.nan, 0.0], [2e-15, np.inf, 5.0, -1.0]]))
    big = be.upload(np.array([[1.0, 2e15]]))
    tiny = be.upload(np.array([[1.0, -5e-16]]))
    flags = [bool(be.range_flag(a).item()) for a in (ok, big, tiny)]
    assert flags == [False, True, True]
    assert bool(be.any_flag([be.range_flag(ok), be.range_flag(big)]).item())


def test_fused_chain_matches_composition(be, data):
    """逐元素算子链融合成一个内核（后缀程序）：单个算子、带常量的链都与 panelops 逐步组合一致。"""
    from ashare.factor.mining.gpu.kernels import ELEM_OPS as OP

    x, y, gx, gy = data
    for op, tol in BINARY:
        got, _ = be.fuse(
            [(OP["load"], 0), (OP["load"], 1), (OP[op], 0)], [gx, gy], [], (T, N)
        )
        r = compare(be.to_host(got), panelops.elementwise(op, x, y))
        assert r["missing"] == 0 and r["rel"] <= tol, (op, r)
    for op in ("negate", "abs", "sign", "log"):
        got, _ = be.fuse([(OP["load"], 0), (OP[op], 0)], [gy], [], (T, N))
        r = compare(be.to_host(got), panelops.elementwise(op, y))
        assert r["missing"] == 0 and r["rel"] <= MEAN_TOL, (op, r)
    # log(abs(x / y) + 1.5) - max(y, -x)
    prog = [(OP["load"], 0), (OP["load"], 1), (OP["divide"], 0), (OP["abs"], 0),
            (OP["const"], 0), (OP["add"], 0), (OP["log"], 0), (OP["load"], 1),
            (OP["load"], 0), (OP["negate"], 0), (OP["maximum"], 0), (OP["subtract"], 0)]  # fmt: skip
    got, flag = be.fuse(prog, [gx, gy], [1.5], (T, N))
    e = panelops.elementwise
    want = e("subtract", e("log", e("add", e("abs", e("divide", x, y)), 1.5)),
             e("maximum", y, e("negate", x)))  # fmt: skip
    r = compare(be.to_host(got), want)
    assert r["missing"] == 0 and r["rel"] <= DELTA_TOL, r
    assert not bool(flag.item())


def test_fused_range_flag(be):
    """乘除结果超出 float32 安全范围、非零相乘下溢成 0 都要标记；除以 0 不标记（与 float64 同为缺失）。"""
    from ashare.factor.mining.gpu.kernels import ELEM_OPS as OP

    def run(op, a, b):
        ga, gb = be.upload(np.array([a])), be.upload(np.array([b]))
        _, f = be.fuse(
            [(OP["load"], 0), (OP["load"], 1), (OP[op], 0)], [ga, gb], [], (1, len(a))
        )
        return bool(f.item())

    assert not run("multiply", [3.0, 1e7], [4.0, 1e7])
    assert run("multiply", [3.0, 1e8], [4.0, 1e8])  # 1e16 > 1e15
    assert run("multiply", [1e-20, 1.0], [1e-20, 1.0])  # 1e-40：float32 下溢
    assert run("divide", [1.0, 2.0], [1e-16, 1.0])
    assert not run("divide", [1.0, 2.0], [0.0, 1.0])
    assert not run("add", [1e14, 0.0], [1e14, 0.0])
