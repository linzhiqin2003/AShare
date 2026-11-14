"""ashare.factor.groupbt.weights：手算例子、逐行逐列参考实现与随机对拍。"""

import numpy as np
import pytest

from ashare.factor.groupbt import weights as W

# ---- 逐行逐列的笨参考实现 --------------------------------------------------------


# Rust 内核对 numpy 参考实现：真值为 0 的格子两边都是 ±1e-17 量级的舍入噪声，求和顺序随平台
# （x86 与 ARM 的 numpy SIMD）不同，纯相对误差会判成 100%–200%（2026-11-14 第一次在 Linux CI 上跑）。
# 权重量级为 1，1e-12 远小于任何有意义的差别。
ATOL = 1e-12


def ref_cs_demean(x):
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        vals = [x[i, j] for j in range(x.shape[1]) if not np.isnan(x[i, j])]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        for j in range(x.shape[1]):
            if not np.isnan(x[i, j]):
                out[i, j] = x[i, j] - mean
    return out


def ref_group_demean(x, groups):
    g = np.asarray(groups)
    if g.ndim == 1:
        g = np.tile(g, (x.shape[0], 1))
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.isnan(x[i, j]) or g[i, j] < 0:
                continue
            vals = [
                x[i, k]
                for k in range(x.shape[1])
                if not np.isnan(x[i, k]) and g[i, k] == g[i, j]
            ]
            out[i, j] = x[i, j] - sum(vals) / len(vals)
    return out


def ref_cap_buckets(logcap, n):
    out = np.full(logcap.shape, -1, dtype=np.int64)
    for i in range(logcap.shape[0]):
        cols = [j for j in range(logcap.shape[1]) if not np.isnan(logcap[i, j])]
        # 稳定插入排序：同值保持列下标顺序
        for a in range(1, len(cols)):
            key = cols[a]
            b = a - 1
            while b >= 0 and logcap[i, cols[b]] > logcap[i, key]:
                cols[b + 1] = cols[b]
                b -= 1
            cols[b + 1] = key
        c = len(cols)
        for r, j in enumerate(cols):
            out[i, j] = r * n // c
    return out


def ref_vector_neut(x, y):
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        cols = [
            j
            for j in range(x.shape[1])
            if np.isfinite(x[i, j]) and np.isfinite(y[i, j])
        ]
        if len(cols) < 2:
            continue
        mx = sum(x[i, j] for j in cols) / len(cols)
        my = sum(y[i, j] for j in cols) / len(cols)
        den = 0.0
        num = 0.0
        for j in cols:
            cx = x[i, j] - mx
            cy = y[i, j] - my
            den += cy * cy
            num += cx * cy
        beta = num / den if den != 0 else 0.0
        for j in cols:
            r = (x[i, j] - mx) - beta * (y[i, j] - my)
            out[i, j] = r if np.isfinite(r) else np.nan
    return out


def ref_normalize(x):
    d = ref_cs_demean(x)
    out = np.full(d.shape, np.nan)
    for i in range(d.shape[0]):
        cols = [j for j in range(d.shape[1]) if not np.isnan(d[i, j])]
        if len(cols) < 2:
            continue
        total = sum(abs(d[i, j]) for j in cols)
        if total == 0:
            continue
        for j in cols:
            out[i, j] = d[i, j] / total
    return out


def ref_decay_linear(w, n):
    w = np.asarray(w, dtype=np.float64)
    if n <= 1:
        return w.copy()
    w0 = np.where(np.isnan(w), 0.0, w)
    out = np.zeros(w.shape)
    for i in range(w.shape[0]):
        k_max = min(n - 1, i)
        den = sum(n - k for k in range(k_max + 1))
        for j in range(w.shape[1]):
            num = sum((n - k) * w0[i - k, j] for k in range(k_max + 1))
            out[i, j] = num / den
            if np.isnan(w[i, j]):
                out[i, j] = np.nan
    return out


def ref_build_weights(
    factor,
    universe,
    *,
    industry=None,
    logcap=None,
    group_neutralize=None,
    n_cap_buckets=10,
    vector=None,
    decay=0,
):
    x = np.array(factor, dtype=np.float64, copy=True)
    if group_neutralize == "industry":
        x = ref_group_demean(x, industry)
    elif group_neutralize == "cap":
        x = ref_group_demean(x, ref_cap_buckets(logcap, n_cap_buckets))
    elif group_neutralize == "cap+industry":
        x = ref_vector_neut(ref_group_demean(x, industry), logcap)
    if vector is not None:
        ys = vector if isinstance(vector, (list, tuple)) else [vector]
        for y in ys:
            x = ref_vector_neut(x, y)
    x = np.where(universe, x, np.nan)
    alpha = ref_cs_demean(x)
    return {"alpha": alpha, "weights": ref_decay_linear(ref_normalize(alpha), decay)}


def _panel(rng, shape, nan_frac=0.2):
    x = rng.standard_normal(shape)
    x[rng.random(shape) < nan_frac] = np.nan
    return x


# ---- 手算例子 -------------------------------------------------------------------


def test_cs_demean_hand():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, np.nan, 4.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    expected = np.array(
        [
            [-1.5, -0.5, 0.5, 1.5],
            [-1.0, np.nan, 1.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    np.testing.assert_allclose(W.cs_demean(x), expected, equal_nan=True, atol=ATOL)


def test_group_demean_hand():
    x = np.array([[1.0, 3.0, 10.0, 14.0], [2.0, np.nan, 5.0, 10.0]])
    groups = np.array([0, 0, -1, 1])
    expected = np.array([[-1.0, 1.0, np.nan, 0.0], [0.0, np.nan, np.nan, 0.0]])
    np.testing.assert_allclose(
        W.group_demean(x, groups), expected, equal_nan=True, atol=ATOL
    )


def test_cap_buckets_hand():
    logcap = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, np.nan, 1.0, 3.0, 3.0]])
    expected = np.array([[0, 0, 0, 1, 1], [1, -1, 0, 0, 1]], dtype=np.int64)
    got = W.cap_buckets(logcap, n=2)
    assert got.dtype == np.int64
    np.testing.assert_allclose(got, expected, equal_nan=True, atol=ATOL)
    # 只有一只有效股票：名次 0、桶号 0
    one = np.array([[np.nan, 7.0, np.nan]])
    np.testing.assert_array_equal(W.cap_buckets(one), [[-1, 0, -1]])


def test_vector_neut_hand():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, np.nan, 5.0, 8.0]])
    y = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, np.nan, 2.0, 3.0]])
    expected = np.array([[-1.0, -1.0, 1.0, 1.0], [0.0, np.nan, 0.0, 0.0]])
    np.testing.assert_allclose(W.vector_neut(x, y), expected, equal_nan=True, atol=ATOL)


def test_vector_neut_inf_becomes_nan():
    x = np.array([[1.0, 2.0, np.inf, 4.0]])
    y = np.array([[0.0, 1.0, 0.0, 1.0]])
    expected = np.array([[0.0, -1.0, np.nan, 1.0]])
    np.testing.assert_allclose(
        W.vector_neut(x, y), expected, atol=1e-15, equal_nan=True
    )


def test_normalize_hand():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, np.nan, 4.0, np.nan]])
    expected = np.array(
        [
            [-0.375, -0.125, 0.125, 0.375],
            [-0.5, np.nan, 0.5, np.nan],
        ]
    )
    np.testing.assert_allclose(W.normalize(x), expected, equal_nan=True, atol=ATOL)
    # 只有一只有效股票：不足 2 个有效值，整行 NaN
    assert np.all(np.isnan(W.normalize(np.array([[3.0, np.nan, np.nan]]))))


def test_decay_linear_hand():
    w = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]])
    expected = np.array(
        [
            [1.0, 2.0],
            [7.0 / 3.0, 10.0 / 3.0],
            [np.nan, 16.0 / 3.0],
        ]
    )
    np.testing.assert_allclose(
        W.decay_linear(w, 2), expected, equal_nan=True, atol=ATOL
    )


def test_decay_linear_short_history_not_renormalized():
    # n=3、t=1 时 K=1，分母仍是 3+2；过去的 NaN 当 0，不重新归一化
    w = np.array([[np.nan], [5.0]])
    expected = np.array([[np.nan], [3.0]])
    np.testing.assert_allclose(
        W.decay_linear(w, 3), expected, equal_nan=True, atol=ATOL
    )


def test_decay_linear_n_le_1_is_copy():
    w = np.array([[1.0, np.nan]])
    for n in (1, 0, -2):
        got = W.decay_linear(w, n)
        np.testing.assert_allclose(got, w, equal_nan=True, atol=ATOL)
        assert got is not w


def test_build_weights_hand_order():
    # 全市场先按行业中性化：第 0 组全市场均值 2（含池外的股票 3），
    # 再抽池（去掉第 2 列），最后池内去均值。若先抽池再中性化，
    # 第 0 组均值为 1.5，结果会不同。
    factor = np.array([[1.0, 2.0, 3.0, 100.0]])
    industry = np.array([[0, 0, 0, 1]])
    universe = np.array([[True, True, False, True]])
    out = W.build_weights(
        factor, universe, industry=industry, group_neutralize="industry"
    )
    np.testing.assert_allclose(
        out["alpha"],
        np.array([[-2.0 / 3.0, 1.0 / 3.0, np.nan, 1.0 / 3.0]]),
        equal_nan=True,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        out["weights"],
        np.array([[-0.5, 0.25, np.nan, 0.25]]),
        equal_nan=True,
        atol=ATOL,
    )


# ---- 随机数据对拍 ---------------------------------------------------------------


def test_random_cs_demean():
    rng = np.random.default_rng(0)
    x = _panel(rng, (6, 5))
    np.testing.assert_allclose(
        W.cs_demean(x), ref_cs_demean(x), equal_nan=True, atol=ATOL
    )


def test_random_group_demean():
    rng = np.random.default_rng(0)
    x = _panel(rng, (6, 5))
    groups = rng.integers(-1, 3, size=(6, 5))
    np.testing.assert_allclose(
        W.group_demean(x, groups),
        ref_group_demean(x, groups),
        equal_nan=True,
        atol=ATOL,
    )
    # (N,) 的组标签对每行相同
    row_groups = rng.integers(-1, 3, size=5)
    np.testing.assert_allclose(
        W.group_demean(x, row_groups),
        ref_group_demean(x, row_groups),
        equal_nan=True,
        atol=ATOL,
    )


def test_random_cap_buckets_with_ties():
    rng = np.random.default_rng(0)
    logcap = np.round(_panel(rng, (6, 5), 0.15), 1)
    for n in (2, 3, 10):
        np.testing.assert_allclose(
            W.cap_buckets(logcap, n),
            ref_cap_buckets(logcap, n),
            equal_nan=True,
            atol=ATOL,
        )


def test_random_vector_neut():
    rng = np.random.default_rng(0)
    x = _panel(rng, (6, 5))
    y = _panel(rng, (6, 5))
    np.testing.assert_allclose(
        W.vector_neut(x, y), ref_vector_neut(x, y), equal_nan=True, atol=ATOL
    )


def test_random_normalize():
    rng = np.random.default_rng(0)
    x = _panel(rng, (6, 5))
    np.testing.assert_allclose(
        W.normalize(x), ref_normalize(x), equal_nan=True, atol=ATOL
    )


def test_random_decay_linear():
    rng = np.random.default_rng(0)
    w = _panel(rng, (6, 5))
    for n in (1, 2, 3, 5, 9):
        np.testing.assert_allclose(
            W.decay_linear(w, n), ref_decay_linear(w, n), equal_nan=True, atol=ATOL
        )


def test_random_build_weights():
    rng = np.random.default_rng(0)
    factor = _panel(rng, (6, 5))
    industry = rng.integers(0, 3, size=(6, 5))
    logcap = _panel(rng, (6, 5))
    vector = _panel(rng, (6, 5))
    universe = rng.random((6, 5)) > 0.25
    configs = [
        dict(),
        dict(industry=industry, group_neutralize="industry"),
        dict(logcap=logcap, group_neutralize="cap"),
        dict(industry=industry, logcap=logcap, group_neutralize="cap+industry"),
        dict(vector=[vector, factor], decay=2),
        dict(
            industry=industry,
            logcap=logcap,
            group_neutralize="cap+industry",
            vector=vector,
            decay=3,
        ),
    ]
    for cfg in configs:
        got = W.build_weights(factor, universe, **cfg)
        exp = ref_build_weights(factor, universe, **cfg)
        np.testing.assert_allclose(
            got["alpha"], exp["alpha"], equal_nan=True, atol=ATOL
        )
        np.testing.assert_allclose(
            got["weights"], exp["weights"], equal_nan=True, atol=ATOL
        )


# ---- 性质与边界 -----------------------------------------------------------------


def test_vector_neut_residual_properties():
    rng = np.random.default_rng(0)
    x = _panel(rng, (5, 6), 0.15)
    y = _panel(rng, (5, 6), 0.15)
    resid = W.vector_neut(x, y)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 2:
            assert np.all(np.isnan(resid[i]))
            continue
        cy = y[i, m] - y[i, m].mean()
        assert abs(resid[i, m].sum()) < 1e-9
        assert abs(np.dot(resid[i, m], cy)) < 1e-9


def test_vector_neut_constant_y_equals_cs_demean():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, np.nan, 7.0, 9.0]])
    y = np.ones_like(x)
    y[np.isnan(x)] = np.nan
    np.testing.assert_allclose(
        W.vector_neut(x, y), W.cs_demean(x), equal_nan=True, atol=ATOL
    )


def test_normalize_positive_sum_is_half():
    rng = np.random.default_rng(0)
    got = W.normalize(_panel(rng, (6, 5)))
    for i in range(got.shape[0]):
        row = got[i][~np.isnan(got[i])]
        if row.size == 0:
            continue
        assert abs(np.abs(row).sum() - 1.0) < 1e-12
        assert abs(row[row > 0].sum() - 0.5) < 1e-12


def test_boundaries():
    all_nan = np.full((2, 4), np.nan)
    x = np.array([[np.nan, 7.0, np.nan, np.nan]])
    np.testing.assert_allclose(W.cs_demean(all_nan), all_nan, equal_nan=True, atol=ATOL)
    np.testing.assert_allclose(
        W.group_demean(all_nan, np.zeros((2, 4), int)),
        all_nan,
        equal_nan=True,
        atol=ATOL,
    )
    np.testing.assert_array_equal(W.cap_buckets(all_nan), np.full((2, 4), -1))
    np.testing.assert_allclose(
        W.vector_neut(all_nan, all_nan), all_nan, equal_nan=True, atol=ATOL
    )
    np.testing.assert_allclose(W.normalize(all_nan), all_nan, equal_nan=True, atol=ATOL)
    # 单只有效股票：去均值后为 0，但 normalize/vector_neut 不足 2 个有效值为 NaN
    np.testing.assert_allclose(
        W.cs_demean(x),
        np.array([[np.nan, 0.0, np.nan, np.nan]]),
        equal_nan=True,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        W.group_demean(x, np.zeros((1, 4), int)),
        np.array([[np.nan, 0.0, np.nan, np.nan]]),
        equal_nan=True,
        atol=ATOL,
    )
    assert np.all(np.isnan(W.normalize(x)))
    assert np.all(np.isnan(W.vector_neut(x, x)))


def test_build_weights_errors():
    factor = np.ones((1, 3))
    universe = np.ones((1, 3), dtype=bool)
    with pytest.raises(ValueError):
        W.build_weights(factor, universe, group_neutralize="industry")
    with pytest.raises(ValueError):
        W.build_weights(factor, universe, group_neutralize="cap")
    with pytest.raises(ValueError):
        W.build_weights(
            factor,
            universe,
            industry=np.zeros((1, 3), int),
            group_neutralize="cap+industry",
        )
    with pytest.raises(ValueError):
        W.build_weights(factor, universe, group_neutralize="sector")


def test_inputs_not_modified():
    rng = np.random.default_rng(0)
    factor = _panel(rng, (5, 4))
    universe = rng.random((5, 4)) > 0.3
    industry = rng.integers(0, 2, size=(5, 4))
    logcap = _panel(rng, (5, 4))
    vec = _panel(rng, (5, 4))
    copies = [a.copy() for a in (factor, universe, industry, logcap, vec)]
    W.cs_demean(factor)
    W.group_demean(factor, industry)
    W.cap_buckets(logcap)
    W.vector_neut(factor, vec)
    W.normalize(factor)
    W.decay_linear(factor, 3)
    W.build_weights(
        factor,
        universe,
        industry=industry,
        logcap=logcap,
        group_neutralize="cap+industry",
        vector=vec,
        decay=2,
    )
    for a, b in zip((factor, universe, industry, logcap, vec), copies, strict=True):
        np.testing.assert_array_equal(a, b)
