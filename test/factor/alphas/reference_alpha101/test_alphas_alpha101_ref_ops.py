"""Alpha101 参考实现算子的测试：手算小例子，加上与朴素循环版本的对拍。

情形覆盖：并列、NaN、常数窗口、窗口不满、非整数窗口取整、两种 argmax 读法。
"""

import math

import numpy as np
import pytest

from ashare.factor.alphas.reference import alpha101_ref_ops as op

nan = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def same(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- 手算小例子


def test_window_floor():
    assert op.window(3.92795) == 3
    assert op.window(1.06157) == 1
    assert op.window(5) == 5
    with pytest.raises(ValueError):
        op.window(0.9)


def test_delay_delta():
    x = col(1, 2, 4, nan, 8)
    same(op.delay(x, 1), col(nan, 1, 2, 4, nan))
    same(op.delta(x, 2), col(nan, nan, 3, nan, 4))
    same(op.delta(x, 2.9), col(nan, nan, 3, nan, 4))  # 取整为 2
    same(op.delay(x, 9), col(nan, nan, nan, nan, nan))


def test_ts_sum_min_max_product():
    x = col(1, 2, 3, nan, 5, 6, 7)
    same(op.ts_sum(x, 3), col(nan, nan, 6, nan, nan, nan, 18))
    same(op.ts_min(x, 2), col(nan, 1, 2, nan, nan, 5, 6))
    same(op.ts_max(x, 2.5), col(nan, 2, 3, nan, nan, 6, 7))
    same(op.ts_product(x, 3), col(nan, nan, 6, nan, nan, nan, 210))
    same(op.ts_sum(x, 1), x)


def test_stddev_hand():
    x = col(1, 2, 3, 4, 4, 4, 4)
    out = op.stddev(x, 4)
    assert np.isnan(out[:3]).all()
    assert out[3, 0] == pytest.approx(
        math.sqrt(5 / 3), abs=1e-15
    )  # 偏差平方和 5，分母 3
    assert out[6, 0] == 0.0  # 常数窗口精确为 0


def test_covariance_correlation_hand():
    x = col(1, 2, 3, 4, 5)
    y = col(2, 4, 6, 8, 11)
    cov = op.covariance(x, y, 3)
    same(cov[2:], col(2, 2, 2.5))  # 例：x=1,2,3 y=2,4,6 → Σ(dx·dy)=4，/2
    corr = op.correlation(x, y, 3)
    same(corr[2:4], col(1, 1))
    # x=3,4,5 y=6,8,11：dx=-1,0,1 dy=-7/3,-1/3,8/3 → Σdxdy=5，Σdx²=2，Σdy²=114/9
    assert corr[4, 0] == pytest.approx(5 / math.sqrt(2 * 114 / 9), abs=1e-14)


def test_correlation_constant_nan_and_not_full():
    x = col(1, 1, 1, 2, 3, nan, 4, 5)
    y = col(3, 1, 2, 2, 2, 4, 5, 9)
    out = op.correlation(x, y, 3)
    assert np.isnan(out[:2]).all()  # 窗口不满
    assert np.isnan(out[2, 0])  # x 常数
    assert np.isnan(out[4, 0])  # y 常数（2,2,2）
    assert not np.isnan(out[3, 0])
    assert np.isnan(out[5:8]).all()  # 窗口里有 NaN
    # 反相关
    same(op.correlation(col(1, 2, 3), col(3, 2, 1), 3)[2], [-1.0])


def test_decay_linear_hand():
    x = col(1, 2, 3, nan, 1, 1)
    out = op.decay_linear(x, 3)
    assert out[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6, abs=1e-15)
    assert np.isnan(out[3:6]).all()
    same(op.decay_linear(col(5, 7), 2)[1], [(5 + 14) / 3])


def test_ts_rank_hand_ties():
    x = col(1, 2, 2, 3, 3, 3, nan, 1)
    out = op.ts_rank(x, 3)
    assert np.isnan(out[:2]).all()
    same(out[2], [(1 + 1.5) / 3])  # [1,2,2]：今天 2，有 1 个更小，2 个相等 → 平均秩 2.5
    same(out[3], [1.0])  # [2,2,3]
    same(out[4], [(1 + 1.5) / 3])  # [2,3,3]
    same(out[5], [2 / 3])  # [3,3,3]：常数窗口 → 平均秩 2
    assert np.isnan(out[6:]).all()


def test_ts_argmax_two_readings_hand():
    x = col(1, 3, 2, 3, 0)
    a = op.ts_argmax(x, 3, how="days_ago")
    b = op.ts_argmax(x, 3, how="position")
    same(a[2:], col(1, 0, 1))  # [1,3,2]：1 天前；[3,2,3]：并列取最近 → 0；[2,3,0] → 1
    same(b[2:], col(2, 1, 2))  # 位置从 1 起，并列取最早
    m = op.ts_argmin(col(2, 1, 1, 5), 3, how="days_ago")
    same(m[2:], col(0, 1))  # [2,1,1]：并列取最近 → 0
    p = op.ts_argmin(col(2, 1, 1, 5), 3, how="position")
    same(p[2:], col(2, 1))
    with pytest.raises(ValueError):
        op.ts_argmax(x, 3, how="other")
    assert np.isnan(op.ts_argmax(col(1, nan, 2, 3), 2)[1:3]).all()


def test_rank_ties_nan():
    x = np.array([[3, 1, nan, 3], [nan, nan, nan, nan], [2, 2, 2, 2]], float)
    out = op.rank(x)
    same(out[0], [2.5 / 3, 1 / 3, nan, 2.5 / 3])
    assert np.isnan(out[1]).all()
    same(out[2], [0.625] * 4)  # 全并列：平均秩 2.5 / 4
    same(op.rank(np.array([[1.0, np.inf, 2.0]])), [[0.5, nan, 1.0]])


def test_scale():
    same(op.scale(np.array([[1, -3, nan]], float)), [[0.25, -0.75, nan]])
    same(op.scale(np.array([[1, -3, nan]], float), 2), [[0.5, -1.5, nan]])
    assert np.isnan(op.scale(np.array([[0.0, 0.0, nan]]))).all()


def test_indneutralize():
    x = np.array([[1, 3, 5, 7, nan], [2, nan, 4, 4, 1]], float)
    g = np.array([[1, 1, 2, nan, 2], [1, 1, 2, 2, nan]], float)
    out = op.indneutralize(x, g)
    same(out[0], [-1, 1, 0, nan, nan])  # 第 5 只 x 缺失不参与组 2 的均值
    same(out[1], [0, nan, 0, 0, nan])  # 单只成组为 0


def test_elementwise_nan_rules():
    a = np.array([[1, nan, 3, 0]], float)
    b = np.array([[2, 1, nan, 0]], float)
    same(op.lt(a, b), [[1, nan, nan, 0]])
    same(op.eq(a, b), [[0, nan, nan, 1]])
    same(op.or_(a, b), [[1, nan, nan, 0]])
    same(op.where(op.lt(a, b), 10, b), [[10, nan, nan, 0]])
    same(op.where(np.array([[1.0, 0.0]]), np.array([[nan, 1.0]]), 5), [[nan, 5]])
    same(op.div(np.array([[1.0, 0.0, -1.0]]), 0), [[nan, nan, nan]])
    same(op.log(np.array([[1.0, 0.0, -1.0]])), [[0, nan, nan]])
    same(op.power(np.array([[-8.0, 4.0, 0.0]]), 0.5), [[nan, 2, 0]])
    same(op.power(np.array([[-2.0]]), 2), [[4]])
    same(op.signedpower(np.array([[-2.0, 3.0, 0.0]]), 2), [[-4, 9, 0]])
    same(op.signedpower(np.array([[-4.0, 4.0]]), np.array([[0.5, -0.5]])), [[-2, 0.5]])
    same(op.sign(np.array([[-2.0, 0.0, nan]])), [[-1, 0, nan]])
    same(op.maximum(a, b), [[2, nan, nan, 0]])
    same(op.minimum(a, b), [[1, nan, nan, 0]])


def test_adv_is_amount_mean():
    amount = col(10, 20, 30, nan, 50)
    same(op.adv(amount, 2), col(nan, 15, 25, nan, nan))


# ---------------------------------------------------------------- 朴素循环对拍


def _naive(x, d, func):
    d = math.floor(d)
    t, n = x.shape
    out = np.full((t, n), np.nan)
    for j in range(n):
        for i in range(d - 1, t):
            w = x[i - d + 1 : i + 1, j]
            if np.isnan(w).any():
                continue
            out[i, j] = func(w)
    return out


def _naive2(x, y, d, func):
    d = math.floor(d)
    t, n = x.shape
    out = np.full((t, n), np.nan)
    for j in range(n):
        for i in range(d - 1, t):
            a = x[i - d + 1 : i + 1, j]
            b = y[i - d + 1 : i + 1, j]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            out[i, j] = func(a, b)
    return out


def _naive_rank_last(w):
    last = w[-1]
    ranks = [k + 1 for k, v in enumerate(sorted(w)) if v == last]
    return sum(ranks) / len(ranks) / len(w)


def _naive_corr(a, b):
    if a.max() == a.min() or b.max() == b.min():
        return np.nan
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sab = sum((p - ma) * (q - mb) for p, q in zip(a, b, strict=True))
    saa = sum((p - ma) ** 2 for p in a)
    sbb = sum((q - mb) ** 2 for q in b)
    return sab / math.sqrt(saa * sbb)


def _naive_argmax(w, how, best):
    target = best(w)
    hits = [k for k, v in enumerate(w) if v == target]
    if how == "days_ago":
        return len(w) - 1 - hits[-1]
    return hits[0] + 1


@pytest.fixture
def rough():
    """带并列（取整）、NaN、常数列、整段常数和小样本的随机面板。"""
    rng = np.random.default_rng(7)
    t, n = 60, 9
    x = np.round(rng.normal(0, 2, (t, n)), 0)
    y = np.round(rng.normal(0, 2, (t, n)), 1)
    x[rng.uniform(size=(t, n)) < 0.05] = np.nan
    y[rng.uniform(size=(t, n)) < 0.05] = np.nan
    x[:, 3] = 2.0
    y[20:35, 4] = 1.5
    x[:, 5] = np.nan
    return x, y


@pytest.mark.parametrize("d", [1, 2, 3, 5.7, 10])
def test_rolling_against_naive(rough, d):
    x, y = rough
    tol = dict(rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.allclose(op.ts_sum(x, d), _naive(x, d, np.sum), **tol)
    assert np.allclose(op.ts_mean(x, d), _naive(x, d, np.mean), **tol)
    assert np.allclose(op.ts_min(x, d), _naive(x, d, np.min), **tol)
    assert np.allclose(op.ts_max(x, d), _naive(x, d, np.max), **tol)
    assert np.allclose(op.ts_product(y, d), _naive(y, d, np.prod), **tol)
    w = math.floor(d)
    weights = np.arange(1, w + 1)
    assert np.allclose(
        op.decay_linear(y, d),
        _naive(y, d, lambda v: sum(v * weights) / weights.sum()),
        **tol,
    )
    assert np.allclose(op.ts_rank(x, d), _naive(x, d, _naive_rank_last), **tol)
    for how in op.ARGMAX_READINGS:
        assert np.allclose(
            op.ts_argmax(x, d, how=how),
            _naive(x, d, lambda v, h=how: _naive_argmax(v, h, max)),
            **tol,
        )
        assert np.allclose(
            op.ts_argmin(x, d, how=how),
            _naive(x, d, lambda v, h=how: _naive_argmax(v, h, min)),
            **tol,
        )
    if w >= 2:
        assert np.allclose(
            op.stddev(y, d), _naive(y, d, lambda v: np.std(v, ddof=1)), **tol
        )
        assert np.allclose(
            op.covariance(x, y, d),
            _naive2(x, y, d, lambda a, b: np.cov(a, b, ddof=1)[0, 1]),
            **tol,
        )
        assert np.allclose(
            op.correlation(x, y, d), _naive2(x, y, d, _naive_corr), **tol
        )


def test_rolling_block_split_is_invisible(rough, monkeypatch):
    """按列分块不改变结果：把分块上限压到每块 1 列再算一遍。"""
    x, y = rough
    whole = op.correlation(x, y, 5), op.ts_rank(x, 4), op.decay_linear(y, 3)
    monkeypatch.setattr(op, "_BLOCK_BYTES", 1)
    split = op.correlation(x, y, 5), op.ts_rank(x, 4), op.decay_linear(y, 3)
    for a, b in zip(whole, split, strict=True):
        np.testing.assert_array_equal(a, b)


def test_window_longer_than_data():
    x = col(1, 2, 3)
    assert np.isnan(op.ts_sum(x, 5)).all()
    assert np.isnan(op.correlation(x, x, 5)).all()


def test_rank_matches_naive(rough):
    x, _ = rough
    out = op.rank(x)
    for i in range(x.shape[0]):
        row = x[i]
        valid = row[~np.isnan(row)]
        for j, v in enumerate(row):
            if np.isnan(v):
                assert np.isnan(out[i, j])
                continue
            ranks = [k + 1 for k, u in enumerate(sorted(valid)) if u == v]
            assert out[i, j] == pytest.approx(sum(ranks) / len(ranks) / len(valid))


# ---------------------------------------------------------------- 2026-11-13 裁决：幂的缺失、窗口和与顺序无关


def test_power_missing_operand_is_missing():
    # np.power 给 1 ** NaN = 1、NaN ** 0 = 1；按「任一操作数缺失则结果缺失」都应为 NaN。
    x = np.array([[1.0, nan, 2.0, 1.0]])
    y = np.array([[nan, 0.0, 3.0, 2.0]])
    out = op.power(x, y)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[0, 2] == 8.0 and out[0, 3] == 1.0
    sp = op.signedpower(np.array([[-1.0, nan]]), np.array([[nan, 0.0]]))
    assert np.isnan(sp).all()


def test_window_sum_independent_of_order():
    # 窗口里进出的值相同（[a, b, c, a] 的两个 3 日窗口 {a,b,c} 与 {b,c,a}）：两个窗口和逐位相等，
    # 于是 2 日相关里那一侧方差为精确的 0、结果为 NaN，而不是舍入噪声给出的 ±0.707。
    a, b, c = 33.411433, 14.750406000000002, 3.1222730000000003
    s = op.ts_sum(col(a, b, c, a), 3)
    assert s[2, 0] == s[3, 0]
    corr = op.correlation(s, col(0.0, 0.0, 1.0, 2.0), 2)
    assert np.isnan(corr[3, 0])
    # 同一组数任意排列，均值、标准差逐位相同
    vals = [0.1, 0.7, 1e8, -3.3, 2.2]
    m = [op.ts_mean(col(*p), 5)[4, 0] for p in (vals, vals[::-1], vals[2:] + vals[:2])]
    sd = [op.stddev(col(*p), 5)[4, 0] for p in (vals, vals[::-1], vals[2:] + vals[:2])]
    assert len(set(m)) == 1 and len(set(sd)) == 1
