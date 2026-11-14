"""国君 191 参考实现辅助函数的手算小例子和朴素循环对拍。"""

import math

import numpy as np
import pytest

from ashare.factor.alphas.reference import gtja191_ref_ops as op

NAN = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def same(a, b):
    np.testing.assert_allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12, equal_nan=True
    )


# ------------------------------------------------------------ 逐元素


def test_div_log_pow_non_finite_to_nan():
    same(op.div(col(1, 0, -1, NAN), col(0, 0, 2, 1)), col(NAN, NAN, -0.5, NAN))
    same(op.log(col(1, 0, -1, NAN, math.e)), col(0, NAN, NAN, NAN, 1))
    same(op.pw(col(-8, 4, 0, -2), col(1 / 3, 0.5, -1, 3)), col(NAN, 2, NAN, -8))


def test_compare_and_logic_nan():
    a = col(1, 2, NAN, 3)
    b = col(1, 1, 1, NAN)
    same(op.gt(a, b), col(0, 1, NAN, NAN))
    same(op.ge(a, b), col(1, 1, NAN, NAN))
    same(op.eq(a, b), col(1, 0, NAN, NAN))
    same(op.le(a, b), col(1, 0, NAN, NAN))
    same(op.lt(a, b), col(0, 0, NAN, NAN))
    same(op.and_(col(1, 0, 2, NAN), col(1, 1, 0, 0)), col(1, 0, 0, NAN))
    same(op.or_(col(1, 0, 0, NAN), col(0, 0, 3, 1)), col(1, 0, 1, NAN))


def test_iff_only_selected_branch():
    cond = col(1, 0, NAN, 1, 0)
    yes = col(10, 10, 10, NAN, NAN)
    no = col(20, NAN, 20, 20, 20)
    same(op.iff(cond, yes, no), col(10, NAN, NAN, NAN, 20))
    same(op.iff(cond, 1, -1), col(1, -1, NAN, 1, -1))


def test_emax_emin_sign():
    same(op.emax(col(1, NAN, -2), col(0, 0, -3)), col(1, NAN, -2))
    same(op.emin(col(1, NAN, -2), 0), col(0, NAN, -2))
    same(op.sign(col(-2, 0, 3, NAN)), col(-1, 0, 1, NAN))


def test_delay_delta():
    x = col(1, 2, NAN, 4, 7)
    same(op.delay(x, 2), col(NAN, NAN, 1, 2, NAN))
    same(op.delta(x, 1), col(NAN, 1, NAN, NAN, 3))
    same(op.delay(x, 2.7), op.delay(x, 2))  # 非整数窗口向下取整


def test_rank_ties_nan():
    x = np.array([[3, 1, NAN, 3], [NAN, NAN, NAN, NAN], [5, 5, 5, 5]], float)
    same(op.rank(x), [[2.5 / 3, 1 / 3, NAN, 2.5 / 3], [NAN] * 4, [0.625] * 4])


# ------------------------------------------------------------ 滚动：手算


def test_sum_mean_full_window():
    x = col(1, 2, 3, NAN, 5, 6, 7)
    same(op.ts_sum(x, 3), col(NAN, NAN, 6, NAN, NAN, NAN, 18))
    same(op.ts_mean(x, 2), col(NAN, 1.5, 2.5, NAN, NAN, 5.5, 6.5))
    same(op.ts_prod(x, 2), col(NAN, 2, 6, NAN, NAN, 30, 42))
    same(op.ts_max(x, 2), col(NAN, 2, 3, NAN, NAN, 6, 7))
    same(op.ts_min(x, 2), col(NAN, 1, 2, NAN, NAN, 5, 6))
    same(op.ts_sum(col(1, 2), 3), col(NAN, NAN))  # 窗口不满


def test_std_constant_exact_zero():
    x = col(0.1, 0.1, 0.1, 1, 2, 3, 4)
    out = op.ts_std(x, 3)
    assert out[2, 0] == 0.0
    same(out[6], [math.sqrt(1.0)])
    same(op.ts_std(col(1, 2, 3, 4), 4)[3], [math.sqrt(5 / 3)])


def test_cov_corr():
    a = col(1, 2, 3, 5, 5, 5)
    b = col(1, 3, 2, 1, 2, 3)
    same(op.ts_cov(a, b, 3)[2], [0.5])
    same(op.ts_corr(a, b, 3)[2], [0.5])  # cov 0.5 / (1 · 1)
    same(op.ts_corr(a, b, 3)[5], [NAN])  # a 窗口常数
    same(op.ts_corr(col(1, 2, 3), col(2, 4, 6), 3)[2], [1.0])
    same(op.ts_corr(col(1, NAN, 3, 4), col(1, 2, 3, 4), 2), col(NAN, NAN, NAN, 1))


def test_tsrank_ties_nan():
    x = col(1, 2, 2, NAN, 5, 4, 3)
    out = op.ts_rank(x, 3)
    same(out, col(NAN, NAN, 2.5 / 3, NAN, NAN, NAN, 1 / 3))
    same(op.ts_rank(col(7, 7, 7), 3)[2], [2 / 3])  # 常数窗口：平均秩 2


def test_decay_linear_wma():
    same(op.decay_linear(col(1, 2, 3), 3)[2], [14 / 6])
    w = np.array([0.81, 0.9, 1.0])
    same(op.wma(col(1, 2, 3), 3)[2], [(w @ [1, 2, 3]) / w.sum()])
    same(op.wma(col(1, NAN, 3), 2), col(NAN, NAN, NAN))


def test_highday_lowday_ties_recent():
    same(op.highday(col(5, 3, 5), 3)[2], [0])
    same(op.highday(col(5, 5, 3), 3)[2], [1])
    same(op.highday(col(9, 1, 2), 3)[2], [2])
    same(op.lowday(col(1, 1, 4), 3)[2], [1])
    same(op.lowday(col(2, 2, 2), 3)[2], [0])
    same(op.lowday(col(1, NAN, 4), 3)[2], [NAN])


def test_count_sumif():
    cond = col(1, 0, 1, NAN, 0, 0, 0)
    a = col(10, 20, 30, 40, NAN, 60, 70)
    same(op.count(cond, 3), col(NAN, NAN, 2, NAN, NAN, NAN, 0))
    # 条件为假但 A 为 NaN（第 4 行）的窗口也是 NaN；全假时和为 0
    same(op.sumif(a, 2, cond), col(NAN, 10, 30, NAN, NAN, NAN, 0))


def test_regbeta_regresi_seq():
    same(op.regbeta_seq(col(2, 4, 6), 3)[2], [2.0])
    same(op.regbeta_seq(col(1, 3, 2), 3)[2], [0.5])
    same(op.regresi_seq(col(1, 3, 2), 3)[2], [-0.5])
    same(op.regbeta_seq(col(5, 5, 5), 3)[2], [0.0])
    same(op.regbeta_seq(col(1, NAN, 2, 3), 2), col(NAN, NAN, NAN, 1))


def test_regbeta_pairs_min_pairs_and_constant_x():
    y = col(NAN, 2, 4, NAN, 8, 3)
    x = col(1, 1, 2, 3, 4, 4)
    out = op.regbeta_pairs(y, x, 4, min_pairs=2)
    # 第 2 行窗口只有配对 (1,2)、(2,4)：斜率 2；第 1 行只有 1 对
    same(out[:3], col(NAN, NAN, 2))
    # 第 5 行窗口 (2,4)(4,8)(4,3)：x̄=10/3、ȳ=5
    xs, ys = np.array([2, 4, 4.0]), np.array([4, 8, 3.0])
    beta = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
    same(out[5], [beta])
    same(op.regbeta_pairs(col(1, 2, 3), col(5, 5, 5), 3, min_pairs=2)[2], [NAN])


def test_sma_recursion_hand():
    x = col(NAN, 2, 4, NAN, 6, 8)
    out = op.sma(x, 2, 1)
    # 初值 2（第 1 个有效值，有效数 1 < 2 不出数）；3；停牌保持；(6+3)/2；(8+4.5)/2
    same(out, col(NAN, NAN, 3, NAN, 4.5, 6.25))
    out = op.sma(col(1, 2, 3), 3, 2)
    same(out, col(NAN, NAN, (2 * 3 + 1 * ((2 * 2 + 1) / 3)) / 3))


def test_filter():
    same(op.filter_(col(1, 2, 3, 4), col(1, 0, NAN, 2)), col(1, NAN, NAN, 4))


# ------------------------------------------------------------ 滚动：朴素循环对拍


def _naive(fn, arrays, n):
    t, k = arrays[0].shape
    out = np.full((t, k), NAN)
    for j in range(k):
        for i in range(n - 1, t):
            wins = [a[i - n + 1 : i + 1, j] for a in arrays]
            if any(np.isnan(w).any() for w in wins):
                continue
            out[i, j] = fn(*wins)
    return out


def _naive_rank(w):
    last = w[-1]
    ranks = []
    order = sorted(w)
    positions = [p + 1 for p, v in enumerate(order) if v == last]
    ranks = sum(positions) / len(positions)
    return ranks / len(w)


def _naive_corr(a, b):
    if len(set(a)) == 1 or len(set(b)) == 1:
        return NAN
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def _naive_std(a):
    if len(set(a)) == 1:
        return 0.0
    m = sum(a) / len(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))


def _naive_days(w, best):
    target = best(w)
    for d, v in enumerate(reversed(w)):
        if v == target:
            return d
    raise AssertionError


@pytest.fixture
def rough(monkeypatch):
    """随机数据，带 NaN、并列（取整）和常数段；把分块调小以覆盖多块。"""
    monkeypatch.setattr(op, "_BLOCK_ELEMENTS", 50)
    rng = np.random.default_rng(3)
    a = np.round(rng.normal(0, 2, (60, 7)), 0)
    b = np.round(rng.normal(0, 2, (60, 7)), 1)
    a[rng.uniform(size=a.shape) < 0.05] = NAN
    b[rng.uniform(size=b.shape) < 0.05] = NAN
    a[20:30, 3] = 4.0
    return a, b


@pytest.mark.parametrize("n", [2, 3, 6])
def test_rolling_against_naive(rough, n):
    a, b = rough
    same(op.ts_sum(a, n), _naive(sum, [a], n))
    same(op.ts_mean(a, n), _naive(lambda w: sum(w) / len(w), [a], n))
    same(op.ts_max(a, n), _naive(max, [a], n))
    same(op.ts_min(a, n), _naive(min, [a], n))
    same(op.ts_rank(a, n), _naive(_naive_rank, [a], n))
    same(op.ts_std(a, n), _naive(_naive_std, [a], n))
    same(op.ts_corr(a, b, n), _naive(_naive_corr, [a, b], n))
    same(op.highday(a, n), _naive(lambda w: _naive_days(list(w), max), [a], n))
    same(op.lowday(a, n), _naive(lambda w: _naive_days(list(w), min), [a], n))
    weights = np.arange(1, n + 1)
    same(
        op.decay_linear(a, n),
        _naive(lambda w: (w * weights).sum() / weights.sum(), [a], n),
    )
    cond = op.gt(b, 0)
    same(op.count(cond, n), _naive(lambda c: float((c != 0).sum()), [cond], n))
    same(
        op.sumif(a, n, cond),
        _naive(
            lambda x, c: sum(v for v, k in zip(x, c, strict=True) if k), [a, cond], n
        ),
    )
    xs = np.arange(1, n + 1, dtype=float)

    def slope(w):
        return ((xs - xs.mean()) * (w - w.mean())).sum() / ((xs - xs.mean()) ** 2).sum()

    same(op.regbeta_seq(a, n), _naive(slope, [a], n))


def test_sma_against_naive(rough):
    a, _ = rough
    n, m = 5, 2
    expected = np.full(a.shape, NAN)
    for j in range(a.shape[1]):
        state, seen = None, 0
        for i in range(a.shape[0]):
            v = a[i, j]
            if np.isnan(v):
                continue
            state = v if state is None else (m * v + (n - m) * state) / n
            seen += 1
            if seen >= n:
                expected[i, j] = state
    same(op.sma(a, n, m), expected)


def test_regbeta_pairs_against_naive(rough):
    a, b = rough
    n, k = 10, 4
    t, cols = a.shape
    expected = np.full(a.shape, NAN)
    for j in range(cols):
        for i in range(t):
            lo = max(0, i - n + 1)
            pairs = [
                (x, y)
                for x, y in zip(b[lo : i + 1, j], a[lo : i + 1, j], strict=True)
                if not (np.isnan(x) or np.isnan(y))
            ]
            if len(pairs) < k or len({x for x, _ in pairs}) == 1:
                continue
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            expected[i, j] = ((xs - xs.mean()) * (ys - ys.mean())).sum() / (
                (xs - xs.mean()) ** 2
            ).sum()
    same(op.regbeta_pairs(a, b, n, min_pairs=k), expected)


def test_benchmark_column_broadcasts():
    stock = np.arange(12, dtype=float).reshape(4, 3)
    bench = col(1, 2, 3, 4)
    same(
        op.ts_corr(stock, bench, 3),
        np.where(np.arange(4)[:, None] >= 2, 1.0, NAN) * np.ones((4, 3)),
    )


# ---------------------------------------------------------------- 2026-11-13 裁决：幂的缺失、窗口和与顺序无关


def test_pw_missing_operand_is_missing():
    # np.power 给 1 ** NaN = 1、NaN ** 0 = 1；按「任一操作数缺失则结果缺失」都应为 NaN。
    a = np.array([[1.0, np.nan, 2.0, 1.0]])
    b = np.array([[np.nan, 0.0, 3.0, 2.0]])
    out = op.pw(a, b)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[0, 2] == 8.0 and out[0, 3] == 1.0


def test_window_sum_independent_of_order():
    # 窗口里进出的值相同时，两个数学上相等的窗口和逐位相等（国君 113 的零方差判定依赖这一点）。
    a, b, c = 33.411433, 14.750406000000002, 3.1222730000000003
    x = np.array([a, b, c, a]).reshape(-1, 1)
    s = op.ts_sum(x, 3)
    assert s[2, 0] == s[3, 0]
    corr = op.ts_corr(s, np.array([0.0, 0.0, 1.0, 2.0]).reshape(-1, 1), 2)
    assert np.isnan(corr[3, 0])
    vals = [0.1, 0.7, 1e8, -3.3, 2.2]
    perms = (vals, vals[::-1], vals[2:] + vals[:2])
    for f in (lambda v: op.ts_mean(v, 5), lambda v: op.ts_std(v, 5)):
        got = {f(np.array(p).reshape(-1, 1))[4, 0] for p in perms}
        assert len(got) == 1
    sums = {
        op.sumif(np.array(p).reshape(-1, 1), 5, np.ones((5, 1)))[4, 0] for p in perms
    }
    assert len(sums) == 1
