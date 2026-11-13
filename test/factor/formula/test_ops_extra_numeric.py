"""补充算子的数值对拍：每个可构造的算子与一份直白的 NumPy 写法逐格比较。

参考写法按 ``ops_extra.FORMULAS`` 的定义逐窗口、逐截面循环，不调用 ``panelops``。数据里
有零散缺失、开头整段缺失和中间长缺口。新加构造器而不在这里补参考写法，
``test_every_builder_has_a_reference`` 会失败。
"""

import numpy as np
import pytest

import ashare.factor as af
from ashare.factor.formula import ops_extra

T, N, W = 80, 12, 5
_rng = np.random.default_rng(7)
x = _rng.normal(0, 1, (T, N))
y = _rng.normal(0, 1, (T, N))
p = np.exp(_rng.normal(0, 0.3, (T, N))) + 0.5
x[_rng.random((T, N)) < 0.08] = np.nan
x[:7, 3] = np.nan
x[30:44, 5] = np.nan
y[_rng.random((T, N)) < 0.05] = np.nan
cond = (_rng.random((T, N)) < 0.5).astype(float)
groups = np.repeat(np.arange(4, dtype=np.int64), 3)[None, :].repeat(T, 0)

X, Y, P, C = (af.field(k) for k in ("x", "y", "p", "c"))
G = af.field("g", dtype="int64")


class _Source:
    dates = np.arange(T)
    symbols = np.array([f"s{i}" for i in range(N)])
    version = "ops-extra-numeric"

    def __getitem__(self, key):
        return {"x": x, "y": y, "p": p, "c": cond, "g": groups}[key]


def _windows(*arrays, w=W):
    """逐 (t, j) 给出各输入的完整窗口；窗口里有缺失的格跳过（结果留 NaN）。"""
    for t in range(w - 1, T):
        for j in range(N):
            cols = [a[t - w + 1 : t + 1, j] for a in arrays]
            if all(np.isfinite(c).all() for c in cols):
                yield (t, j, *cols)


def _ts(fn, *arrays, w=W):
    out = np.full((T, N), np.nan)
    for t, j, *cols in _windows(*arrays, w=w):
        out[t, j] = fn(*cols)
    return out


def _lag(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _moment(c, k):
    d = c - c.mean()
    return (d**k).mean() / (d**2).mean() ** (k / 2)


def _rank01(c):
    """最后一个值在窗口内的平均秩，映射到 [0, 1]。"""
    r = (c < c[-1]).sum() + ((c == c[-1]).sum() + 1) / 2
    return (r - 1) / (len(c) - 1)


def _decay_linear_zero_filled(a, w=W):
    v = np.where(np.isfinite(a), a, 0.0)
    k = np.arange(1, w + 1.0)
    out = np.full_like(a, np.nan)
    for t in range(w - 1, T):
        out[t] = (v[t - w + 1 : t + 1] * k[:, None]).sum(0) / k.sum()
    return out


def _backfill(a, lookback):
    out = a.copy()
    for t in range(T):
        for j in range(N):
            if not np.isfinite(a[t, j]):
                win = a[max(0, t - lookback + 1) : t + 1, j]
                ok = win[np.isfinite(win)]
                out[t, j] = ok[-1] if len(ok) else np.nan
    return out


def _count_nans(a, w=W):
    out = np.full_like(a, np.nan)
    for t in range(T):
        out[t] = (~np.isfinite(a[max(0, t - w + 1) : t + 1])).sum(0)
    return out


def _cs(fn, *arrays, need=1):
    """逐日在各输入都有值的股票上算 fn，其余留 NaN。"""
    out = np.full((T, N), np.nan)
    for t in range(T):
        m = np.all([np.isfinite(a[t]) for a in arrays], axis=0)
        if m.sum() >= need:
            out[t, m] = fn(*[a[t, m] for a in arrays])
    return out


def _cs_rank01(v):
    order = np.array([(v < u).sum() + ((v == u).sum() + 1) / 2 for u in v])
    return (order - 1) / (len(v) - 1)


def _ols_residual(a, b):
    beta = ((a - a.mean()) * (b - b.mean())).sum() / ((b - b.mean()) ** 2).sum()
    return a - a.mean() - beta * (b - b.mean())


def _group_demean(a):
    out = np.full((T, N), np.nan)
    for t in range(T):
        for k in np.unique(groups[t]):
            m = (groups[t] == k) & np.isfinite(a[t])
            if m.any():
                out[t, m] = a[t, m] - a[t, m].mean()
    return out


with np.errstate(all="ignore"):
    CASES = {
        # 逐元素
        "exp": ((X,), {}, np.exp(x)),
        "sign": ((X,), {}, np.sign(x)),
        "log": ((P,), {}, np.log(p)),
        "max": ((X, Y, P), {}, np.maximum(np.maximum(x, y), p)),
        "min": ((X, Y), {}, np.minimum(x, y)),
        "abs": ((X,), {}, np.abs(x)),
        "sigmoid": ((X,), {}, 1 / (1 + np.exp(-x))),
        "divide": ((X, P), {}, x / p),
        "signed_power": ((X, 2.0), {}, np.sign(x) * np.abs(x) ** 2),
        "inverse": ((P,), {}, 1 / p),
        "sqrt": ((P,), {}, np.sqrt(p)),
        "s_log_1p": ((X,), {}, np.sign(x) * np.log1p(np.abs(x))),
        "reverse": ((X,), {}, -x),
        "log_diff": ((P,), {}, np.log(p) - np.log(_lag(p, 1))),
        "power": ((P, 3.0), {}, p**3),
        "if_else": ((C, X, Y), {}, np.where(cond > 0, x, y)),
        # 时序
        "ts_sum": ((X,), {"window": W}, _ts(np.sum, x)),
        "ts_mean": ((X,), {"window": W}, _ts(np.mean, x)),
        "ts_std_dev": ((X,), {"window": W}, _ts(lambda c: c.std(ddof=1), x)),
        "ts_min": ((X,), {"window": W}, _ts(np.min, x)),
        "ts_max": ((X,), {"window": W}, _ts(np.max, x)),
        "ts_median": ((X,), {"window": W}, _ts(np.median, x)),
        "ts_range": ((X,), {"window": W}, _ts(np.ptp, x)),
        "ts_zscore": (
            (X,),
            {"window": W},
            _ts(lambda c: (c[-1] - c.mean()) / c.std(ddof=1), x),
        ),
        "ts_ir": ((X,), {"window": W}, _ts(lambda c: c.mean() / c.std(ddof=1), x)),
        "ts_av_diff": ((X,), {"window": W}, _ts(lambda c: c[-1] - c.mean(), x)),
        "ts_min_diff": ((X,), {"window": W}, _ts(lambda c: c[-1] - c.min(), x)),
        "ts_max_diff": ((X,), {"window": W}, _ts(lambda c: c[-1] - c.max(), x)),
        "ts_min_max_diff": (
            (X,),
            {"window": W, "f": 0.5},
            _ts(lambda c: c[-1] - 0.5 * (c.min() + c.max()), x),
        ),
        "ts_min_max_cps": (
            (X,),
            {"window": W, "f": 2.0},
            _ts(lambda c: c.min() + c.max() - 2.0 * c[-1], x),
        ),
        "ts_scale": (
            (X,),
            {"window": W, "constant": 0.25},
            _ts(lambda c: (c[-1] - c.min()) / (c.max() - c.min()) + 0.25, x),
        ),
        "ts_skewness": ((X,), {"window": W}, _ts(lambda c: _moment(c, 3), x)),
        "ts_kurtosis": ((X,), {"window": W}, _ts(lambda c: _moment(c, 4) - 3, x)),
        "ts_arg_max": (
            (X,),
            {"window": W},
            _ts(lambda c: len(c) - 1 - np.argmax(c), x),
        ),
        "ts_arg_min": (
            (X,),
            {"window": W},
            _ts(lambda c: len(c) - 1 - np.argmin(c), x),
        ),
        "ts_rank": ((X,), {"window": W, "constant": 0.1}, _ts(_rank01, x) + 0.1),
        "ts_corr": (
            (X, Y),
            {"window": W},
            _ts(lambda a, b: np.corrcoef(a, b)[0, 1], x, y),
        ),
        "ts_vector_neut": (
            (X, Y),
            {"window": W},
            _ts(lambda a, b: a[-1] - (a @ b) / (b @ b) * b[-1], x, y),
        ),
        "ts_decay_exp_window": (
            (X,),
            {"window": W, "factor": 0.5},
            _ts(
                lambda c: (
                    (c[::-1] * 0.5 ** np.arange(W)).sum() / (0.5 ** np.arange(W)).sum()
                ),
                x,
            ),
        ),
        "ts_decay_linear": ((X,), {"window": W}, _decay_linear_zero_filled(x)),
        "ts_delay": ((X,), {"window": 2}, _lag(x, 2)),
        "ts_delta": ((X,), {"window": 2}, x - _lag(x, 2)),
        "ts_returns": ((P,), {"window": 3}, p / _lag(p, 3) - 1),
        "ts_weighted_decay": ((X,), {"k": 0.3}, 0.3 * x + 0.7 * _lag(x, 1)),
        "ts_count_nans": ((X,), {"window": W}, _count_nans(x)),
        "ts_backfill": ((X,), {"lookback": 6}, _backfill(x, 6)),
        # 截面
        "rank": ((X,), {}, _cs(_cs_rank01, x, need=2)),
        "zscore": ((X,), {}, _cs(lambda v: (v - v.mean()) / v.std(), x, need=2)),
        "normalize": ((X,), {}, _cs(lambda v: v - v.mean(), x)),
        "scale": ((X,), {"scale": 2.0}, _cs(lambda v: 2.0 * v / np.abs(v).sum(), x)),
        "vector_neut": (
            (X, Y),
            {},
            _cs(lambda a, b: a - (a @ b) / (b @ b) * b, x, y, need=2),
        ),
        "regression_neut": ((X, Y), {}, _cs(_ols_residual, x, y, need=3)),
        "group_neutralize": ((X, G), {}, _group_demean(x)),
    }


def _evaluate(expression):
    return np.asarray(af.Evaluator().evaluate(expression, _Source()), dtype=float)


def test_every_builder_has_a_reference():
    assert set(CASES) == set(ops_extra.BUILDERS)


@pytest.mark.parametrize("name", sorted(CASES))
def test_builder_matches_reference(name):
    args, params, reference = CASES[name]
    got = _evaluate(ops_extra.build(name, *args, **params))
    assert got.shape == reference.shape
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(reference))
    both = np.isfinite(reference)
    assert both.sum() > 100  # 参考不是整片缺失
    np.testing.assert_allclose(got[both], reference[both], rtol=1e-9, atol=1e-12)


def test_ts_returns_mode_two_uses_the_midpoint():
    got = _evaluate(ops_extra.build("ts_returns", P, window=3, mode=2))
    past = _lag(p, 3)
    reference = (p - past) / ((p + past) / 2)
    both = np.isfinite(reference)
    np.testing.assert_allclose(got[both], reference[both], rtol=1e-12)


def test_normalize_with_std_and_limit():
    got = _evaluate(ops_extra.build("normalize", X, use_std=True, limit=1.0))
    reference = _cs(lambda v: np.clip((v - v.mean()) / v.std(), -1.0, 1.0), x, need=2)
    both = np.isfinite(reference)
    np.testing.assert_allclose(got[both], reference[both], rtol=1e-9)


def test_ts_decay_linear_dense_ignores_missing():
    """dense=True 交给内核：窗口内有缺失时当天缺失，不把缺失当 0。"""
    got = _evaluate(ops_extra.build("ts_decay_linear", X, window=W, dense=True))
    k = np.arange(1, W + 1.0)
    reference = _ts(lambda c: (c * k).sum() / k.sum(), x)
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(reference))
    both = np.isfinite(reference)
    np.testing.assert_allclose(got[both], reference[both], rtol=1e-9)


def test_ts_backfill_window_includes_today():
    """lookback 天的窗口含当天：缺口第 lookback 天起不再回填；lookback=1 原样返回。"""
    gap = x[30:44, 5]
    assert not np.isfinite(gap).any() and np.isfinite(x[29, 5])
    got = _evaluate(ops_extra.build("ts_backfill", X, lookback=6))
    assert np.all(got[30:35, 5] == x[29, 5])
    assert not np.isfinite(got[35:44, 5]).any()
    same = _evaluate(ops_extra.build("ts_backfill", X, lookback=1))
    np.testing.assert_array_equal(np.isfinite(same), np.isfinite(x))
    with pytest.raises(ValueError):
        ops_extra.build("ts_backfill", X, lookback=0)


def test_vector_neut_is_not_the_intercept_regression():
    """不带截距的投影与带截距的回归残差不是一回事，两个算子不能互相替代。"""
    projection = _evaluate(ops_extra.build("vector_neut", X, Y))
    regression = _evaluate(ops_extra.build("regression_neut", X, Y))
    both = np.isfinite(projection) & np.isfinite(regression)
    assert np.abs(projection[both] - regression[both]).max() > 0.05
    # 投影残差与 y 正交；回归残差均值为 0
    for t in (10, 40, 70):
        m = both[t]
        assert abs(projection[t, m] @ y[t, m]) < 1e-9
        assert abs(regression[t, m].mean()) < 1e-12
