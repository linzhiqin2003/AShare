"""组合算子与朴素 pandas/NumPy 实现在随机数据（含 NaN、并列、常数窗口）上对拍，加手算小例。"""

import numpy as np
import pandas as pd
import pytest

from ashare.factor import Evaluator, call, field
from ashare.factor.formula import ops


class Source:
    def __init__(self, **arrays):
        first = next(iter(arrays.values()))
        self.dates = np.arange(first.shape[0])
        self.symbols = np.arange(first.shape[1])
        self.version = "test"
        self.arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}

    def __getitem__(self, name):
        return self.arrays[name]


def evaluate(expression, **arrays):
    return np.asarray(Evaluator().evaluate(expression, Source(**arrays)))


def random_panel(seed=0, t=120, n=9, ties=True):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, (t, n)).astype(float) if ties else rng.normal(size=(t, n))
    x[rng.uniform(size=(t, n)) < 0.1] = np.nan
    x[:, 0] = 3.0  # 常数
    x[20:50, 1] = np.nan  # 长缺口
    x[:, 2] = rng.normal(size=t)
    return x


X = field("x")


def test_keep_where_is_bitwise_and_masks():
    x = np.array([[1.0, -0.0, 3.5, np.nan], [1e-310, 2.0, -7.0, 4.0]])
    keep = np.array([[1.0, 1.0, 0.0, 1.0], [1.0, np.nan, 1.0, 0.0]])
    got = evaluate(ops.keep_where(field("k"), X), x=x, k=keep)
    expected = np.where(keep == 1, x, np.nan)
    np.testing.assert_array_equal(
        got.view(np.uint64)[keep == 1], expected.view(np.uint64)[keep == 1]
    )
    assert np.isnan(got[keep != 1]).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cs_pct_rank_matches_pandas(seed):
    x = random_panel(seed)
    x[7] = np.nan  # 全空的一行
    got = evaluate(ops.cs_pct_rank(X), x=x)
    expected = pd.DataFrame(x).rank(axis=1, pct=True, method="average").to_numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    count = evaluate(ops.cs_valid_count(X), x=x)
    expected_count = np.isfinite(x).sum(axis=1).astype(float)
    expected_count[expected_count == 0] = np.nan
    np.testing.assert_array_equal(count[:, 0], expected_count)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("window", [1, 3, 10, 30])
def test_ts_pct_rank_and_valid_count_match_pandas(seed, window):
    x = random_panel(seed)
    frame = pd.DataFrame(x)
    got = evaluate(ops.ts_pct_rank(X, window, 1), x=x)
    expected = frame.rolling(window, min_periods=1).rank(pct=True).to_numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    count = evaluate(ops.ts_valid_count(X, window), x=x)
    np.testing.assert_array_equal(
        count, frame.notna().astype(float).rolling(window, min_periods=1).sum()
    )
    full = evaluate(ops.ts_pct_rank(X, window, window), x=x)
    expected_full = frame.rolling(window).rank(pct=True).to_numpy()
    np.testing.assert_allclose(full, expected_full, rtol=0, atol=1e-15)


def test_ts_pct_rank_hand_case():
    x = np.array([[1.0], [3.0], [3.0], [2.0], [np.nan], [5.0]])
    got = evaluate(ops.ts_pct_rank(X, 3, 1), x=x)[:, 0]
    # 窗口 [1]、[1,3]、[1,3,3]、[3,3,2]、当天缺失、[2,nan,5]
    np.testing.assert_array_equal(got, [1.0, 1.0, 2.5 / 3, 1 / 3, np.nan, 1.0])


@pytest.mark.parametrize("n, m", [(3, 1), (12, 2), (61, 2)])
def test_ewm_matches_pandas_recursion(n, m):
    rng = np.random.default_rng(n)
    x = rng.normal(size=(200, 4))
    x[[10, 11, 50], 1] = np.nan
    x[:5, 2] = np.nan
    got = evaluate(ops.ewm(X, n, m, 1), x=x)
    # 缺失日保持状态（ignore_na=True），且那天输出 NaN。
    state = (
        pd.DataFrame(x).ewm(alpha=m / n, adjust=False, ignore_na=True).mean().to_numpy()
    )
    expected = np.where(np.isfinite(x), state, np.nan)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
    strict = evaluate(ops.ewm(X, n, m, n), x=x)
    counts = np.cumsum(np.isfinite(x), axis=0)
    assert np.isnan(strict[counts < n]).all()
    np.testing.assert_array_equal(strict[counts >= n], got[counts >= n])


def test_ewm_hand_case():
    x = np.array([[1.0], [2.0], [np.nan], [4.0]])
    got = evaluate(ops.ewm(X, 2, 1, 1), x=x)[:, 0]  # α = 1/2
    np.testing.assert_array_equal(got, [1.0, 1.5, np.nan, 2.75])


def test_decayed_mean_matches_direct_weights():
    x = random_panel(3, ties=False)
    n = 12
    got = evaluate(ops.decayed_mean(X, n), x=x)
    w = 0.9 ** np.arange(n)  # 今天 i=0
    expected = np.full_like(x, np.nan)
    for t in range(n - 1, len(x)):
        window = x[t - n + 1 : t + 1][::-1]  # 今天在前
        expected[t] = (w[:, None] * window).sum(axis=0) / w.sum()
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
    assert np.isnan(got[: n - 1]).all()


def test_filter_out():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    cond = np.array([[1.0, 0.0, np.nan, -2.0]])
    got = evaluate(ops.filter_out(X, field("c")), x=x, c=cond)
    np.testing.assert_array_equal(got, [[1.0, np.nan, np.nan, 4.0]])


# ---- Qlib 组合：与 pandas 的 Qlib 写法逐只股票对拍 ----------------------------


def listed_panel(seed=0, t=150, n=8):
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(10, 1, (t, n)), 1)  # 并列多
    listed = np.ones((t, n), bool)
    listed[:40, 1] = False
    listed[100:, 2] = False
    listed[:, 3] = False
    x[~listed] = np.nan
    x[rng.uniform(size=(t, n)) < 0.08] = np.nan  # 停牌
    x[60:70, 4] = np.nan
    x[:, 5] = np.where(listed[:, 5], 7.0, np.nan)  # 常数
    return x, listed


def per_stock(x, listed, fn):
    out = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        rows = np.flatnonzero(listed[:, j])
        if len(rows):
            out[rows, j] = fn(pd.Series(x[rows, j])).to_numpy(dtype=float)
    return out


@pytest.mark.parametrize(
    "op, method", [("ts_idxmax", "argmax"), ("ts_idxmin", "argmin")]
)
@pytest.mark.parametrize("window", [1, 5, 20])
def test_qlib_idx_matches_rolling_apply(op, method, window):
    x, listed = listed_panel()
    got = evaluate(ops.qlib_idx(X, window, op), x=x, is_listed=listed)
    expected = per_stock(
        x,
        listed,
        lambda s: s.rolling(window, min_periods=1).apply(
            lambda a: getattr(a, method)() + 1, raw=True
        ),
    )
    np.testing.assert_array_equal(got, expected)


def test_qlib_idx_without_nan_rule_ignores_missing():
    x, listed = listed_panel(1)
    got = evaluate(
        ops.qlib_idx(X, 10, "ts_idxmax", nan_first=False), x=x, is_listed=listed
    )

    def first_max(a):
        valid = np.where(np.isnan(a), -np.inf, a)
        return valid.argmax() + 1

    expected = per_stock(
        x, listed, lambda s: s.rolling(10, min_periods=1).apply(first_max, raw=True)
    )
    np.testing.assert_array_equal(got, expected)


def test_qlib_compare_counts_missing_as_false():
    x, listed = listed_panel(2)
    expr = call(
        "ts_mean",
        ops.qlib_compare("gt", X, call("ts_delay", X, periods=1)),
        window=5,
        min_periods=1,
    )
    got = evaluate(expr, x=x, is_listed=listed)
    expected = per_stock(
        x, listed, lambda s: np.greater(s, s.shift(1)).rolling(5, min_periods=1).mean()
    )
    # 退市后 Qlib 没有行；组合本身只保证上市区间内一致（转换程序在最外层再乘上市区间）。
    np.testing.assert_allclose(got[listed], expected[listed], rtol=0, atol=1e-15)


@pytest.mark.parametrize("window", [2, 5, 20])
def test_qlib_corr_and_rsquare_guards(window):
    x, listed = listed_panel(3)
    rng = np.random.default_rng(9)
    y = np.where(np.isfinite(x), x * 0.5 + rng.normal(0, 0.2, x.shape), np.nan)
    y[:, 6] = np.where(listed[:, 6], 1.0, np.nan)  # 另一侧常数
    got = evaluate(ops.qlib_corr(X, field("y"), window), x=x, y=y, is_listed=listed)

    def corr(a, b):
        res = a.rolling(window, min_periods=1).corr(b)
        small = np.isclose(
            a.rolling(window, min_periods=1).std(), 0, atol=2e-05
        ) | np.isclose(b.rolling(window, min_periods=1).std(), 0, atol=2e-05)
        res[small] = np.nan
        return res

    expected = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        rows = np.flatnonzero(listed[:, j])
        if len(rows):
            expected[rows, j] = corr(pd.Series(x[rows, j]), pd.Series(y[rows, j]))
    # 退市后的行由转换程序在最外层置 NaN，这里只比上市区间。
    np.testing.assert_array_equal(
        np.isfinite(got[listed]), np.isfinite(expected[listed])
    )
    np.testing.assert_allclose(got[listed], expected[listed], rtol=1e-9, atol=1e-9)
    rsq = evaluate(ops.qlib_rsquare(X, window), x=x, is_listed=listed)
    assert np.isnan(rsq[:, 5]).all()  # 常数列：标准差 0 置 NaN
