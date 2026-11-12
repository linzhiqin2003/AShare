"""翻译用语言每条语义说明的守护测试（手算小例为主）。"""

import numpy as np
import pandas as pd
import pytest

from ashare.factor import Evaluator, call, field
from ashare.factor.formula import dsl as d
from ashare.factor.formula.testing import synthetic_source


class Source:
    def __init__(self, **arrays):
        first = next(iter(v for v in arrays.values() if np.ndim(v) == 2))
        self.dates = np.arange(first.shape[0])
        self.symbols = np.arange(first.shape[1])
        self.version = "test"
        self.arrays = arrays

    def __getitem__(self, name):
        return np.ascontiguousarray(self.arrays[name])


def ev(expression, **arrays):
    return np.asarray(Evaluator().evaluate(expression, Source(**arrays)))


def col(*values):
    return np.array(values, float).reshape(-1, 1)


X, Y = field("x"), field("y")
nan = np.nan


def test_days_are_floored_and_validated():
    assert d.days(8.22237) == 8 and d.days(1.06157) == 1 and d.days(5) == 5
    assert d.ts_sum(X, 4.9) == d.ts_sum(X, 4)
    with pytest.raises(TypeError):
        d.days("5")
    with pytest.raises(ValueError):
        d.ts_sum(X, 0.5)


def test_delay_delta_direction_and_full_windows():
    x = col(1, 3, 3, 2, nan, 5, 5, 4)
    np.testing.assert_array_equal(
        ev(d.delay(X, 2), x=x)[:, 0], [nan, nan, 1, 3, 3, 2, nan, 5]
    )
    np.testing.assert_array_equal(
        ev(d.delta(X, 1), x=x)[:, 0], [nan, 2, 0, -1, nan, nan, 0, -1]
    )
    np.testing.assert_array_equal(ev(d.delay(X, 0.9), x=x)[:, 0], x[:, 0])
    # 窗口必须满：含 NaN 的窗口都是 NaN，前 d−1 行 NaN。
    np.testing.assert_array_equal(
        ev(d.ts_sum(X, 3), x=x)[:, 0], [nan, nan, 7, 8, nan, nan, nan, 14]
    )
    np.testing.assert_array_equal(
        ev(d.ts_max(X, 2), x=x)[:, 0], [nan, 3, 3, 3, nan, nan, 5, 5]
    )


def test_ranks_are_percentiles():
    x = np.array([[3.0, 1.0, nan, 1.0, 2.0]])
    np.testing.assert_array_equal(ev(d.rank(X), x=x), [[1.0, 0.375, nan, 0.375, 0.75]])
    s = col(1, 3, 3, 2, nan, 5)
    np.testing.assert_array_equal(
        ev(d.ts_rank(X, 3), x=s)[:, 0], [nan, nan, 2.5 / 3, 1 / 3, nan, nan]
    )
    np.testing.assert_array_equal(ev(d.ts_rank(X, 3.7), x=s), ev(d.ts_rank(X, 3), x=s))


def test_argmax_days_ago_latest_ties_and_highday():
    x = col(1, 3, 3, 2, 5, 5)
    np.testing.assert_array_equal(
        ev(d.ts_argmax(X, 3), x=x)[:, 0], [nan, nan, 0, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        ev(d.ts_argmin(X, 3), x=x)[:, 0], [nan, nan, 2, 0, 1, 2]
    )
    np.testing.assert_array_equal(ev(d.highday(X, 3), x=x), ev(d.ts_argmax(X, 3), x=x))
    np.testing.assert_array_equal(ev(d.lowday(X, 3), x=x), ev(d.ts_argmin(X, 3), x=x))


def test_returns_missing_on_listing_day_only():
    """D-03：上市首日（上市区间第一行）收益为 NaN；样本开始前就已上市的不受影响。"""
    source = synthetic_source(seed=4)
    a = {k: np.asarray(v, float) for k, v in source.arrays.items()}
    got = Evaluator().evaluate(d.returns, source)
    # 第 1、2 只在样本中途上市，第 10 只在样本第一天上市。
    for j, row in [(1, 30), (2, 75), (10, 0)]:
        assert a["listing_age"][row, j] == 1 and np.isnan(got[row, j])
        assert np.isfinite(a["close"][row, j] / a["previous_close"][row, j])
    # 第 0 只在样本开始前已上市：第一行（有前收盘时）照常有收益。
    assert a["listing_age"][0, 0] > 1
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = a["close"] / a["previous_close"] - 1
    keep = a["listing_age"] != 1
    np.testing.assert_array_equal(got[keep], raw[keep])
    assert (a["listing_age"] == 1).sum() == 3


def test_argmax_position_reading():
    """D-13 读法 B：窗口内位置，最早一天为 1，并列取最早；只在 with 块内生效。"""
    x = col(1, 3, 3, 2, 5, 5)
    with d.argmax_reading("position"):
        amax, amin = d.ts_argmax(X, 3), d.ts_argmin(X, 3.9)
    np.testing.assert_array_equal(ev(amax, x=x)[:, 0], [nan, nan, 2, 1, 3, 2])
    np.testing.assert_array_equal(ev(amin, x=x)[:, 0], [nan, nan, 1, 3, 2, 1])
    # with 块外回到默认读法；highday/lowday 不受读法影响（D-14）。
    assert d.ts_argmax(X, 3) == call("ts_argmax", X, window=3, min_periods=3)
    with d.argmax_reading("position"):
        assert d.highday(X, 3) == call("ts_argmax", X, window=3, min_periods=3)
    with pytest.raises(ValueError):
        with d.argmax_reading("index"):
            pass


@pytest.mark.parametrize("reading", ["days_ago", "position"])
def test_argmax_readings_match_numpy(reading):
    """两种读法对照 numpy 直算（含缺失、并列）。"""
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(size=(80, 6)), 1)
    x[rng.uniform(size=x.shape) < 0.05] = nan
    w = 5
    with d.argmax_reading(reading):
        got_max, got_min = ev(d.ts_argmax(X, w), x=x), ev(d.ts_argmin(X, w), x=x)
    for got, sign in [(got_max, 1), (got_min, -1)]:
        expected = np.full(x.shape, nan)
        for t in range(w - 1, x.shape[0]):
            for j in range(x.shape[1]):
                window = sign * x[t - w + 1 : t + 1, j]
                if np.isnan(window).any():
                    continue
                if reading == "days_ago":
                    ago = window[::-1]
                    expected[t, j] = int(np.argmax(ago))  # 最近的并列先出现
                else:
                    expected[t, j] = int(np.argmax(window)) + 1  # 最早的并列先出现
        np.testing.assert_array_equal(got, expected)


def test_regbeta_min_pairs():
    """Q-G-149：窗口里有效配对不少于 min_pairs 才出数；默认窗口必须满。"""
    rng = np.random.default_rng(5)
    t, w = 40, 10
    xs = rng.normal(size=(t, 1))
    ys = 2 * xs + rng.normal(scale=0.1, size=(t, 1))
    xs[rng.uniform(size=xs.shape) < 0.4] = nan
    got = ev(d.regbeta(Y, X, w, min_pairs=4), x=xs, y=ys)[:, 0]
    full = ev(d.regbeta(Y, X, w), x=xs, y=ys)[:, 0]
    for k in range(t):
        lo = max(0, k - w + 1)
        a, b = ys[lo : k + 1, 0], xs[lo : k + 1, 0]
        ok = np.isfinite(a) & np.isfinite(b)
        if k < w - 1 and lo != 0:
            raise AssertionError
        if ok.sum() >= 4:
            slope = np.polyfit(b[ok], a[ok], 1)[0]
            np.testing.assert_allclose(got[k], slope, rtol=1e-9)
        else:
            assert np.isnan(got[k])
        if ok.sum() < w or k < w - 1:
            assert np.isnan(full[k])
    for bad in (1, 11, 2.5, True):
        with pytest.raises(ValueError):
            d.regbeta(Y, X, w, min_pairs=bad)


def test_weighted_and_recursive_means():
    x = col(1, 3, 3, 2)
    np.testing.assert_allclose(
        ev(d.decay_linear(X, 3), x=x)[:, 0], [nan, nan, 16 / 6, 15 / 6]
    )
    w = 0.9 ** np.arange(3)
    np.testing.assert_allclose(
        ev(d.wma(X, 3), x=x)[:, 0],
        [
            nan,
            nan,
            (3 * w[0] + 3 * w[1] + 1 * w[2]) / w.sum(),
            (2 + 3 * 0.9 + 3 * 0.81) / w.sum(),
        ],
    )
    # SMA(A, 3, 1)：α = 1/3，首个有效值作初值，至少 3 个有效值才出数。
    y = col(3, 6, 9, nan, 12)
    got = ev(d.sma(X, 3, 1), x=y)[:, 0]
    s1, s2 = 3 + (6 - 3) / 3, None
    s2 = s1 + (9 - s1) / 3
    s3 = s2 + (12 - s2) / 3
    np.testing.assert_allclose(got, [nan, nan, s2, nan, s3], rtol=1e-15)
    with pytest.raises(ValueError):
        d.sma(X, 3, 4)


def test_statistics_degenerate_cases():
    c = col(1, 1, 1, 2, 3)
    y = col(1, 2, 3, 4, 5)
    np.testing.assert_allclose(
        ev(d.correlation(X, Y, 3), x=c, y=y)[:, 0], [nan, nan, nan, np.sqrt(0.75), 1.0]
    )
    np.testing.assert_allclose(
        ev(d.stddev(X, 3), x=c)[:, 0], [nan, nan, 0, np.sqrt(1 / 3), 1.0]
    )
    np.testing.assert_allclose(
        ev(d.covariance(X, Y, 3), x=c, y=y)[:, 0], [nan, nan, 0, 0.5, 1.0]
    )
    np.testing.assert_allclose(ev(d.ts_product(X, 2), x=y)[:, 0], [nan, 2, 6, 12, 20])


def test_count_sumif_filter_and_regressions():
    cond = col(1, 0, 1, 1, 0)
    x = col(1, 2, 3, 4, 5)
    np.testing.assert_array_equal(
        ev(d.ts_count(X, 3), x=cond)[:, 0], [nan, nan, 2, 2, 2]
    )
    np.testing.assert_array_equal(
        ev(d.ts_sumif(X, Y, 3), x=x, y=cond)[:, 0], [nan, nan, 4, 7, 7]
    )
    np.testing.assert_array_equal(
        ev(d.filter_(X, Y), x=x, y=cond)[:, 0], [1, nan, 3, 4, nan]
    )
    y = col(2, 4, 6, 8, 10.5)
    np.testing.assert_allclose(
        ev(d.regbeta(Y, X, 3), x=x, y=y)[:, 0], [nan, nan, 2, 2, 2.25]
    )
    np.testing.assert_allclose(ev(d.ts_slope(Y, 3), y=y)[:, 0], [nan, nan, 2, 2, 2.25])
    z = col(1, 2, 4, 4, 5)
    np.testing.assert_allclose(
        ev(d.ts_residual(X, 3), x=z)[:, 0], [nan, nan, 1 / 6, -1 / 3, 1 / 6], atol=1e-14
    )
    np.testing.assert_allclose(
        ev(d.regresi(X, Y, 3), x=z, y=x)[:, 0],
        [nan, nan, 1 / 6, -1 / 3, 1 / 6],
        atol=1e-14,
    )


def test_cross_section_scale_and_neutralize():
    x = np.array([[1.0, -3.0, nan, 2.0]])
    np.testing.assert_allclose(ev(d.scale(X), x=x), [[1 / 6, -0.5, nan, 1 / 3]])
    np.testing.assert_allclose(ev(d.scale(X, 2), x=x), [[1 / 3, -1.0, nan, 2 / 3]])
    groups = np.array([[1, 1, 2, -1]], np.int64)
    got = ev(
        d.indneutralize(X, "sector"),
        x=np.array([[1.0, 3.0, 5.0, 7.0]]),
        industry_l1=groups,
    )
    np.testing.assert_array_equal(got, [[-1.0, 1.0, 0.0, nan]])
    with pytest.raises(ValueError):
        d.industry("market")


def test_elementwise_missing_rules():
    a = np.array([[1.0, 0.0, -1.0, nan, 2.0]])
    b = np.array([[0.0, 0.0, 0.0, 1.0, nan]])
    np.testing.assert_array_equal(ev(X / Y, x=a, y=b), [[nan] * 5])
    np.testing.assert_array_equal(ev(d.gt(X, Y), x=a, y=b), [[1, 0, 0, nan, nan]])
    np.testing.assert_array_equal(ev(d.eq(X, Y), x=a, y=b), [[0, 1, 0, nan, nan]])
    np.testing.assert_array_equal(ev(d.or_(X, Y), x=a, y=b), [[1, 0, 1, nan, nan]])
    np.testing.assert_array_equal(ev(d.and_(X, Y), x=a, y=b), [[0, 0, 0, nan, nan]])
    np.testing.assert_array_equal(ev(d.log(X), x=a)[0, [1, 2, 3]], [nan, nan, nan])
    np.testing.assert_array_equal(ev(d.sign(X), x=a), [[1, 0, -1, nan, 1]])
    np.testing.assert_array_equal(ev(d.maximum(X, Y), x=a, y=b), [[1, 0, 0, nan, nan]])
    cond = np.array([[1.0, 0.0, nan]])
    got = ev(
        d.where(X, Y, field("z")),
        x=cond,
        y=np.array([[5.0, nan, 5.0]]),
        z=np.array([[nan, 6.0, 6.0]]),
    )
    np.testing.assert_array_equal(got, [[5, 6, nan]])
    base = np.array([[-8.0, 4.0, 0.0]])
    np.testing.assert_array_equal(ev(d.power(X, 0.5), x=base), [[nan, 2.0, 0.0]])
    np.testing.assert_allclose(
        ev(d.signed_power(X, Y), x=base, y=np.array([[1 / 3, 0.5, 2]])),
        [[-2.0, 2.0, 0.0]],
    )


def test_fields_follow_conventions():
    source = synthetic_source(seed=4)
    a = {k: np.asarray(v, float) for k, v in source.arrays.items()}
    e = Evaluator()
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_array_equal(
            e.evaluate(d.close, source), a["close"] * a["adj_factor"]
        )
        np.testing.assert_array_equal(
            e.evaluate(d.vwap, source), a["vwap"] * a["adj_factor"]
        )
        np.testing.assert_array_equal(
            e.evaluate(d.volume, source), a["volume"] / a["adj_factor"]
        )
        raw = a["close"] / a["previous_close"] - 1
        first_day = a["listing_age"] == 1
        np.testing.assert_array_equal(
            e.evaluate(d.returns, source), np.where(first_day, nan, raw)
        )
    adv = pd.DataFrame(a["amount"]).rolling(5).mean().to_numpy()
    np.testing.assert_allclose(e.evaluate(d.adv(5), source), adv, rtol=1e-12)
    bench = e.evaluate(d.benchmark_close, source)
    np.testing.assert_array_equal(
        bench[:, 3], source.benchmark_close
    )  # 市场序列：每列相同
    assert np.isnan(bench[5]).all()
    hd = e.evaluate(d.hd, source)
    high = a["high"] * a["adj_factor"]
    np.testing.assert_array_equal(hd[1:], high[1:] - high[:-1])
    tr = e.evaluate(d.tr, source)
    close = a["close"] * a["adj_factor"]
    low = a["low"] * a["adj_factor"]
    prev = np.vstack([np.full((1, close.shape[1]), nan), close[:-1]])
    expected = np.fmax(np.fmax(high - low, np.abs(high - prev)), np.abs(low - prev))
    expected[~(np.isfinite(high) & np.isfinite(low) & np.isfinite(prev))] = nan
    np.testing.assert_array_equal(tr, expected)


def test_exact_equality_zero_denominator_and_nan_extrema():
    """补充裁决：== 不加容差；分母为 0 得 NaN 并向后传播；逐元素 max/min 任一侧 NaN 为 NaN。"""
    x = col(0.1, 1.0, 0.05, 2.0)
    y = col(0.3 - 0.2, 1.0, 0.05, nan)
    np.testing.assert_array_equal(ev(d.eq(X, Y), x=x, y=y)[:, 0], [0, 1, 1, nan])
    np.testing.assert_array_equal(ev(d.ne(X, Y), x=x, y=y)[:, 0], [1, 0, 0, nan])
    num, den = col(1.0, 2.0, 0.0, 3.0), col(1.0, 0.0, 0.0, 2.0)
    ratio = X / Y
    np.testing.assert_array_equal(ev(ratio, x=num, y=den)[:, 0], [1, nan, nan, 1.5])
    # NaN 向后传播：窗口里有分母为 0 的一天，窗口统计为 NaN；排名里不参与。
    np.testing.assert_array_equal(
        ev(d.ts_sum(ratio, 2), x=num, y=den)[:, 0], [nan, nan, nan, nan]
    )
    a, b = col(1.0, nan, 3.0), col(2.0, 0.0, nan)
    np.testing.assert_array_equal(ev(d.maximum(X, Y), x=a, y=b)[:, 0], [2, nan, nan])
    np.testing.assert_array_equal(ev(d.minimum(X, Y), x=a, y=b)[:, 0], [1, nan, nan])


def test_identity_forms_are_not_simplified():
    """补充裁决：恒等写法照字面保留在表达式里（求值结果与化简后相同或只差舍入）。"""
    assert d.ts_sum(X, 1) != X and d.ts_sum(X, 1).op == "ts_sum"
    assert d.ts_product(X, 1).op == "ts_product"
    assert d.power(X, 1).op == "power"
    assert d.delta(X - 1, 3) != d.delta(X, 3)
    x = col(1.0, 2.5, nan, 4.0)
    np.testing.assert_array_equal(ev(d.ts_sum(X, 1), x=x), x)
    np.testing.assert_array_equal(ev(d.power(X, 1), x=x), x)


def test_mask_missing():
    """ref 缺失处为 NaN，其余逐位不变；用于「当天停牌时回归不出数」。"""
    x, ref = col(1.5, 2.0, nan, 4.0), col(0.1, nan, 3.0, -1.0)
    np.testing.assert_array_equal(
        ev(d.mask_missing(X, Y), x=x, y=ref)[:, 0], [1.5, nan, nan, 4.0]
    )
    rng = np.random.default_rng(9)
    xs, ys = rng.normal(size=(60, 1)), rng.normal(size=(60, 1))
    ys[[10, 40, 41]] = nan
    slope = ev(d.regbeta(X, Y, 30, min_pairs=20), x=xs, y=ys)
    masked = ev(d.mask_missing(d.regbeta(X, Y, 30, min_pairs=20), Y), x=xs, y=ys)
    assert np.isnan(masked[[10, 40, 41]]).all() and np.isfinite(slope[40])
    keep = np.isfinite(ys[:, 0])
    np.testing.assert_array_equal(masked[keep], slope[keep])
    # 面板开头不足窗口：已有的行里配对够 20 对就出数。
    assert np.isnan(slope[18]) and np.isfinite(slope[20])
