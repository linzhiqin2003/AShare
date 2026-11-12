"""P-006：特质方差逐日递推与从头计算的容差（相对 1e-12）、确定性和模式开关。"""

from dataclasses import replace

import numpy as np
import pytest

from ashare.risk.fundamental import FundamentalConfig
from ashare.risk.series import model_series
from ashare.risk.specific import SpecificMoments, direct_variances, reference_variances
from test.risk.test_risk_review import config
from test.risk.test_series import source

TOLERANCE = 1e-12


def adversarial_panel(t, n, seed):
    """随机残差加对抗列：停牌复牌、整列缺失、常数、量级跳变、大均值、晚上市。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, (t, n)) * rng.uniform(0.2, 3, n)
    x[rng.random((t, n)) < 0.05] = np.nan
    x[t // 3 : t // 3 + 25, 0] = np.nan  # 停牌 25 天后复牌
    x[:, 1] = np.nan  # 从未有残差
    x[:, 2] = 0.0125  # 常数序列
    x[t // 2 :, 3] *= 1e5  # 量级跳变
    x[:, 4] = 50 + rng.normal(0, 1e-4, t)  # 均值远大于波动
    x[: t - 7, 5] = np.nan  # 晚上市：只有最后 7 天
    x[:, 6] = np.nan
    x[t - 1, 6] = 0.01  # 只有一个有效值
    x[: 2 * t // 3, 7] = 0.0
    x[2 * t // 3 :, 7] = 1e6 + rng.normal(0, 1e-3, t - 2 * t // 3)  # 锚点后均值漂走
    x[:, 8] = np.where(np.arange(t) % 2, 1e-7, -1e-7)  # 强负自相关，NW 几乎抵消
    # 交替 ±1e-2 加 1e-6 噪声：NW 方差约 1e-12，各项却是 1e-4 量级（只有条件数闸能拦）。
    x[:, 9] = np.where(np.arange(t) % 2, 1e-2, -1e-2) + rng.normal(0, 1e-6, t)
    return x


def relative_error(actual, expected):
    scale = np.maximum(np.abs(expected), 1e-300)
    error = np.abs(actual - expected) / scale
    error[(actual == 0) & (expected == 0)] = 0
    return error


@pytest.mark.parametrize(
    ("t", "n", "history", "period", "seed"),
    (
        (160, 40, 60, 17, 1),  # N≤64：从头算走 numpy 参考内核
        (160, 90, 60, 17, 2),  # N>64：从头算走 Rust
        (90, 70, 120, 252, 3),  # 窗口始终不满
        (40, 70, 3, 5, 4),  # 窗口比 NW 滞后还短
    ),
)
def test_incremental_matches_direct_within_tolerance(t, n, history, period, seed):
    c = replace(FundamentalConfig(), history=history, specific_reanchor_days=period)
    x = adversarial_panel(t, n, seed)
    moments = SpecificMoments(n, c)
    worst = 0.0
    for day in range(t):
        moments.push(day, x[day])
        window = x[max(0, day - history + 1) : day + 1]
        base, corr0, corrnw, count = moments.variances(day)
        for actual, expected in zip(
            (base, corr0, corrnw), direct_variances(window, c), strict=True
        ):
            worst = max(worst, relative_error(actual, expected).max())
        np.testing.assert_array_equal(count, np.isfinite(window).sum(axis=0))
    assert worst <= TOLERANCE
    assert moments.fallback_columns > 0  # 对抗列确实触发了从头算


def test_rust_moments_agree_with_numpy_reference():
    c = replace(FundamentalConfig(), history=50, specific_reanchor_days=11)
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.02, (120, 30))
    x[rng.random(x.shape) < 0.1] = np.nan
    moments = SpecificMoments(30, c)
    for day in range(120):
        moments.push(day, x[day])
    got = moments.variances(119)[:3]
    expected = reference_variances(x[70:], c)
    for a, b in zip(got, expected, strict=True):
        assert relative_error(a, b).max() <= TOLERANCE


def test_lazy_catch_up_is_bitwise_identical_to_daily_updates():
    c = replace(FundamentalConfig(), history=40, specific_reanchor_days=13)
    x = adversarial_panel(150, 80, 11)
    daily, lazy = SpecificMoments(80, c), SpecificMoments(80, c)
    for day in range(150):
        daily.push(day, x[day])
        lazy.push(day, x[day])
        a = daily.variances(day)
        if day in (61, 149):  # 追加路径：只在少数日子请求，中间靠补算
            for u, v in zip(a, lazy.variances(day), strict=True):
                np.testing.assert_array_equal(u, v)


def test_window_bounds_are_enforced():
    c = replace(FundamentalConfig(), history=5, specific_reanchor_days=3)
    moments = SpecificMoments(4, c)
    with pytest.raises(ValueError, match="逐日连续"):
        moments.push(1, np.zeros(4))
    for day in range(20):
        moments.push(day, np.full(4, day, dtype=float))
    with pytest.raises(ValueError, match="覆盖"):
        moments.window(14, 16)
    with pytest.raises(ValueError, match="不在缓冲内"):
        moments.window(13, 19)
    np.testing.assert_array_equal(moments.window(15, 19)[:, 0], np.arange(15, 20))
    with pytest.raises(ValueError, match="最新一天"):
        moments.variances(18)


def test_model_series_modes_cache_key_and_append(tmp_path):
    p = source(70, 90)
    c = config(history=30, specific_reanchor_days=9)
    scratch = list(model_series(p, config=replace(c, specific_incremental=False)))
    fast = list(model_series(p, config=c))
    worst = 0.0
    for a, b in zip(fast, scratch, strict=True):
        np.testing.assert_array_equal(a.B, b.B)
        assert a.metadata["parameter_hash"] != b.metadata["parameter_hash"]
        worst = max(worst, relative_error(a.D, b.D).max())
        np.testing.assert_array_equal(a.F, b.F)  # 特质方差不进入 F
    assert 0 < worst <= 1e-10
    # 同一模式下，追加一天与全量重建逐位相同（追加要从最近的重算点补算）。
    list(model_series(p, config=c, cache=tmp_path, end=69))
    appended = list(model_series(p, config=c, cache=tmp_path, start=69))
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(
            getattr(appended[0], name), getattr(fast[-1], name)
        )
    assert appended[0].metadata == fast[-1].metadata
    other = list(model_series(p, config=replace(c, specific_reanchor_days=10)))
    assert other[-1].metadata["parameter_hash"] != fast[-1].metadata["parameter_hash"]


@pytest.mark.parametrize("value", (0, -3, 2.5))
def test_reanchor_days_validated(value):
    with pytest.raises(ValueError):
        FundamentalConfig(specific_reanchor_days=value)


def test_segment_floor_matches_sliced_window():
    import portopt

    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 7))
    x[rng.random(x.shape) < 0.2] = np.nan
    shift = rng.normal(size=7)
    floored = portopt.ew_moments_segment(x, 6, 20, shift, 9.0, 4, floor=6)
    sliced = portopt.ew_moments_segment(x[6:], 0, 14, shift, 9.0, 4)
    np.testing.assert_array_equal(floored, sliced)
    full = portopt.ew_moments_segment(x, 6, 20, shift, 9.0, 4)
    assert not np.array_equal(full, sliced)  # 不设下限时会多出跨界的滞后对


def test_branch_decisions_match_direct_next_to_variance_floor():
    """把每列缩放到 base、corr0 恰好落在 variance_floor 附近，分支判断必须与从头算一致。"""
    c = replace(FundamentalConfig(), history=40, specific_reanchor_days=7)
    rng = np.random.default_rng(23)
    t, n = 60, 400
    x = rng.normal(0, 1, (t, n))
    moments = SpecificMoments(n, c)
    window = x[t - 40 :]
    base, corr0, _ = direct_variances(window, c)
    which = np.arange(n) % 2 == 0  # 一半对齐 base，一半对齐 corr0
    scale = np.sqrt(c.variance_floor / np.where(which, base, corr0))
    scale *= 1 + rng.integers(-4, 5, n) * 2.2e-16
    x = x * scale
    for day in range(t):
        moments.push(day, x[day])
        got = moments.variances(day)
    expected = direct_variances(x[t - 40 :], c)
    for a, b in zip(got[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(a > c.variance_floor, b > c.variance_floor)
    assert np.sum(np.abs(expected[0] / c.variance_floor - 1) < 1e-12) > 50


def test_condition_gate_covers_a_stale_shift():
    """移位远离当前均值时抵消吃掉有效位，条件数闸必须让这些列改用从头算。"""
    c = replace(FundamentalConfig(), history=50, specific_reanchor_days=50)
    rng = np.random.default_rng(31)
    x = rng.normal(0, 0.02, (80, 70))
    moments = SpecificMoments(70, c)
    for day in range(60):
        moments.push(day, x[day])
    moments.variances(59)
    moments.shift = np.full(70, 1e4)  # 人为让移位失效，并清掉按旧移位算好的块
    moments.blocks.clear()
    moments.sums.clear()
    moments.push(60, x[60])
    got = moments.variances(60)
    for a, b in zip(got[:3], direct_variances(x[11:61], c), strict=True):
        assert relative_error(a, b).max() <= TOLERANCE
