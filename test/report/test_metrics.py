"""指标公式：手算的小例子写死核对。"""

import math

import numpy as np
import pytest

from ashare.report import metrics as m


def test_annual_return_hand():
    r = np.array([0.10, -0.05])
    assert m.annual_return(r, 252) == pytest.approx((1.10 * 0.95) ** 126 - 1)
    assert m.annual_return(np.array([0.01] * 252), 252) == pytest.approx(1.01**252 - 1)
    assert m.annual_return(np.array([]), 252) is None


def test_volatility_and_sharpe_hand():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    mean = 0.005
    var = ((0.005**2) + (0.015**2) + (0.015**2) + (0.005**2)) / 3
    assert m.volatility(r, 252) == pytest.approx(math.sqrt(var) * math.sqrt(252))
    assert m.sharpe(r, 252) == pytest.approx(mean / math.sqrt(var) * math.sqrt(252))
    rf = 0.02
    assert m.sharpe(r, 252, rf) == pytest.approx(
        (mean - rf / 252) / math.sqrt(var) * math.sqrt(252)
    )
    assert m.sharpe(np.zeros(5), 252) is None
    assert m.volatility(np.array([0.1]), 252) is None


def test_sortino_hand():
    r = np.array([0.02, -0.01, 0.0, -0.03])
    dn = math.sqrt((0.01**2 + 0.03**2) / 4)  # 下行偏差按全部日子求均值
    assert m.sortino(r, 252) == pytest.approx(np.mean(r) * math.sqrt(252) / dn)
    assert m.sortino(np.array([0.01, 0.02]), 252) is None


def test_max_drawdown_hand():
    # 1 → 1.1 → 0.88 → 0.968：峰 1.1，谷 0.88，回撤 −20%
    r = np.array([0.10, -0.20, 0.10])
    assert m.max_drawdown_of_returns(r) == pytest.approx(-0.2)
    assert m.max_drawdown_of_returns(np.array([0.01, 0.02])) == 0.0
    nav = np.array([1.0, 1.2, 0.9, 1.3])
    np.testing.assert_allclose(m.drawdown(nav), [0, 0, -0.25, 0])


def test_beta_hand():
    b = np.array([0.01, -0.02, 0.03, 0.0])
    r = 0.5 * b + 0.001
    assert m.beta(r, b) == pytest.approx(0.5)
    assert m.beta(r, np.zeros(4)) is None


def test_episodes_hand():
    nav = np.array([1.0, 1.2, 0.9, 1.3, 1.1, 1.25])
    d = [f"d{i}" for i in range(len(nav))]
    eps = m.episodes(nav, d)
    assert eps == [
        {
            "peak": "d1",
            "trough": "d2",
            "recovery": "d3",
            "depth": -0.25,
            "days_to_trough": 1,
            "duration_days": 2,
            "recovered": True,
        },
        {
            "peak": "d3",
            "trough": "d4",
            "recovery": None,
            "depth": round(1.1 / 1.3 - 1, 6),
            "days_to_trough": 1,
            "duration_days": 2,
            "recovered": False,
        },
    ]
    assert m.episodes(np.array([1.0, 1.1, 1.2]), ["a", "b", "c"]) == []


def test_episode_recovery_counts_equal_peak():
    # 回到与峰值相等即算恢复
    eps = m.episodes(np.array([1.0, 0.9, 1.0, 0.95]), list("abcd"))
    assert eps[0]["recovery"] == "c" and eps[1]["peak"] == "c"


def test_rolling_matches_loop():
    rng = np.random.default_rng(3)
    r = np.r_[0.0, rng.normal(0, 0.01, 40)]
    b = np.r_[0.0, rng.normal(0, 0.01, 40)]
    out = m.rolling(r, b, 10, 252)
    assert out["date_index"][0] == 10 and out["date_index"][-1] == 40
    for k, i in enumerate(out["date_index"]):
        w, wb = r[i - 9 : i + 1], b[i - 9 : i + 1]
        assert out["annual_return"][k] == pytest.approx(np.prod(1 + w) - 1, abs=1e-5)
        assert out["volatility"][k] == pytest.approx(
            np.std(w, ddof=1) * math.sqrt(252), abs=1e-5
        )
        assert out["sharpe"][k] == pytest.approx(
            w.mean() / np.std(w, ddof=1) * math.sqrt(252), abs=1e-4
        )
        assert out["beta"][k] == pytest.approx(
            np.cov(w, wb)[0, 1] / np.var(wb, ddof=1), abs=1e-4
        )


def test_rolling_short_history_is_empty():
    out = m.rolling(np.zeros(100), None, 252, 252)
    assert out["date_index"] == [] and out["sharpe"] == []


def test_rolling_beta_none_where_benchmark_missing():
    rng = np.random.default_rng(1)
    r = np.r_[0.0, rng.normal(0, 0.01, 20)]
    b = r.copy()
    b[:8] = np.nan
    out = m.rolling(r, b, 5, 252)
    assert out["beta"][0] is None and out["beta"][-1] == pytest.approx(1.0)
