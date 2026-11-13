"""groupbt.portfolio 的测试：手算小例子、笨循环参考实现、边界与随机性质。"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ashare"
    / "factor"
    / "groupbt"
    / "portfolio.py"
)
_SPEC = importlib.util.spec_from_file_location("gbt_portfolio", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
portfolio = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = portfolio
_SPEC.loader.exec_module(portfolio)

GroupingError = portfolio.GroupingError
assign_groups = portfolio.assign_groups
group_targets = portfolio.group_targets
long_short_targets = portfolio.long_short_targets
simulate = portfolio.simulate

_TOL = 1e-12


# ---- 笨参考实现：逐行逐列 --------------------------------------------------------


def _ref_assign_groups(alpha, n_groups):
    a = np.asarray(alpha, dtype=np.float64)
    n_rows, n_cols = a.shape
    out = np.full((n_rows, n_cols), -1, dtype=np.int64)
    bad = []
    for t in range(n_rows):
        cols = [j for j in range(n_cols) if not np.isnan(a[t, j])]
        count = len(cols)
        if count == 0:
            continue
        if count < n_groups or len({a[t, j] for j in cols}) < n_groups:
            bad.append(t)
            continue
        for j in cols:
            rank = 0
            for k in cols:
                if a[t, k] > a[t, j] or (a[t, k] == a[t, j] and k < j):
                    rank += 1
            out[t, j] = rank * n_groups // count
    if bad:
        raise GroupingError(bad)
    return out


def _ref_group_targets(groups, g):
    grp = np.asarray(groups)
    n_rows, n_cols = grp.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for t in range(n_rows):
        members = [j for j in range(n_cols) if grp[t, j] == g]
        if members:
            weight = 1.0 / len(members)
            for j in members:
                out[t, j] = weight
    return out


def _ref_long_short_targets(groups, n_groups):
    return 0.5 * _ref_group_targets(groups, 0) - 0.5 * _ref_group_targets(
        groups, n_groups - 1
    )


def _ref_simulate(target, fwd_ret, can_buy=None, can_sell=None):
    tgt = np.asarray(target, dtype=np.float64)
    ret = np.asarray(fwd_ret, dtype=np.float64)
    n_rows, n_cols = tgt.shape
    buy = (
        np.ones((n_rows, n_cols), dtype=bool)
        if can_buy is None
        else np.asarray(can_buy, dtype=bool)
    )
    sell = (
        np.ones((n_rows, n_cols), dtype=bool)
        if can_sell is None
        else np.asarray(can_sell, dtype=bool)
    )

    held = np.zeros((n_rows, n_cols), dtype=np.float64)
    pnl = np.zeros(n_rows, dtype=np.float64)
    turnover = np.zeros(n_rows, dtype=np.float64)
    prev = [0.0] * n_cols

    for t in range(n_rows):
        row = [0.0 if math.isnan(tgt[t, j]) else tgt[t, j] for j in range(n_cols)]
        rr = [0.0 if math.isnan(ret[t, j]) else ret[t, j] for j in range(n_cols)]
        if can_buy is None and can_sell is None:
            cand = list(row)
        else:
            frozen = [False] * n_cols
            budget_long = sum(v for v in row if v > 0)
            budget_short = -sum(v for v in row if v < 0)
            while True:
                used_long = sum(
                    prev[j] for j in range(n_cols) if frozen[j] and prev[j] > 0
                )
                used_short = sum(
                    -prev[j] for j in range(n_cols) if frozen[j] and prev[j] < 0
                )
                free_long = sum(
                    row[j] for j in range(n_cols) if not frozen[j] and row[j] > 0
                )
                free_short = sum(
                    -row[j] for j in range(n_cols) if not frozen[j] and row[j] < 0
                )
                scale_long = (
                    max(budget_long - used_long, 0.0) / free_long
                    if free_long > 0
                    else 0.0
                )
                scale_short = (
                    max(budget_short - used_short, 0.0) / free_short
                    if free_short > 0
                    else 0.0
                )
                cand = [0.0] * n_cols
                for j in range(n_cols):
                    if frozen[j]:
                        cand[j] = prev[j]
                    elif row[j] > 0:
                        cand[j] = row[j] * scale_long
                    elif row[j] < 0:
                        cand[j] = row[j] * scale_short
                violate = []
                for j in range(n_cols):
                    if frozen[j]:
                        continue
                    if cand[j] > prev[j] + _TOL and not buy[t, j]:
                        violate.append(j)
                    elif cand[j] < prev[j] - _TOL and not sell[t, j]:
                        violate.append(j)
                if not violate:
                    break
                for j in violate:
                    frozen[j] = True
        held[t] = cand
        pnl[t] = sum(cand[j] * rr[j] for j in range(n_cols))
        turnover[t] = sum(abs(cand[j] - prev[j]) for j in range(n_cols))
        prev = cand

    return {"weights": held, "pnl": pnl, "turnover": turnover}


# ---- assign_groups -------------------------------------------------------------


def test_assign_groups_hand_six_stocks_three_groups():
    alpha = np.array([[0.5, 0.1, 0.9, 0.3, 0.7, 0.2]])
    got = assign_groups(alpha, 3)
    # 从大到小：0.9(2) 0.7(4) 0.5(0) 0.3(3) 0.2(5) 0.1(1)，每组 2 只。
    expected = np.array([[1, 2, 0, 1, 0, 2]], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.int64


def test_assign_groups_hand_with_ties():
    alpha = np.array([[0.9, 0.5, 0.5, 0.5, 0.1]])
    got = assign_groups(alpha, 3)
    # 0.5 并列，按列下标 1、2、3 依次占名次 1、2、3；组号 r*3//5。
    expected = np.array([[0, 0, 1, 1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)


def test_assign_groups_all_nan_row_is_not_an_error():
    alpha = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    got = assign_groups(alpha, 2)
    expected = np.array(
        [
            [-1, -1, -1, -1],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(got, expected)


def test_assign_groups_single_valid_stock_with_one_group():
    alpha = np.array(
        [
            [np.nan, 0.7, np.nan, np.nan],
            [np.nan, 0.7, np.nan, np.nan],
        ]
    )
    got = assign_groups(alpha, 1)
    expected = np.array(
        [
            [-1, 0, -1, -1],
            [-1, 0, -1, -1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(got, expected)


def test_assign_groups_insufficient_rows_raise_with_all_rows():
    alpha = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [1.0, 1.0, 1.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [1.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    with pytest.raises(GroupingError) as excinfo:
        assign_groups(alpha, 3)
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert err.rows == [1, 3, 4]
    assert "3 行" in str(err)
    assert "[1, 3, 4]" in str(err)


def test_assign_groups_matches_brute_force_reference():
    rng = np.random.default_rng(0)
    alpha = rng.normal(size=(8, 12))
    alpha[rng.random(alpha.shape) < 0.05] = np.nan
    got = assign_groups(alpha, 3)
    expected = _ref_assign_groups(alpha, 3)
    np.testing.assert_array_equal(got, expected)


# ---- group_targets / long_short_targets ----------------------------------------


def test_group_targets_hand():
    groups = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, -1],
        ]
    )
    np.testing.assert_allclose(
        group_targets(groups, 0),
        np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        group_targets(groups, 1),
        np.array(
            [
                [0.0, 0.0, 0.5, 0.5],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
            ]
        ),
    )


def test_long_short_targets_hand():
    groups = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, -1],
        ]
    )
    expected = np.array(
        [
            [0.25, 0.25, -0.25, -0.25],
            [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, 0.0],
        ]
    )
    np.testing.assert_allclose(long_short_targets(groups, 2), expected)


def test_group_targets_and_long_short_match_reference():
    rng = np.random.default_rng(0)
    groups = rng.integers(-1, 4, size=(7, 10))
    for g in range(4):
        np.testing.assert_allclose(
            group_targets(groups, g), _ref_group_targets(groups, g)
        )
    np.testing.assert_allclose(
        long_short_targets(groups, 4), _ref_long_short_targets(groups, 4)
    )


# ---- simulate：手算 ------------------------------------------------------------


def test_simulate_no_limits_hand():
    target = np.array(
        [
            [0.5, -0.5],
            [0.25, np.nan],
            [0.0, 0.6],
        ]
    )
    fwd_ret = np.array(
        [
            [0.1, 0.2],
            [0.4, np.nan],
            [np.nan, -0.5],
        ]
    )
    res = simulate(target, fwd_ret)
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.5, -0.5],
                [0.25, 0.0],
                [0.0, 0.6],
            ]
        ),
    )
    np.testing.assert_allclose(res["pnl"], np.array([-0.05, 0.1, -0.3]))
    np.testing.assert_allclose(res["turnover"], np.array([1.0, 0.75, 0.85]))
    assert np.isnan(target[1, 1])
    assert np.isnan(fwd_ret[1, 1])


def test_simulate_suspended_holding_keeps_weight():
    target = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    fwd_ret = np.zeros_like(target)
    can_buy = np.array(
        [
            [True, True, True, True],
            [False, True, True, True],
        ]
    )
    can_sell = can_buy.copy()
    res = simulate(target, fwd_ret, can_buy=can_buy, can_sell=can_sell)
    # 第 0 只停牌，保持 0.25；剩余多头额度 0.75 按 0.5:0.5 缩放。
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.0, 0.375, 0.375],
            ]
        ),
    )
    np.testing.assert_allclose(res["turnover"], np.array([1.0, 0.5]))


def test_simulate_limit_up_can_only_sell():
    target = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.0],
            [0.1, 0.3],
        ]
    )
    fwd_ret = np.zeros_like(target)
    can_buy = np.array(
        [
            [True, True],
            [False, True],
            [False, True],
        ]
    )
    res = simulate(target, fwd_ret, can_buy=can_buy)
    # 涨停只能卖：想加仓被冻结在 0.2；想减仓照常从 0.2 降到 0.1。
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.2, 0.2],
                [0.2, 0.0],
                [0.1, 0.3],
            ]
        ),
    )
    np.testing.assert_allclose(res["turnover"], np.array([0.4, 0.2, 0.4]))


def test_simulate_limit_down_can_only_buy():
    target = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.0],
            [0.1, 0.3],
        ]
    )
    fwd_ret = np.zeros_like(target)
    can_sell = np.array(
        [
            [True, True],
            [True, False],
            [True, False],
        ]
    )
    res = simulate(target, fwd_ret, can_sell=can_sell)
    # 跌停只能买：想减仓被冻结在 0.2；想加仓照常从 0.2 加到 0.3。
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.2, 0.2],
                [0.2, 0.2],
                [0.1, 0.3],
            ]
        ),
    )
    np.testing.assert_allclose(res["turnover"], np.array([0.4, 0.0, 0.2]))


def test_simulate_limit_up_without_position_gets_no_weight():
    target = np.array([[0.5, 0.3, 0.2]])
    fwd_ret = np.zeros_like(target)
    can_buy = np.array([[False, True, True]])
    res = simulate(target, fwd_ret, can_buy=can_buy)
    # 第 0 只涨停不能买，剩下的多头额度 1.0 按 0.3:0.2 分给另外两只。
    np.testing.assert_allclose(res["weights"], np.array([[0.0, 0.6, 0.4]]))
    np.testing.assert_allclose(res["turnover"], np.array([1.0]))


def test_simulate_frozen_short_occupies_short_budget_only():
    target = np.array(
        [
            [0.5, 0.5, -0.5, -0.5],
            [0.8, 0.2, -0.3, -0.7],
        ]
    )
    fwd_ret = np.zeros_like(target)
    can_sell = np.array(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    res = simulate(target, fwd_ret, can_sell=can_sell)
    # 第 3 只空头跌停不能再卖（不能把 -0.5 加到 -0.7），冻结占掉 0.5 空头额度；
    # 第 2 只空头用剩下的 0.5 额度从 -0.3 放大到 -0.5，多头照旧 0.8/0.2。
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.5, 0.5, -0.5, -0.5],
                [0.8, 0.2, -0.5, -0.5],
            ]
        ),
    )
    np.testing.assert_allclose(res["turnover"], np.array([2.0, 0.6]))


def test_simulate_cascade_freezes_second_restricted_stock():
    target = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.7, 0.5, 0.0],
        ]
    )
    fwd_ret = np.zeros_like(target)
    can_buy = np.array(
        [
            [True, True, True],
            [False, False, True],
        ]
    )
    res = simulate(target, fwd_ret, can_buy=can_buy)
    # 第一轮：第 0 只想从 0.5 加到 0.7，涨停不能买，冻结在 0.5。
    # 第二轮：剩余额度让第 1 只从 0.5 放大到 0.7，它也不能买，被冻结。
    # 第三轮：两只都保持 0.5，收盘权重 [0.5, 0.5, 0]。
    np.testing.assert_allclose(
        res["weights"],
        np.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(res["turnover"], np.array([1.0, 0.0]))


# ---- simulate：随机对拍与性质 --------------------------------------------------


def test_simulate_matches_brute_force_reference_with_limits():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(7, 6)) * 0.2
    fwd_ret = rng.normal(size=(7, 6)) * 0.01
    target[0, :] = np.nan
    fwd_ret[0, :] = np.nan
    target[2, 1] = np.nan
    fwd_ret[3, 2] = np.nan
    can_buy = rng.random((7, 6)) > 0.7
    can_sell = rng.random((7, 6)) > 0.7

    got = simulate(target, fwd_ret, can_buy=can_buy, can_sell=can_sell)
    expected = _ref_simulate(target, fwd_ret, can_buy=can_buy, can_sell=can_sell)
    for key in ("weights", "pnl", "turnover"):
        np.testing.assert_allclose(
            got[key], expected[key], equal_nan=True, rtol=1e-14, atol=1e-14
        )


def test_simulate_random_properties():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 20, 15
    target = rng.normal(size=(n_rows, n_cols)) * 0.1
    fwd_ret = rng.normal(size=(n_rows, n_cols)) * 0.01
    target[rng.random(target.shape) < 0.05] = np.nan
    fwd_ret[rng.random(fwd_ret.shape) < 0.05] = np.nan
    can_buy = rng.random(target.shape) > 0.8
    can_sell = rng.random(target.shape) > 0.8

    res = simulate(target, fwd_ret, can_buy=can_buy, can_sell=can_sell)
    held = res["weights"]
    clean = np.where(np.isnan(target), 0.0, target)

    for t in range(n_rows):
        prev = held[t - 1] if t > 0 else np.zeros(n_cols)
        cur = held[t]
        budget_long = float(clean[t][clean[t] > 0].sum())
        prev_long = float(prev[prev > 0].sum())
        cur_long = float(cur[cur > 0].sum())
        assert cur_long <= max(budget_long, prev_long) + 1e-9

        budget_short = float(-clean[t][clean[t] < 0].sum())
        prev_short = float(-prev[prev < 0].sum())
        cur_short = float(-cur[cur < 0].sum())
        assert cur_short <= max(budget_short, prev_short) + 1e-9

        increased = cur > prev + 1e-9
        assert np.all(can_buy[t][increased])
        decreased = cur < prev - 1e-9
        assert np.all(can_sell[t][decreased])
