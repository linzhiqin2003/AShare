"""信号日 → 成交日的错位：正常路径、末尾信号、停牌顺延、与手写 +1 逐位相同。"""

import numpy as np
import panelbt
import panelops
import pytest

from ashare.backtest import execution_days, execution_targets


def same(a, b):
    for name in ("days", "indptr", "cols", "weights"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and np.array_equal(x, y), name
    assert a.kind == b.kind


def test_dense_shifts_rows_and_drops_tail():
    w = np.full((5, 3), np.nan)
    w[1] = [0.5, 0.5, 0.0]
    w[4] = [1.0, 0.0, 0.0]  # 末日信号：没有成交日
    plan = execution_targets(w)
    assert plan.days.tolist() == [2]
    assert plan.cols.tolist() == [0, 1] and plan.weights.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="没有对应的成交日"):
        execution_targets(w, tail="error")
    w[4] = np.nan
    same(execution_targets(w, tail="error"), plan)


def test_dense_lag_two_and_lag_longer_than_panel():
    w = np.full((4, 2), np.nan)
    w[0] = [1.0, 0.0]
    assert execution_targets(w, lag=2).days.tolist() == [2]
    assert execution_targets(w, lag=9).days.tolist() == []


def test_empty_row_means_close_everything_and_survives_shift():
    w = np.full((3, 2), np.nan)
    w[0] = 0.0
    plan = execution_targets(w)
    assert plan.days.tolist() == [1] and plan.indptr.tolist() == [0, 0]


def test_sparse_input_requires_t_and_keeps_kind():
    s = panelbt.Targets([0, 3], [0, 1, 2], [1, 0], [100.0, 200.0], "shares")
    with pytest.raises(ValueError, match="t=面板行数"):
        execution_targets(s)
    plan = execution_targets(s, t=4)
    assert plan.kind == "shares"
    assert plan.days.tolist() == [1] and plan.cols.tolist() == [1]
    with pytest.raises(ValueError, match="没有成交日"):
        execution_targets(s, t=4, tail="error")
    full = execution_targets(s, t=5, tail="error")
    assert full.days.tolist() == [1, 4] and full.weights.tolist() == [100.0, 200.0]


@pytest.mark.parametrize("lag", [0, -1, 1.0, True])
def test_bad_lag(lag):
    with pytest.raises(ValueError, match="lag"):
        execution_targets(np.zeros((2, 2)), lag=lag)


def test_execution_days_checks():
    assert execution_days(np.array([0, 2, 3]), 4).tolist() == [1, 3]
    with pytest.raises(ValueError, match="严格递增"):
        execution_days(np.array([2, 1]), 4)
    with pytest.raises(ValueError, match="行号必须在"):
        execution_days(np.array([4]), 4)
    with pytest.raises(ValueError, match="tail"):
        execution_days(np.array([0]), 4, tail="keep")


def test_matches_handwritten_plus_one_bit_for_bit():
    """旧写法：panelops.targets 的 days + 1 直接构造 Targets（bench/e2e 曾这样写）。"""
    rng = np.random.default_rng(7)
    score = rng.standard_normal((300, 40))
    score[rng.random(score.shape) < 0.1] = np.nan
    days = np.arange(5, 299, 7, dtype=np.int64)
    sparse = panelops.targets(score, fraction=0.2, days=days)
    old = panelbt.Targets(sparse.days + 1, sparse.indptr, sparse.cols, sparse.weights)
    same(execution_targets(sparse, t=300), old)
    dense = old.to_dense(300, 40)
    signal = np.full_like(dense, np.nan)
    signal[:-1] = dense[1:]
    same(execution_targets(signal), panelbt.Targets.from_dense(dense))


def test_suspended_on_execution_day_carries_to_next_day():
    """成交日停牌不在错位里处理：引擎拦下，carry_blocked 次日重试。"""
    t, n = 6, 2
    price = np.full((t, n), 10.0)
    buy = np.ones((t, n), bool)
    buy[2, 0] = False  # 信号日 1 → 成交日 2 停牌
    market = panelbt.Market(price, price, buy, np.ones((t, n), bool))
    signal = np.full((t, n), np.nan)
    signal[1] = [0.5, 0.5]
    config = panelbt.Config(min_qty=0.0, qty_step=0.0, record_trades=True)
    costs = panelbt.Costs(
        commission_rate=0,
        commission_min=None,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    result = panelbt.run(market, execution_targets(signal), costs, config)
    held = result.positions
    assert result.diagnostics["blocked_buy"] == 1
    assert result.diagnostics["carry_trade"] == 1
    assert held[1].tolist() == [0, 0]  # 信号日当天不成交
    assert held[2, 0] == 0 and held[2, 1] > 0
    assert held[3, 0] > 0
