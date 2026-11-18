"""groupbt.rank_ic 的测试：手算样例、停牌前填、top 组选择与独立参考实现对拍。"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gbt_rank_ic", ROOT / "src/ashare/factor/groupbt/rank_ic.py"
)
rank_ic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rank_ic)


def forward_filled(price, adjustment):
    """独立参考：逐列前填复权价。"""
    out = np.full(price.shape, np.nan)
    last = np.full(price.shape[1], np.nan)
    for row in range(price.shape[0]):
        value = price[row] * adjustment[row]
        ok = np.isfinite(value) & (value > 0.0)
        last[ok] = value[ok]
        out[row] = last
    return out


def _ranks_corr(a, b):
    rank_a = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rank_b = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = math.sqrt(float((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
    return float((rank_a * rank_b).sum() / denom) if denom > 0 else math.nan


def reference(factor, pool, price, adjustment, first, min_pool=20, min_top=10):
    """逐日逐股的笨办法：前填价格、手写速率相关、手写 top 组。"""
    px = forward_filled(price, adjustment)
    rows = px.shape[0]
    out_pool = np.full(rows, np.nan)
    out_top = np.full(rows, np.nan)
    for t in range(first, rows - 1):
        signal = t - 1 - first
        if signal < 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            forward = px[t + 1] / px[t] - 1.0
        x = factor[signal]
        valid = pool[signal] & np.isfinite(x) & np.isfinite(forward)
        if valid.sum() >= min_pool:
            out_pool[t] = _ranks_corr(x[valid], forward[valid])
        columns = np.flatnonzero(pool[signal] & np.isfinite(x))
        if columns.size:
            order = np.argsort(-x[columns], kind="stable")
            size = max(1, math.ceil(columns.size / 10))
            top = columns[order[:size]]
            keep = np.isfinite(forward[top])
            if keep.sum() >= min_top:
                out_top[t] = _ranks_corr(x[top][keep], forward[top][keep])
    return out_pool, out_top


def test_hand_example_and_padding():
    price = np.array([[10.0, 10.0], [np.nan, 11.0], [12.0, 12.0]])
    adjustment = np.ones_like(price)
    factor = np.array([[1.0, 2.0], [0.5, 0.5], [0.0, 0.0]])
    pool = np.ones((3, 2), dtype=bool)
    out_pool, out_top = rank_ic.series(
        factor, pool, price, adjustment, 0, min_pool=2, min_top=1
    )
    assert out_pool.shape == (3,) and out_top.shape == (3,)
    assert np.isnan(out_pool[0]) and np.isnan(out_pool[2])
    # t=1：因子 [1, 2] 与收益 [0.2, 12/11-1] 的秩相反，相关 -1。
    assert out_pool[1] == -1.0
    assert np.isnan(out_top[1])  # top 组只有 1 只，min_top=1 时秩相关无定义


def test_top_group_is_top_tenth_with_ties_by_column():
    price = np.array([[100.0] * 20, [100.0] * 20, [100.0] * 20])
    adjustment = np.ones((3, 20))
    factor = np.zeros((3, 20))
    factor[0] = np.arange(20, dtype=float)
    pool = np.ones((3, 20), dtype=bool)
    price[2] = 100.0 * (1.0 + 0.001 * factor[0])
    out_pool, out_top = rank_ic.series(
        factor, pool, price, adjustment, 0, min_pool=2, min_top=2
    )
    assert out_pool[1] == 1.0
    assert out_top[1] == 1.0  # top 2 只（因子 18、19）与收益完全同序
    # 并列的因子按列顺序：全 0 的一行，前 2 列进 top 组。
    tied = np.zeros((3, 20))
    px = np.full((3, 20), 100.0)
    px[2, :] = 100.0 * (1.0 + 0.001 * np.arange(20))
    out_pool, out_top = rank_ic.series(
        tied, pool, px, adjustment, 0, min_pool=2, min_top=2
    )
    assert out_pool[1] == 1.0
    assert out_top[1] == 1.0  # top 组是第 0、1 列，组内因子并列、按列顺序


def test_suspension_uses_previous_price():
    price = np.array(
        [[10.0, 10.0], [np.nan, 10.0], [10.5, 10.2], [11.0, 10.4], [11.0, 10.6]]
    )
    adjustment = np.ones_like(price)
    factor = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    pool = np.ones((5, 2), dtype=bool)
    out_pool, _ = rank_ic.series(
        factor, pool, price, adjustment, 0, min_pool=2, min_top=2
    )
    assert np.isfinite(out_pool[1:4]).all()


def test_random_matches_reference_and_inputs_unchanged():
    rng = np.random.default_rng(7)
    rows, columns, first = 40, 80, 6
    price = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (rows, columns)), axis=0))
    price[rng.random(price.shape) < 0.04] = np.nan
    adjustment = np.ones((rows, columns))
    adjustment[:, : columns // 4] = 1.3
    factor = rng.normal(size=(rows - first, columns))
    factor[rng.random(factor.shape) < 0.05] = np.nan
    pool = rng.random((rows - first, columns)) > 0.1
    copies = [array.copy() for array in (factor, pool, price, adjustment)]

    got = rank_ic.series(factor, pool, price, adjustment, first)
    want = reference(factor, pool, price, adjustment, first)
    for actual, expected in zip(got, want, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)

    for array, copy in zip((factor, pool, price, adjustment), copies, strict=True):
        np.testing.assert_allclose(array, copy, rtol=0, atol=0, equal_nan=True)


def test_window_offsets_and_shape_errors():
    rows, columns = 12, 40
    price = np.full((rows, columns), 100.0)
    adjustment = np.ones_like(price)
    pool = np.ones((rows - 4, columns), dtype=bool)
    factor = np.random.default_rng(0).normal(size=(rows - 4, columns))
    out_pool, out_top = rank_ic.series(factor, pool, price, adjustment, 4)
    assert np.isnan(out_pool[:5]).all() and np.isnan(out_pool[-1])
    assert np.isnan(out_top[:5]).all() and np.isnan(out_top[-1])

    with np.testing.assert_raises(ValueError):
        rank_ic.series(factor, pool, price, adjustment[:-1], 4)
    with np.testing.assert_raises(ValueError):
        rank_ic.series(factor[:-1], pool[:-1], price, adjustment, 4)


def test_cumulative():
    values = np.array([0.1, np.nan, -0.05, 0.02])
    copy = values.copy()
    np.testing.assert_allclose(
        rank_ic.cumulative(values), [0.1, 0.1, 0.05, 0.07], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(values, copy, rtol=0, atol=0, equal_nan=True)
