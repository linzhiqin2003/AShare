"""标签：时间对齐与回测成交约定一致、可买性、相对收益与排名，与逐日循环的参考实现对拍。"""

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402
from ashare.backtest import execution_days, market_inputs  # noqa: E402
from ashare.ml import labels as L  # noqa: E402

from .synthetic import SynthPanel, make  # noqa: E402


def _panel(t=60, n=12, seed=0):
    rng = np.random.default_rng(seed)
    price = np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0)) * 10
    price[rng.random((t, n)) < 0.05] = np.nan
    adj = np.cumprod(np.where(rng.random((t, n)) < 0.02, 1.1, 1.0), axis=0)
    hi, lo = price * 1.02, price * 0.98
    up, down = price * 1.1, price * 0.9
    lock = rng.random((t, n)) < 0.05
    up[lock] = price[lock]
    arrays = {
        "open": price,
        "close": price * 1.001,
        "vwap": price,
        "adj_factor": adj,
        "high": hi,
        "low": lo,
        "upper_limit": up,
        "lower_limit": down,
        "is_suspended": rng.random((t, n)) < 0.03,
        "has_quote": np.ones((t, n), bool),
        "volume": np.ones((t, n)),
        "split": np.full((t, n), np.nan),
        "cash_dividend": np.full((t, n), np.nan),
        "delist": np.zeros((t, n), bool),
        "industry": rng.integers(0, 4, (t, n)).astype(np.uint32),
    }
    return SynthPanel(
        arrays, [f"2020{i:04d}" for i in range(t)], [f"S{i}" for i in range(n)]
    )


def _naive(panel, h, lag, exec="close"):
    """逐格循环：信号日 t → 成交日 execution_days(t) 买入，再过 h 行卖出。"""
    p = np.asarray(panel[exec]) * np.asarray(panel["adj_factor"])
    t, n = p.shape
    out = np.full((t, n), np.nan)
    signal = np.arange(t)
    buy = execution_days(signal, t, lag=lag)  # 回测的成交日
    for s, b in zip(signal[: len(buy)], buy, strict=True):
        e = b + h
        if e >= t:
            continue
        for j in range(n):
            a, z = p[b, j], p[e, j]
            if np.isfinite(a) and np.isfinite(z) and a > 0 and z > 0:
                out[s, j] = z / a - 1
    return out


@pytest.mark.parametrize("h,lag", [(1, 1), (5, 1), (3, 2)])
def test_label_alignment_matches_backtest_timing(h, lag):
    panel = _panel()
    universe = np.ones(panel.shape, bool)
    lab = ml.label("ret1", horizon=h, lag=lag, exec="close")
    values = L.compute(panel, lab, universe)
    ref = _naive(panel, h, lag)
    np.testing.assert_array_equal(np.isnan(values.raw), np.isnan(ref))
    np.testing.assert_allclose(values.raw, ref, rtol=0, atol=0, equal_nan=True)
    assert lab.span == h + lag
    assert np.isnan(values.raw[-lab.span :]).all()
    assert np.isfinite(values.raw[: -lab.span]).any()


@pytest.mark.parametrize("exec", ["open", "close", "vwap"])
def test_tradable_same_as_market_inputs(exec):
    panel = _panel()
    can_buy, can_sell = L.tradable(panel, exec)
    inputs = market_inputs(panel, exec)
    np.testing.assert_array_equal(can_buy, inputs["can_buy"])
    np.testing.assert_array_equal(can_sell, inputs["can_sell"])


def test_buyable_aligned_to_buy_day_and_blocked_excluded():
    panel = _panel()
    universe = np.ones(panel.shape, bool)
    lab = ml.label("ret5", exec="close")
    values = L.compute(panel, lab, universe)
    can_buy, _ = L.tradable(panel, "close")
    t = panel.shape[0]
    np.testing.assert_array_equal(
        values.buyable[: t - lab.span], can_buy[1 : t - lab.span + 1]
    )
    assert (~values.buyable[: t - lab.span]).any(), "用例里必须真有买不进的格子"
    train = values.trainable(universe)
    assert not (train & ~values.buyable).any()
    keep = L.compute(panel, ml.label(lab, blocked="keep"), universe).trainable(universe)
    assert (keep & ~values.buyable).any()


def test_sell_blocked_drop():
    panel = _panel()
    universe = np.ones(panel.shape, bool)
    lab = ml.label("ret1", exec="close", sell_blocked="drop")
    values = L.compute(panel, lab, universe)
    _, can_sell = L.tradable(panel, "close")
    t = panel.shape[0]
    blocked = ~can_sell[lab.span :]
    assert blocked.any()
    assert np.isnan(values.raw[: t - lab.span][blocked]).all()


def test_relative_and_rank_against_daily_loop():
    panel = _panel()
    universe = np.random.default_rng(3).random(panel.shape) < 0.8
    raw = L.compute(panel, ml.label("ret5", exec="close"), universe).raw
    market = L.compute(
        panel, ml.label("ret5", exec="close", relative="market"), universe
    ).target
    ind = L.compute(
        panel, ml.label("ret5", exec="close", relative="industry"), universe
    ).target
    rank = L.compute(panel, ml.label("ret5", exec="close", rank=True), universe).target
    groups = np.asarray(panel["industry"])
    for d in range(panel.shape[0]):
        ok = universe[d] & np.isfinite(raw[d])
        if not ok.any():
            assert not np.isfinite(market[d]).any()
            continue
        np.testing.assert_allclose(
            market[d][ok], raw[d][ok] - raw[d][ok].mean(), atol=1e-15
        )
        for g in np.unique(groups[d][ok]):
            sel = ok & (groups[d] == g)
            np.testing.assert_allclose(
                ind[d][sel], raw[d][sel] - raw[d][sel].mean(), atol=1e-15
            )
        x = raw[d][ok]
        order = x.argsort().argsort()  # 无并列
        expect = order / (len(x) - 1) - 0.5 if len(x) > 1 else np.zeros(1)
        np.testing.assert_allclose(rank[d][ok], expect, atol=1e-12)


def test_presets_and_validation():
    assert ml.label("ret20").horizon == 20
    assert ml.label("ret5", relative="market").relative == "market"
    with pytest.raises(ValueError):
        ml.label("ret7")
    with pytest.raises(ValueError):
        ml.Label(horizon=0)
    with pytest.raises(ValueError):
        ml.Label(relative="sector")
    assert ml.label("ret5").digest != ml.label("ret5", lag=2).digest


def test_synthetic_label_is_planted_return():
    panel, feats, beta = make(t=80, n=10, k=2, noise=0.0, scale=0.01)
    values = L.compute(panel, ml.label("ret1"), np.ones(panel.shape, bool))
    planted = 0.01 * (beta[0] * feats["f0"] + beta[1] * feats["f1"])
    ok = np.isfinite(values.raw)
    np.testing.assert_allclose(np.log1p(values.raw[ok]), planted[ok], atol=1e-12)


def _naive_carry(panel, h, lag, k, exec="close"):
    """逐格循环：买入日 b，从 b+h 起找第一个可卖且有价的日子，最多到 b+h+k。"""
    p = np.asarray(panel[exec]) * np.asarray(panel["adj_factor"])
    _, can_sell = L.tradable(panel, exec)
    t, n = p.shape
    out, end = np.full((t, n), np.nan), np.full((t, n), -1)
    span = lag + h + k
    for s in range(t - span):
        b = s + lag
        for j in range(n):
            a = p[b, j]
            if not (np.isfinite(a) and a > 0):
                continue
            for e in range(b + h, b + h + k + 1):
                if can_sell[e, j] and np.isfinite(p[e, j]) and p[e, j] > 0:
                    out[s, j], end[s, j] = p[e, j] / a - 1, e
                    break
    return out, end


@pytest.mark.parametrize("h,lag,k", [(1, 1, 3), (5, 1, 2), (3, 2, 1)])
def test_sell_blocked_carry_against_daily_loop(h, lag, k):
    panel = _panel(t=80)
    universe = np.ones(panel.shape, bool)
    lab = ml.label(
        "ret1", horizon=h, lag=lag, exec="close", sell_blocked="carry", max_carry=k
    )
    assert lab.span == lag + h + k and lab.carry == k
    values = L.compute(panel, lab, universe)
    ref, end = _naive_carry(panel, h, lag, k)
    np.testing.assert_array_equal(values.raw, ref)
    np.testing.assert_array_equal(values.end, end)
    _, can_sell = L.tradable(panel, "close")
    hold = lag + h
    t = panel.shape[0]
    carried = np.isfinite(ref[: t - lab.span]) & ~can_sell[hold : t - lab.span + hold]
    assert carried.any(), "用例里必须真有顺延卖出的样本"
    assert (
        end[: t - lab.span][carried]
        > np.arange(t - lab.span)[:, None].repeat(panel.shape[1], 1)[carried] + hold
    ).all()
    # 不顺延的口径：结束日就是 t+lag+h
    price = L.compute(panel, ml.label(lab, sell_blocked="price"), universe)
    ok = np.isfinite(price.raw)
    rows = np.nonzero(ok)[0]
    np.testing.assert_array_equal(price.end[ok], rows + hold)
    assert (price.end[~ok] == -1).all()


def test_carry_widens_minimum_gap():
    lab = ml.label("ret5", sell_blocked="carry", max_carry=4)
    days = np.array(
        [
            f"{2010 + i // 240}{(i // 20) % 12 + 1:02d}{i % 20 + 1:02d}"
            for i in range(900)
        ]
    )
    with pytest.raises(ValueError, match="不能小于标签跨度"):
        ml.schedule(days, lab.span, start=600, gap=1 + 5)  # 旧的 lag+h 不够
    sched = ml.schedule(days, lab.span, start=600)
    ml.check_schedule(sched, lab.span)
    for f in sched.folds:
        assert f.train[1] - 1 + lab.span < (f.valid or f.test)[0]


def test_zscore_target_against_daily_loop():
    panel = _panel()
    universe = np.random.default_rng(4).random(panel.shape) < 0.8
    raw = L.compute(panel, ml.label("ret5", exec="close"), universe).raw
    z = L.compute(panel, ml.label("ret5", exec="close", zscore=True), universe)
    np.testing.assert_array_equal(z.raw, raw)  # 评估仍用原始收益
    for d in range(panel.shape[0]):
        ok = universe[d] & np.isfinite(raw[d])
        if ok.sum() < 2:
            continue
        x = raw[d][ok]
        np.testing.assert_allclose(
            z.target[d][ok], (x - x.mean()) / x.std(ddof=0), atol=1e-12
        )
    with pytest.raises(ValueError, match="只能选一个"):
        ml.label("ret5", rank=True, zscore=True)


def test_new_label_options_keep_existing_identity():
    """后加的选项取默认值时定义不变，旧试验的编号不变。"""
    base = ml.label("ret5")
    assert "zscore" not in base.describe() and "max_carry" not in base.describe()
    assert (
        ml.label("ret5", max_carry=3).digest == base.digest
    )  # 不顺延时 max_carry 无效
    assert ml.label("ret5", zscore=True).digest != base.digest
    carry = ml.label("ret5", sell_blocked="carry")
    assert carry.digest != ml.label("ret5", sell_blocked="carry", max_carry=3).digest
