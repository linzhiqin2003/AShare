"""分组回测入口：与一份逐日逐股循环的独立参考实现对拍，并守住时点、股票池和交易限制。"""

import numpy as np
import pytest

from ashare.factor import groupbt
from ashare.factor.groupbt import api


class FakePanel:
    def __init__(self, arrays, dates, symbols):
        self._a, self.dates, self.symbols = arrays, dates, symbols

    def __getitem__(self, name):
        return self._a[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


def make_panel(seed=0, t=70, n=60, limits=True):
    rng = np.random.default_rng(seed)
    dates = np.array(
        [f"{2020 + i // 40}{1 + (i % 40) // 20:02d}{1 + i % 20:02d}" for i in range(t)]
    )
    ret = rng.normal(0, 0.03, (t, n))
    close = 10 * np.exp(np.cumsum(ret, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, (t, n)))
    adj = np.ones((t, n))
    adj[t // 2 :, : n // 3] = 1.25  # 一次除权
    susp = rng.random((t, n)) < (0.06 if limits else 0.0)
    listed = np.ones((t, n), dtype=bool)
    listed[:10, -3:] = False  # 后上市
    quote = listed & ~susp
    open_[~quote], close[~quote] = np.nan, np.nan
    buy = quote & ~(rng.random((t, n)) < (0.08 if limits else 0.0))
    sell = quote & ~(rng.random((t, n)) < (0.08 if limits else 0.0))
    up = np.where(rng.random((t, n)) < (0.08 if limits else 0.0), close, close * 1.1)
    down = np.where(rng.random((t, n)) < (0.08 if limits else 0.0), close, close * 0.9)
    member = lambda lo, hi: np.tile((np.arange(n) >= lo) & (np.arange(n) < hi), (t, 1))  # noqa: E731
    m1000 = member(30, 50)
    m1000[:15] = False  # 中证1000 的成分数据晚开始
    arrays = dict(
        open=open_,
        close=close,
        adj_factor=adj,
        upper_limit=up,
        lower_limit=down,
        total_market_cap=np.exp(rng.normal(23, 1.5, (t, n))),
        industry=(801000 + 10 * rng.integers(0, 4, (t, n))).astype(np.uint32),
        is_listed=listed,
        is_suspended=susp,
        has_quote=quote,
        can_buy_open=buy,
        can_sell_open=sell,
        member_csi300=member(0, 12),
        member_csi500=member(12, 30),
        member_csi1000=m1000,
    )
    return FakePanel(arrays, dates, np.array([f"S{i}" for i in range(n)]))


def reference(factor, panel, universe, trade_at, limited, n_groups):
    """逐日逐股的笨办法：只做「不中性化」的情形，结构与实现无关。"""
    t, n = panel.shape
    names = {"TOP800": ["member_csi300", "member_csi500"]}
    names["TOP1800"] = names["TOP800"] + ["member_csi1000"]
    px = np.full((t, n), np.nan)
    for j in range(n):
        last = np.nan
        for i in range(t):
            v = panel[trade_at][i, j] * panel["adj_factor"][i, j]
            if np.isfinite(v) and v > 0:
                last = v
            px[i, j] = last
    if trade_at == "open":
        buy, sell = panel["can_buy_open"], panel["can_sell_open"]
    else:
        buy = np.zeros((t, n), dtype=bool)
        sell = np.zeros((t, n), dtype=bool)
        for i in range(t):
            for j in range(n):
                c = panel["close"][i, j]
                ok = (
                    np.isfinite(c)
                    and panel["has_quote"][i, j]
                    and not panel["is_suspended"][i, j]
                )
                buy[i, j] = ok and not c >= panel["upper_limit"][i, j] - 0.005
                sell[i, j] = ok and not c <= panel["lower_limit"][i, j] + 0.005
    targets = []  # 每个信号日：各组和多空的目标 {股票: 权重}
    for i in range(t):
        pool = []
        for j in range(n):
            if not panel["is_listed"][i, j] or not np.isfinite(factor[i, j]):
                continue
            if universe != "ALL":
                parts = [panel[f][i] for f in names[universe]]
                if not all(p.any() for p in parts) or not any(p[j] for p in parts):
                    continue
            pool.append(j)
        ranked = sorted(pool, key=lambda j: (-factor[i, j], j))
        books = [dict() for _ in range(n_groups + 1)]
        for r, j in enumerate(ranked):
            books[r * n_groups // len(ranked)][j] = 0.0
        for g in range(n_groups):
            for j in books[g]:
                books[g][j] = 1.0 / len(books[g])
        for j in books[0]:
            books[n_groups][j] = 0.5 / len(books[0])
        for j in books[n_groups - 1] if ranked else []:
            books[n_groups][j] = -0.5 / len(books[n_groups - 1])
        targets.append(books)
    pnl = np.zeros((n_groups + 1, t))
    turn = np.zeros((n_groups + 1, t))
    for b in range(n_groups + 1):
        prev = np.zeros(n)
        for s in range(t):
            want = np.zeros(n)
            if s > 0:
                for j, w in targets[s - 1][b].items():
                    want[j] = w
            hold = want.copy()
            if limited:
                frozen = set()
                while True:
                    hold = np.zeros(n)
                    room = {
                        1: sum(w for w in want if w > 0),
                        -1: sum(-w for w in want if w < 0),
                    }
                    free = {1: 0.0, -1: 0.0}
                    for j in range(n):
                        if j in frozen:
                            hold[j] = prev[j]
                            if prev[j] != 0:
                                room[int(np.sign(prev[j]))] -= abs(prev[j])
                        elif want[j] != 0:
                            free[int(np.sign(want[j]))] += abs(want[j])
                    for j in range(n):
                        if j not in frozen and want[j] != 0:
                            side = int(np.sign(want[j]))
                            hold[j] = want[j] * max(room[side], 0.0) / free[side]
                    bad = {
                        j
                        for j in range(n)
                        if j not in frozen
                        and (
                            (hold[j] > prev[j] + 1e-12 and not buy[s, j])
                            or (hold[j] < prev[j] - 1e-12 and not sell[s, j])
                        )
                    }
                    if not bad:
                        break
                    frozen |= bad
            for j in range(n):
                if s + 1 < t and hold[j] != 0:
                    r = px[s + 1, j] / px[s, j] - 1
                    if np.isfinite(r):
                        pnl[b, s] += hold[j] * r
                turn[b, s] += abs(hold[j] - prev[j])
            prev = hold
    return pnl, turn


@pytest.mark.parametrize("universe", ["TOP800", "TOP1800", "ALL"])
@pytest.mark.parametrize("trade_at", ["open", "close"])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("n_groups", [5, 10])
def test_matches_loop_reference(universe, trade_at, limited, n_groups):
    panel = make_panel(seed=3)
    factor = np.random.default_rng(7).normal(size=panel.shape)
    factor[:2] = np.nan  # 开头没有因子
    got = groupbt.run(
        factor,
        panel,
        universe,
        trade_at=trade_at,
        trade_limitation=limited,
        n_groups=n_groups,
        sub_universes=False,
    )
    pnl, turn = reference(factor, panel, universe, trade_at, limited, n_groups)
    live = ~np.isnan(got.pnl["long_short"])
    assert live.sum() > 40
    np.testing.assert_allclose(
        got.group_pnl[:, live], pnl[:n_groups][:, live], atol=1e-12
    )
    np.testing.assert_allclose(
        got.pnl["long_short"][live], pnl[n_groups][live], atol=1e-12
    )
    np.testing.assert_allclose(
        got.turnover["long_short"][live], turn[n_groups][live], atol=1e-12
    )
    np.testing.assert_allclose(got.turnover["top"][live], turn[0][live], atol=1e-12)
    # 统计开始之前（没有持仓）参考实现的盈亏也是 0
    assert np.all(pnl[:, ~live] == 0)


def test_no_lookahead():
    """改动第 k 行及以后的因子，k 日及以前的结果不变；改动 k+1 行以后的价格，k-1 日及以前不变。"""
    panel = make_panel(seed=1)
    rng = np.random.default_rng(2)
    factor = rng.normal(size=panel.shape)
    base = groupbt.run(factor, panel, "ALL", sub_universes=False)
    k = 40
    other = factor.copy()
    other[k:] = rng.normal(size=other[k:].shape)
    moved = groupbt.run(other, panel, "ALL", sub_universes=False)
    np.testing.assert_array_equal(
        base.group_pnl[:, : k + 1], moved.group_pnl[:, : k + 1]
    )
    assert not np.allclose(base.group_pnl[:, k + 1 :], moved.group_pnl[:, k + 1 :])
    arrays = dict(panel._a)
    arrays["open"] = arrays["open"].copy()
    arrays["open"][k + 1 :] *= 1.5
    later = groupbt.run(
        factor,
        FakePanel(arrays, panel.dates, panel.symbols),
        "ALL",
        sub_universes=False,
    )
    np.testing.assert_array_equal(
        base.pnl["long_short"][:k], later.pnl["long_short"][:k]
    )
    assert base.pnl["long_short"][k] != later.pnl["long_short"][k]


def test_universes_and_subs():
    panel = make_panel(seed=4, limits=False)
    top800 = api.universe_mask(panel, "TOP800")
    assert top800[20].sum() == 30 and top800[20, :30].all()
    top1800 = api.universe_mask(panel, "TOP1800")
    assert not top1800[:15].any() and top1800[20].sum() == 50
    assert api.universe_mask(panel, "ALL")[0].sum() == 57
    factor = np.random.default_rng(5).normal(size=panel.shape)
    res = groupbt.run(factor, panel, "ALL", n_groups=5)
    assert list(res.subs) == ["TOP1800", "TOP800"]
    alone = groupbt.run(factor, panel, "TOP800", n_groups=5)
    np.testing.assert_array_equal(res.subs["TOP800"].group_pnl, alone.group_pnl)
    with pytest.raises(ValueError):
        groupbt.run(factor, panel, "TOP500")


def test_neutralize_on_all_then_demean_in_pool():
    """行业中性化在全市场做：池外股票的因子值变了，池内结果跟着变；不做中性化时不变。"""
    panel = make_panel(seed=6, limits=False)
    factor = np.random.default_rng(8).normal(size=panel.shape)
    other = factor.copy()
    other[:, 50:] += 3.0  # 只改 TOP800 之外的股票
    a = groupbt.run(factor, panel, "TOP800", n_groups=5)
    b = groupbt.run(other, panel, "TOP800", n_groups=5)
    np.testing.assert_array_equal(a.group_pnl, b.group_pnl)
    a = groupbt.run(factor, panel, "TOP800", n_groups=5, group_neutralize="industry")
    b = groupbt.run(other, panel, "TOP800", n_groups=5, group_neutralize="industry")
    assert not np.allclose(np.nan_to_num(a.group_pnl), np.nan_to_num(b.group_pnl))
    for mode in ("cap", "cap+industry"):
        groupbt.run(factor, panel, "TOP800", n_groups=5, group_neutralize=mode)
    cap = np.log(panel["total_market_cap"])
    groupbt.run(factor, panel, "TOP800", n_groups=5, vector_neut=cap)


def test_ungroupable_days_raise():
    panel = make_panel(seed=9, limits=False)
    factor = np.random.default_rng(1).normal(size=panel.shape)
    factor[30] = np.round(factor[30])  # 取值太少
    factor[30] = np.where(np.arange(panel.shape[1]) % 2, 1.0, 0.0)
    with pytest.raises(groupbt.GroupingError) as err:
        groupbt.run(factor, panel, "ALL")
    assert err.value.rows == [30]


def test_outputs():
    panel = make_panel(seed=2)
    factor = np.random.default_rng(3).normal(size=panel.shape)
    res = groupbt.run(
        factor, panel, "TOP1800", n_groups=5, weights_portfolio=True, by="quarter"
    )
    table = res.table("long_short")
    assert table.columns == [
        "period",
        "return",
        "sharpe",
        "turnover",
        "maxdrawdown",
        "margin",
    ]
    assert table["period"][-1] == "ALL" and table["period"][0].endswith("Q1")
    assert res.table("top", by="year")["period"].to_list()[-1] == "ALL"
    assert set(res.exposure) == {"size", "reversal_20", "reversal_60", "momentum"}
    assert set(res.pnl) == {"long_short", "top", "weights"}
    curve = res.curves()
    np.testing.assert_allclose(curve["cumsum"][-1], np.nansum(res.pnl["long_short"]))
    assert res.chart().startswith("<svg") and "TOP800" in res.html()
    assert "long_short" in res.text() and "TOP800" in res.to_dict()["subs"]


def test_top1800_starts_in_2015():
    """涉及 TOP1800 的回测从 2015-01-01 起：之前即使有中证1000 的成分也不出结果。"""
    panel = make_panel(seed=4, limits=False)
    t = panel.shape[0]
    dates = np.array(
        [f"201412{i + 1:02d}" for i in range(30)]
        + [f"2015{1 + (i // 20):02d}{1 + i % 20:02d}" for i in range(t - 30)]
    )
    panel = FakePanel(panel._a, dates, panel.symbols)
    assert panel["member_csi1000"][20].any()
    mask = api.universe_mask(panel, "TOP1800")
    assert not mask[:30].any() and mask[30].sum() == 50
    assert api.universe_mask(panel, "TOP800")[5].sum() == 30
    factor = np.random.default_rng(5).normal(size=panel.shape)
    res = groupbt.run(factor, panel, "TOP1800", n_groups=5)
    first = lambda r: int(np.flatnonzero(~np.isnan(r.pnl["long_short"]))[0])  # noqa: E731
    assert first(res) == 31 and first(res.subs["TOP800"]) == 31  # 信号日 30，成交日 31
    whole = groupbt.run(factor, panel, "ALL", n_groups=5)
    assert first(whole) == 1 and first(whole.subs["TOP800"]) == 1
    assert first(whole.subs["TOP1800"]) == 31
    alone = groupbt.run(factor, panel, "TOP800", n_groups=5)
    assert first(alone) == 1
