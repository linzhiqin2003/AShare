"""研究主链路：数据层读取 → 因子表达式求值 → 分组回测 → 报告产物。

每一步都和独立算法对照：因子值对 NumPy 手算，分组多空收益对「持仓 × 数据层未来收益」，
绩效表对按定义手算的指标，报告产物对内存里的结果；另有截断未来数据不改变结果的前视检查。
"""

import json
import math

import numpy as np
import pytest

import ashare.data
from ashare.factor import Evaluator, groupbt
from ashare.factor.groupbt.api import _adjusted, _forward_return, close_masks
from ashare.platform.language import parse

SOURCE = "-ts_std(returns, 10)"
WINDOW = 10


@pytest.fixture(scope="module")
def expression():
    return parse(SOURCE)[0].expression


@pytest.fixture(scope="module")
def panel(datadir, expression):
    data = ashare.data.open(datadir.root)
    fields = sorted(set(groupbt.FIELDS) | set(expression.fields) | {"previous_close"})
    return data.panel(fields)


@pytest.fixture(scope="module")
def alpha(panel, expression):
    values = Evaluator().evaluate(expression, panel)
    values.flags.writeable = False
    return values


def reference_alpha(panel):
    """独立手算（docs/PLATFORM.md 口径）：returns = close / previous_close − 1，上市首日为空；
    取窗口满 10 天的样本标准差再取负。前收已是除权参考价，所以除权日收益连续。"""
    price = np.asarray(panel["close"], dtype=np.float64)
    returns = price / np.asarray(panel["previous_close"]) - 1
    returns[np.asarray(panel["listing_age"]) == 1] = np.nan
    out = np.full_like(price, np.nan)
    for t in range(WINDOW - 1, len(price)):
        block = returns[t - WINDOW + 1 : t + 1]
        full = np.isfinite(block).all(axis=0)
        out[t, full] = -np.std(block[:, full], axis=0, ddof=1)
    return out


def dense_weights(holdings, shape):
    """稀疏调仓快照展开成逐日权重：非调仓日沿用上一次（内核不随价格漂移）。"""
    weights = np.zeros(shape)
    days = np.asarray(holdings["days"])
    offsets = np.asarray(holdings["offsets"])
    snapshot = np.zeros(shape[1])
    k = 0
    for t in range(shape[0]):
        if k < len(days) and days[k] == t:
            snapshot = np.zeros(shape[1])
            lo, hi = offsets[k : k + 2]
            snapshot[holdings["columns"][lo:hi]] = holdings["weights"][lo:hi]
            k += 1
        weights[t] = snapshot
    assert k == len(days)
    return weights


def test_factor_matches_numpy_reference(panel, alpha):
    expected = reference_alpha(panel)
    assert alpha.shape == panel.shape
    assert np.array_equal(np.isnan(alpha), np.isnan(expected))
    finite = np.isfinite(expected)
    assert finite.sum() > 0.8 * expected.size
    np.testing.assert_allclose(alpha[finite], expected[finite], rtol=1e-9, atol=1e-15)


CASES = [
    dict(universe="TOP1800"),
    dict(universe="TOP1800", trade_at="close"),
    dict(universe="ALL", rebalance=5, group_neutralize="industry"),
    dict(universe="TOP800", n_groups=5, group_neutralize="cap+industry"),
]


@pytest.mark.parametrize("window", [groupbt.IS, groupbt.VALIDATION], ids=["is", "val"])
@pytest.mark.parametrize(
    "settings", CASES, ids=lambda c: "-".join(map(str, c.values()))
)
def test_long_short_pnl_equals_holdings_times_forward_returns(
    panel, alpha, settings, window
):
    """分组多空日收益 = 当日实际持仓 · 数据层的下一成交时点收益（与内核独立的重算）。"""
    result = groupbt.run(
        alpha,
        panel,
        record_holdings=True,
        sub_universes=False,
        window=window,
        **settings,
    )
    trade_at = settings.get("trade_at", "open")
    rows = len(result.dates)
    cut = _Rows(panel, rows)
    weights = dense_weights(result.holdings, (rows, panel.shape[1]))
    forward = _forward_return(_adjusted(cut, trade_at))
    estimate = (weights * np.nan_to_num(forward)).sum(axis=1)
    pnl = result.pnl["long_short"]
    valid = np.isfinite(pnl)
    assert valid.sum() > 30
    np.testing.assert_allclose(pnl[valid], estimate[valid], rtol=0, atol=1e-12)
    # 窗口起点之前只作预热：没有收益。
    assert not np.isfinite(pnl[np.asarray(result.dates) < window[0]]).any()
    # 每次调仓多空两侧各 0.5（有持仓时）。
    for k in range(len(result.holdings["days"])):
        lo, hi = result.holdings["offsets"][k : k + 2]
        w = result.holdings["weights"][lo:hi]
        if len(w):
            assert w[w > 0].sum() == pytest.approx(0.5, abs=1e-9)
            assert w[w < 0].sum() == pytest.approx(-0.5, abs=1e-9)

    # 换仓必须可成交：加仓的股票当天可买，减仓的当天可卖（数据层掩码 → 回测行为）。
    if trade_at == "open":
        buy = np.asarray(cut["can_buy_open"])
        sell = np.asarray(cut["can_sell_open"])
    else:
        buy, sell = close_masks(cut)
    delta = np.diff(weights, axis=0)
    assert not ((delta > 1e-12) & ~buy[1:]).any()
    assert not ((delta < -1e-12) & ~sell[1:]).any()
    # 停牌期间持仓冻结。
    suspended = np.asarray(cut["is_suspended"])[1:] | ~np.asarray(cut["has_quote"])[1:]
    assert not (np.abs(delta) > 0)[suspended].any()


class _Rows:
    def __init__(self, panel, rows):
        self.panel, self.rows = panel, rows
        self.dates = np.asarray(panel.dates)[:rows]
        self.symbols = panel.symbols

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def __getitem__(self, name):
        return np.asarray(self.panel[name])[: self.rows]


class _Poisoned(_Rows):
    """窗口止日及以后的每个字段都换成垃圾：结果不变说明没有读未来。"""

    def __init__(self, panel, stop):
        super().__init__(panel, len(panel.dates))
        self.stop = int(np.searchsorted(self.dates, stop))

    def __getitem__(self, name):
        value = np.array(self.panel[name], copy=True)
        if value.dtype == bool:
            value[self.stop :] = ~value[self.stop :]
        else:
            value[self.stop :] = 7.0
        value.flags.writeable = False
        return value


@pytest.mark.parametrize("window", [groupbt.IS, groupbt.VALIDATION], ids=["is", "val"])
def test_backtest_does_not_read_beyond_window(panel, alpha, window):
    clean = groupbt.run(alpha, panel, "TOP1800", record_holdings=True, window=window)
    poisoned_alpha = np.array(alpha, copy=True)
    stop = int(np.searchsorted(np.asarray(panel.dates), window[1]))
    poisoned_alpha[stop:] = -poisoned_alpha[stop:]
    dirty = groupbt.run(
        poisoned_alpha,
        _Poisoned(panel, window[1]),
        "TOP1800",
        record_holdings=True,
        window=window,
    )
    assert clean.group_pnl.tobytes() == dirty.group_pnl.tobytes()
    for name in clean.pnl:
        assert clean.pnl[name].tobytes() == dirty.pnl[name].tobytes()
    assert clean.subs["TOP800"].pnl["long_short"].tobytes() == (
        dirty.subs["TOP800"].pnl["long_short"].tobytes()
    )


def independent_performance(pnl, turnover, periods=240):
    keep = np.isfinite(pnl)
    p, t = pnl[keep], np.nan_to_num(turnover[keep])
    mean, sd = p.mean(), p.std(ddof=1)
    cumulative = np.cumsum(p)
    peak = np.maximum(np.maximum.accumulate(cumulative), 0)
    return dict(
        ret=mean * periods,
        sharpe=mean / sd * math.sqrt(periods),
        turnover=t.mean(),
        maxdrawdown=(peak - cumulative).max(),
    )


def test_report_artifacts_match_backtest(panel, alpha, tmp_path):
    result = groupbt.run(alpha, panel, "TOP1800", weights_portfolio=True, decay=3)
    table = result.table().to_dicts()
    overall = table[-1]
    assert overall["period"] == "ALL"
    expected = independent_performance(
        result.pnl["long_short"], result.turnover["long_short"]
    )
    assert overall["return"] == pytest.approx(expected["ret"], rel=1e-12)
    assert overall["sharpe"] == pytest.approx(expected["sharpe"], rel=1e-12)
    assert overall["turnover"] == pytest.approx(expected["turnover"], rel=1e-12)
    assert overall["maxdrawdown"] == pytest.approx(expected["maxdrawdown"], rel=1e-12)
    # 分年行覆盖的日期并起来正好是全部有收益的日子。
    years = {row["period"] for row in table[:-1]}
    dates = np.asarray(result.dates)[np.isfinite(result.pnl["long_short"])]
    assert years == {d[:4] for d in dates}

    # 文本、JSON、HTML 三种产物，数值都来自同一结果。
    html = tmp_path / "groupbt.html"
    html.write_text(result.html(), encoding="utf-8")
    content = html.read_text(encoding="utf-8")
    assert content.count("<svg") == 1 + len(result.subs)
    assert f"{overall['sharpe']:.2f}" in content
    record = json.loads(json.dumps(result.to_dict()))
    assert record["long_short"][-1]["sharpe"] == overall["sharpe"]
    assert set(record["subs"]) == {"TOP800"}
    assert record["exposure_mean"] == pytest.approx(result.exposure_mean())
    text = result.text()
    assert text.splitlines()[0].startswith("TOP1800")
    assert "TOP800" in text and "risk exposure" in text
    # 权重组合打开时图里多一条线，表里的 weights 序列有效。
    assert np.isfinite(result.pnl["weights"]).sum() > 30
    curves = result.curves()
    assert curves["cumsum"][-1] == pytest.approx(
        np.nansum(result.pnl["long_short"]), rel=1e-12
    )
