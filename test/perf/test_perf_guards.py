"""性能守卫：只比同一测试内的相对比例和复杂度，不比绝对耗时，换机器、有负载都成立。

每条阈值旁边写了录入时（2026-11-14，本机负载很高）实测的比例，阈值留了几倍余量。
来源：2026-11-14 测试体系补全（性能回归测试，pytest 里的比例与复杂度守卫）。
"""

import numpy as np
import panelbt as bt
import panelops as po
import pytest

from ashare.factor import Evaluator, groupbt
from ashare.platform.language import parse
from bench.compare.common import inputs
from bench.platform.synthetic import panel

from .timing import describe, paired_ratio


@pytest.fixture(scope="module")
def source():
    return panel(n=300, full=True)


@pytest.fixture(scope="module")
def alpha(source):
    expression = parse("-ts_std(returns, 20)", factors={})[0].expression
    return Evaluator().evaluate(expression, source)


@pytest.fixture
def single_thread():
    po.set_num_threads(1)
    yield
    po.set_num_threads(0)


def _variants(configurations, alpha, source):
    def shared():
        groupbt.run_many(
            [alpha] * len(configurations), source, configurations, workers=1
        )

    def separate():
        for c in configurations:
            groupbt.run(alpha, source, **c)

    return paired_ratio(separate, shared, rounds=5)


def test_run_many_shares_pools_and_exposures(alpha, source):
    """8 个不同中性化 × 调仓周期的变体：共享窗口市值、股票池和风险均值，单线程也要省。

    实测 run_many / 逐个 run ≈ 0.58。"""
    configurations = [
        dict(universe="ALL", group_neutralize=g, rebalance=r, sub_universes=False)
        for g in (None, "industry", "cap", "cap+industry")
        for r in (1, 5)
    ]
    ratio, separate, shared = _variants(configurations, alpha, source)
    assert ratio <= 0.85, describe(("逐个", "共享"), ratio, separate, shared)


def test_run_many_batches_ledger_scan(alpha, source):
    """同一信号只扫调仓周期和权重组合：合进一个 Rust 批量调用。实测比例 ≈ 0.21。"""
    configurations = [
        dict(universe="ALL", rebalance=r, weights_portfolio=w, sub_universes=False)
        for r in (1, 2, 3, 5)
        for w in (False, True)
    ]
    ratio, separate, shared = _variants(configurations, alpha, source)
    assert ratio <= 0.5, describe(("逐个", "批量"), ratio, separate, shared)


# 线性的规模守卫只设上限：数据翻倍时线性约 2 倍、平方 4 倍，上限 3 倍。比 2 小只说明
# 固定开销占比高，不是回归，不设下限（负载下下限会误报）。


@pytest.mark.parametrize("op", ["std", "mean", "sum", "min", "max", "rank"])
def test_rolling_operator_linear_in_rows(single_thread, op):
    """行数翻倍，耗时约翻倍（实测 1.93–2.14），不能变成平方。"""
    x = np.random.default_rng(1).normal(size=(4800, 1500))
    half = np.ascontiguousarray(x[:2400])
    fn = getattr(po, "ts_" + op)
    ratio, small, large = paired_ratio(lambda: fn(half, 20), lambda: fn(x, 20))
    assert ratio <= 3.0, f"ts_{op}: {describe(('T', '2T'), ratio, small, large)}"


@pytest.mark.parametrize("op", ["std", "mean", "sum", "min", "max"])
def test_rolling_cost_independent_of_window(single_thread, op):
    """滑动累加和单调队列：窗口从 20 到 250 耗时基本不变（实测 1.00–1.19），不能按窗口重扫。"""
    x = np.random.default_rng(2).normal(size=(2400, 1500))
    fn = getattr(po, "ts_" + op)
    ratio, short, long = paired_ratio(lambda: fn(x, 20), lambda: fn(x, 250))
    assert ratio <= 2.5, f"ts_{op}: {describe(('w20', 'w250'), ratio, short, long)}"


def test_weight_backtest_linear_in_rows():
    """权重模式记账按天线性（实测 1.97）。"""
    rng = np.random.default_rng(3)
    weights, returns = rng.random((4800, 800)), rng.normal(0, 0.01, (4800, 800))
    half = [np.ascontiguousarray(a[:2400]) for a in (weights, returns)]
    ratio, small, large = paired_ratio(
        lambda: bt.run_weights(*half), lambda: bt.run_weights(weights, returns)
    )
    assert ratio <= 3.0, describe(("T", "2T"), ratio, small, large)


def test_trade_backtest_linear_in_rows():
    """成交模式（现金、手数、费用）按天线性（实测 2.09）。"""
    a, b = inputs(1200, 1500, interval=1), inputs(2400, 1500, interval=1)
    ratio, small, large = paired_ratio(lambda: bt.run(*a), lambda: bt.run(*b))
    assert ratio <= 3.0, describe(("T", "2T"), ratio, small, large)


def test_group_backtest_linear_in_window(alpha, source):
    """分组回测窗口从 5 年到 10 年，耗时不超过 3 倍（线性约 2 倍，含固定开销会更低）。"""
    short = ("20150101", "20200101")
    long = ("20150101", "20250101")
    ratio, small, large = paired_ratio(
        lambda: groupbt.run(alpha, source, "ALL", sub_universes=False, window=short),
        lambda: groupbt.run(alpha, source, "ALL", sub_universes=False, window=long),
    )
    assert ratio <= 3.0, describe(("5y", "10y"), ratio, small, large)


def test_expression_cache_hit_much_faster_than_cold(tmp_path):
    """表达式求值缓存命中比冷算（求值 + 写缓存）快得多：实测约 25 倍，要求至少 4 倍。"""
    import shutil

    big = panel(n=1000, full=True)
    expression = parse(
        "rank(ts_corr(close, volume, 20)) - ts_rank(returns, 60)", factors={}
    )[0].expression
    warm = tmp_path / "warm"
    expected = Evaluator(warm).evaluate(expression, big)

    def cold():
        shutil.rmtree(tmp_path / "cold", ignore_errors=True)
        evaluator = Evaluator(tmp_path / "cold")
        evaluator.evaluate(expression, big)
        assert not evaluator.cache_hit

    def hit():
        evaluator = Evaluator(warm)
        result = evaluator.evaluate(expression, big)
        assert evaluator.cache_hit
        assert result.tobytes() == expected.tobytes()

    ratio, first, cached = paired_ratio(cold, hit, rounds=5, minimum=0)
    assert ratio <= 0.25, describe(("冷", "命中"), ratio, first, cached)


def test_benchmark_switch_reuses_results_without_rerun(tmp_path, monkeypatch):
    """切换基准只重算展示指标：不求值、不回测；同一基准第二次直接命中服务端缓存。"""
    from fastapi.testclient import TestClient

    from ashare.platform import analysis, benchmarks
    from ashare.platform.analysis import run_variant
    from ashare.platform.app import create_app
    from ashare.platform.config import DEFAULTS
    from bench.platform.synthetic import write_indexes, write_panel

    data = tmp_path / "data"
    small = write_panel(data)
    write_indexes(data, small, codes=["000001.SH", "000300.SH"])
    variant = parse("rank(close)", factors={})[0]
    computed = run_variant(variant.expression, small, DEFAULTS)
    calls = []
    real = benchmarks.Benchmarks.comparison

    def counting(self, store, runs, symbol=None):
        calls.append(symbol)
        return real(self, store, runs, symbol)

    monkeypatch.setattr(benchmarks.Benchmarks, "comparison", counting)
    app = create_app(tmp_path / "results", data_root=str(data))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        store = app.state.store
        job, runs = store.begin("tab", "alpha", [(variant, DEFAULTS)], {})
        store.save(runs[0]["id"], *computed, 0, "synthetic")
        store.finish(job, "complete")

        def forbidden(*args, **kwargs):
            raise AssertionError("切换基准触发了重新求值或回测")

        monkeypatch.setattr(groupbt, "run", forbidden)
        monkeypatch.setattr(groupbt, "run_many", forbidden)
        monkeypatch.setattr(analysis, "run_variant", forbidden)
        monkeypatch.setattr(Evaluator, "evaluate", forbidden)
        url = f"/api/jobs/{job}/benchmark"
        first = client.get(url, params={"symbol": "000001.SH"})
        second = client.get(url, params={"symbol": "000300.SH"})
        again = client.get(url, params={"symbol": "000001.SH"})
        assert first.status_code == second.status_code == again.status_code == 200
        assert first.headers["etag"] == again.headers["etag"] != second.headers["etag"]
        assert first.json()["symbol"] == "000001.SH"
        assert second.json()["symbol"] == "000300.SH"
        assert calls == ["000001.SH", "000300.SH"]
