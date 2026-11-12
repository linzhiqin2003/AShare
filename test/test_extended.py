"""Hand-computed rules, independent binding preparation and round-two fuzzing."""

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.api import _prepare
from panelbt.markets import ASHARE, FeeSchedule, MarketProfile, ashare_margin
from panelbt.metrics import calculate
from panelbt.models import EXTRA_COUNTERS, METRICS
from panelbt.reference import run as reference

ROOT = Path(__file__).resolve().parents[1]
ZERO = bt.Costs(
    commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
)


def example(prices, targets, *, costs=ZERO, kind="shares", margin=None, **kwargs):
    p = np.asarray(prices, float)
    market = bt.Market(
        p,
        kwargs.pop("close", p),
        kwargs.pop("can_buy", np.ones_like(p, bool)),
        kwargs.pop("can_sell", np.ones_like(p, bool)),
        **kwargs.pop("market", {}),
    )
    cfg = bt.Config(
        initial_cash=1000,
        min_qty=0,
        qty_step=0,
        record_trades=True,
        margin=margin or bt.MarginModel(),
        **kwargs,
    )
    return bt.run(market, bt.Targets.from_dense(targets, kind=kind), costs, cfg)


def test_preparation_content_and_sparse_conversion():
    market = bt.Market(
        [[10, 20], [11, 21]],
        [[10, 20], [11, 21]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 1]],
        delist=[[0, 0], [0, 1]],
    )
    target = bt.Targets.from_dense([[np.nan, np.nan], [3, -2]], kind="shares")
    costs = bt.Costs(
        tax_sell=[0.1, 0.2],
        fee_rate_buy=0.03,
        fee_per_share_sell=0.04,
        slippage=[[0.01, 0.02], [0.03, 0.04]],
    )
    cfg = bt.Config(
        min_qty=[100, 200],
        qty_step=[100, 1],
        cash_flow=[5, -7],
        margin=bt.MarginModel(allow_short=True, borrow_rate=[0.001, 0.002]),
    )
    f, m, p, v, s, flags, first = _prepare(market, [target], costs, cfg)
    np.testing.assert_array_equal(p[0][0], [1])
    np.testing.assert_array_equal(p[0][1], [0, 2])
    np.testing.assert_array_equal(p[0][2], [0, 1])
    np.testing.assert_array_equal(p[0][3], [3, -2])
    assert p[0][4]
    np.testing.assert_array_equal(v[0], [0.1, 0.2])
    np.testing.assert_array_equal(v[3], [0.03, 0.03])
    np.testing.assert_array_equal(v[4], [0.04, 0.04])
    np.testing.assert_array_equal(v[6], [100, 200])
    np.testing.assert_array_equal(v[7], [100, 1])
    np.testing.assert_array_equal(v[10], [5, -7])
    np.testing.assert_array_equal(first, [2, 1])
    np.testing.assert_array_equal(f[5], [[0.01, 0.02], [0.03, 0.04]])
    np.testing.assert_array_equal(f[7], [[0.001], [0.002]])
    assert flags[2] and not flags[-1] and s[5] == 1e7
    assert m[2].shape == (2, 2)


def test_legacy_bitwise_fixture():
    from bench.review_legacy import capture, digest

    from .generators import case

    with np.load(
        ROOT / "test/fixtures/legacy_sha256.npz", allow_pickle=False
    ) as fixture:
        # 夹具只存 SHA-256 的前 8 字节：足够拦住回归，体积是完整摘要的四分之一。
        expected = fixture["corrected"]
        assert expected.shape == (20000, 8)
        for seed in range(20000):
            actual = np.frombuffer(digest(capture(bt.run(*case(seed)))), dtype=np.uint8)
            np.testing.assert_array_equal(
                actual[:8], expected[seed], err_msg=f"seed={seed}"
            )


def test_shares_bypass_lots_and_trade_band_exceptions():
    p = np.full((3, 1), 10.0)
    m = bt.Market(p, p, p > 0, p > 0)
    cfg = bt.Config(initial_cash=1000, min_qty=100, qty_step=100, record_trades=True)
    r = bt.run(m, bt.Targets.from_dense([[1.25], [2.5], [0]], kind="shares"), ZERO, cfg)
    np.testing.assert_array_equal(r.positions[:, 0], [1.25, 2.5, 0])
    r = example(
        [[10], [10], [10], [10]],
        [[0.5], [0.51], [0], [0.01]],
        kind="weights",
        trade_band=0.02,
    )
    np.testing.assert_array_equal(r.positions[:, 0], [50, 50, 0, 1])
    assert r.coverage["trade_band"] == 1


def test_commission_cap_directional_share_fees():
    c = replace(
        ZERO,
        commission_per_share=0.1,
        commission_min=5,
        commission_max_rate=0.01,
        fee_per_share_sell=0.2,
        tax_buy=0.03,
        tax_sell=0.04,
    )
    r = example([[2], [2]], [[10], [0]], costs=c)
    np.testing.assert_array_equal(r.commission, [0.2, 0.2])
    np.testing.assert_array_equal(r.tax, [0.6, 0.8])
    np.testing.assert_array_equal(r.other_fee, [0, 2])
    assert r.cash[-1] == pytest.approx(996.2)


def test_impact_cash_and_panel_slippage():
    r = example(
        [[10], [10]],
        [[100], [0]],
        costs=replace(
            ZERO, impact_k=np.full((2, 1), 0.5), slippage=np.array([[0.01], [0.02]])
        ),
        market={"volume": np.full((2, 1), 1000.0)},
    )
    buy, sell = r.trades
    assert buy.price == pytest.approx(10 * 1.01 * (1 + 0.5 * np.sqrt(buy.qty / 1000)))
    assert sell.price == pytest.approx(
        10 * 0.98 * (1 - 0.5 * np.sqrt(-sell.qty / 1000))
    )
    assert np.all(r.cash >= 0) and r.impact_cost.sum() > 0
    with pytest.raises(ValueError, match="volume"):
        example([[10]], [[1]], costs=replace(ZERO, impact_k=np.ones((1, 1))))


def test_dividend_flow_interest_and_contribution():
    r = example(
        [[10], [10], [10]],
        [[10], [np.nan], [0]],
        market={"cash_div": [[0], [2], [0]]},
        dividend_tax_rate=0.25,
        cash_flow=[100, -50, -2000],
        cash_rate=0.01,
    )
    np.testing.assert_array_equal(r.dividend_gross, [0, 20, 0])
    np.testing.assert_array_equal(r.dividend_tax, [0, 5, 0])
    assert r.cash_interest[0] == 10.0
    assert (
        r.cash_flow[0] == 100
        and r.cash_flow[1] == -50
        and r.withdrawal_shortfall[2] > 0
    )
    np.testing.assert_allclose(
        r.returns, r.equity / (np.r_[1000, r.equity[:-1]] + r.cash_flow) - 1
    )
    np.testing.assert_allclose(r.unit_nav, np.cumprod(1 + r.returns))
    assert r.contribution.sum() + r.cash_interest.sum() == pytest.approx(
        r.equity[-1] - 1000 - r.cash_flow.sum()
    )


def test_short_permission_dividend_borrow_cover_and_delist():
    mm = bt.MarginModel(allow_short=True, borrow_rate=0.01, short_initial=1)
    r = example(
        [[10], [10], [10]],
        [[-20], [-20], [0]],
        margin=mm,
        market={"can_short": [[0], [1], [1]], "cash_div": [[0], [0], [2]]},
    )
    np.testing.assert_array_equal(r.positions[:, 0], [0, -20, 0])
    assert r.dividend_gross[-1] == -40 and r.dividend_tax.sum() == 0
    assert r.borrow_cost[1] == 2 and r.coverage["cover_short"] == 1
    r = example(
        [[10], [999]],
        [[-20], [np.nan]],
        margin=mm,
        market={"delist": [[0], [1]]},
        delist_recovery=0.3,
    )
    assert r.delist_events[0].proceeds == -60 and r.positions[-1, 0] == 0


def test_margin_scaling_financing_and_account_identity():
    mm = bt.MarginModel(
        allow_short=True,
        long_initial=0.5,
        short_initial=0.5,
        allow_financing=True,
        short_proceeds_usable=True,
        financing_rate=0.01,
    )
    r = example([[10, 10]], [[200, -200]], margin=mm)
    np.testing.assert_allclose(r.positions, [[100, -100]], atol=1e-12)
    assert r.coverage["margin_scaled"] == 1
    np.testing.assert_allclose(r.equity, r.cash + r.long_value - r.short_value)
    assert r.margin_usage[0] <= 1
    r = example([[10]], [[150]], margin=mm)
    assert r.financing_interest[0] == 5 and r.cash[0] == -505


def test_margin_call_liquidate_record_and_bankruptcy():
    mm = bt.MarginModel(allow_short=True, short_initial=1, short_maintenance=0.8)
    r = example(
        [[10], [16], [16], [16]], [[-80], [np.nan], [np.nan], [np.nan]], margin=mm
    )
    assert (
        r.margin_call[1] and r.liquidation_events and r.liquidation_events[0].day == 2
    )
    assert r.positions[2, 0] > r.positions[1, 0] and r.margin_usage[2] <= 1 + 1e-14
    record = example(
        [[10], [16], [16]],
        [[-80], [np.nan], [np.nan]],
        margin=replace(mm, margin_call="record"),
    )
    assert not record.liquidation_events and record.positions[-1, 0] == -80
    broke = example(
        [[10], [30], [30], [1]], [[-80], [np.nan], [np.nan], [1]], margin=mm
    )
    assert broke.bankruptcy_events == [1] and broke.positions[-1, 0] == 0


def test_long_short_mirror_and_flat_hedge():
    mm = bt.MarginModel(
        allow_short=True,
        allow_financing=True,
        long_initial=0,
        short_initial=0,
        short_proceeds_usable=True,
    )
    a = example([[10], [12], [9]], [[20], [np.nan], [0]], margin=mm)
    b = example([[10], [12], [9]], [[-20], [np.nan], [0]], margin=mm)
    np.testing.assert_allclose(a.equity - 1000, 1000 - b.equity, atol=1e-12)
    r = example(
        [[10, 10], [10, 10]],
        [[20, -20], [0, 0]],
        margin=mm,
        costs=replace(ZERO, commission_rate=0.01),
    )
    np.testing.assert_allclose(r.equity, 1000 - np.cumsum(r.commission))


def test_metrics_manual():
    r = np.array([0.1, -0.1, 0.0])
    x = calculate(
        r,
        periods_per_year=3,
        benchmark=[0, -0.05, 0.05],
        turnover=[0.1, 0.2, 0.3],
        cost_ratio=[0.01, 0.02, 0.03],
    )
    assert x["cumulative_return"] == pytest.approx(-0.01)
    assert x["annual_return"] == pytest.approx(-0.01)
    assert x["annual_volatility"] == pytest.approx(0.1 * np.sqrt(3))
    assert x["sharpe"] == 0 and x["sortino"] == 0
    assert x["max_drawdown"] == pytest.approx(0.1) and x["drawdown_duration"] == 2
    assert x["calmar"] == pytest.approx(-0.1)
    assert x["annual_turnover"] == pytest.approx(0.6) and x[
        "annual_cost_ratio"
    ] == pytest.approx(0.06)
    assert x["beta"] == pytest.approx(1) and x["alpha"] == pytest.approx(0)
    assert x["excess_annual_return"] == pytest.approx(0.99 / 0.9975 - 1)
    assert x["information_ratio"] == pytest.approx(0)
    assert x["tracking_error"] == pytest.approx(0.15)
    assert x["excess_max_drawdown"] == pytest.approx(1 - 0.9 / 0.95 / 1.05)


def test_result_roundtrip(tmp_path):
    r = example([[10], [11]], [[10], [0]])
    path = tmp_path / "result.npz"
    r.to_npz(path)
    a = bt.Result.from_npz(path)
    for key in (*METRICS, "contribution", "positions"):
        np.testing.assert_array_equal(getattr(a, key), getattr(r, key))
    assert (
        a.trades == r.trades
        and a.diagnostics == r.diagnostics
        and a.coverage == r.coverage
    )


def test_hk_us_style_profiles():
    dates = np.array(["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    p = np.array([[10, 20], [11, 19], [12, 18]])
    for profile, lots, overrides in [
        (
            MarketProfile(
                "HK-style example",
                "HKD",
                252,
                FeeSchedule((("2025-01-01", {"tax_buy": 0.001, "tax_sell": 0.001}),)),
            ),
            [100, 200],
            {"commission_rate": 0.001, "commission_min": 0},
        ),
        (
            MarketProfile(
                "US-style example",
                "USD",
                252,
                FeeSchedule((("2025-01-01", {"fee_per_share_sell": 0.001}),)),
            ),
            [0, 0],
            {
                "commission_rate": 0,
                "commission_per_share": 0.005,
                "commission_min": 1,
                "commission_max_rate": 0.01,
            },
        ),
    ]:
        m = profile.market(p, p)
        c = profile.costs(dates, slippage=0, **overrides)
        g = profile.config(lot_sizes=lots, initial_cash=10000, record_trades=True)
        target = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
        a, b = bt.run(m, target, c, g), reference(m, target, c, g)
        assert a.trades
        for key in METRICS:
            np.testing.assert_allclose(
                getattr(a, key), getattr(b, key), atol=1e-9, rtol=1e-9
            )
    np.testing.assert_array_equal(
        ASHARE.costs(["2023-08-25", "2023-08-28"]).tax_sell, [0.001, 0.0005]
    )
    with pytest.raises(TypeError):
        ashare_margin()


def extended_case(seed):
    from .generators import case

    m, target, c, g = case(seed)
    rng = np.random.default_rng(10000 + seed)
    t, n = m.shape
    m = replace(m, can_short=rng.random((t, n)) > 0.25)
    w = target.to_dense(t, n)
    w *= rng.choice([-1, 1], w.shape) * rng.choice([1, 2, 4])
    kind = "shares" if seed % 4 == 0 else "weights"
    if kind == "shares":
        w *= g.initial_cash / 20
    target = bt.Targets.from_dense(w, kind=kind)
    margin = bt.MarginModel(
        allow_short=True,
        long_initial=rng.choice([0.3, 0.5, 1], t),
        short_initial=rng.choice([0.5, 1, 1.5], t),
        long_maintenance=0.3,
        short_maintenance=0.8,
        allow_financing=seed % 2 == 0,
        short_proceeds_usable=seed % 3 == 0,
        financing_rate=0.001,
        borrow_rate=rng.choice([0, 0.001, 0.01], (t, n)),
        margin_call="record" if seed % 5 == 0 else "liquidate",
    )
    g = replace(
        g,
        margin=margin,
        trade_band=0.02,
        dividend_tax_rate=0.2,
        cash_flow=rng.choice([0, 0, g.initial_cash * 0.03, -g.initial_cash * 0.05], t),
        cash_rate=0.0001,
    )
    c = replace(
        c,
        commission_per_share=0.001,
        commission_max_rate=0.003,
        fee_per_share_sell=0.002,
        slippage=rng.uniform(0, 0.002, (t, n)),
        impact_k=np.full((t, n), 0.01) if m.volume is not None else None,
    )
    return m, target, c, g


def test_1000_extended_random_cases(tmp_path):
    total = dict.fromkeys(EXTRA_COUNTERS, 0)
    post_checks = {
        "metrics_valid": 0,
        "metrics_invalid_rejected": 0,
        "result_roundtrip": 0,
        "contribution_invariant": 0,
    }
    for seed in range(1000):
        m, t, c, g = extended_case(seed)
        a, b = bt.run(m, t, c, g), reference(m, t, c, g)
        for key in (*METRICS, "contribution", "positions"):
            np.testing.assert_allclose(
                getattr(a, key),
                getattr(b, key),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"seed={seed} {key}",
            )
        assert a.diagnostics == b.diagnostics, seed
        assert a.coverage == b.coverage, (seed, a.coverage, b.coverage)
        assert [(x.day, x.col) for x in a.trades] == [
            (x.day, x.col) for x in b.trades
        ], seed
        assert a.bankruptcy_events == b.bankruptcy_events
        np.testing.assert_allclose(
            a.equity, a.cash + a.long_value - a.short_value, rtol=1e-12, atol=1e-9
        )
        np.testing.assert_allclose(
            a.contribution.sum() + a.cash_interest.sum() - a.financing_interest.sum(),
            a.equity[-1] - g.initial_cash - a.cash_flow.sum(),
            rtol=1e-9,
            atol=1e-8,
        )
        post_checks["contribution_invariant"] += 1
        archive = io.BytesIO()
        a.to_npz(archive)
        archive.seek(0)
        loaded = bt.Result.from_npz(archive)
        for key in (*METRICS, "contribution", "positions"):
            np.testing.assert_array_equal(getattr(loaded, key), getattr(a, key))
        assert loaded.trades == a.trades
        post_checks["result_roundtrip"] += 1
        if np.all(a.returns >= -1):
            left, right = calculate(a.returns), calculate(b.returns)
            np.testing.assert_allclose(
                list(left.values()), list(right.values()), rtol=1e-9, atol=1e-9
            )
            post_checks["metrics_valid"] += 1
        else:
            with pytest.raises(ValueError, match="returns"):
                calculate(a.returns)
            post_checks["metrics_invalid_rejected"] += 1
        for k, v in a.coverage.items():
            total[k] += v
    assert all(v > 0 for v in total.values()), total
    (tmp_path / "coverage_extended.json").write_text(
        json.dumps({**total, **post_checks}, indent=2) + "\n"
    )
    print("Round-two coverage:", total)


def test_margin_prefix_constraints_and_costed_liquidation():
    rng = np.random.default_rng(6201)
    for usable in (False, True):
        margin = bt.MarginModel(
            allow_short=True,
            long_initial=0.7,
            short_initial=1.2,
            short_proceeds_usable=usable,
        )
        for _ in range(30):
            targets = rng.uniform(-150, 150, (4, 5))
            result = example(
                np.full((4, 5), 10.0),
                targets,
                margin=margin,
                costs=replace(ZERO, commission_rate=0.01),
            )
            cash, shares = 1000.0, np.zeros(5)
            for trade in result.trades:
                old = shares[trade.col]
                cash -= (
                    trade.qty * trade.price
                    + trade.commission
                    + trade.tax
                    + trade.other_fee
                )
                shares[trade.col] += trade.qty
                long, short = (
                    np.maximum(shares, 0).sum() * 10,
                    -np.minimum(shares, 0).sum() * 10,
                )
                if abs(shares[trade.col]) > abs(old):
                    assert 0.7 * long + 1.2 * short <= cash + long - short + 1e-10
                assert cash - (0 if usable else short) >= -1e-10
    result = example(
        [[10], [16], [16], [16]],
        [[-80], [np.nan], [np.nan], [np.nan]],
        margin=bt.MarginModel(allow_short=True, short_initial=1, short_maintenance=0.8),
        costs=replace(ZERO, commission_rate=0.01),
    )
    assert result.margin_usage[2] <= 1 + 1e-14
    assert result.liquidation_events[0].qty > 0


def test_extended_batch_bitwise_and_serialization(tmp_path):
    for seed in range(10):
        m, target, c, g = extended_case(seed)
        plans = [target, bt.Targets([], [0], [], [], kind="shares")]
        serial = bt.run_batch(m, plans, c, g, threads=1)
        parallel = bt.run_batch(m, plans, c, g, threads=0)
        for key in (*METRICS, "contribution", "positions"):
            np.testing.assert_array_equal(getattr(serial, key), getattr(parallel, key))
        assert serial.liquidation_events == parallel.liquidation_events
        path = tmp_path / "batch.npz"
        serial.to_npz(path)
        restored = bt.Result.from_npz(path)
        assert restored.trades == serial.trades
        assert restored.bankruptcy_events == serial.bankruptcy_events
        for key in EXTRA_COUNTERS:
            np.testing.assert_array_equal(restored.coverage[key], serial.coverage[key])


def test_pandas_is_optional_and_frame_matches():
    import subprocess
    import sys

    script = 'import sys, panelbt; assert "pandas" not in sys.modules'
    subprocess.run([sys.executable, "-c", script], check=True)
    result = example([[10], [11]], [[10], [0]])
    pytest.importorskip("pandas")
    frame = result.to_frame(index=["a", "b"])
    assert frame.index.tolist() == ["a", "b"]
    np.testing.assert_array_equal(frame["nav"], result.nav)


def test_per_share_commission_without_cap():
    result = example(
        [[10], [10]], [[10], [0]], costs=replace(ZERO, commission_per_share=0.03)
    )
    np.testing.assert_allclose(result.commission, [0.3, 0.3], rtol=1e-14)


def test_metrics_nonzero_excess_mean():
    result = calculate([0.2, -0.1], periods_per_year=2)
    assert result["annual_return"] == pytest.approx(0.08)
    assert result["annual_volatility"] == pytest.approx(0.3)
    assert result["sharpe"] == pytest.approx(1 / 3)
    assert result["sortino"] == pytest.approx(1)
    assert result["calmar"] == pytest.approx(0.8)


def test_optional_commission_floor():
    result = example(
        [[10]],
        [[1]],
        costs=replace(ZERO, commission_min=None, commission_per_share=0.01),
    )
    assert result.commission[0] == 0.01
