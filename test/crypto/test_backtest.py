"""来源：C3/4.17 的账户、时间轴、事件与防前视合同；全部合成数据。"""

import copy

import numpy as np
import polars as pl
import pytest

from ashare.data.crypto.dataset import write_panel
from ashare.data.crypto.events import normalize_event
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import DAY, HOUR, millis


def sample(tmp_path, *, kind="spot", hours=72, events=(), change=None):
    start = millis("2024-01-01")
    profiles, data = [], {}
    for j, coin in enumerate(("BTC", "ETH")):
        name = f"{kind}:{coin}USDT"
        p = dict(
            instrument=name,
            symbol=coin + "USDT",
            market=kind,
            base_asset=coin,
            coin=coin,
            coin_units=1,
            quote_asset="USDT",
            margin_asset="USDT",
            underlying_type="COIN",
            listed_at=start,
            contract_type="SPOT" if kind == "spot" else "PERPETUAL",
            source_url="https://example.test/synthetic",
            multiplier=0.01 if kind == "usdm" else 1.0,
            rules=[
                dict(
                    known_at=start,
                    effective_at=start,
                    qty_step=0.01,
                    min_qty=0.01,
                    min_notional=5.0,
                    source="snapshot",
                    source_url="https://example.test/synthetic",
                )
            ],
        )
        profiles.append(p)
        bars = [
            dict(
                time=start + i * HOUR,
                open=100.0 + j,
                high=102.0 + j,
                low=99.0 + j,
                close=101.0 + j,
                volume=10000.0,
                quote_volume=1000000.0 + j,
                taker_buy_quote_volume=500000.0,
            )
            for i in range(hours + 1)
        ]
        data[name] = dict(
            klines=bars,
            markPriceKlines=copy.deepcopy(bars),
            indexPriceKlines=copy.deepcopy(bars),
            fundingRate=[
                dict(time=start + i * HOUR, interval=8, rate=0.0001)
                for i in range(0, hours + 1, 8)
            ],
            funding_prices=[
                dict(fundingTime=start + i * HOUR, fundingRate=".0001", markPrice="100")
                for i in range(0, hours + 1, 8)
            ],
        )
    args = dict(
        profiles=profiles,
        data=data,
        events=list(events),
        start=start,
        end=start + hours * HOUR,
    )
    if change:
        change(args)
    frame, notes = build_frame(**args, allow_missing=True)
    return write_panel(
        tmp_path, frame, profiles=profiles, data=data, events=list(events), notes=notes
    )


def adapter():
    from ashare.backtest.crypto import crypto_market, daily_targets, top_n

    return crypto_market, daily_targets, top_n


def test_daily_targets_one_shift_and_blocked_not_carried(tmp_path):
    market, daily, _ = adapter()

    def change(a):
        for bars in a["data"].values():
            bars["klines"][24]["volume"] = 0

    ds = sample(tmp_path, change=change)
    m = market(ds, ["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-01-04")
    plan = daily((["2024-01-01", "2024-01-02"], [[0.5, 0.4], [0.5, 0.4]]))
    result = m.run(plan, record_trades=True)
    assert result.positions[:48].sum() == 0
    assert result.positions[48].sum() > 0
    assert {tr.day for tr in result.trades} == {48}
    np.testing.assert_array_equal(plan.align(m.times, m.symbols)[24], [0.5, 0.4])
    assert np.isnan(m.signal_prices[0]).all()
    assert m.signal_prices[1, 0] == 101
    daily_frame = ds.daily(trade_hour=1)
    assert (daily_frame["available_at"] - daily_frame["day"]).to_list() == [
        DAY
    ] * daily_frame.height


def test_daily_trade_hour_and_final_signal_no_extra_shift():
    _, daily, _ = adapter()
    plan = daily((["2024-01-01", "2024-01-02"], [[1.0], [0.0]]), trade_hour=1)
    times = np.arange(millis("2024-01-01"), millis("2024-01-04"), HOUR)
    w = plan.align(times, ["spot:BTCUSDT"])
    assert np.flatnonzero(np.isfinite(w[:, 0])).tolist() == [25, 49]
    assert w[25, 0] == 1 and w[49, 0] == 0


def test_missing_active_data_errors_with_name_time(tmp_path):
    market, _, _ = adapter()

    def change(a):
        a["data"]["spot:BTCUSDT"]["klines"].pop(5)

    ds = sample(tmp_path, change=change)
    with pytest.raises(ValueError, match="BTCUSDT.*2024-01-01T05"):
        market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04")
    m = market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04", allow_missing=True)
    assert not m.market.can_buy[5, 0] and m.market.close[5, 0] == m.market.close[4, 0]
    assert m.diagnostics["missing_intervals"]


def delist(known=25, effective=30):
    t = millis("2024-01-01")
    return normalize_event(
        dict(
            event_id="synthetic_delist",
            instrument="spot:BTCUSDT",
            kind="delist",
            known_at=t + known * HOUR,
            effective_at=t + effective * HOUR,
            source_url="https://example.test/delist",
            original_text="Synthetic notice",
        )
    )


def test_delist_known_at_hourly_retry_only_that_column_and_unrealized(tmp_path):
    market, daily, _ = adapter()

    def change(a):
        for b in a["data"]["spot:BTCUSDT"]["klines"]:
            if b["time"] >= a["start"] + 25 * HOUR:
                b["volume"] = 1.0

    ds = sample(tmp_path, events=[delist()], change=change)
    m = market(
        ds,
        ["BTCUSDT", "ETHUSDT"],
        "2024-01-01",
        "2024-01-04",
        initial_cash=10000,
        max_participation=0.1,
    )
    r = m.run(daily((["2024-01-01"], [[0.5, 0.4]])), record_trades=True)
    assert r.positions[24, 0] == 50
    np.testing.assert_allclose(np.diff(r.positions[24:30, 0]), [-0.1] * 5, atol=1e-12)
    np.testing.assert_array_equal(r.positions[24:, 1], np.full(48, r.positions[24, 1]))
    assert all(t.day >= 25 for t in r.trades if t.col == 0 and t.qty < 0)
    assert not r.delist_events
    assert r.cash[30] == r.cash[29]
    d = r.diagnostics["crypto"]
    assert d["unrealized"][0]["quantity"] == pytest.approx(49.5)
    assert d["unrealized"][0]["value"] == pytest.approx(49.5 * 101)
    assert r.equity[-1] > r.cash[-1]


def test_delist_overrides_buy_and_missing_announcement_errors(tmp_path):
    market, daily, _ = adapter()
    ds = sample(tmp_path, events=[delist(24, 30)])
    m = market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04")
    r = m.run(daily((["2024-01-01"], [[1.0]])), record_trades=True)
    assert not r.trades

    class NoEvents:
        meta = ds.meta
        profiles = ds.profiles
        read = ds.read

        def inputs(self):
            return dict(ds.inputs(), events=[])

    with pytest.raises(ValueError, match="BTCUSDT.*公告"):
        market(NoEvents(), ["BTCUSDT"], "2024-01-01", "2024-01-04")


def test_usdm_wiring_funding_and_tier_excess_error(tmp_path):
    market, _, _ = adapter()
    ds = sample(tmp_path, kind="usdm")
    with pytest.raises(ValueError, match="assume"):
        market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04", kind="usdm")
    m = market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-04",
        kind="usdm",
        assume="previous",
        initial_cash=100000,
        leverage=2,
    )
    assert m.market.futures.exec_at_open
    assert m.market.volume[0, 0] == 1000000
    assert m.market.futures.funding[7, 0] == 0.0001 * 100 * 0.01
    w = np.full((72, 1), np.nan)
    w[0] = 1
    with pytest.raises(ValueError, match="tier_cap_exceeded"):
        m.run(w)
    r = m.run(w, allow_tier_exceeded=True)
    assert r.diagnostics["crypto"]["valid"] is False
    assert r.diagnostics["crypto"]["rule_assumptions"]


def test_top_n_completed_time_window_coin_mapping_and_truncation(tmp_path):
    _, _, top = adapter()
    ds = sample(tmp_path, kind="usdm")
    f = ds.read().with_columns(
        pl.when(pl.col("time") >= millis("2024-01-03"))
        .then(1e12)
        .otherwise(pl.col("quote_volume"))
        .alias("quote_volume")
    )
    selector = top(1, lookback="1D", min_age="1D")
    at = millis("2024-01-03")
    names = selector.select(f, ds.profiles, at, events=[])
    shorter = selector.select(f.filter(pl.col("time") < at), ds.profiles, at, events=[])
    assert names == shorter == ["ETH"]
    assert top(2, min_age="30D").select(f, ds.profiles, at, events=[]) == []
    # 只有窗口内的已完成小时参与排名；边界恰好 at 的巨额不能进。
    f = f.with_columns(
        pl.when((pl.col("instrument") == "usdm:BTCUSDT") & (pl.col("time") < at - DAY))
        .then(1e15)
        .otherwise(pl.col("quote_volume"))
        .alias("quote_volume")
    )
    assert selector.select(f, ds.profiles, at, events=[]) == ["ETH"]
    profiles = copy.deepcopy(ds.profiles)
    profiles[1]["margin_asset"] = "USDC"
    assert selector.select(f, profiles, at, events=[]) == ["BTC"]


def test_quarter_roll_is_calendar_duration_and_forward_only():
    from ashare.backtest.crypto import quarter_chain, roll_time

    assert roll_time("2024-06-28T08:00", roll_before="3D", trade_hour=1) == millis(
        "2024-06-25T01:00"
    )
    times = np.arange(millis("2024-06-24"), millis("2024-06-29"), HOUR)
    profiles = [
        dict(
            instrument="usdm:A",
            coin="BTC",
            contract_type="CURRENT_QUARTER",
            listed_at=millis("2024-01-01"),
            delivery_at=millis("2024-06-28T08:00"),
        ),
        dict(
            instrument="usdm:B",
            coin="BTC",
            contract_type="NEXT_QUARTER",
            listed_at=millis("2024-03-01"),
            delivery_at=millis("2024-09-27T08:00"),
        ),
    ]
    held = quarter_chain(profiles, times, trade_hour=1, roll_before="3D")
    assert held[24, 0] == 0 and held[25, 0] == 1
    assert (np.diff(held[:, 0]) >= 0).all()


def test_liquidation_diagnostics_intraday_not_lost(tmp_path):
    """开发自检：内核计数名是 intraday，展示别名不能静默归零。"""
    market, _, _ = adapter()

    def crash(a):
        a["data"]["usdm:BTCUSDT"]["markPriceKlines"][0]["low"] = 1.0

    ds = sample(tmp_path, kind="usdm", change=crash)
    m = market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-04",
        kind="usdm",
        assume="previous",
        leverage=10,
        initial_cash=1000,
    )
    w = np.full((72, 1), np.nan)
    w[0] = 5.0
    r = m.run(w, allow_tier_exceeded=True)
    assert r.coverage["liquidation_intraday"] == 1
    assert r.diagnostics["crypto"]["liquidation_counts"]["intraday"] == 1
    assert r.diagnostics["crypto"]["clearance_fee"] > 0


def test_migration_carries_residual_without_cash_sale(tmp_path):
    """来源：4.17 带 known_at 的币转换接续；不凭未来映射换仓。"""
    market, daily, _ = adapter()
    t = millis("2024-01-01")
    event = normalize_event(
        dict(
            event_id="conversion",
            instrument="spot:BTCUSDT",
            kind="migration",
            known_at=t + 28 * HOUR,
            effective_at=t + 31 * HOUR,
            replacement="ETH",
            ratio=2.0,
            source_url="https://example.test/swap",
            original_text="Synthetic conversion",
        )
    )

    def change(a):
        for b in a["data"]["spot:BTCUSDT"]["klines"]:
            if b["time"] >= t + 25 * HOUR:
                b["volume"] = 0.0

    ds = sample(tmp_path, events=[delist(), event], change=change)
    m = market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04", max_participation=0.1)
    r = m.run(daily((["2024-01-01"], [[0.5]])), record_trades=True)
    assert r.cash[31] == r.cash[30]
    assert r.positions[31, 0] == r.positions[30, 0]
    assert m.market.close[30, 0] == 101.0
    assert m.market.close[31, 0] == 204.0
    assert m.market.can_buy[31, 0]
    assert m.diagnostics["migrations"][0]["ratio"] == 2.0
    assert r.diagnostics["crypto"]["unrealized"] == []


def test_numpy_daily_dates_from_dataset_are_accepted():
    """真实例子回归：ds.daily().to_numpy() 返回 np.int64，不能按 datetime 读 tzinfo。"""
    _, daily, _ = adapter()
    days = np.array([millis("2024-01-01")], dtype=np.int64)
    p = daily((days, [[0.5]]))
    assert p.times[0] == millis("2024-01-02")


def test_c2_shared_fixture_lifecycle_settlement_and_filters(tmp_path, fixture):
    market, _, _ = adapter()
    frame, notes = build_frame(**fixture)
    ds = write_panel(
        tmp_path,
        frame,
        profiles=fixture["profiles"],
        data=fixture["data"],
        events=fixture["events"],
        notes=notes,
    )
    rules = {
        p["instrument"]: dict(
            maint_rate=0.004,
            tier_cap=50000,
            max_leverage=125,
            clearance_fee_rate=0.0125,
            reason="synthetic rule fixture",
            source_url="https://example.test/rules",
        )
        for p in fixture["profiles"]
    }
    names = ["usdm:PERPUSDT", "usdm:FUTUSDT_240628"]
    m = market(
        ds,
        names,
        fixture["start"],
        fixture["end"],
        kind="usdm",
        assume=dict(rules=rules),
        initial_cash=1000.0,
    )
    assert np.isnan(m.market.futures.liquidation.mark_open[0, 0])
    assert m.market.futures.settlement[0, 0] == 103.5
    assert np.isnan(m.signal_prices[:3, 0]).all()
    assert m.market.futures.expiry.tolist() == [12, 7]
    assert m.market.futures.settlement[7, 1] == 199
    w = np.full((12, 2), np.nan)
    w[0] = [1.0, 2.0]
    w[2] = [1.0, 2.0]
    r = m.run(w, kind="shares", roll_before=None, record_trades=True)
    assert (r.positions[:, 0] == 0).all()  # 最小数量 2 张，输入 1 张被拦下。
    assert r.diagnostics["crypto"]["filtered_orders"]["futures_rounded_to_zero"] > 0
    assert len(r.expiry_events) == 1 and r.expiry_events[0].day == 7


def test_event_revision_in_future_does_not_change_old_trades(tmp_path):
    market, daily, _ = adapter()
    first = delist(25, 30)
    later = normalize_event(
        dict(
            first,
            event_id=first["event_id"],
            known_at=millis("2024-01-03"),
            effective_at=millis("2024-01-04"),
            supersedes=first["version_id"],
            version_id=None,
            status="cancelled",
        )
    )
    a = sample(tmp_path / "old", events=[first])
    b = sample(tmp_path / "future", events=[first, later])
    ma = market(a, ["BTCUSDT"], "2024-01-01", "2024-01-02T12:00")
    mb = market(b, ["BTCUSDT"], "2024-01-01", "2024-01-02T12:00")
    plan = daily((["2024-01-01"], [[0.5]]))
    ra = ma.run(plan, record_trades=True)
    rb = mb.run(plan, record_trades=True)
    np.testing.assert_array_equal(ra.cash, rb.cash)
    assert ra.trades == rb.trades


def test_top_n_merges_quote_units_and_keeps_historical_delisted_coin(tmp_path):
    _, _, top = adapter()
    ds = sample(tmp_path, kind="usdm")
    f = ds.read()
    extra = f.filter(pl.col("instrument") == "usdm:BTCUSDT").with_columns(
        pl.lit("usdm:1000BTCUSDT").alias("instrument"),
        pl.lit(1000.0).alias("coin_units"),
    )
    profile = dict(ds.profiles[0], instrument="usdm:1000BTCUSDT", coin_units=1000.0)
    event = normalize_event(
        dict(
            event_id="future_delist",
            instrument="usdm:BTCUSDT",
            kind="settlement",
            known_at=millis("2024-01-03"),
            effective_at=millis("2024-01-04"),
            source_url="https://example.test/event",
            original_text="Synthetic event",
        )
    )
    assert top(1, min_age="1D").select(
        pl.concat([f, extra]), [*ds.profiles, profile], "2024-01-02", events=[event]
    ) == ["BTC"]
    # 名单按币；quote_volume 不乘 coin_units，两个永续的实际成交额相加。
    extra = extra.with_columns(pl.lit(0.0001).alias("quote_volume"))
    assert top(1, min_age="1D").select(
        pl.concat([f, extra]), [*ds.profiles, profile], "2024-01-02", events=[event]
    ) == ["ETH"]


def test_hand_example_funds_execution_to_mark_loss(tmp_path):
    """真实手算例回归：成交/标记开盘不同，原例仅开出 995 张，不能按 1000 张手算。"""
    from bench.crypto_examples import initial_margin_cash

    market, _, _ = adapter()

    def change(a):
        b = a["data"]["usdm:BTCUSDT"]["markPriceKlines"][0]
        b.update(open=90.0, low=89.0)

    ds = sample(tmp_path, kind="usdm", change=change)
    capital = initial_margin_cash(100.0, 90.0, 1.0, 10.0, 0.0005)
    m = market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-04",
        kind="usdm",
        assume="previous",
        initial_cash=capital,
        leverage=10.0,
    )
    w = np.full((72, 1), np.nan)
    w[0] = 100
    r = m.run(w, kind="shares", record_trades=True)
    assert r.trades[0].qty == 100
    assert capital == pytest.approx(19.05)


def test_signal_prices_never_read_current_hour(tmp_path):
    market, _, _ = adapter()

    def change(a):
        for j, b in enumerate(a["data"]["spot:BTCUSDT"]["klines"]):
            b.update(close=100.0 + j, high=200.0 + j)

    ds = sample(tmp_path, change=change)
    m = market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04")
    np.testing.assert_array_equal(m.signal_prices[1:, 0], np.arange(100.0, 171.0))


@pytest.mark.parametrize("ordinary_on_roll", [False, True])
def test_quarter_roll_preserves_units_with_different_multipliers(
    tmp_path, ordinary_on_roll
):
    """开发自检：普通张数目标路由与换月专用行都必须换算乘数。"""
    market, daily, _ = adapter()
    start = millis("2024-01-01")
    event = normalize_event(
        dict(
            event_id="delivery",
            instrument="usdm:BTCUSDT",
            kind="delivery",
            known_at=start,
            effective_at=start + 56 * HOUR,
            source_url="https://example.test/delivery",
            original_text="Synthetic quarterly date",
        )
    )

    def change(a):
        for j, p in enumerate(a["profiles"]):
            p.update(
                coin="BTC",
                contract_type="CURRENT_QUARTER" if j == 0 else "NEXT_QUARTER",
                delivery_at=start + (56 if j == 0 else 120) * HOUR,
            )
            a["data"][p["instrument"]]["fundingRate"] = []
            a["data"][p["instrument"]]["funding_prices"] = []
        a["profiles"][0]["multiplier"] = 0.1
        a["profiles"][0]["rules"][0].update(qty_step=0.1, min_qty=0.1)
        a["data"]["usdm:BTCUSDT"]["deliveries"] = [
            dict(deliveryTime=start + 56 * HOUR, deliveryPrice=100.0)
        ]

    ds = sample(tmp_path, kind="usdm", events=[event], change=change)
    m = market(
        ds,
        ["BTCUSDT", "ETHUSDT"],
        "2024-01-01",
        "2024-01-04",
        kind="usdm",
        assume="previous",
    )
    days = ["2023-12-31", "2024-01-01"] if ordinary_on_roll else ["2023-12-31"]
    r = m.run(
        daily((days, [[2.0, 0.0]] * len(days)), trade_hour=1),
        kind="shares",
        roll_before="1D",
        record_trades=True,
    )
    np.testing.assert_array_equal(r.positions[24], [2.0, 0.0])
    np.testing.assert_array_equal(r.positions[25], [0.0, 20.0])
    assert {tr.day for tr in r.trades} == {1, 25}
    assert all(t.col == 1 for t in r.trades if t.day > 25)


@pytest.mark.parametrize(
    "label", [np.datetime64("2024-01-01"), np.datetime64("2024-01-01", "ns")]
)
def test_daily_numpy_datetime_labels(label):
    """接口自检：带日期标签的 numpy 宽表不能被当作 datetime.tzinfo 对象。"""
    _, daily, _ = adapter()
    assert daily(([label], [[0.5]])).times[0] == millis("2024-01-02")


def test_daily_polars_date_labels():
    """接口自检：Polars Date 长表也属于带 day 标签的输入。"""
    from datetime import date

    _, daily, _ = adapter()
    plan = daily(
        pl.DataFrame(dict(day=[date(2024, 1, 1)], instrument=["BTCUSDT"], weight=[0.5]))
    )
    assert plan.times[0] == millis("2024-01-02")


def test_spot_share_mode_cannot_bypass_snapshot_lots(tmp_path):
    """接口自检：内核现货 shares 绕过步长；C3 不能默默绕过档案约束。"""
    market, _, _ = adapter()
    m = market(sample(tmp_path), ["BTCUSDT"], "2024-01-01", "2024-01-04")
    w = np.full((72, 1), np.nan)
    w[0] = 0.015
    with pytest.raises(ValueError, match="现货.*权重"):
        m.run(w, kind="shares")


def test_migration_incompatible_fixed_spot_lot_is_rejected(tmp_path):
    """接口自检：转换后固定旧单位无法落在新币格点时不得生成非法订单。"""
    market, _, _ = adapter()
    t = millis("2024-01-01")
    event = normalize_event(
        dict(
            event_id="fractional_swap",
            instrument="spot:BTCUSDT",
            kind="migration",
            known_at=t,
            effective_at=t + HOUR,
            replacement="ETH",
            ratio=0.75,
            source_url="https://example.test/swap",
            original_text="Synthetic fractional conversion",
        )
    )
    ds = sample(tmp_path, events=[event])
    with pytest.raises(ValueError, match="转换.*数量格点"):
        market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04")
