"""来源：C6 4.18 v5 验收；全部价格与结算为程序生成合成夹具。"""

import copy
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ashare.backtest.crypto import daily_targets
from ashare.data.crypto.signals import contract_metadata, signal_rows, write_signals
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.factor.crypto import FACTOR_NAMES, crypto_factors, daily_factor

START = millis("2023-10-01")


class Synthetic:
    version = "synthetic-panel"

    def __init__(self, n=130 * 24):
        self.time = START + np.arange(n) * HOUR
        self.meta = dict(start=START, end=int(self.time[-1] + HOUR))
        self.values = dict(
            close=np.column_stack([np.full(n, 100.0), 101 + np.arange(n) / n]),
            mark_close=np.column_stack([np.full(n, np.nan), np.full(n, 103.0)]),
            index_close=np.column_stack([np.full(n, np.nan), np.full(n, 100.0)]),
            quality=np.zeros((n, 2), int),
            funding_rate=np.full((n, 2), np.nan),
            funding_interval_hours=np.full((n, 2), np.nan),
            open=np.column_stack(
                [100 + np.arange(n) * 0.01, 101 + np.arange(n) * 0.01]
            ),
            tradable=np.ones((n, 2), bool),
            funding_price=np.full((n, 2), 100.0),
        )
        for i in range(n):
            hour = i + 1
            interval = 8 if hour < 80 else 4 if hour < 120 else 1 if hour < 144 else 8
            if hour % interval == 0:
                self.values["funding_rate"][i, 1] = 0.0001 + hour * 1e-8
                self.values["funding_interval_hours"][i, 1] = interval

    def arrays(self, fields, **kwargs):
        return dict(time=self.time, **{k: self.values[k] for k in fields})

    def cut(self, at):
        obj = copy.deepcopy(self)
        keep = self.time + HOUR <= at
        obj.time = self.time[keep]
        obj.values = {k: v[keep].copy() for k, v in self.values.items()}
        return obj


def quarters(ds):
    at = ds.time + HOUR
    meta = [
        dict(
            symbol="BTCUSDT_231229",
            pair="BTCUSDT",
            listed_at=START - 20 * DAY,
            delivery_at=millis("2023-12-29T08:00:00Z"),
        ),
        dict(
            symbol="BTCUSDT_240329",
            pair="BTCUSDT",
            listed_at=START,
            delivery_at=millis("2024-03-29T08:00:00Z"),
        ),
    ]
    rows = [
        dict(
            time=int(t),
            symbol=m["symbol"],
            close=105 + float(i) * 0.0001 + (3 if j else 0),
            quality=0,
        )
        for j, m in enumerate(meta)
        for i, t in enumerate(ds.time)
    ]
    frame = pl.DataFrame(rows)
    return SimpleNamespace(
        version="synthetic-signals",
        metadata=meta,
        read=lambda: frame,
        at=at,
        frame=frame,
    )


def test_all_factors_hand_calculated():
    ds = Synthetic()
    q = quarters(ds)
    r = crypto_factors(ds, q, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert set(r) == set(FACTOR_NAMES)
    at = r["funding_last"][0]
    for i in [72, 121, 143, 700, len(at) - 1]:
        event = np.flatnonzero(np.isfinite(ds.values["funding_rate"][: i + 1, 1]))[-1]
        assert (
            r["funding_last"][1][i, 0]
            == ds.values["funding_rate"][event, 1]
            * 8760
            / ds.values["funding_interval_hours"][event, 1]
        )
        for days in (3, 7, 30):
            expected = (
                np.nan
                if i + 1 < days * 24
                else sum(
                    ds.values["funding_rate"][j, 1]
                    for j in range(i - days * 24 + 1, i + 1)
                    if np.isfinite(ds.values["funding_rate"][j, 1])
                )
                * 8760
                / (days * 24)
            )
            np.testing.assert_allclose(
                r[f"funding_mean_{days}d"][1][i, 0], expected, rtol=1e-12
            )
    i = len(at) - 1
    refs = r["funding_mean_7d"][1][i - 24 * np.arange(1, 91), 0]
    expected = (r["funding_mean_7d"][1][i, 0] - refs.mean()) / refs.std(ddof=1)
    np.testing.assert_allclose(r["funding_z_90d"][1][i, 0], expected, rtol=1e-10)
    for days in (1, 7):
        np.testing.assert_allclose(
            r[f"premium_{days}d"][1][i, 0],
            np.mean(ds.values["close"][i - days * 24 + 1 : i + 1, 1] / 100 - 1),
        )
        np.testing.assert_allclose(r[f"mark_premium_{days}d"][1][i, 0], 0.03)
    selected = r.selected_contract[1][i, 0]
    m = next(m for m in q.metadata if m["symbol"] == selected)
    px = q.frame.filter(
        (pl.col("time") == ds.time[i]) & (pl.col("symbol") == selected)
    )["close"].item()
    expected = (px / 100 - 1) * 8760 / ((m["delivery_at"] - at[i]) / HOUR)
    assert r["basis_q"][1][i, 0] == expected
    oldpx = q.frame.filter(
        (pl.col("time") == ds.time[i - 168]) & (pl.col("symbol") == selected)
    )["close"].item()
    old = (oldpx / 100 - 1) * 8760 / ((m["delivery_at"] - at[i - 168]) / HOUR)
    np.testing.assert_allclose(r["basis_q_chg_7d"][1][i, 0], expected - old)
    np.testing.assert_allclose(
        r["carry_spread"][1][i, 0], r["funding_mean_7d"][1][i, 0] - expected
    )


@pytest.mark.parametrize("flag", [1, 8, 128, 512, 1024, 2048])
def test_price_quality_both_sides_and_two_thirds(flag):
    ds = Synthetic(300)
    ds.values["quality"][100:109, 0] = flag
    r = crypto_factors(ds, None, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert np.isnan(r["premium_1d"][1][123, 0])  # 15/24
    assert np.isfinite(r["premium_1d"][1][124, 0])  # 16/24
    ds.values["quality"][:] = 0
    ds.values["quality"][100:109, 1] = flag
    r = crypto_factors(ds, None, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert np.isnan(r["mark_premium_1d"][1][123, 0])


def test_roll_strict_336_and_no_fallback_and_same_contract_change():
    ds = Synthetic()
    q = quarters(ds)
    switch = q.metadata[0]["delivery_at"] - 336 * HOUR
    index = int((switch - START) // HOUR) - 1
    near = q.metadata[0]["symbol"]
    frame = q.frame.with_columns(
        pl.when((pl.col("time") == ds.time[index - 1]) & (pl.col("symbol") == near))
        .then(1)
        .otherwise(pl.col("quality"))
        .alias("quality")
    )
    q.read = lambda: frame
    r = crypto_factors(ds, q, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert r.selected_contract[1][index - 1, 0] == near
    assert np.isnan(r["basis_q"][1][index - 1, 0])
    assert r.selected_contract[1][index, 0] == "BTCUSDT_240329"
    assert r.roll[1][index, 0]
    assert not r.roll[1][index - 1, 0]
    assert np.isfinite(r["basis_q_chg_7d"][1][index, 0])
    q.metadata[1]["listed_at"] = switch - 7 * DAY
    r = crypto_factors(ds, q, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert np.isnan(r["basis_q_chg_7d"][1][index, 0])


@pytest.mark.parametrize("flag", [1, 2, 4, 512, 2048])
def test_quarter_bad_hour_does_not_choose_far(flag):
    ds = Synthetic(300)
    q = quarters(ds)
    frame = q.frame.with_columns(
        pl.when(
            (pl.col("time") == ds.time[200]) & (pl.col("symbol") == "BTCUSDT_231229")
        )
        .then(flag)
        .otherwise(pl.col("quality"))
        .alias("quality")
    )
    q.read = lambda: frame
    r = crypto_factors(ds, q, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert r.selected_contract[1][200, 0] == "BTCUSDT_231229"
    assert np.isnan(r["basis_q"][1][200, 0])


@pytest.mark.parametrize("hour", [0, 1, 8, 23])
def test_daily_factor_daily_targets_month_year(hour):
    at = millis("2023-12-30") + np.arange(120) * HOUR
    vals = np.arange(len(at))[:, None]
    days, x = daily_factor((at, vals), hour)
    targets = daily_targets((days, x), hour)
    np.testing.assert_array_equal(targets.times, at[at % DAY == hour * HOUR])
    aligned = targets.align(at, ["BTCUSDT"])
    np.testing.assert_array_equal(aligned[at % DAY == hour * HOUR], x)


@pytest.mark.parametrize("cut", [150, 800, 2000, 3000])
def test_available_at_raw_event_truncation(cut):
    ds = Synthetic()
    q = quarters(ds)
    at = ds.time[cut] + HOUR
    whole = crypto_factors(ds, q, ["BTCUSDT"], at - 2 * HOUR, at + HOUR)
    truncated = ds.cut(at)
    frame = q.frame.filter(pl.col("time") + HOUR <= at)
    qs = SimpleNamespace(version=q.version, metadata=q.metadata, read=lambda: frame)
    part = crypto_factors(truncated, qs, ["BTCUSDT"], at - 2 * HOUR, at + HOUR)
    for name in FACTOR_NAMES:
        np.testing.assert_allclose(whole[name][1], part[name][1], atol=1e-12)
    np.testing.assert_array_equal(whole.selected_contract[1], part.selected_contract[1])


def raw_row(t, close=100, volume=1):
    return dict(
        time=t,
        close_time=t + HOUR - 1,
        time_quality="normal",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        quote_volume=volume * close,
        count=int(volume),
        taker_buy_volume=0.0,
        taker_buy_quote_volume=0.0,
    )


def test_signal_listing_metadata_conflicts_and_version(tmp_path):
    symbol = "BTCUSDT_240329"
    rows = [raw_row(START + i * HOUR, 100 + i) for i in range(3)]
    archives = ["synthetic-2023-10.zip"]
    m = contract_metadata(symbol, rows, archives, coverage_end=START + 3 * HOUR)
    assert m["listed_at"] == START
    with pytest.raises(ValueError, match="deliveryDate"):
        contract_metadata(symbol, rows, archives, dict(deliveryDate=START))
    with pytest.raises(ValueError, match="晚于最早"):
        contract_metadata(
            symbol,
            rows,
            archives,
            dict(onboardDate=START + HOUR, listed_source="synthetic"),
            coverage_end=START + 3 * HOUR,
        )
    with pytest.raises(ValueError, match="晚于交割"):
        contract_metadata(symbol, [raw_row(m["delivery_at"] + HOUR)], archives)
    a = write_signals(
        tmp_path,
        raw={symbol: rows},
        metadata=[m],
        references={},
        start=START + HOUR,
        end=START + 3 * HOUR,
    )
    assert a.metadata[0]["listed_at"] == START
    m["evidence"] = dict(onboardDate=START - HOUR, listed_source="synthetic-explicit")
    b = write_signals(
        tmp_path,
        raw={symbol: rows},
        metadata=[m],
        references={},
        start=START + HOUR,
        end=START + 3 * HOUR,
    )
    assert a.version != b.version
    assert a.read().height == 2


def test_signal_zero_frozen_gap_partial():
    symbol = "BTCUSDT_240329"
    rows = [
        raw_row(START),
        raw_row(START + HOUR),
        raw_row(START + 2 * HOUR, volume=0),
        raw_row(START + 4 * HOUR, 101),
    ]
    rows[-1]["time_quality"] = "partial_halt"
    m = contract_metadata(
        symbol, rows, ["synthetic.zip"], coverage_end=START + 5 * HOUR
    )
    with pytest.raises(ValueError, match="缺小时"):
        signal_rows(m, rows, START, START + 5 * HOUR)
    out = signal_rows(
        m,
        rows,
        START,
        START + 5 * HOUR,
        [
            dict(
                symbol=symbol,
                start=START + 3 * HOUR,
                end=START + 4 * HOUR,
                reason="synthetic gap",
            )
        ],
    )
    assert [r["quality"] for r in out] == [0, 4, 6, 1, 512]


def test_quarter_before_panel_does_not_fill_first_hour():
    # 来源：开发复核，searchsorted 对面板前历史返回 0，不能覆盖第一行。
    ds = Synthetic(300)
    q = quarters(ds)
    symbol = q.metadata[0]["symbol"]
    old = pl.DataFrame([dict(time=START - HOUR, symbol=symbol, close=123.0, quality=0)])
    frame = pl.concat(
        [
            old,
            q.frame.filter(~((pl.col("time") == START) & (pl.col("symbol") == symbol))),
        ]
    )
    q.read = lambda: frame
    r = crypto_factors(ds, q, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert np.isnan(r["basis_q"][1][0, 0])


def test_constant_funding_z_is_missing():
    # 来源：C6 零标准差要求，累计和消减误差不应制造 z 分数。
    ds = Synthetic()
    ds.values["funding_rate"][:] = np.nan
    ds.values["funding_interval_hours"][:] = np.nan
    ds.values["funding_rate"][7::8, 1] = 0.0001
    ds.values["funding_interval_hours"][7::8, 1] = 8
    r = crypto_factors(ds, None, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert np.isnan(r["funding_z_90d"][1][-24:, 0]).all()


def test_string_quality_exception_normalized_for_independent_verify(tmp_path):
    # 来源：开发复核，生产接受 ISO 字符串例外，但独立核验需要规范化整数。
    from ashare.data.crypto.verify_signals import verify_signals

    symbol = "BTCUSDT_240329"
    raw = [raw_row(START, 100), raw_row(START + 2 * HOUR, 101)]
    m = contract_metadata(symbol, raw, ["synthetic.zip"], coverage_end=START + 3 * HOUR)
    ds = write_signals(
        tmp_path,
        raw={symbol: raw},
        metadata=[m],
        references={},
        start=START,
        end=START + 3 * HOUR,
        exceptions=[
            dict(
                symbol=symbol,
                start="2023-10-01T01:00:00Z",
                end="2023-10-01T02:00:00Z",
                reason="synthetic gap",
            )
        ],
    )
    assert verify_signals(tmp_path, ds.version)["verdict"] == "pass"


def test_funding_initial_event_and_lifecycle_coverage():
    # 来源：开发复核，面板左端的已结算事件没有本面板的前一行，必须从冻结事件取。
    ds = Synthetic(200)
    ds.inputs = lambda: dict(
        data={
            "usdm:BTCUSDT": dict(
                fundingRate=[dict(time=START, rate=0.0004, interval=4)]
            )
        }
    )
    ds.values["quality"][40:50, 1] = 128
    r = crypto_factors(ds, None, ["BTCUSDT"], START, ds.meta["end"] + HOUR)
    assert r["funding_last"][1][0, 0] == pytest.approx(0.0004 * 8760 / 4)
    assert np.isnan(r["funding_last"][1][45, 0])
    assert np.isnan(r["funding_mean_3d"][1][100, 0])


@pytest.mark.parametrize("offset", [121, 744, 1700, 2400])
def test_rebuild_from_raw_events_cut_by_available_at(offset):
    # 来源：4.18 前视验收。先截断两类原始事件，再重新装配输入；不切已有面板。
    original = Synthetic(2500)
    raw_bars = [
        dict(
            available_at=int(t + HOUR),
            time=int(t),
            close=original.values["close"][i].copy(),
            mark_close=original.values["mark_close"][i].copy(),
            index_close=original.values["index_close"][i].copy(),
            quality=original.values["quality"][i].copy(),
        )
        for i, t in enumerate(original.time)
    ]
    raw_funding = [
        dict(
            time=int(t + HOUR),
            rate=float(original.values["funding_rate"][i, 1]),
            interval=int(original.values["funding_interval_hours"][i, 1]),
        )
        for i, t in enumerate(original.time)
        if np.isfinite(original.values["funding_rate"][i, 1])
    ]
    original_q = quarters(original)
    raw_quarter = list(original_q.frame.iter_rows(named=True))

    def rebuild(cutoff):
        bars = [r for r in raw_bars if r["available_at"] <= cutoff]
        settlements = [r for r in raw_funding if r["time"] <= cutoff]
        ds = Synthetic.__new__(Synthetic)
        ds.time = np.array([r["time"] for r in bars])
        ds.meta = dict(start=START, end=cutoff)
        ds.values = {
            key: np.array([r[key] for r in bars])
            for key in ["close", "mark_close", "index_close", "quality"]
        }
        ds.values["funding_rate"] = np.full((len(bars), 2), np.nan)
        ds.values["funding_interval_hours"] = np.full((len(bars), 2), np.nan)
        for event in settlements:
            row = int((event["time"] - START) // HOUR) - 1
            ds.values["funding_rate"][row, 1] = event["rate"]
            ds.values["funding_interval_hours"][row, 1] = event["interval"]
        ds.inputs = lambda: dict(data={"usdm:BTCUSDT": dict(fundingRate=settlements)})
        quarter = pl.DataFrame([r for r in raw_quarter if r["time"] + HOUR <= cutoff])
        q = SimpleNamespace(
            version="raw-synthetic", metadata=original_q.metadata, read=lambda: quarter
        )
        return ds, q

    end = START + 2500 * HOUR
    cut = START + offset * HOUR
    a, qa = rebuild(end)
    b, qb = rebuild(cut)
    full = crypto_factors(a, qa, ["BTCUSDT"], cut - 2 * HOUR, cut + HOUR)
    limited = crypto_factors(b, qb, ["BTCUSDT"], cut - 2 * HOUR, cut + HOUR)
    for name in FACTOR_NAMES:
        np.testing.assert_allclose(full[name][1], limited[name][1], atol=1e-12)
