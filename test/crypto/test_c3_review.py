"""来源：Claude C3 第一轮九条审查反例；先在返工前代码上确认失败。"""

import copy

import numpy as np
import polars as pl
import pytest

from ashare.backtest.crypto import crypto_market
from ashare.data.crypto.dataset import write_panel
from ashare.data.crypto.events import EventLedger, normalize_event
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame
from bench import crypto_examples as examples
from test.crypto.test_announcements import PUBLISHED, article, crawl
from test.crypto.test_backtest import delist, sample
from test.crypto.test_c3_data_fixes import halt_fixture


def event(name, kind, known, effective, **kwargs):
    return normalize_event(
        dict(
            event_id=f"{name}:{kind}",
            instrument=name,
            kind=kind,
            known_at=known,
            effective_at=effective,
            source_url="https://example.test/c3-review",
            original_text="Synthetic review event",
            **kwargs,
        )
    )


def migration_events():
    t = millis("2024-01-01")
    return [
        event("spot:BTCUSDT", "migration", t, t + HOUR, replacement="ETH", ratio=1),
        event("spot:ETHUSDT", "delist", t + 3 * HOUR, t + 100 * HOUR),
    ]


def test_r1_p0_1_migrated_delist_and_future_announcement_invariance(tmp_path):
    events = migration_events()
    results = []
    for k, extra in enumerate(([], [delist(100, 120)], [delist(50, 120)])):
        ds = sample(tmp_path / str(k), events=[*events, *extra])
        m = crypto_market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04")
        targets = np.full((72, 1), np.nan)
        targets[0] = 0.5
        results.append(m.run(targets, record_trades=True))
    for result in results:
        assert [t.day for t in result.trades if t.qty < 0] == [3]
        assert result.positions[2, 0] == 50
        assert result.positions[3, 0] == 0
        np.testing.assert_array_equal(result.nav, results[0].nav)


def test_r1_p0_1_missing_notice_checks_actual_execution_symbol(tmp_path):
    events = migration_events()
    events[1] = event(
        "spot:ETHUSDT", "delist", millis("2024-01-01T03"), millis("2024-01-02")
    )
    # 原始币有公告不能替代新币的公告。
    events.append(delist(50, 120))
    ds = sample(tmp_path, events=events)

    class HiddenNotice:
        read = ds.read

        def inputs(self):
            return dict(ds.inputs(), events=[events[0], events[2]])

    with pytest.raises(ValueError, match="ETHUSDT.*公告"):
        crypto_market(HiddenNotice(), ["BTCUSDT"], "2024-01-01", "2024-01-04")


@pytest.mark.parametrize("partial_source", ["indexPriceKlines", "klines"])
def test_r1_p0_2_normal_market_data_refutes_inferred_halt(fixture, partial_source):
    f = halt_fixture(fixture, "usdm")
    name = f["profiles"][0]["instrument"]
    data = f["data"][name]
    for field in ("klines", "markPriceKlines", "indexPriceKlines"):
        if field != partial_source:
            first = dict(
                data[field][0], time_quality="normal", close_time=f["start"] + HOUR - 1
            )
            data[field] = [
                dict(
                    first,
                    time=f["start"] + i * HOUR,
                    close_time=f["start"] + (i + 1) * HOUR - 1,
                )
                for i in range(3)
            ] + data[field][1:]
    data["markPriceKlines"][1]["low"] = 1.0
    with pytest.raises(ValueError, match="缺少|反证"):
        build_frame(**f)


def test_r1_p0_2_independent_verify_rejects_false_halt(fixture):
    f = halt_fixture(fixture, "usdm")
    frame, _ = build_frame(**f)
    name = f["profiles"][0]["instrument"]
    data = f["data"][name]
    # 给原始输入补正常成交和真实暴跌标记；旧持平面板必须被独立校验拒绝。
    for field in ("klines", "markPriceKlines"):
        first = dict(
            data[field][0], time_quality="normal", close_time=f["start"] + HOUR - 1
        )
        data[field] = [
            dict(
                first,
                time=f["start"] + i * HOUR,
                close_time=f["start"] + (i + 1) * HOUR - 1,
            )
            for i in range(3)
        ] + data[field][1:]
    data["markPriceKlines"][1]["low"] = 1.0
    # 部分标记仍由指数首行支持，避免仅靠首行 time_quality 差异杀死反例。
    report = verify_frame(frame, f)
    assert report["verdict"] == "fail"
    assert any(
        e["check"] == "halt_evidence" and e["time"] == f["start"] + HOUR
        for e in report["errors"]
    )


def test_r1_p0_2_counterevidence_breaks_entire_gap_chain(fixture):
    """返工排查：缺口中途已有正常标记，后续缺口不能重新启用此前半截线证据。"""
    f = halt_fixture(fixture, "usdm")
    name = f["profiles"][0]["instrument"]
    marks = f["data"][name]["markPriceKlines"]
    marks.insert(
        1,
        dict(
            marks[0],
            time=f["start"] + HOUR,
            close_time=f["start"] + 2 * HOUR - 1,
            time_quality="normal",
        ),
    )
    frame, _ = build_frame(**f, allow_missing=True)
    assert not any(frame["quality"][i] & 1024 for i in (1, 2))
    forged = frame.with_columns(
        pl.when(pl.col("time") == f["start"] + 2 * HOUR)
        .then(pl.col("quality") | 1024)
        .otherwise(pl.col("quality"))
        .alias("quality")
    )
    report = verify_frame(forged, f)
    assert any(
        e["check"] == "halt_evidence" and e["time"] == f["start"] + 2 * HOUR
        for e in report["errors"]
    )


def quarter_market(tmp_path, fixture, *, corrupt=None):
    f = copy.deepcopy(fixture)
    name = "usdm:FUTUSDT_240628"
    f["data"][name]["klines"] = [
        b for b in f["data"][name]["klines"] if b["time"] != f["start"] + 7 * HOUR
    ]
    frame, notes = build_frame(**f, allow_missing=True)
    if corrupt:
        frame = frame.with_columns(
            pl.when((pl.col("instrument") == name) & pl.col("expires"))
            .then(pl.lit(corrupt[1]))
            .otherwise(pl.col(corrupt[0]))
            .alias(corrupt[0])
        )
    ds = write_panel(
        tmp_path,
        frame,
        profiles=f["profiles"],
        data=f["data"],
        events=f["events"],
        notes=notes,
    )
    return crypto_market(
        ds,
        [name],
        f["start"],
        f["end"],
        kind="usdm",
        allow_missing=True,
        initial_cash=100,
        assume={
            "rules": {
                name: dict(
                    maint_rate=0.004,
                    tier_cap=50000,
                    max_leverage=100,
                    clearance_fee_rate=0.0125,
                    reason="synthetic",
                    source_url="https://example.test",
                )
            }
        },
    )


def test_r1_p0_3_missing_trade_bar_preserves_delivery_price(tmp_path, fixture):
    m = quarter_market(tmp_path, fixture)
    assert m.futures.settlement[7, 0] == 199
    w = np.full(m.market.shape, np.nan)
    w[0] = 2
    r = m.run(w, kind="shares", roll_before=None)
    assert r.cash[-1] == pytest.approx(99.7601)


@pytest.mark.parametrize(
    "corrupt", [("settlement", float("nan")), ("settlement_source", "missing_carry")]
)
def test_r1_p0_3_unproven_expiry_price_is_rejected(tmp_path, fixture, corrupt):
    with pytest.raises(ValueError, match="结算"):
        quarter_market(tmp_path, fixture, corrupt=corrupt)


def test_r1_p0_4_resume_consumes_old_body_before_new_revision(tmp_path, monkeypatch):
    a = article(
        body="The mock venue will conduct an automatic settlement on FAKEUSDT at 2024-06-11 09:00 (UTC)."
    )
    store = RawStore(tmp_path)
    append = EventLedger.append

    def interrupted(self, e):
        raise RuntimeError("body saved, ledger interrupted")

    try:
        monkeypatch.setattr(EventLedger, "append", interrupted)
        with pytest.raises(RuntimeError, match="ledger interrupted"):
            crawl(store, a, "2024-06-04")
        monkeypatch.setattr(EventLedger, "append", append)
        a["body"] = a["body"].replace("2024-06-11", "2024-06-12")
        get_json = PublicClient.json

        def check_drained(self, url, **kwargs):
            if "/detail/query" in url:
                assert len(EventLedger(tmp_path).read()) == 1
            return get_json(self, url, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(PublicClient, "json", check_drained)
            crawl(store, a, "2024-06-05")
        versions = EventLedger(tmp_path).read()
        assert len(versions) == 2
        assert [e["effective_at"] for e in versions] == [
            millis("2024-06-11T09"),
            millis("2024-06-12T09"),
        ]
        assert [e["known_at"] for e in versions] == [PUBLISHED, millis("2024-06-05")]
        assert versions[1]["supersedes"] == versions[0]["version_id"]
        assert crawl(store, a, "2024-06-06")["imported"] == 0
    finally:
        store.close()


def shift_sample(args, start):
    delta = millis(start) - args["start"]
    args["start"] += delta
    args["end"] += delta
    for p in args["profiles"]:
        p["listed_at"] += delta
        for rule in p["rules"]:
            rule["known_at"] += delta
            rule["effective_at"] += delta
    for datasets in args["data"].values():
        for rows in datasets.values():
            for row in rows:
                row["fundingTime" if "fundingTime" in row else "time"] += delta


def test_r1_p1_5_delivery_fee_uses_right_boundary(tmp_path):
    t = millis("2023-05-31T23")
    e = event("usdm:BTCUSDT", "settlement", t, t + HOUR, settlement_price=100)

    def change(a):
        shift_sample(a, t)
        for d in a["data"].values():
            d["fundingRate"] = [dict(time=t + HOUR, interval=8, rate=0.0001)]
            d["funding_prices"] = [
                dict(fundingTime=t + HOUR, fundingRate=".0001", markPrice="100")
            ]

    ds = sample(tmp_path, kind="usdm", hours=1, events=[e], change=change)
    m = crypto_market(
        ds, ["BTCUSDT"], t, t + HOUR, kind="usdm", assume="previous", initial_cash=1000
    )
    r = m.run(np.array([[100.0]]), kind="shares")
    assert r.commission.sum() == pytest.approx(0.04)
    assert r.delivery_fee.sum() == pytest.approx(0.05)
    assert m.diagnostics["delivery_fee_provenance"]


def test_r1_p1_6_daily_signal_excludes_invalid_close(tmp_path):
    def change(a):
        b = a["data"]["spot:BTCUSDT"]["klines"][23]
        b.update(
            close=10000.0,
            high=10000.0,
            time_quality="invalid_close_time",
            close_time=b["time"] - 1,
        )

    ds = sample(tmp_path, change=change)
    m = crypto_market(ds, ["BTCUSDT"], "2024-01-01", "2024-01-04", allow_missing=True)
    assert m.market.close[23, 0] == m.signal_prices[24, 0] == 101
    assert m.daily()["close"][0] == 101
    with pytest.raises(ValueError, match="异常"):
        ds.daily(instruments=["spot:BTCUSDT"])
    assert ds.read(instruments=["spot:BTCUSDT"])["close"][23] == 10000


def test_r1_p1_7_summary_counts_clearance_once(tmp_path):
    def crash(a):
        a["data"]["usdm:BTCUSDT"]["markPriceKlines"][0]["low"] = 1.0

    ds = sample(tmp_path, kind="usdm", change=crash)
    m = crypto_market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-04",
        kind="usdm",
        assume="previous",
        leverage=10,
        initial_cash=1000,
    )
    w = np.full(m.market.shape, np.nan)
    w[0] = 5
    r = m.run(w, allow_tier_exceeded=True)
    assert r.clearance_fee.sum() > 0
    expected = sum(
        np.sum(getattr(r, f))
        for f in (
            "commission",
            "tax",
            "other_fee",
            "delivery_fee",
            "slippage_cost",
            "impact_cost",
        )
    )
    assert examples.summary(r, 1000, 72)["fees"] == pytest.approx(expected)


def test_r1_p1_8_raw_funding_oracle_catches_adapter_scale(tmp_path, monkeypatch):
    def change(a):
        shift_sample(a, "2024-06-01")
        for p in a["profiles"]:
            p["rules"][0]["min_notional"] = 0

    ds = sample(tmp_path, kind="usdm", hours=24, change=change)
    assert examples.funding_check(ds)["expected_funding"] == pytest.approx(0.0003)
    original = examples.crypto_market

    def wrong_funding(*args, **kwargs):
        m = original(*args, **kwargs)
        same = np.shares_memory(m.arrays["funding"], m.futures.funding)
        m.arrays["funding"] *= 10
        if not same:
            m.futures.funding[:] *= 10
        return m

    monkeypatch.setattr(examples, "crypto_market", wrong_funding)
    with pytest.raises(AssertionError):
        examples.funding_check(ds)


def test_r1_p2_9_risk_rounding_metadata_reaches_diagnostics(tmp_path):
    ds = sample(
        tmp_path,
        kind="usdm",
        hours=4,
        change=lambda a: shift_sample(a, "2023-12-24T08"),
    )
    m = crypto_market(
        ds,
        ["ETHUSDT"],
        "2023-12-24T08",
        "2023-12-24T12",
        kind="usdm",
        assume="previous",
    )
    np.testing.assert_array_equal(
        m.futures.maint_rate[:, 0], [0.005, 0.005, 0.004, 0.004]
    )
    notes = m.diagnostics["rule_approximation_intervals"]
    assert any(
        n.get("parameter") == "maint_rate"
        and n.get("effective_at") == "2023-12-24T09:35:00.000"
        and n.get("applied_at") == "2023-12-24T10:00:00.000"
        for n in notes
    )
