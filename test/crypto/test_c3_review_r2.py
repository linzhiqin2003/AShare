"""来源：Claude C3 第二轮复核；缺陷先复现失败，另补快慢路径覆盖。"""

import numpy as np
import panelbt as bt
import pytest

from ashare.backtest.crypto import crypto_market
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame
from test.crypto.test_backtest import delist, sample
from test.crypto.test_c3_data_fixes import halt_fixture
from test.crypto.test_c3_review import migration_events


def flat_spot(a):
    for datasets in a["data"].values():
        for b in datasets["klines"]:
            b.update(open=100.0, high=100.0, low=100.0, close=100.0)


def spot_market(ds, symbols=("BTCUSDT",), hours=4):
    return crypto_market(
        ds,
        symbols,
        "2024-01-01",
        millis("2024-01-01") + hours * HOUR,
        config=bt.Config(initial_cash=10000, carry_blocked=True),
    )


def assert_execution_identical(left, right):
    """所有逐行结果逐位相同；并核对成交、事件和内核诊断。"""
    for name, value in vars(left).items():
        other = getattr(right, name)
        if isinstance(value, np.ndarray):
            assert value.dtype == other.dtype and value.shape == other.shape, name
            assert value.tobytes() == other.tobytes(), name
        elif name == "diagnostics":
            assert {k: v for k, v in value.items() if k != "crypto"} == {
                k: v for k, v in other.items() if k != "crypto"
            }
        else:
            assert value == other, name


def test_r2_p1_delist_cancels_flat_pending_buy(tmp_path):
    def change(a):
        flat_spot(a)
        bars = a["data"]["spot:BTCUSDT"]["klines"]
        bars[0]["volume"] = 0
        for b in bars[2:]:
            b.update(open=50.0, high=50.0, low=50.0, close=50.0)

    ds = sample(tmp_path, hours=4, events=[delist(1, 6)], change=change)
    w = np.full((4, 1), np.nan)
    w[0] = 0.5
    r = spot_market(ds).run(w, record_trades=True)
    np.testing.assert_array_equal(r.nav, np.ones(4))
    np.testing.assert_array_equal(r.positions, np.zeros((4, 1)))
    assert not r.trades
    assert r.n_blocked_buy[0] == 1


@pytest.mark.parametrize("other_action", ["buy", "reduce", "sell"])
def test_r2_p1_cancel_only_delisted_pending_buy(tmp_path, other_action):
    def change(a):
        flat_spot(a)
        a["data"]["spot:BTCUSDT"]["klines"][1]["volume"] = 0
        for i in (1, 2):
            a["data"]["spot:ETHUSDT"]["klines"][i]["volume"] = 0

    ds = sample(tmp_path, hours=4, events=[delist(2, 6)], change=change)
    m = spot_market(ds, ("BTCUSDT", "ETHUSDT"))
    w = np.full((4, 2), np.nan)
    w[0] = [0, 0 if other_action == "buy" else 0.4]
    w[1] = [0.5, {"buy": 0.4, "reduce": 0.2, "sell": 0}[other_action]]
    r = m.run(w, record_trades=True)
    control = w.copy()
    control[1, 0] = 0  # 从未发出待取消买单；另一列所有目标、阻碍完全相同。
    expected = m.run(control, record_trades=True)
    assert not [t for t in r.trades if t.col == 0]
    np.testing.assert_array_equal(r.positions, expected.positions)
    np.testing.assert_array_equal(r.cash, expected.cash)
    assert r.trades == expected.trades
    resumed = [t for t in r.trades if t.col == 1 and t.day == 3]
    assert len(resumed) == 1
    assert (resumed[0].qty > 0) == (other_action == "buy")


@pytest.mark.parametrize("pending_target", [0.2, 0.0])
def test_r2_p1_delisted_pending_reduction_still_exits(tmp_path, pending_target):
    def change(a):
        flat_spot(a)
        bars = a["data"]["spot:BTCUSDT"]["klines"]
        bars[1]["volume"] = bars[2]["volume"] = 0

    ds = sample(tmp_path, hours=4, events=[delist(2, 6)], change=change)
    w = np.array([[0.5], [pending_target], [np.nan], [np.nan]])
    r = spot_market(ds).run(w, record_trades=True)
    np.testing.assert_array_equal(r.positions[:, 0], [50, 50, 50, 0])
    assert [(t.day, t.qty) for t in r.trades] == [(0, 50), (3, -50)]


def normal_mark_at_partial_trade(f):
    data = f["data"][f["profiles"][0]["instrument"]]
    data["markPriceKlines"][0].update(
        time_quality="normal", close_time=f["start"] + HOUR - 1
    )


def test_r2_p2_normal_mark_before_gap_does_not_refute_build(fixture):
    f = halt_fixture(fixture, "usdm")
    normal_mark_at_partial_trade(f)
    frame, notes = build_frame(**f)
    assert frame["quality"][0] & 512
    for i in (1, 2):
        assert frame["quality"][i] & 1024
        assert not frame["tradable"][i]
        assert frame["settlement"][i] == frame["settlement"][0]
    assert frame["tradable"][3] and not frame["quality"][3] & 1024
    assert notes[f["profiles"][0]["instrument"]]["halts"] == [
        dict(
            start=f["start"] + HOUR,
            end=f["start"] + 3 * HOUR,
            evidence="preceding_partial_trade_kline",
        )
    ]
    prefix, _ = build_frame(**dict(f, end=f["start"] + 3 * HOUR))
    assert prefix.equals(frame.head(3))


def test_r2_p2_independent_verify_accepts_mark_before_gap(fixture):
    f = halt_fixture(fixture, "usdm")
    frame, _ = build_frame(**f)
    # 校验器单独受测：面板先按原始正例构建，再仅改变半截线同小时标记的时间质量。
    normal_mark_at_partial_trade(f)
    report = verify_frame(frame, f)
    assert report["verdict"] == "pass", report["errors"]
    assert [
        d["time"] for d in report["diagnostics"] if d["check"] == "inferred_halt"
    ] == [f["start"] + HOUR, f["start"] + 2 * HOUR]


@pytest.mark.parametrize("carry_blocked", [False, True])
@pytest.mark.parametrize("exit_after_migration", [False, True])
def test_r2_p0_1_migration_fast_slow_bitwise(
    tmp_path, monkeypatch, carry_blocked, exit_after_migration
):
    events = migration_events()

    def change(a):
        for datasets in a["data"].values():
            for b in datasets["klines"]:
                b["volume"] = 5.0

    ds = sample(tmp_path, hours=8, events=events, change=change)
    fast = crypto_market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-01T08",
        config=bt.Config(
            initial_cash=10000, carry_blocked=carry_blocked, max_participation=0.1
        ),
    )
    slow = crypto_market(
        ds,
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-01T08",
        config=fast.config,
    )
    # 快路径只保留迁移和未来公告；慢路径在迁移后才得知旧币下架，不应再退出新币。
    fast.events = [events[0], delist(100, 120)]
    slow.events = events if exit_after_migration else [events[0], delist(3, 100)]
    w = np.full((8, 1), np.nan)
    w[0] = 0.5
    reference = w.copy()
    if exit_after_migration:
        # 原 P0-1 场景：新币公告产生的逐时退出，与同一已知退出计划走快路径对照。
        reference[3:] = 0
    calls = []
    run, simulator = bt.run, bt.Simulator

    def run_spy(*args, **kwargs):
        calls.append("fast")
        return run(*args, **kwargs)

    def simulator_spy(*args, **kwargs):
        calls.append("slow")
        return simulator(*args, **kwargs)

    monkeypatch.setattr(bt, "run", run_spy)
    monkeypatch.setattr(bt, "Simulator", simulator_spy)
    fast_result = fast.run(reference, record_trades=True)
    assert calls == ["fast"]
    slow_result = slow.run(w, record_trades=True)
    assert calls == ["fast", "slow"]
    assert fast.arrays["execution_symbol"][:, 0].tolist() == [
        "spot:BTCUSDT",
        *["spot:ETHUSDT"] * 7,
    ]
    assert_execution_identical(fast_result, slow_result)
