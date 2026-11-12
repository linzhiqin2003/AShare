"""第三轮：跨停机日、结算截止、时钟回拨和资源保留的合成回归。"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import paperbroker
import polars as pl
import pytest
from paperbroker import Broker, Costs, MarketConfig, money

import ashare.paper
from ashare.paper.__main__ import main
from ashare.paper.market import timestamp
from ashare.paper.profiles import instrument
from ashare.paper.runner import LiveRunner
from ashare.paper.schedule import Schedule
from ashare.paper.source import LiveSource, SourceError
from ashare.paper.storage import ParquetStore, disk_usage

from .test_revision import at
from .test_sources import FakeClock

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)

CAL = [
    dict(
        cal_date=(datetime(2026, 11, 11) + timedelta(days=i)).strftime("%Y%m%d"),
        is_open=int((datetime(2026, 11, 11) + timedelta(days=i)).weekday() < 5),
    )
    for i in range(20)
]


class Client:
    def __init__(self, clock, symbols=("A",)):
        self.clock, self.symbols = clock, symbols
        self.volume = 0
        self.actions = {}
        self.queries = []
        self.prices = {s: 10 for s in symbols}
        self.stale = set()
        self.early = False
        self.fail_day = None

    def call(self, name, params=None, fields=""):
        params = params or {}
        if name == "trade_cal":
            return CAL
        if name == "suspend_d":
            return []
        if name == "dividend":
            self.queries.append(params["ex_date"])
            if params["ex_date"] == self.fail_day:
                raise SourceError("合成公司行为查询失败")
            return self.actions.get(params["ex_date"], [])
        if name == "stk_limit":
            return [
                dict(ts_code=s, up_limit=p * 1.1, down_limit=p * 0.9)
                for s, p in self.prices.items()
            ]
        if name == "daily":
            return [
                dict(
                    ts_code=s,
                    close=p if s not in self.stale else p + 1,
                    open=p,
                    vol=self.volume / 100,
                    amount=self.volume * p / 1000,
                )
                for s, p in self.prices.items()
            ]
        if name == "stk_auction":
            return []
        assert name == "rt_k"
        self.volume += 10000
        now = self.clock.now()
        final = self.early and now.hour >= 14 and (now.hour == 15 or now.minute == 59)
        if final:
            self.volume = 40000
        return [
            dict(
                ts_code=s,
                close=p,
                open=p,
                high=p,
                low=p,
                pre_close=p,
                trade_time=(
                    now.replace(hour=14, minute=50, second=0)
                    if s in self.stale
                    else now.replace(hour=15, minute=0, second=0)
                    if final
                    else now
                ).isoformat(),
                vol=self.volume,
                amount=self.volume * p,
                ask_price1=p,
                ask_volume1=10000,
                bid_price1=p,
                bid_volume1=10000,
            )
            for s, p in self.prices.items()
        ]


def action(day, cash=1, stock=1):
    return dict(
        ts_code="A",
        ex_date=day,
        div_proc="实施",
        cash_div_tax=cash,
        stk_div=stock,
        end_date="20260630",
    )


def build(tmp_path, symbols=("A",), clock=None):
    clock = clock or FakeClock("2026-11-11T09:00:00+08:00")
    client = Client(clock, symbols)
    store = ParquetStore(tmp_path / "quotes")
    cfg = MarketConfig(t_plus_one=False, costs=Costs(0, 0, 0))
    broker = Broker(list(symbols), config=cfg, journal=tmp_path, quote_store=store)
    runner = LiveRunner(
        broker,
        LiveSource(client, list(symbols)),
        Schedule(CAL),
        clock,
        store,
        output=tmp_path / "reports",
    )
    return runner, client, clock, broker


def held(tmp_path, cleared=False):
    r, c, clock, b = build(tmp_path)
    at(clock, "2026-11-11", "09:30:00")
    r.step()
    b.order(timestamp(clock.now()), "A", "buy", 1000, 11)
    for t in ("09:30:03", "09:30:06"):
        at(clock, "2026-11-11", t)
        r.step()
    assert b.account()["positions"][0]["total"] == 1000
    if cleared:
        b.order(timestamp(clock.now()), "A", "sell", 1000, 9)
        for t in ("09:30:09", "09:30:12"):
            at(clock, "2026-11-11", t)
            r.step()
        assert b.account()["positions"][0]["total"] == 0
    at(clock, "2026-11-11", "15:00:01")
    r.step()
    b.close()
    return c, clock


@pytest.mark.parametrize(
    "mode", ["one_day", "many_days", "consecutive_actions", "cleared"]
)
def test_outage_actions_in_date_order(tmp_path, mode):
    _, clock = held(tmp_path, mode == "cleared")
    return_day = "2026-11-13" if mode == "one_day" else "2026-11-16"
    at(clock, return_day, "09:30:00")
    r, c, _, b = build(tmp_path, clock=clock)
    c.actions["20261112"] = [action("20261112")]
    c.prices["A"] = 4.5
    if mode == "consecutive_actions":
        c.actions["20261113"] = [action("20261113", cash=0.5)]
        c.prices["A"] = 2
    r.step()
    assert c.queries == (
        ["20261112", "20261113"]
        if mode == "one_day"
        else ["20261112", "20261113", "20261116"]
    )
    corp = [e for e in b.history if e["type"] == "corporate"]
    expected = (
        0 if mode == "cleared" else 4000 if mode == "consecutive_actions" else 2000
    )
    assert b.account()["positions"][0]["total"] == expected
    assert b.account()["equity"] == money(1_000_000)
    assert [e["ex_day"] for e in corp] == (
        []
        if mode == "cleared"
        else [20261112, 20261113]
        if mode == "consecutive_actions"
        else [20261112]
    )
    # Same-day retries cannot duplicate either past action.
    before = b.account()
    r.metadata_day = ""
    r.step()
    assert b.account() == before
    b.close()


def test_outage_query_failure_blocks_trading_and_resumes_cursor(tmp_path):
    _, clock = held(tmp_path)
    at(clock, "2026-11-16", "09:30:00")
    r, c, _, b = build(tmp_path, clock=clock)
    c.actions = {
        "20261112": [action("20261112")],
        "20261113": [action("20261113", cash=0.5)],
    }
    c.prices["A"] = 2
    c.fail_day = "20261113"
    with pytest.raises(SourceError, match="查询失败"):
        r.step()
    assert json.loads(b.dump())["phase"] == "preopen"
    assert json.loads(b.dump())["actions_through"] == 20261112
    b.close()
    r, c, _, b = build(tmp_path, clock=clock)
    c.prices["A"] = 2
    c.actions = {"20261113": [action("20261113", cash=0.5)]}
    r.step()
    assert c.queries == ["20261113", "20261116"]
    assert b.account()["equity"] == money(1_000_000)
    assert b.account()["positions"][0]["total"] == 4000
    b.close()


@pytest.mark.parametrize("stale", [{"B"}, {"A", "B"}])
def test_close_deadline_is_per_security(tmp_path, stale):
    r, c, clock, b = build(tmp_path, ("A", "B"))
    c.prices["B"] = 7
    c.stale = stale
    at(clock, "2026-11-11", "14:57:00")
    r.step()
    for symbol in ("A", "B"):
        assert b.order(
            timestamp(clock.now()) + 1, symbol, "buy", 1000, c.prices[symbol] * 1.1
        )["accepted"]
    at(clock, "2026-11-11", "14:58:00")
    r.step()
    at(clock, "2026-11-11", "15:00:01")
    with pytest.raises(SourceError, match="收盘快照"):
        r.step()
    assert b.account()["phase"] == "close_auction"
    assert not (tmp_path / "reports/20261111.json").exists()
    at(clock, "2026-11-11", "15:03:01")
    r.step()
    assert json.loads(b.dump())["settled_day"] == 20261111
    result = json.loads((tmp_path / "reports/20261111.json").read_bytes())
    assert result["stale_close_symbols"] == sorted(stale)
    proxies = pl.concat(
        [pl.read_parquet(p) for p in (tmp_path / "quotes").glob("*.parquet")]
    ).filter(pl.col("source") == "rt_k_close_volume_proxy")
    for symbol in stale:
        assert proxies.filter(pl.col("symbol") == b.symbol_ids[symbol])[
            "auction_volume"
        ].to_list() == [0]
    assert [f["symbol"] for f in result["fills"]] == ([] if "A" in stale else [0])
    state = json.loads(b.dump())
    assert state["marks"] == [money(10), money(7)]
    assert all(o["status"] != "open" for o in state["orders"])
    b.close()


def test_early_close_quote_can_be_enriched_once(tmp_path):
    r, c, clock, b = build(tmp_path)
    c.early = True
    at(clock, "2026-11-11", "14:57:00")
    r.step()
    b.order(timestamp(clock.now()) + 1, "A", "buy", 1000, 11)
    at(clock, "2026-11-11", "14:59:58")
    r.step()
    assert not [e for e in b.history if e["type"] == "fill"]
    at(clock, "2026-11-11", "15:00:05")
    r.step()
    fills = [e for e in b.history if e["type"] == "fill"]
    assert len(fills) == 1 and fills[0]["auction"] and fills[0]["qty"] == 1000
    r.step()
    assert [e for e in b.history if e["type"] == "fill"] == fills
    b.close()


def test_clock_rollback_run_survives_and_targets_back_off(tmp_path, monkeypatch):
    r, c, clock, b = build(tmp_path)
    at(clock, "2026-11-11", "09:30:00")
    r.step()
    clock.value -= timedelta(seconds=5)
    r.run(duration=9)
    assert b.account()["phase"] == "continuous"
    r.targets = dict(at="09:30:00", weights={"A": 0.5}, rebalance_seconds=30)
    calls = []

    def reject(*args, **kwargs):
        calls.append(clock.monotonic())
        return [
            dict(
                accepted=False,
                request_id=999,
                events=[dict(type="rejected", reason="cash")],
            )
        ]

    monkeypatch.setattr(b, "target_weights", reject)
    r.run(duration=65)
    assert len(calls) == 3
    assert all(y - x >= 30 for x, y in zip(calls, calls[1:], strict=False))
    b.close()


def test_reports_not_rewritten_status_throttled_gc_cli(tmp_path, capsys):
    r, c, clock, b = build(tmp_path)
    for t in ("09:00:00", "09:30:00", "09:30:03"):
        at(clock, "2026-11-11", t)
        r.step()
    status = tmp_path / "status.json"
    stamp = status.stat().st_mtime_ns
    for t in ("09:30:06", "09:30:09", "09:30:12"):
        at(clock, "2026-11-11", t)
        r.step()
    assert status.stat().st_mtime_ns == stamp
    at(clock, "2026-11-11", "15:00:01")
    r.step()
    report = tmp_path / "reports/20261111.json"
    stamp = report.stat().st_mtime_ns
    b.close()
    r, c, _, b = build(tmp_path, clock=clock)
    r.step()
    assert report.stat().st_mtime_ns == stamp
    b.close()
    main(["gc", str(tmp_path), "--quote-retention-days", "0"])
    result = json.loads(capsys.readouterr().out)
    assert result["quotes_deleted"] == 0
    main(["status", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["disk_bytes"] == disk_usage(tmp_path)
    assert result["disk_bytes"]["wal"] > 0


@pytest.mark.parametrize(
    "symbol,step,minimum,tick",
    [
        ("688001.SH", 1, 200, 100),
        ("689009.SH", 1, 200, 100),
        ("300001.SZ", 100, 0, 100),
        ("301001.SZ", 100, 0, 100),
        ("302001.SZ", 100, 0, 100),
        ("920001.BJ", 1, 100, 100),
        ("830001.BJ", 1, 100, 100),
        ("430001.BJ", 1, 100, 100),
        ("510001.SH", 100, 0, 10),
        ("520001.SH", 100, 0, 10),
        ("560001.SH", 100, 0, 10),
        ("588001.SH", 100, 0, 10),
        ("501001.SH", 100, 0, 10),
        ("158001.SZ", 100, 0, 10),
        ("159001.SZ", 100, 0, 10),
        ("160001.SZ", 100, 0, 10),
        ("169999.SZ", 100, 0, 10),
    ],
)
def test_code_profiles(symbol, step, minimum, tick):
    profile = instrument(symbol)
    assert (profile.lots.step, profile.lots.minimum, profile.tick) == (
        step,
        minimum,
        tick,
    )
