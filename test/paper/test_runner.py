import json
from datetime import datetime
from pathlib import Path

import paperbroker
import pytest
from paperbroker import Broker, money

import ashare.paper
from ashare.paper.__main__ import main
from ashare.paper.market import rows
from ashare.paper.runner import LiveRunner, load_targets
from ashare.paper.schedule import Schedule
from ashare.paper.source import Batch, LiveSource
from ashare.paper.storage import ParquetStore

from .test_sources import FakeClock, fixture

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)


class StubClient:
    def __init__(self, clock):
        self.clock, self.calls = clock, []
        self.volume = 0

    def call(self, api_name, params=None, fields=""):
        self.calls.append(api_name)
        if api_name == "rt_k":
            self.volume += 10000
            return [
                dict(
                    ts_code="A",
                    close=10,
                    open=10,
                    high=10,
                    low=10,
                    pre_close=10,
                    vol=self.volume,
                    amount=self.volume * 10,
                    trade_time=self.clock.now().isoformat(),
                    ask_price1=10,
                    ask_volume1=10000,
                    bid_price1=10,
                    bid_volume1=10000,
                )
            ]
        if api_name == "stk_limit":
            return [dict(ts_code="A", up_limit=11, down_limit=9)]
        if api_name in ("suspend_d", "dividend"):
            return []
        if api_name == "daily":
            return [
                dict(
                    ts_code="A",
                    close=10,
                    open=10,
                    vol=self.volume / 100,
                    amount=self.volume * 10 / 1000,
                )
            ]
        if api_name == "stk_auction":
            return [dict(ts_code="A", price=10, vol=1000, amount=10000)]
        raise AssertionError(api_name)


def make_runner(tmp_path, now="2026-11-11T09:00:00+08:00"):
    clock = FakeClock(now)
    client = StubClient(clock)
    store = ParquetStore(tmp_path / "quotes")
    broker = Broker(["A"], journal=tmp_path, quote_store=store)
    source = LiveSource(client, ["A"])
    runner = LiveRunner(
        broker,
        source,
        Schedule(rows(fixture("trade_cal_2026q4"))),
        clock,
        store,
        dict(at="09:30:00", weights={"A": 0.1}),
        tmp_path / "reports",
    )
    return runner, client, clock, broker


def test_clock_schedule_no_lunch_or_weekend_poll_and_eod(tmp_path):
    runner, client, clock, broker = make_runner(tmp_path)
    for time, should_poll in (
        ("09:00", False),
        ("09:26", False),
        ("09:30", True),
        ("09:31", True),
        ("11:30", False),
        ("12:30", False),
        ("13:00", True),
        ("14:57", True),
        ("14:58", True),
        ("15:00", True),
    ):
        clock.value = datetime.fromisoformat(f"2026-11-11T{time}:00+08:00")
        before = client.calls.count("rt_k")
        runner.step()
        assert (client.calls.count("rt_k") > before) == should_poll
    assert broker.account()["day"] == 20261111
    assert json.loads(broker.dump())["settled_day"] == 20261111
    assert (tmp_path / "reports" / "20261111.json").exists()
    before = list(client.calls)
    clock.value = datetime.fromisoformat("2026-11-14T10:00:00+08:00")
    runner.step()
    assert client.calls == before
    broker.close()


def test_cli_status_replay_and_targets(tmp_path, capsys):
    runner, client, clock, broker = make_runner(tmp_path)
    runner.step()
    clock.value = datetime.fromisoformat("2026-11-11T09:30:00+08:00")
    runner.step()
    clock.sleep(3)
    runner.step()
    expected = broker.account()
    broker.close()
    main(["status", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert output["account"]["equity"] == expected["equity"]
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            dict(
                symbols=["A"],
                events=[dict(type="session", time=0, day=20261111, phase="continuous")],
            )
        )
    )
    result = tmp_path / "replay.json"
    main(
        [
            "replay",
            str(tmp_path / "quotes"),
            "--day",
            "20261111",
            "--script",
            str(script),
            "--output",
            str(result),
        ]
    )
    assert json.loads(result.read_text())["account"]["equity"] == money(1_000_000)
    script.write_text(json.dumps(dict(version=1, at="09:35:00", weights={"A": 0.4})))
    assert load_targets(script)["weights"] == {"A": 0.4}
    script.write_text(json.dumps(dict(version=1, at="09:35:00", weights={"A": 1.1})))
    with pytest.raises(ValueError):
        load_targets(script)


def test_run_uses_injected_clock(tmp_path):
    runner, client, clock, broker = make_runner(tmp_path, "2026-11-11T11:30:00+08:00")
    runner.run(12)
    assert clock.elapsed == 12
    assert "rt_k" not in client.calls
    broker.close()


def test_closed_quote_does_not_fill_and_restart_state(tmp_path):
    runner, _, clock, broker = make_runner(tmp_path)
    runner.step()
    clock.value = datetime.fromisoformat("2026-11-11T09:30:00+08:00")
    runner.step()
    broker.session(int(clock.now().timestamp() * 1000) + 1, 20261111, "closed")
    q = json.loads(broker.dump())["quotes"][0]
    import numpy as np

    q[1] += 1000
    q[7] += 100000
    runner.consume(Batch(q[1], np.array([q], np.int64), "closed"))
    assert broker.account()["positions"][0]["total"] == 0
    broker.close()


def test_status_available_while_writer_running_and_periodic_snapshot(tmp_path, capsys):
    runner, _, clock, broker = make_runner(tmp_path)
    runner.step()
    main(["status", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["account"]["day"] == 20261111
    for i in range(101):
        broker.transfer(int(clock.now().timestamp() * 1000) + i, 0)
    runner._drain()
    assert list(tmp_path.glob("state-*.json"))
    broker.close()


def test_smoke_cli_offline_runs_orders_and_records(tmp_path, monkeypatch, capsys):
    import ashare.paper.__main__ as cli

    clock = FakeClock("2026-11-11T09:30:00+08:00")

    class Client(StubClient):
        def __init__(self, lock, *, clock):
            super().__init__(clock)

        def call(self, api_name, params=None, fields=""):
            if api_name == "trade_cal":
                return rows(fixture("trade_cal_2026q4"))
            return super().call(api_name, params, fields)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cli, "SystemClock", lambda: clock)
    monkeypatch.setattr(cli, "TushareClient", Client)
    cli.main(
        [
            "smoke",
            "--directory",
            str(tmp_path),
            "--symbols",
            "A",
            "--duration",
            "9",
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["fills"]) == 2
    assert sum(f["qty"] for f in result["fills"]) == 2000
    assert not list(tmp_path.iterdir())


def test_postclose_verifier_offline(tmp_path, monkeypatch, capsys):
    import sys

    from ashare.paper import verify_live

    (tmp_path / "manifest.json").write_text(json.dumps(dict(symbols=["600000.SH"])))
    (tmp_path / "quotes").mkdir()
    calls = []

    class Client:
        def __init__(self, lock):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def call(self, name, params):
            calls.append((name, params))
            return rows(
                fixture("dividend_600000" if name == "dividend" else name + "_20261111")
            )

    monkeypatch.setattr(verify_live, "TushareClient", Client)
    monkeypatch.setattr(
        sys, "argv", ["verify_live", "--directory", str(tmp_path), "--day", "20261111"]
    )
    verify_live.main()
    result = json.loads(capsys.readouterr().out)
    assert len(calls) == 7 and all(
        params
        == (
            {"ex_date": "20261111"}
            if name == "dividend"
            else {"trade_date": "20261111"}
        )
        for name, params in calls
    )
    assert result["official_rows"]["stk_auction_c"] == 5714
    assert (tmp_path / "official-20261111.json").exists()
