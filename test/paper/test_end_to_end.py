import json
from pathlib import Path

import paperbroker
from paperbroker import Broker, money

import ashare.paper
from ashare.paper.market import normalize, rows, timestamp
from ashare.paper.profiles import configuration
from ashare.paper.runner import report
from ashare.paper.source import MinuteSource
from ashare.paper.storage import ParquetStore

from .test_sources import fixture

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)


def test_recorded_four_stock_full_day(tmp_path):
    symbols = ["600000.SH", "000001.SZ", "300750.SZ", "688981.SH"]
    day = "20261111"
    daily = {r["ts_code"]: r for r in rows(fixture("daily_20261111"))}
    limits = {
        r["ts_code"]: (r["up_limit"], r["down_limit"])
        for r in rows(fixture("stk_limit_20261111"))
    }
    suspended = {
        r["ts_code"]
        for r in rows(fixture("suspend_d_20261111"))
        if r["suspend_type"] == "S"
    }
    store = ParquetStore(tmp_path / "quotes")
    b = Broker(
        symbols, config=configuration(symbols), journal=tmp_path, quote_store=store
    )
    all_events = []

    def ts(t):
        return timestamp(f"2026-11-11 {t}")

    def feed(quotes, source):
        store.received_at, store.source = int(quotes[:, 1].max()), source
        b.feed(quotes, received_at=store.received_at)
        all_events.extend(b.events())

    b.session(ts("09:14:00"), int(day), "preopen")
    initial = [
        dict(
            ts_code=s,
            close=daily[s]["pre_close"],
            vol=0,
            amount=0,
            trade_time="2026-11-11 09:14:00",
        )
        for s in symbols
    ]
    feed(
        normalize(initial, symbols, limits=limits, suspended=suspended),
        "preopen_previous_close",
    )
    b.session(ts("09:15:00"), int(day), "open_auction")
    for s in symbols:
        b.order(ts("09:20:00"), s, "buy", 200, limits[s][0])
    auction = rows(fixture("stk_auction_20261111"))
    for r in auction:
        r.update(
            trade_time="2026-11-11 09:25:00",
            close=r["price"],
            open=r["price"],
            auction_volume=r["vol"],
        )
    feed(normalize(auction, symbols, limits=limits, auction_kind=1), "stk_auction")
    b.session(ts("09:30:00"), int(day), "continuous")
    minutes = sum(
        (rows(fixture(f"stk_mins_{s.replace('.', '_')}_20261111")) for s in symbols), []
    )
    source = MinuteSource(minutes, symbols, limits=limits, suspended=suspended)
    closing = {r["ts_code"]: r for r in rows(fixture("stk_auction_c_20261111"))}
    phase = "continuous"
    while (batch := source.poll()) is not None:
        t = batch.received_at
        if t == ts("11:30:00"):
            # 11:30 的分钟是上午最后一分钟，先撮合再午休。
            feed(batch.quotes, batch.source)
            b.session(t, int(day), "break")
            phase = "break"
            continue
        if phase == "break" and t >= ts("13:00:00"):
            b.session(ts("13:00:00"), int(day), "continuous")
            phase = "continuous"
        if t >= ts("14:57:00") and phase != "close_auction":
            b.session(ts("14:57:00"), int(day), "close_auction")
            for s in symbols:
                b.order(
                    ts("14:57:00"),
                    s,
                    "buy",
                    200 if s.startswith("688") else 100,
                    limits[s][0],
                )
            phase = "close_auction"
        if t == ts("15:00:00"):
            batch.quotes[:, 16] = [int(closing[s]["vol"]) for s in symbols]
            # normalize 按 source 原行次序输出，按 symbol id 对齐而非依赖行序。
            for row in batch.quotes:
                row[16] = int(closing[symbols[int(row[0])]]["vol"])
                row[17] = 2
        feed(batch.quotes, batch.source)
        if t == ts("09:30:00"):
            for s in symbols:
                b.order(
                    t + 1, s, "buy", 200 if s.startswith("688") else 100, limits[s][0]
                )
    b.session(ts("15:01:00"), int(day), "closed")
    b.settle(ts("15:02:00"), int(day))
    all_events.extend(b.events())
    b.checkpoint()
    state = b.dump()
    result = report(b, all_events)
    fs = result["fills"]
    assert len(fs) == 12
    by_key = {(f["symbol"], f["auction"], f["qty"]): f for f in fs}
    # 人工可算的两笔：浦发竞价 200 股 × 10.22 元；9:31 建立下单后的量基线，9:32 再买 100 股 × 10.24 元。
    assert by_key[(0, True, 200)]["price"] == money(10.22)
    assert by_key[(0, True, 200)]["fees"]["commission"] == money(5)
    assert by_key[(0, False, 100)]["price"] == money(10.24)
    assert by_key[(0, False, 100)]["fees"]["commission"] == money(5)
    for i, symbol in enumerate(symbols):
        p = result["account"]["positions"][i]
        assert p["total"] == p["sellable"] == (600 if symbol.startswith("688") else 400)
        assert p["mark"] == money(daily[symbol]["close"])
        assert p["market_value"] == (600 if symbol.startswith("688") else 400) * money(
            daily[symbol]["close"]
        )
    expected_cash = money(1_000_000) - sum(
        f["qty"] * f["price"] + sum(f["fees"].values()) for f in fs
    )
    assert result["account"]["cash"] == expected_cash
    assert result["account"]["equity"] == expected_cash + sum(
        (600 if s.startswith("688") else 400) * money(daily[s]["close"])
        for s in symbols
    )
    assert result["account"]["frozen"] == 0
    b.close()
    with Broker(
        symbols,
        config=configuration(symbols),
        journal=tmp_path,
        quote_store=ParquetStore(tmp_path / "quotes"),
    ) as recovered:
        assert recovered.dump() == state
    print(
        "PAPER_E2E="
        + json.dumps(
            {
                "fills": len(fs),
                "cash": expected_cash,
                "equity": result["account"]["equity"],
                "fees": result["account"]["fees"],
            },
            sort_keys=True,
        )
    )
