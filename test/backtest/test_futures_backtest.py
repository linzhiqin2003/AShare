"""ashare.backtest.futures：换月展开、个人持仓期限、外部长表入口（合成数据，不读真实行情）。"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import panelbt as bt
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))

from futures_fixtures import DAYS, FakeTushare  # noqa: E402

import ashare.data as ad  # noqa: E402
from ashare.backtest import futures as bf  # noqa: E402
from ashare.data.futures_panel import build_futures  # noqa: E402
from ashare.data.futures_sync import sync_futures  # noqa: E402
from ashare.data.store import Store  # noqa: E402


@pytest.fixture
def fut(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHARE_DATA_DIR", str(tmp_path / "default-data"))
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    store = Store(tmp_path)
    asyncio.run(sync_futures(FakeTushare(), store, DAYS[0], DAYS[-1]))
    build_futures(store)
    return ad.open(tmp_path).futures()


def test_personal_last_hold_rules():
    calendar = np.array(
        [f"202606{d:02}" for d in range(1, 31)] + ["20260701", "20260702"]
    )
    contracts = pl.DataFrame(
        dict(
            symbol=[
                "SC2607.INE",
                "LU2607.INE",
                "M2607.DCE",
                "IF2606.CFFEX",
                "EC2607.INE",
            ],
            exchange=["INE", "INE", "DCE", "CFFEX", "INE"],
            product=["SC", "LU", "M", "IF", "EC"],
            last_trade_date=[
                "20260630",
                "20260630",
                "20260702",
                "20260619",
                "20260629",
            ],
            delivery_month=["202607", "202607", "202607", "202606", "202607"],
        )
    )
    out = bf.last_hold_dates(contracts, calendar)
    # SC：最后交易日前第 8 个交易日（20260622）收市后为 0 → 最后过夜 20260621。
    assert out["SC2607.INE"] == "20260621"
    assert out["LU2607.INE"] == "20260624"
    # 大商所：不得进入交割月 → 交割月前最后一个交易日。
    assert out["M2607.DCE"] == "20260630"
    assert out["IF2606.CFFEX"] is None and out["EC2607.INE"] is None


def test_portfolio_holds_contracts_and_rolls(fut):
    pf = bf.futures_portfolio(
        fut,
        ["RB"],
        missing=dict(margin_rate=0.1, fee_rate=1e-4, fee_per_contract=0.0),
        config=bt.Config(initial_cash=1e6, record_trades=True),
        account=bt.FuturesAccount(),
        slippage=0.0,
        infer_locked=False,
        personal=False,
    )
    t = len(pf.dates)
    held = pf.held[:, 0]
    # 第 0 行没有前一天的持仓量、最后一行是最后交易日：都不持有。
    assert held[0] == -1 and held[-1] == -1
    assert (held[1:-1] >= 0).all() and len(set(held[held >= 0])) >= 2
    w = np.full((t, 1), np.nan)
    w[1] = 3.0
    r = pf.run(w, kind="lots")
    # 每行只持有当行的合约，3 手；换月当行旧合约降为 0。
    for day in range(1, t - 1):
        row = r.positions[day]
        assert row[held[day]] == 3 and np.count_nonzero(row) == 1, day
    assert not r.positions[-1].any()
    # 盈亏只按具体合约的结算价：逐日权益变化 = 持仓 × 乘数 × 结算价变化 − 费用（换月行另算成交）。
    f = pf.book.market.futures
    rolls = pf.rolls["date"].to_list()
    assert len(rolls) >= 2
    for day in range(2, t - 1):
        if pf.dates[day] in rolls:
            continue
        h = held[day]
        expect = 3 * f.multiplier[h] * (f.settlement[day, h] - f.settlement[day - 1, h])
        assert np.isclose(r.equity[day] - r.equity[day - 1], expect - r.commission[day])
    # 换月行：旧合约平仓、新合约开仓，各付一次手续费。
    roll_day = list(pf.dates).index(rolls[1])
    fills = [x for x in r.trades if x.day == roll_day]
    assert sorted(x.qty for x in fills) == [-3.0, 3.0]
    assert all(x.commission > 0 for x in fills)


def test_personal_override_moves_to_next_contract(fut, monkeypatch):
    monkeypatch.setitem(bf.PERSONAL_RULES, ("SHFE", "RB"), ("before_last_trade", 3))
    kw = dict(
        missing=dict(margin_rate=0.1, fee_rate=1e-4, fee_per_contract=0.0),
        slippage=0.0,
        infer_locked=False,
    )
    pf = bf.futures_portfolio(fut, ["RB"], **kw)
    plain = bf.futures_portfolio(fut, ["RB"], personal=False, **kw)
    for day in range(len(pf.dates)):
        h = pf.held[day, 0]
        if h < 0:
            continue
        hold = bf._rows(pf.dates, pf.last_hold[pf.symbols[h].split("@")[0]])
        assert hold >= day, (day, pf.symbols[h])
    assert not np.array_equal(pf.held, plain.held)
    # 不能持有的那一行起，引擎在这一行收盘按结算价强平（若仍持有）。
    assert (pf.book.end_row <= pf.book.market.futures.expiry).all()
    assert pf.book.personal.any()


def test_lots_follow_unit_changes():
    # 手工构造：同一合约在第 2 行交易单位 25 → 5，拆成两列。
    s = np.full((4, 2), 100.0)
    book = bt.FuturesBook(
        settlement=s,
        close=s,
        exec_price=s,
        multiplier=[25.0, 5.0],
        expiry=[1, 4],
        margin_rate=0.1,
    )
    pf = bf.FuturesPortfolio(
        book=book,
        dates=np.array(["1", "2", "3", "4"]),
        symbols=np.array(["A", "A@3"]),
        products=["A"],
        held=np.array([[0], [0], [1], [1]]),
        signal_prices=s[:, :1],
        contracts=pl.DataFrame(),
        splits=[],
        rolls=pl.DataFrame(),
        last_hold={},
    )
    w = np.full((4, 1), np.nan)
    w[0] = 2.0
    t = pf.targets(w, "lots").to_dense(4, 2)
    assert t[0].tolist() == [2.0, 0.0]
    assert t[2].tolist() == [0.0, 10.0]  # 2 手 × 25 / 5
    assert np.isnan(t[1]).all() and np.isnan(t[3]).all()
    notional = pf.targets(w * 0.5, "notional").to_dense(4, 2)
    assert notional[2].tolist() == [0.0, 1.0]


def test_book_from_table_matches_arrays_and_example():
    from futures_wti_example import chain, run_example

    ext, r = run_example()
    assert r.diagnostics["trades"] > 10 and r.equity[-1] != 1e6
    assert len(r.expiry_events) == 0  # 都在最后交易日前换掉了
    daily, contracts = chain()
    ext2 = bf.book_from_table(
        daily,
        contracts,
        margin_rate=0.10,
        fee_per_contract=1.5,
        slippage=1e-4,
        initial_cash=1e6,
    )
    m = ext2.book.market
    j = ext2.column(contracts["symbol"][3])
    sub = daily.filter(pl.col("symbol") == contracts["symbol"][3])
    rows = [list(ext2.dates).index(d) for d in sub["date"]]
    assert np.array_equal(m.futures.settlement[rows, j], sub["settlement"].to_numpy())
    assert np.array_equal(m.exec_price[rows, j], sub["open"].to_numpy())
    assert m.futures.multiplier[j] == 1000.0
    with pytest.raises(ValueError, match="settlement"):
        bf.book_from_table(daily.drop("settlement"), contracts, margin_rate=0.1)
    with pytest.raises(ValueError, match="margin_rate"):
        bf.book_from_table(daily, contracts)
    with pytest.raises(ValueError, match="合约表没有"):
        bf.book_from_table(daily, contracts.head(3), margin_rate=0.1)
