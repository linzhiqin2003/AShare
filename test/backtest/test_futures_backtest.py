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
    # 大商所：交割月前一个月最后一个交易日（20260630）收市后须为 0 → 最后过夜 20260629。
    assert out["M2607.DCE"] == "20260629"
    assert out["IF2606.CFFEX"] is None and out["EC2607.INE"] is None
    # 提前量：每个合约的期限再往前挪 lead 个交易日。
    early = bf.last_hold_dates(contracts, calendar, lead=2)
    assert early["SC2607.INE"] == "20260619" and early["M2607.DCE"] == "20260627"
    assert early["IF2606.CFFEX"] is None
    # 日历截在到期之前：用周一到周五补齐并警告；节假日当成交易日，算出的期限只会更晚。
    days = np.arange(np.datetime64("2026-06-01"), np.datetime64("2026-07-10"))
    days = np.char.replace(days[np.is_busday(days)].astype("U10"), "-", "").astype("U8")
    holiday = days[(days <= "20260612") | (days >= "20260622")]  # 停市一周
    full = bf.last_hold_dates(contracts, holiday)
    with pytest.warns(UserWarning, match="偏晚"):
        cut = bf.last_hold_dates(contracts, holiday[holiday <= "20260612"])
    for symbol in ("SC2607.INE", "LU2607.INE", "M2607.DCE"):
        assert cut[symbol] >= full[symbol]
    assert cut["SC2607.INE"] > full["SC2607.INE"]


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


def _hand_portfolio(settle, mult, expiry, held, products, record=True):
    """手写的合约层面组合：settle 是 T×N 结算价（也当成交价），held 是 T×P。"""
    t, n = settle.shape
    book = bt.FuturesBook(
        settlement=settle,
        close=settle,
        exec_price=settle,
        multiplier=mult,
        expiry=expiry,
        margin_rate=0.1,
        fee_rate=1e-4,
        config=bt.Config(initial_cash=1e6, record_trades=record),
    )
    return bf.FuturesPortfolio(
        book=book,
        dates=np.array([f"2024{i + 1:04d}" for i in range(t)]),
        symbols=np.array([f"C{j}" for j in range(n)]),
        products=products,
        held=np.asarray(held, dtype=np.int64),
        signal_prices=settle[:, : len(products)],
        contracts=pl.DataFrame(),
        splits=[],
        rolls=pl.DataFrame(),
        last_hold={},
    )


def test_lots_follow_unit_changes():
    # 同一合约在第 2 行交易单位 25 → 5，拆成两列：前一列第 1 行收盘被引擎了结，
    # 后一列第 2 行按「旧手数 × 旧乘数 / 新乘数」接上：2 手 → 10 手。
    s = np.full((4, 2), 100.0)
    pf = _hand_portfolio(s, [25.0, 5.0], [1, 4], [[0], [0], [1], [1]], ["A"])
    w = np.full((4, 1), np.nan)
    w[0] = 2.0
    r = pf.run(w, "lots")
    assert r.positions[:, 0].tolist() == [2.0, 0.0, 0.0, 0.0]
    assert r.positions[:, 1].tolist() == [0.0, 0.0, 10.0, 10.0]
    # 名义目标同样保持手数：第 0 行开出的手数在分界后按乘数折算。
    w[0] = 0.5
    r = pf.run(w, "notional")
    q = r.positions[0, 0]
    assert q == np.trunc(0.5 * 1e6 / (100.0 * 25.0))
    assert r.positions[2, 1] == q * 5


@pytest.mark.parametrize("kind", ["lots", "notional", "margin"])
def test_first_row_target_opens_where_the_contract_appears(kind):
    # 品种在第 0 行还没有合约（held = -1）：目标挂起，第 1 行按原目标开仓，不被乘数缩小。
    s = np.array([[np.nan], [100.0], [101.0], [102.0]])
    pf = _hand_portfolio(s, [10.0], [4], [[-1], [0], [0], [0]], ["A"])
    w = np.full((4, 1), np.nan)
    w[0] = {"lots": 5.0, "notional": 0.05, "margin": 0.005}[kind]
    r = pf.run(w, kind)
    # 保证金目标 = 名义 / 保证金率（0.005 / 0.1 在浮点里略小于 0.05，取整后 49 手）。
    expect = {"lots": 5.0, "notional": 50.0, "margin": np.trunc(0.005 / 0.1 * 1e3)}[
        kind
    ]
    assert r.positions[:, 0].tolist() == [0.0, expect, expect, expect]


@pytest.mark.parametrize("kind", ["lots", "notional", "margin"])
def test_round_trip_without_rolls_equals_book(kind):
    # 没有换月时，逐行推进与直接把同一份合约层面目标交给引擎逐位相同。
    rng = np.random.default_rng(3)
    t = 40
    s = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (t, 2)), axis=0))
    pf = _hand_portfolio(s, [10.0, 5.0], [t, t], np.tile([0, 1], (t, 1)), ["A", "B"])
    w = np.full((t, 2), np.nan)
    scale = {"lots": 20.0, "notional": 0.6, "margin": 0.06}[kind]
    w[::7] = np.round(rng.normal(0, 1, (len(w[::7]), 2)), 2) * scale
    if kind == "lots":
        w = np.trunc(w)
    a = pf.run(w, kind)
    b = bt.run(pf.book.market, pf.book.targets(w, kind), pf.book.costs, pf.book.config)
    assert a.equity.tobytes() == b.equity.tobytes()
    assert a.positions.tobytes() == b.positions.tobytes()


@pytest.mark.parametrize("kind", ["lots", "notional", "margin"])
def test_roll_row_moves_only_the_rolling_product(kind):
    # 两个品种，A 在第 5 行从列 0 换到列 1；B（列 2）在这一行不成交，A 平旧开新保持手数。
    t = 10
    rows = np.arange(t)[:, None]
    s = np.column_stack(
        [100 + rows[:, 0], 103 + 2 * rows[:, 0], 50 + 0.5 * np.sin(rows[:, 0])]
    )
    held = np.array([[0, 2]] * 5 + [[1, 2]] * 5)
    pf = _hand_portfolio(s, [10.0, 10.0, 20.0], [6, t, t], held, ["A", "B"])
    w = np.full((t, 2), np.nan)
    w[0] = {"lots": [4.0, -3.0], "notional": [0.3, -0.2], "margin": [0.03, -0.02]}[kind]
    r = pf.run(w, kind)
    fills = [(x.day, x.col, x.qty) for x in r.trades]
    q0, q2 = r.positions[0, 0], r.positions[0, 2]
    assert q0 > 0 and q2 < 0
    assert sorted(fills) == sorted([(0, 0, q0), (0, 2, q2), (5, 0, -q0), (5, 1, q0)])
    assert (r.positions[5:, 2] == q2).all()


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


def _tiny_table(last_trade_date):
    days = ["20200102", "20200103", "20200106", "20200107"]
    daily = pl.DataFrame(
        dict(
            date=days,
            symbol=["A"] * 4,
            settlement=[50.0, 51.0, 52.0, 53.0],
            close=[50.0, 51.0, 52.0, 53.0],
            open=[50.0, 51.0, 52.0, 53.0],
        )
    )
    contracts = pl.DataFrame(
        dict(symbol=["A"], multiplier=[1000.0], last_trade_date=[last_trade_date])
    )
    return days, daily, contracts


def test_book_from_table_settles_on_the_last_sample_day():
    # 最后交易日正好是样本最后一天：那一行按结算价了结、收费（不能当成「在样本之后」）。
    days, daily, contracts = _tiny_table("20200107")
    ext = bf.book_from_table(daily, contracts, margin_rate=0.1, fee_per_contract=1.0)
    assert ext.expiry.tolist() == [3]
    r = ext.book.run(np.array([[2.0], [np.nan], [np.nan], [np.nan]]), "lots")
    assert [(e.day, e.qty, e.settlement) for e in r.expiry_events] == [(3, 2.0, 53.0)]
    assert r.other_fee[3] == 2.0 and r.futures_margin[3] == 0
    # 晚于样本的最后交易日不了结。
    _, daily, contracts = _tiny_table("20200108")
    ext = bf.book_from_table(daily, contracts, margin_rate=0.1)
    assert ext.expiry.tolist() == [4]


def test_book_from_table_margin_takes_the_larger_side():
    days, daily, contracts = _tiny_table("20200107")
    params = pl.DataFrame(
        dict(
            date=days,
            symbol=["A"] * 4,
            margin_long=[0.10, 0.12, 0.08, 0.10],
            margin_short=[0.11, 0.09, 0.15, 0.10],
        )
    )
    ext = bf.book_from_table(daily, contracts, params=params)
    assert ext.book.margin_rate[:, 0].tolist() == [0.11, 0.12, 0.15, 0.10]


class _FakePanel:
    def __init__(self, dates, symbols, arrays):
        self.dates, self.symbols, self._arrays = dates, symbols, arrays

    def __getitem__(self, name):
        return self._arrays[name]


class _FakeFutures:
    """只有 ``_held`` 用到的几个公开入口。"""

    def __init__(self, panel, held, last):
        self._panel, self._held, self._last = panel, held, last

    def held_columns(self, product, **kw):
        return self._panel, self._held

    def contracts(self, products=None):
        return pl.LazyFrame(
            dict(symbol=list(self._last), last_trade_date=list(self._last.values()))
        )

    @staticmethod
    def adjust_prices(held, price, mode):
        from ashare.data.futures_continuous import adjust_prices

        return adjust_prices(held, price, mode)


def test_personal_override_does_not_look_at_the_switch_row():
    # A 第 2 行之后不能持有；B 第 3 行才有第一条行情，C 一直有。第 3 行换月只能看第 2 行的行情：
    # 选 C，不能因为 B 当行有了行情就选 B（那是前视）。
    dates = np.array([f"2024010{i}" for i in range(1, 7)])
    symbols = np.array(["A", "B", "C"])
    quote = np.ones((6, 3), dtype=bool)
    quote[:3, 1] = False
    panel = _FakePanel(
        dates, symbols, dict(has_quote=quote, settlement=np.full((6, 3), 100.0))
    )
    fake = _FakeFutures(
        panel,
        np.zeros(6, dtype=np.int64),
        dict(A="20240110", B="20240120", C="20240130"),
    )
    limit = dict(A=2, B=10, C=10)
    names, _ = bf._held(fake, "X", "open_interest", None, None, dates, limit.get)
    assert names.tolist() == ["A", "A", "A", "C", "C", "C"]


def test_roll_keeps_lots_across_different_multipliers():
    # 换月时新旧合约乘数不同（10 → 5）：保持「手数 × 乘数」，3 手 → 6 手。
    t = 6
    s = np.full((t, 2), 100.0)
    held = np.array([[0]] * 3 + [[1]] * 3)
    pf = _hand_portfolio(s, [10.0, 5.0], [4, t], held, ["A"])
    w = np.full((t, 1), np.nan)
    w[0] = 3.0
    r = pf.run(w, "lots")
    assert r.positions[:, 0].tolist() == [3.0, 3.0, 3.0, 0.0, 0.0, 0.0]
    assert r.positions[:, 1].tolist() == [0.0, 0.0, 0.0, 6.0, 6.0, 6.0]
