"""数据层公开入口 ``FuturesData.held_columns`` / ``adjust_prices`` / ``panelbt_inputs``
与内部函数逐位相同（回测层只走这三个公开入口）。"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from futures_fixtures import DAYS, FakeTushare  # noqa: E402

import ashare.data as ad  # noqa: E402
from ashare.data.futures_continuous import adjust_prices, held_columns  # noqa: E402
from ashare.data.futures_panel import build_futures  # noqa: E402
from ashare.data.futures_panelbt import panelbt_inputs  # noqa: E402
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


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def test_held_columns_public_equals_internal(fut):
    for rule in ("open_interest", "volume"):
        for rank in (1, 2):
            p1, h1 = fut.held_columns("RB", rule=rule, rank=rank)
            p2, h2 = held_columns(fut, "RB", rule=rule, rank=rank)
            _same(h1, h2)
            _same(p1.dates, p2.dates)
            assert list(p1.symbols) == list(p2.symbols)
            _same(p1["settlement"], p2["settlement"])


def test_adjust_prices_public_equals_internal(fut):
    panel, held = held_columns(fut, "RB")
    price = np.asarray(panel["settlement"])
    for mode in ("none", "difference", "ratio"):
        r1, a1, k1 = fut.adjust_prices(held, price, mode)
        r2, a2, k2 = adjust_prices(held, price, mode)
        _same(r1, r2)
        _same(a1, a2)
        assert repr(k1) == repr(k2)
    assert type(fut).adjust_prices(held, price, "ratio")[1].tobytes() == a2.tobytes()


def test_panelbt_inputs_public_equals_internal(fut):
    kw = dict(
        products=["RB"],
        missing=dict(margin_rate=0.1, fee_rate=1e-4, fee_per_contract=0.0),
        multiplier_change="split",
    )
    a, b = fut.panelbt_inputs(**kw), panelbt_inputs(fut, **kw)
    assert a.keys() == b.keys()
    assert list(a["symbols"]) == list(b["symbols"])
    _same(a["dates"], b["dates"])
    assert a["splits"] == b["splits"]
    assert a["market"].keys() == b["market"].keys()
    for k in a["market"]:
        _same(a["market"][k], b["market"][k])
    for name in ("settlement", "multiplier", "expiry", "margin_rate", "fee_rate"):
        _same(getattr(a["futures"], name), getattr(b["futures"], name))
    assert a["contracts"].equals(b["contracts"])


def test_portfolio_across_unit_change_reads_bars_once(tmp_path):
    # 回归：拆列后的两列共用同一合约的行情，曾因重复代码读面板报错。
    from test_futures_rework import spec_store

    from ashare.backtest import futures as bf

    store = spec_store(tmp_path)
    build_futures(store)
    fut = ad.open(tmp_path).futures()
    pf = bf.futures_portfolio(
        fut,
        ["PB"],
        start="20130828",
        end="20130910",
        personal=False,
        infer_locked=False,
        missing=dict(margin_rate=0.1, fee_rate=0.0, fee_per_contract=3.0),
        config=bt.Config(initial_cash=1e6, record_trades=True),
    )
    assert pf.splits, "样本区间应跨过交易单位分界日"
    labels = list(pf.symbols)
    dates = list(pf.dates)
    o = np.asarray(pf.book.market.open)
    for split in pf.splits:
        a, b = labels.index(split["before"]), labels.index(split["after"])
        # 两列的开盘价都等于数据层这一合约自己的开盘价（独立读一次面板对照）。
        raw = fut.panel(
            ["open"], symbols=[split["symbol"]], start=dates[0], end=dates[-1]
        )
        expect = np.asarray(raw["open"])[:, 0]
        _same(o[:, a], expect)
        _same(o[:, b], expect)
        # 回测：分界日前持有 2 手（25 吨），分界日起按乘数折成 10 手（5 吨）。
        held = pf.held[:, 0]
        first = int(np.argmax(held >= 0))
        w = np.full((len(dates), 1), np.nan)
        w[first] = 2.0
        r = pf.run(w, "lots")
        cut = dates.index(split["date"])
        mult = pf.book.market.futures.multiplier
        assert mult[a] == 25.0 and mult[b] == 5.0
        assert r.positions[cut - 1, a] == 0  # 前一列在分界日前一行收盘了结
        assert r.positions[cut, b] == 10.0
