"""截断不变性：以 end=D 构建的面板与全量构建相比，D 及以前的格逐位相同。

例外只有 ``LOOKAHEAD_FIELDS`` 列出的字段，而且只在「这一列在 D 及以前最后一次成交之后」的格：那段
时间里下一个成交日还没到，事件执不执行、是否退市要看 D 之后的数据。其余字段（价格、成交、上市、港股通、
拆段、清算回收率……）不许依赖 D 之后。
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest
from test_hk_us_fixtures import DAYS, US_DAYS, FakeOverseas

import ashare
from ashare.data import overseas_panel as op
from ashare.data.overseas_panel import LOOKAHEAD_FIELDS, build_overseas
from ashare.data.overseas_sync import select_sample, sync_overseas
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def _store(tmp_path, market):
    days = DAYS if market == "hk" else US_DAYS
    store = Store(tmp_path / "root", market)
    source = FakeOverseas()
    basics = [f"{market}_calendar", f"{market}_securities"]
    asyncio.run(sync_overseas(source, store, days[0], days[-1], basics))
    asyncio.run(select_sample(source, store, days[-1]))
    asyncio.run(sync_overseas(source, store, days[0], days[-1]))
    return store


def _load(store, meta):
    path = store.root / "panel_all" / meta["version"]
    return {name: np.load(path / f"{name}.npy") for name in meta["fields"]}, meta[
        "symbols"
    ]


def _same(a, b):
    if a.dtype.kind == "f":
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


@pytest.mark.parametrize("market", ["hk", "us"])
def test_truncated_build_matches_full_up_to_d(tmp_path, monkeypatch, market):
    # 美股把拆段门槛调低，让夹具里的长缺口（GAP 跳涨、SHZ 暴跌）也拆段，一起核对。
    monkeypatch.setattr(op, "US_GAP_SPLIT", 10)
    store = _store(tmp_path, market)
    full_meta = build_overseas(store, "all", keep=5)
    full, full_symbols = _load(store, full_meta)
    days = full_meta["dates"]
    allowed_seen = set()
    for D in days[2:-1]:
        meta = build_overseas(store, "all", end=D, keep=5)
        cut, symbols = _load(store, meta)
        T = len(meta["dates"])
        assert meta["dates"] == days[:T] and days[T - 1] == D
        # 截到 D 时出现的每一列，全量里都有（拆段只在 D 及以前切开的那些）。
        assert set(symbols) <= set(full_symbols), D
        jf = [full_symbols.index(c) for c in symbols]
        trade = cut["tradable"]
        rows = np.arange(T)[:, None]
        last = np.where(trade, rows, -1).max(axis=0)
        open_tail = rows > last[None, :]
        for name in meta["fields"]:
            same = _same(full[name][:T][:, jf], cut[name])
            if name in LOOKAHEAD_FIELDS:
                bad = ~same & ~open_tail
                if (~same).any():
                    allowed_seen.add(name)
            else:
                bad = ~same
            where = np.argwhere(bad)[:5]
            assert not bad.any(), (D, name, [(days[i], symbols[j]) for i, j in where])
    # 夹具确实覆盖到了清单里的情形（否则这条测试什么也没证明）。
    want = {"action_source", "action_check", "action_suspect", "split", "adj_factor"}
    want |= {"equivalent_cash"} if market == "hk" else {"delist"}
    assert want <= allowed_seen, allowed_seen
