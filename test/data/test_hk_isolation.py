"""同一个数据根加入港股、美股的原始库与面板（含未完成、失败的分区）前后，A 股侧输出逐位不变。"""

import asyncio

from test_futures_isolation import snapshot
from test_hk_us_fixtures import DAYS, US_DAYS, FakeOverseas

from ashare.data.catalog import ALL_DATASETS, DATASETS, OVERSEAS_DATASETS
from ashare.data.overseas_panel import build_overseas
from ashare.data.overseas_sync import select_sample, sync_overseas
from ashare.data.panel import build
from ashare.data.store import Store
from ashare.data.sync import sync


def add_market(root, market, days):
    store = Store(root, market)
    source = FakeOverseas()
    first = [f"{market}_calendar", f"{market}_securities"]
    asyncio.run(sync_overseas(source, store, days[0], days[-1], first))
    asyncio.run(select_sample(source, store, days[-1]))
    asyncio.run(sync_overseas(source, store, days[0], days[-1]))
    build_overseas(store, "all")
    build_overseas(store, "sample")
    store.mark(
        f"{market}_daily/20260216",
        dict(
            dataset=f"{market}_daily", partition="20260216", params={}, status="pending"
        ),
    )
    store.mark(
        f"{market}_adjfactor/20260216",
        dict(
            dataset=f"{market}_adjfactor",
            partition="20260216",
            params={},
            status="failed",
            error="RuntimeError",
        ),
    )
    return store


def test_stock_outputs_identical_with_overseas_markets(tables, make_store, tmp_path):
    store = make_store(tables)
    build(store)
    root = tmp_path / "root"
    before = snapshot(store, root, tmp_path, "a")
    hk = add_market(root, "hk", DAYS)
    us = add_market(root, "us", US_DAYS)
    after = snapshot(store, root, tmp_path, "b")
    for key in before:
        assert before[key] == after[key], key
    # 港美股确实写进了同一个数据根的兄弟目录，A 股清单里没有它们的条目。
    assert hk.root.parent == us.root.parent == store.root.parent
    assert not [
        v for v in store.manifest().values() if v["dataset"] in OVERSEAS_DATASETS
    ]
    assert len(hk.entries()) > 50 and len(us.entries()) > 50
    assert hk.version != store.version != us.version


def test_catalog_keeps_stock_datasets_apart():
    assert not set(DATASETS) & set(OVERSEAS_DATASETS)
    assert set(OVERSEAS_DATASETS) <= set(ALL_DATASETS)
    assert all(n.startswith(("hk_", "us_", "overseas_")) for n in OVERSEAS_DATASETS)


def test_stock_sync_rejects_overseas_names(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="未知数据集"):
        asyncio.run(
            sync(
                FakeOverseas(),
                Store(tmp_path / "r"),
                "20260105",
                "20260110",
                ["hk_daily"],
            )
        )
