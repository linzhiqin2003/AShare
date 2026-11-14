"""数据层集成：原始库（Store.write）→ 面板构建（panel.build）→ 公开读取（ashare.data.open）。

边界逐个按生成器记下的位置断言面板里的值，而不是只看「能建出来」。
"""

import json

import numpy as np
import polars as pl
import pytest

import ashare.data
from ashare.data.catalog import INDEXES
from test.fixtures import synthetic_datadir as synthetic

FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "previous_close",
    "adj_factor",
    "upper_limit",
    "lower_limit",
    "is_listed",
    "listing_age",
    "delist",
    "is_suspended",
    "has_quote",
    "is_st",
    "can_buy_open",
    "can_sell_open",
    "split",
    "cash_dividend",
    "action_source",
    "industry",
    "member_csi300",
    "member_csi500",
    "weight_csi300",
    "total_market_cap",
    "vwap",
]


@pytest.fixture(scope="module")
def panel(datadir):
    return ashare.data.open(datadir.root).panel(FIELDS)


def column(datadir, panel, key):
    return list(panel.symbols).index(datadir.events[key]["symbol"])


def test_generator_is_deterministic():
    first, events, symbols, dates = synthetic.raw_tables()
    second, events2, symbols2, dates2 = synthetic.raw_tables()
    assert (events, symbols, dates) == (events2, symbols2, dates2)
    for name, table in first.items():
        assert table.equals(second[name]), name
    other, *_ = synthetic.raw_tables(seed=synthetic.SEED + 1)
    assert not other["daily"].equals(first["daily"])


def test_panel_shape_version_and_windows(datadir, panel):
    data = ashare.data.open(datadir.root)
    assert data.version == datadir.version
    assert list(panel.dates) == datadir.dates
    assert list(panel.symbols) == sorted(datadir.symbols)
    # 跨样本内、验证、TEST 三段，平台和分组回测各段都有数据。
    dates = np.asarray(panel.dates)
    assert (dates < "20250101").sum() > 100
    assert ((dates >= "20250101") & (dates < "20260101")).sum() > 60
    assert (dates >= "20260101").sum() > 3
    # 未构建的面板字段与未知字段给清楚的错误。
    with pytest.raises(KeyError):
        data.panel(["no_such_field"])


def test_suspension_blocks_trading_and_quotes(datadir, panel):
    j = column(datadir, panel, "suspended")
    rows = datadir.events["suspended"]["rows"]
    assert panel["is_suspended"][rows, j].all()
    assert not panel["has_quote"][rows, j].any()
    assert np.isnan(panel["close"][rows, j]).all()
    assert not panel["can_buy_open"][rows, j].any()
    assert not panel["can_sell_open"][rows, j].any()
    # 复牌当天：行情回来、可交易，前收是停牌前最后一个收盘价。
    after = rows[-1] + 1
    assert panel["has_quote"][after, j] and not panel["is_suspended"][after, j]
    assert panel["previous_close"][after, j] == panel["close"][rows[0] - 1, j]
    assert panel["is_listed"][rows, j].all()


def test_price_limits_and_open_masks(datadir, panel):
    up = column(datadir, panel, "limit_up_one_price")
    for row in datadir.events["limit_up_one_price"]["rows"]:
        assert panel["open"][row, up] == panel["upper_limit"][row, up]
        assert not panel["can_buy_open"][row, up]
        assert panel["can_sell_open"][row, up]
    down = column(datadir, panel, "limit_down_one_price")
    for row in datadir.events["limit_down_one_price"]["rows"]:
        assert panel["open"][row, down] == panel["lower_limit"][row, down]
        assert panel["can_buy_open"][row, down]
        assert not panel["can_sell_open"][row, down]
    close_up = column(datadir, panel, "limit_up")
    for row in datadir.events["limit_up"]["rows"]:
        assert panel["close"][row, close_up] == panel["upper_limit"][row, close_up]
        # 收在涨停但开盘没涨停：开盘可买。
        assert panel["can_buy_open"][row, close_up]
    # 主板 10%，主板 ST 5%，创业板（换码那只）20%。
    listed = panel["has_quote"] & np.isfinite(panel["previous_close"])
    ratio = np.where(listed, panel["upper_limit"] / panel["previous_close"], np.nan)
    st = column(datadir, panel, "st_always")
    chinext = list(panel.symbols).index(datadir.events["alias"]["new"])
    assert np.nanmax(np.abs(ratio[:, st] - 1.05)) < 0.01
    assert np.nanmax(np.abs(ratio[:, chinext] - 1.2)) < 0.01


def test_st_flags(datadir, panel):
    always = column(datadir, panel, "st_always")
    assert panel["is_st"][:, always].all()
    window = column(datadir, panel, "st_window")
    rows = datadir.events["st_window"]["rows"]
    assert panel["is_st"][rows, window].all()
    assert not panel["is_st"][: rows[0], window].any()
    assert not panel["is_st"][rows[-1] + 1 :, window].any()


def test_listing_and_delisting(datadir, panel):
    new = column(datadir, panel, "listed")
    row = datadir.events["listed"]["row"]
    assert not panel["is_listed"][:row, new].any()
    assert panel["is_listed"][row:, new].all()
    assert panel["listing_age"][row, new] == 1
    assert np.isnan(panel["close"][:row, new]).all()
    gone = column(datadir, panel, "delisted")
    row = datadir.events["delisted"]["row"]
    assert panel["is_listed"][:row, gone].all()
    assert not panel["is_listed"][row:, gone].any()
    assert panel["delist"][row, gone]
    assert panel["delist"][:, gone].sum() == 1


@pytest.mark.parametrize("key", ["split", "cash", "split_cash", "cash_validation"])
def test_corporate_actions_reconcile_with_adjustment_factor(datadir, panel, key):
    event = datadir.events[key]
    j, row = column(datadir, panel, key), event["row"]
    assert panel["split"][row, j] == pytest.approx(1 + event["stk_div"])
    assert panel["cash_dividend"][row, j] == pytest.approx(event["cash"])
    # 公告值与复权因子一致：来源 1（原始公告），没有落到「按因子比合成」。
    assert panel["action_source"][row, j] == 1
    # 其余日子没有事件。
    others = np.delete(np.arange(len(panel.dates)), row)
    assert (panel["split"][others, j][panel["is_listed"][others, j]] == 1).all()
    # 除权参考价 = (前收 − 派现) ÷ (1 + 送转)，两位小数；复权收益等于按参考价再投资的收益。
    previous = panel["close"][row - 1, j]
    reference = (previous - event["cash"]) / (1 + event["stk_div"])
    assert panel["previous_close"][row, j] == pytest.approx(reference, abs=0.006)
    adjusted = panel["close"][:, j] * panel["adj_factor"][:, j]
    assert adjusted[row] / adjusted[row - 1] == pytest.approx(
        panel["close"][row, j] / panel["previous_close"][row, j], rel=1e-12
    )


def test_industry_change_and_index_membership(datadir, panel):
    change = datadir.events["industry_change"]
    j, row = column(datadir, panel, "industry_change"), change["row"]
    before = int(synthetic.INDUSTRIES[change["before"]][0][:6])
    after = int(synthetic.INDUSTRIES[change["after"]][0][:6])
    assert (panel["industry"][:row, j] == before).all()
    assert (panel["industry"][row:, j] == after).all()
    swap = datadir.events["index_swap"]
    out = list(panel.symbols).index(swap["out"])
    into = list(panel.symbols).index(swap["into"])
    dates = np.asarray(panel.dates)
    # 2024-12 月末快照在下一个交易日起生效。
    snapshot = max(d for d in datadir.dates if d[:6] == swap["month"])
    effective = int(np.searchsorted(dates, snapshot, side="right"))
    assert panel["member_csi300"][:effective, out].all()
    assert not panel["member_csi300"][effective:, out].any()
    assert panel["member_csi300"][effective:, into].all()
    assert panel["member_csi500"][effective:, out].all()
    # 每天沪深300 的权重和为 1（百分数换算后）。
    sums = np.nansum(panel["weight_csi300"], axis=1)
    assert np.allclose(sums[sums > 0], 1.0)
    assert len(INDEXES) == 5


def test_symbol_alias_is_stitched_into_new_code(datadir, panel):
    alias = datadir.events["alias"]
    assert alias["old"] not in panel.symbols
    j = list(panel.symbols).index(alias["new"])
    dates = np.asarray(panel.dates)
    seam = int(np.searchsorted(dates, alias["switch"]))
    assert dates[seam - 1] == alias["old_last"]
    # 旧代码的历史并入新代码，接缝两边都有行情且参考价连续。
    assert panel["has_quote"][:, j].all()
    assert panel["previous_close"][seam, j] == panel["close"][seam - 1, j]
    # 新代码沿用原上市日：上市天数与同期老股票相同，不从换码日重新计。
    veteran = column(datadir, panel, "split")
    assert panel["listing_age"][seam, j] == panel["listing_age"][seam, veteran]


def test_public_tables_and_securities(datadir):
    data = ashare.data.open(datadir.root)
    securities = data.securities().collect()
    assert set(securities["symbol"]) == set(datadir.symbols)
    assert (
        securities.filter(pl.col("symbol") == datadir.events["delisted"]["symbol"])[
            "name"
        ]
        .item()
        .startswith("合成")
    )
    indexes = data.indexes()
    assert set(indexes["symbol"]) == set(synthetic.BENCHMARK_CODES)
    daily = data.table("index_daily", symbols=["000300.SH"]).collect()
    assert daily.height == len(datadir.dates)


def test_shared_datadir_is_read_only_and_reads_do_not_write(datadir):
    """2026-11-14：共享库建好后去掉写权限；读路径若偷偷写数据目录，这里和所有读测试都会报错。"""
    root = datadir.root
    files = [p for p in root.rglob("*") if p.is_file()]
    assert files and not any(p.stat().st_mode & 0o222 for p in files)
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    data = ashare.data.open(root)
    data.panel(["close", "industry"])
    data.table("index_classify").collect()
    data.indexes()
    after = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    assert before == after


def test_build_from_raw_is_bitwise_reproducible(datadir, tmp_path):
    """同一原始库重建：版本哈希相同、每个字段逐位相同（版本不含构建时间和路径）。"""
    from ashare.data.panel import build
    from ashare.data.store import Store

    copy = synthetic.copy_datadir(datadir, tmp_path / "copy", panel=False)
    metadata = build(Store(copy.root))
    assert metadata["version"] == datadir.version
    rebuilt = ashare.data.open(copy.root).panel(FIELDS)
    original = ashare.data.open(datadir.root).panel(FIELDS)
    for name in FIELDS:
        a, b = np.asarray(original[name]), np.asarray(rebuilt[name])
        assert a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), name
    assert str(tmp_path) not in json.dumps(metadata, default=str)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="产品问题（2026-11-14 发现）：数据层要求任何覆盖换码接缝日的原始库都有换码两边的行情",
)
def test_build_without_alias_symbols_across_seam(tmp_path):
    """2026-11-14，建合成库时发现：原始库覆盖 2025-02-14 / 02-17（catalog.SYMBOL_ALIASES 里
    300114.SZ→302132.SZ 的接缝）但库里根本没有这两个代码时，`panel.build` 报
    「换码 … 接缝 … 缺行情」而拒绝构建。`aliases.alias_context` 的注释说只在不覆盖接缝时
    跳过，没考虑「两边代码都不在库里」（部分证券的库、测试用的合成库）。两边都没有时
    不需要因子基准，应当跳过而不是报错。修好后去掉 xfail。"""
    try:
        info = synthetic.build_datadir(
            tmp_path / "data", dates=synthetic.calendar(step=6), with_alias=False
        )
    except ValueError as error:
        if "换码" not in str(error):  # 别的 ValueError 不算这个已知问题
            raise RuntimeError(error) from error
        raise
    assert synthetic.SYMBOL_ALIAS["new"] not in info.symbols
