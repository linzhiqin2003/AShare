"""美股原始库、面板与校验（合成数据）。"""

import asyncio
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_hk_us_fixtures import US_DAYS, US_HOLIDAY, FakeOverseas

import ashare
import ashare.data as ad
from ashare.data.overseas_panel import build_overseas
from ashare.data.overseas_sync import select_sample, sync_overseas
from ashare.data.overseas_verify import verify_overseas
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

US_EXCEPTIONS = dict(
    reasons={
        "U1": "代码复用或源重复行",
        "U2": "同一代码几个交易所的因子序列口径不同",
        "U3": "没有交易印证的价格跳变",
    },
    cause_unknown=["U1"],
    entries=[
        dict(
            market="us",
            check="price_jumps",
            ts_code="VUP",
            trade_date=US_DAYS[14],
            kind="相邻",
            reason="U3",
        ),
        dict(
            market="us",
            check="price_jumps",
            ts_code="DRP",
            trade_date=US_DAYS[14],
            kind="相邻",
            reason="U3",
        ),
        dict(
            market="us",
            check="unique_keys",
            dataset="us_adjfactor",
            kind="取值冲突",
            reason="U2",
        ),
        dict(
            market="us",
            check="unique_keys",
            dataset="us_adjfactor",
            kind="同值重复",
            reason="U2",
        ),
        dict(
            market="us",
            check="unique_keys",
            dataset="us_daily",
            ts_code="BBB",
            trade_date=US_DAYS[7],
            reason="U1",
        ),
        dict(
            market="us",
            check="unique_keys",
            dataset="us_daily",
            ts_code="FFF",
            trade_date=US_DAYS[9],
            reason="U1",
        ),
    ],
)


def make(tmp_path):
    store = Store(tmp_path / "root", "us")
    source = FakeOverseas()
    asyncio.run(
        sync_overseas(
            source, store, US_DAYS[0], US_DAYS[-1], ["us_calendar", "us_securities"]
        )
    )
    asyncio.run(select_sample(source, store, US_DAYS[-1]))
    asyncio.run(sync_overseas(source, store, US_DAYS[0], US_DAYS[-1]))
    return store, source


def col(panel, name, code):
    return np.asarray(panel[name])[:, list(panel.symbols).index(code)]


def test_us_calendar_and_paging(tmp_path):
    store, source = make(tmp_path)
    days = sorted(
        {
            p["trade_date"]
            for api, p in source.calls
            if api == "us_daily" and "trade_date" in p
        }
    )
    assert days == US_DAYS and US_HOLIDAY not in days


def test_us_panel_events_duplicates_delist(tmp_path):
    store, _ = make(tmp_path)
    meta = build_overseas(store, "all")
    us = ad.open(tmp_path / "root", market="us")
    p = us.panel(list(us.fields()))
    n = len(US_DAYS)
    assert list(p.dates) == US_DAYS and "connect_eligible" not in us.fields()
    # 范围：优先股、第 5 位为 W 的权证、只有场外因子的证券不进面板。
    assert {"PRF", "WRNTW", "OTCX"}.isdisjoint(p.symbols)
    assert meta["diagnostics"]["us_scope"]["dropped_by_kind"] == 2
    assert meta["diagnostics"]["us_scope"]["without_major_exchange_factor"] == 1
    # AAA 4 拆 1；EEE 10 合 1；DDD 因子比 1.07（现金类）→ 美股不执行。
    assert col(p, "action_source", "AAA")[12] == 1 and col(p, "split", "AAA")[12] == 4.0
    assert col(p, "action_source", "EEE")[18] == 1
    assert col(p, "split", "EEE")[18] == pytest.approx(0.1)
    assert col(p, "action_source", "DDD")[15] == 5 and col(p, "split", "DDD")[15] == 1.0
    assert not col(p, "equivalent_cash", "DDD").any()
    assert col(p, "adj_factor", "DDD")[-1] == 1.0  # 价格收益口径
    # 美股前收不除权。
    assert col(p, "previous_close", "AAA")[12] == pytest.approx(
        col(p, "close", "AAA")[11]
    )
    # BBB 第 7 天两行值不同 → 整组丢弃；FFF 第 9 天两行相同 → 保留一行。
    assert not col(p, "has_quote", "BBB")[7] and col(p, "has_quote", "BBB")[6]
    assert col(p, "has_quote", "FFF")[9]
    assert meta["diagnostics"]["daily_conflicting_keys"] == 1
    assert meta["diagnostics"]["daily_merged_duplicate_rows"] == 2
    # CCC 第 4 天后没有成交：满 20 个交易日后的下一天（第 25 天）才不算上市、记退市。
    assert col(p, "is_listed", "CCC").tolist() == [i <= 24 for i in range(n)]
    assert col(p, "delist", "CCC").tolist() == [i == 25 for i in range(n)]
    assert not col(p, "delist", "AAA").any()
    # MRG：最后一次成交（第 10 天）之后因子 ×80 → 不执行（7）。
    assert col(p, "action_source", "MRG")[11] == 7
    assert (col(p, "split", "MRG") == 1).all() and col(p, "action_suspect", "MRG")[11]
    assert not col(p, "tradable", "MRG")[12] and col(p, "has_quote", "MRG")[12]
    # TOP：NAS 序列到第 8 天，之后只有 ASE 序列，第 9 天 ×10 而价格不动 → 6。
    assert col(p, "action_source", "TOP")[9] == 6
    assert (col(p, "adj_factor", "TOP") == 1).all()
    # ZZZ：因子 1 合 5 而价格不动 → 6。
    assert col(p, "action_source", "ZZZ")[6] == 6 and col(p, "split", "ZZZ")[6] == 1
    # CZO：第 8 天价格 ×80、因子只有 1.097 → 不执行（6），并从第 8 天起拆成 CZO#2。
    assert col(p, "action_source", "CZO#2")[8] == 6
    assert col(p, "has_quote", "CZO").tolist() == [i < 8 for i in range(n)]
    assert col(p, "has_quote", "CZO#2").tolist() == [i >= 8 for i in range(n)]
    # RSQ：第 14 天价格 ×20、成交股数没放大（疑似漏记的合股）→ 拆成 RSQ#2，旧列在那天退市，
    # 清算价取 min(上一成交日收盘, 当天收盘) = 上一成交日收盘（回收率 1，涨幅不给）。
    # AUP：价格 ×10、成交额 ×9 但成交股数没放大（活跃的一天加上漏记的合股）→ 同样拆成 AUP#2。
    assert meta["diagnostics"]["segments_split"] == 3
    assert sorted(meta["diagnostics"]["quiet_jump_cuts"]) == [
        ["AUP", US_DAYS[14]],
        ["RSQ", US_DAYS[14]],
    ]
    assert col(p, "has_quote", "AUP#2").tolist() == [i >= 14 for i in range(n)]
    assert col(p, "delist_recovery", "AUP")[14] == 1.0
    assert col(p, "has_quote", "RSQ#2").tolist() == [i >= 14 for i in range(n)]
    assert col(p, "delist", "RSQ").tolist() == [i == 14 for i in range(n)]
    assert col(p, "delist_recovery", "RSQ")[14] == 1.0
    # PMP 的拉升成交股数 ×50（真实行情）、DRP 的下跳、VUP（成交股数 ×3）、EEE 合股后的真实拉升（基准换算成
    # 新单位后 6.7 倍）、CZO#2 第 12 天的拉升（整个代码的基准下 12 倍）：都不拆列。
    assert {"PMP#2", "DRP#2", "VUP#2", "EEE#2", "CZO#3"}.isdisjoint(p.symbols)
    # STL：因子停了以后只有成交量为 0 的陈旧行，不算无因子成交日，不触发无因子退市。
    assert col(p, "has_quote", "STL").all()
    assert col(p, "tradable", "STL").tolist() == [i <= 10 for i in range(n)]
    assert col(p, "is_listed", "STL").all() and not col(p, "delist", "STL").any()
    # 新旧两列不同时上市：旧列从新列开始那天起为假，并在那天退市；价格是向上跳，清算不打折。
    assert col(p, "is_listed", "CZO").tolist() == [i < 8 for i in range(n)]
    assert col(p, "delist", "CZO").tolist() == [i == 8 for i in range(n)]
    assert col(p, "delist_recovery", "CZO")[8] == 1.0
    assert np.isnan(np.delete(col(p, "delist_recovery", "CZO"), 8)).all()
    assert not (col(p, "is_listed", "CZO") & col(p, "is_listed", "CZO#2")).any()
    # SPX：小窗口里因子 ×0.85 而价格不动 → 不执行（5）；STK 6 拆 5 照常；SMC 1.15 价格印证 → 4。
    assert col(p, "action_source", "SPX")[10] == 5 and col(p, "split", "SPX")[10] == 1
    assert col(p, "action_source", "STK")[14] == 1
    assert col(p, "split", "STK")[14] == pytest.approx(1.2)
    assert col(p, "action_source", "SMC")[16] == 4
    assert col(p, "action_check", "SMC")[16] == 2
    assert (
        col(p, "action_check", "AAA")[12] == 2
        and col(p, "action_check", "AAA")[11] == 0
    )
    assert (
        col(p, "action_check", "MRG")[11] == 6 and col(p, "action_check", "ZZZ")[6] == 5
    )
    # ORF：因子第 15 天后停了（转到场外）。第 16 天起有成交也不可交易；第 20 天（第 5 个没有因子的
    # 成交日）退市，之后不再上市；回收率 = min(1, 当天收盘 / 最后一个可交易日收盘)。第 17 天的错价
    # 不进入可交易的价格。
    assert col(p, "tradable", "ORF").tolist() == [i <= 15 for i in range(n)]
    assert col(p, "has_quote", "ORF").all()
    assert col(p, "delist", "ORF").tolist() == [i == 20 for i in range(n)]
    assert col(p, "is_listed", "ORF").tolist() == [i < 20 for i in range(n)]
    # 引擎在第 20 天按记账价（最后一个有行情的收盘，第 19 天）× 回收率清算，结果等于
    # min(最后一个可交易日收盘, 当天收盘)。
    orf = col(p, "close", "ORF")
    paid = col(p, "delist_recovery", "ORF")[20] * orf[19]
    assert paid == pytest.approx(min(orf[15], orf[20]))
    # HOL：因子偶发缺 3 天（第 10–12 天）：那 3 天不可交易，但不退市。
    assert col(p, "tradable", "HOL").tolist() == [not 10 <= i <= 12 for i in range(n)]
    assert col(p, "is_listed", "HOL").all() and not col(p, "delist", "HOL").any()
    assert meta["diagnostics"]["orphan_delists"] == 1
    # TRF：转板，拼接处（第 15 天）不算事件；NYS 序列里的 2 拆 1 照常执行。
    trf = col(p, "action_source", "TRF")
    assert trf[15] == 0 and trf[20] == 1 and col(p, "split", "TRF")[20] == 2.0
    # WMX：第 22 天前取 NYS（含派息，美股不执行），之后取 NAS，换序列那天不算事件。
    wmx = col(p, "action_source", "WMX")
    assert wmx[10] == 5 and (wmx[np.arange(n) != 10] == 0).all()
    sec = us.securities().collect()
    row = {r["symbol"]: r for r in sec.iter_rows(named=True)}
    assert row["TRF"]["exchange"] == "NYS" and row["TRF"]["exchanges"] == "NAS,NYS"
    assert row["WMX"]["factor_includes_cash"] == "部分"
    assert row["AAA"]["factor_includes_cash"] == "未知"
    assert row["DDD"]["factor_includes_cash"] == "含"
    assert row["MRG"]["action_suspect"] and not row["AAA"]["action_suspect"]
    assert row["TOP"]["action_suspect_dates"] == US_DAYS[9]
    assert row["BBB"]["code_reused_in_basic"]
    assert col(p, "total_market_cap", "AAA")[0] == pytest.approx(
        col(p, "close", "AAA")[0] * 1e9
    )
    assert us.field("close")["currency"] == "USD"
    assert us.fx("USDCNH").collect().height == 30
    assert us.fx("CNYUSD").collect()["rate"][0] == pytest.approx(0.915)


def test_us_long_gap_split_into_segments(tmp_path, monkeypatch):
    """GAP 第 6–20 天没有行情、复牌后价格 ×50 而因子不变：缺口长于门槛时拆成 GAP 与 GAP#2。

    SHZ 同样的缺口，复牌后价格只剩 0.1 倍：旧列在复牌日退市，按复牌价清算（回收率约 0.1），不让
    回测按停牌前的价格了结、躲过暴跌。
    """
    from ashare.data import overseas_panel as op

    monkeypatch.setattr(op, "US_GAP_SPLIT", 10)
    store, _ = make(tmp_path)
    meta = build_overseas(store, "all")
    us = ad.open(tmp_path / "root", market="us")
    p = us.panel(["has_quote", "is_listed", "delist", "close", "delist_recovery"])
    n = len(US_DAYS)
    # GAP、SHZ 两处长缺口，加上 CZO、RSQ、AUP 原有的三处。
    assert "GAP#2" in p.symbols and meta["diagnostics"]["segments_split"] == 5
    assert col(p, "has_quote", "GAP").tolist() == [i <= 5 for i in range(n)]
    assert col(p, "has_quote", "GAP#2").tolist() == [i >= 21 for i in range(n)]
    assert col(p, "is_listed", "GAP").tolist() == [i <= 20 for i in range(n)]
    assert col(p, "delist", "GAP").tolist() == [i == 21 for i in range(n)]
    assert col(p, "is_listed", "GAP#2").tolist() == [i >= 21 for i in range(n)]
    assert col(p, "delist_recovery", "GAP")[21] == 1.0
    shz = col(p, "close", "SHZ#2")[21] / col(p, "close", "SHZ")[5]
    assert shz == pytest.approx(0.1 * 1.005**16, rel=1e-3)
    assert col(p, "delist", "SHZ").tolist() == [i == 21 for i in range(n)]
    assert col(p, "delist_recovery", "SHZ")[21] == pytest.approx(shz)
    sec = us.securities().collect()
    assert sec.filter(pl.col("symbol") == "GAP#2")["segment"][0] == 2
    # 缺口不够长（默认 60 个交易日）就不拆，缺口里满 20 天后也不算上市。
    monkeypatch.setattr(op, "US_GAP_SPLIT", 60)
    build_overseas(store, "all")
    q = ad.open(tmp_path / "root", market="us").panel(["is_listed"])
    assert "GAP#2" not in q.symbols
    assert col(q, "is_listed", "GAP").all()  # 缺口 15 天，不到 20 天


def test_us_verify(tmp_path):
    store, _ = make(tmp_path)
    build_overseas(store, "all")
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(US_EXCEPTIONS))
    report = verify_overseas(store, exceptions_path=path)
    assert report["verdict"] == "pass", {
        k: v for k, v in report["checks"].items() if v["verdict"] == "fail"
    }
    assert report["checks"]["unique_keys"]["verdict"] == "explained"
    assert report["known_exceptions"]["cause_unknown"] == 2
    # 价格跳变：上跳看成交股数、下跳看成交额。GAP（×50）复牌日和 PMP 的拉升成交股数、SHZ（×0.1）复牌日
    # 的成交额都在前 10 日中位数 5 倍以上，自动归为真实行情；DRP 的 ÷3.5（成交额随价格缩小）、VUP（成交股数
    # 只 ×3）必须逐条登记（清单里有）。ORF 第 17 天的错价不可交易，不算；RSQ、AUP 已拆列，不在列内。
    jumps = report["checks"]["price_jumps"]
    assert jumps["verdict"] == "explained" and jumps["explained"] == 2
    assert {
        (e["ts_code"], e["trade_date"], e["kind"]) for e in jumps["explained_examples"]
    } == {("DRP", US_DAYS[14], "相邻"), ("VUP", US_DAYS[14], "相邻")}
    auto = report["checks"]["price_jumps_confirmed"]
    assert auto["verdict"] == "info"
    # EEE 第 21 天：基准里有第 18 天 10 合 1 之前的旧单位股数，换算成新单位后才是 6.7 倍；CZO#2 第 12 天：
    # 基准按整个代码取（含 CZO 那一列的低成交日）是 12 倍，只按 CZO#2 一列取是 1.2 倍。
    assert {(e["ts_code"], e["trade_date"]) for e in auto["examples"]} == {
        ("GAP", US_DAYS[21]),
        ("SHZ", US_DAYS[21]),
        ("PMP", US_DAYS[14]),
        ("EEE", US_DAYS[21]),
        ("CZO#2", US_DAYS[12]),
    }
    # 清单里去掉 DRP、VUP 这两条：跳变检查本身判 fail。
    bare = dict(
        US_EXCEPTIONS,
        entries=[e for e in US_EXCEPTIONS["entries"] if e["check"] != "price_jumps"],
    )
    bare_path = tmp_path / "exceptions_bare.json"
    bare_path.write_text(json.dumps(bare, ensure_ascii=False))
    unexplained = verify_overseas(store, exceptions_path=bare_path)["checks"]
    assert unexplained["price_jumps"]["verdict"] == "fail"
    assert unexplained["price_jumps"]["failed"] == 2
    # 源里因子与价格矛盾的事件只列为信息（面板必须不执行它们）。
    info = report["checks"]["factor_vs_previous_close"]
    assert info["verdict"] == "info"
    assert {e["ts_code"] for e in info["examples"]} == {"TOP", "ZZZ", "CZO"}
    # 面板被改成执行 TOP 的伪拆股 → 持有者价值、因子账目都对不上 → fail。
    current = (
        store.root
        / "panel_all"
        / json.loads((store.root / "panel_all/current.json").read_text())["version"]
    )
    meta = json.loads((current / "metadata.json").read_text())
    j = meta["symbols"].index("TOP")
    for name, value in (
        ("split", 10.0),
        ("action_source", 1),
        ("action_suspect", False),
    ):
        array = np.load(current / f"{name}.npy")
        array[9, j] = value
        np.save(current / f"{name}.npy", array)
    bad = verify_overseas(store, exceptions_path=path)
    assert bad["checks"]["panel"]["verdict"] == "fail"
    problems = {e.get("problem", "") for e in bad["checks"]["panel"]["examples"]}
    assert any("持有者价值" in x for x in problems)


def test_listing_us_orphan_delist_not_repeated():
    """无因子退市只记一次：退市之后再没有带因子的成交时，不再按「最后一次成交后满 20 天」另记一次。"""
    from ashare.data.overseas_panel import _listing_us

    T = 40
    tradable = np.zeros((T, 2), bool)
    orphan = np.zeros((T, 2), bool)
    tradable[:10, :] = True
    orphan[10:15, 0] = True  # 第 0 列：第 10–14 天连续 5 个无因子成交日 → 第 14 天退市
    close = np.full((T, 2), 10.0)
    close[14, 0] = 4.0
    listed, delist, recovery, gone = _listing_us(
        tradable, orphan=orphan, close=close, multiplier=np.ones((T, 2))
    )
    assert gone == 1
    assert delist[:, 0].tolist() == [t == 14 for t in range(T)]
    assert listed[:, 0].tolist() == [t < 14 for t in range(T)]
    assert recovery[14, 0] == pytest.approx(0.4)
    assert np.isnan(np.delete(recovery[:, 0], 14)).all()
    # 对照列：没有无因子成交，照常在最后一次成交（第 9 天）后满 20 天的下一天（第 30 天）退市。
    assert delist[:, 1].tolist() == [t == 30 for t in range(T)]
    assert listed[:, 1].tolist() == [t <= 29 for t in range(T)]
    # 无因子退市后第 20 天又有带因子的成交：重新上市，之后按 20 天规则在第 41 天再退市一次。
    T = 50
    again = np.zeros((T, 1), bool)
    again[:10, 0] = again[20, 0] = True
    lone = np.zeros((T, 1), bool)
    lone[10:15, 0] = True
    listed, delist, _, gone = _listing_us(
        again, orphan=lone, close=np.full((T, 1), 10.0), multiplier=np.ones((T, 1))
    )
    assert gone == 1
    assert delist[:, 0].tolist() == [t in (14, 41) for t in range(T)]
    assert listed[:, 0].tolist() == [t < 14 or 20 <= t <= 40 for t in range(T)]


def test_us_version_key_and_code_version(tmp_path, monkeypatch):
    """元数据记的 code_version 是源码哈希（不是某个代码名）；无因子退市的天数进版本键，改了就换代。"""
    from ashare.data import overseas_panel as op

    store, _ = make(tmp_path)
    first = build_overseas(store, "all")
    assert first["code_version"] == op.code_version()
    monkeypatch.setattr(op, "US_ORPHAN_DAYS", 3)
    second = build_overseas(store, "all")
    assert second["version"] != first["version"]
    # 换代后的面板确实按 3 天建：ORF 第 18 天（第 3 个没有因子的成交日）退市。
    p = ad.open(tmp_path / "root", market="us").panel(["delist"])
    assert col(p, "delist", "ORF").tolist() == [i == 18 for i in range(len(US_DAYS))]


def test_us_sample_panel(tmp_path):
    store, _ = make(tmp_path)
    meta = build_overseas(store, "sample")
    s = ad.open(tmp_path / "root", market="us", universe="sample")
    p = s.panel(["split", "close"], start=US_DAYS[0], symbols=["AAA"])
    assert col(p, "split", "AAA")[12] == 4.0
    # 夹具没有锚定大票的行情与因子：不进面板，元数据列出。
    assert "AAPL" in meta["diagnostics"]["sample_symbols_not_in_panel"]


def test_dedupe_merges_same_prices_and_drops_conflicts():
    from ashare.data.overseas_panel import CORE_DAILY, _dedupe

    frame = pl.DataFrame(
        dict(
            ts_code=["A", "A", "B", "B", "C", "Z"],
            trade_date=["1"] * 6,
            close=[2.0, 2.0, 3.0, 3.5, 4.0, 0.0],
            open=[1.0] * 6,
            high=[1.0] * 6,
            low=[1.0] * 6,
            pre_close=[1.0] * 6,
            vol=[5, 5, 1, 1, 1, 0],
            amount=[1.0] * 6,
            total_mv=[None, 9.0, 1.0, 1.0, 1.0, None],
        )
    )
    diag = {}
    out = _dedupe(frame, ["ts_code", "trade_date"], diag, "daily", CORE_DAILY).sort(
        "ts_code"
    )
    assert out["ts_code"].to_list() == ["A", "C", "Z"]
    assert out.filter(pl.col("ts_code") == "A")["total_mv"][0] == 9.0  # 取非空的估值
    assert (
        diag["daily_conflicting_keys"] == 1 and diag["daily_merged_duplicate_rows"] == 2
    )


def test_choose_series_rules():
    """每天取开始日最早的那条序列，一样早按交易所优先顺序（纽交所、纳斯达克、美交所）；不看序列哪天结束。
    比值只在同一序列内部算，且该序列上一行必须正好是上一个取用日，否则记 1（拼接）。行的先后顺序不影响
    结果；截到某一天再选，与全量选出的结果在那天及以前完全相同（不用以后的数据）。"""
    from ashare.data.overseas_panel import _choose_series

    rows = []

    def add(code, exchange, days, values):
        for d, v in zip(days, values, strict=True):
            rows.append(
                dict(
                    ts_code=code, exchange=exchange, trade_date=str(d), cum_adjfactor=v
                )
            )

    # W：NYS 到第 3 天（含分红），NAS 全程只含拆股，同一天开始 → 1–3 取 NYS（优先），4–5 取 NAS。
    add("W", "NYS", [1, 2, 3], [0.49, 0.5, 0.5])
    add("W", "NAS", [1, 2, 3, 4, 5], [0.5, 0.5, 0.5, 1.0, 1.0])
    # Y：NAS 先开始 → 全程取 NAS（开始日比交易所优先）。
    add("Y", "NYS", [2, 3, 4], [1.0, 2.0, 2.0])
    add("Y", "NAS", [1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0])
    # S：ASE 先开始，NYS 晚一天开始 → 全程取 ASE。
    add("S", "ASE", [1, 2, 3], [1.0, 1.0, 1.0])
    add("S", "NYS", [2, 3], [5.0, 5.0])
    # Z：转板，NAS 第 1–2 天、NYS 第 4–5 天：第 4 天拼接，不算事件。
    add("Z", "NAS", [1, 2], [0.2, 0.2])
    add("Z", "NYS", [4, 5], [1.0, 1.0])
    # K：NYS 只有第 1、3 天，NAS 全程：第 2 天只能取 NAS（比值按 NAS 第 1 天），第 3 天回到 NYS，
    #    NYS 上一行（第 1 天）不是上一个取用日（第 2 天）→ 拼接记 1，不把第 2 天的变化再算一遍。
    add("K", "NYS", [1, 3], [0.5, 1.0])
    add("K", "NAS", [1, 2, 3, 4, 5], [0.5, 1.0, 1.0, 1.0, 1.0])
    # X：场外不在允许的交易所里。
    add("X", "OTC", [1, 2], [1.0, 5.0])
    add("X", "ASE", [2, 3], [1.0, 1.0])
    frame = pl.DataFrame(rows)
    allowed = ("NYS", "NAS", "ASE")
    diag = {}
    out = _choose_series(frame, allowed, diag).sort("ts_code", "trade_date")
    shuffled = _choose_series(
        frame.sample(fraction=1.0, shuffle=True, seed=5), allowed, {}
    )
    assert shuffled.sort("ts_code", "trade_date").equals(out)
    for day in ("2", "3", "4"):
        cut = _choose_series(frame.filter(pl.col("trade_date") <= day), allowed, {})
        assert cut.sort("ts_code", "trade_date").equals(
            out.filter(pl.col("trade_date") <= day)
        ), day

    def pick(code):
        g = out.filter(pl.col("ts_code") == code)
        return g["exchange"].to_list(), g["ratio"].to_list()

    assert pick("W") == (
        ["NYS", "NYS", "NYS", "NAS", "NAS"],
        pytest.approx([1, 0.5 / 0.49, 1, 2, 1]),
    )
    assert pick("Y") == (["NAS"] * 4, [1, 1, 1, 1])
    assert pick("S") == (["ASE"] * 3, [1, 1, 1])
    assert pick("Z") == (["NAS", "NAS", "NYS", "NYS"], [1, 1, 1, 1])
    assert pick("K") == (["NYS", "NAS", "NYS", "NAS", "NAS"], [1, 2, 1, 1, 1])
    assert pick("X") == (["ASE", "ASE"], [1, 1])
    assert diag["factor_series_splices"] == 2  # Z 第 4 天、K 第 3 天


def _us_panel_path(store):
    base = store.root / "panel_all"
    return base / json.loads((base / "current.json").read_text())["version"]


def _set(path, field, code, day, value):
    meta = json.loads((path / "metadata.json").read_text())
    array = np.load(path / f"{field}.npy")
    array[day, meta["symbols"].index(code)] = value
    np.save(path / f"{field}.npy", array)


US_PANEL_FAULTS = {
    # 拆段旧列的清算回收率被改成 1（躲过暴跌）。
    "delist_recovery": lambda p: _set(p, "delist_recovery", "SHZ", 21, 1.0),
    # 新旧两列同一天都上市。
    "is_listed": lambda p: _set(p, "is_listed", "CZO", 8, True),
    # 美股记成「交易所前收印证」（美股前收不除权，不能用）。
    "action_check": lambda p: _set(p, "action_check", "AAA", 12, 1),
    # ORF 第 17 天有成交、没有三大交易所因子（错价那天）被记成可交易。
    "tradable": lambda p: _set(p, "tradable", "ORF", 17, True),
}


@pytest.mark.parametrize("field", sorted(US_PANEL_FAULTS))
def test_us_verify_panel_faults(tmp_path, monkeypatch, field):
    from ashare.data import overseas_panel as op

    monkeypatch.setattr(op, "US_GAP_SPLIT", 10)
    store, _ = make(tmp_path)
    build_overseas(store, "all")
    exc = tmp_path / "exc.json"
    exc.write_text(json.dumps(US_EXCEPTIONS, ensure_ascii=False))
    clean = verify_overseas(store, exceptions_path=exc)["checks"]["panel"]
    assert clean["verdict"] == "pass", clean["examples"]
    US_PANEL_FAULTS[field](_us_panel_path(store))
    panel = verify_overseas(store, exceptions_path=exc)["checks"]["panel"]
    assert panel["verdict"] == "fail"
    assert field in {e.get("field") for e in panel["examples"]}, panel["examples"]


def test_us_verify_catches_unsplit_quiet_jump(tmp_path, monkeypatch):
    """构建侧关掉「疑似漏记的合股拆段」时，校验独立重算出 RSQ、AUP 第 14 天应拆未拆；AUP 的成交额虽然
    放大了，也不能被算成交易印证的真实行情（上跳看成交股数）。"""
    from ashare.data import overseas_panel as op

    monkeypatch.setattr(op, "US_QUIET_JUMP", 1e9)
    store, _ = make(tmp_path)
    meta = build_overseas(store, "all")
    assert (
        "RSQ#2" not in meta["symbols"] and meta["diagnostics"]["quiet_jump_cuts"] == []
    )
    exc = tmp_path / "exc.json"
    exc.write_text(json.dumps(US_EXCEPTIONS, ensure_ascii=False))
    checks = verify_overseas(store, exceptions_path=exc)["checks"]
    panel = checks["panel"]
    assert panel["verdict"] == "fail"
    for code in ("RSQ", "AUP"):
        assert any(
            e.get("ts_code") == code
            and e.get("trade_date") == US_DAYS[14]
            and "漏记的合股" in e.get("problem", "")
            for e in panel["examples"]
        ), (code, panel["examples"])
    confirmed = {e["ts_code"] for e in checks["price_jumps_confirmed"]["examples"]}
    assert "AUP" not in confirmed and "PMP" in confirmed
    assert ("AUP", US_DAYS[14]) in {
        (e["ts_code"], e["trade_date"]) for e in checks["price_jumps"]["examples"]
    }


def test_quiet_split_volume_threshold():
    """构建侧疑似合股的成交股数门槛是前 10 个成交日中位数的 1.5 倍：上跳 ×4 时 1.4 倍拆、1.6 倍不拆；
    上跳不到 3 倍（×2.9）时成交股数再小也不拆。门槛写成字面倍数，改动 ``US_QUIET_VOLUME`` 会被抓到。
    基准里的股数按其后执行了的拆并股换算成跳变日的单位。"""
    from ashare.data import overseas_panel as op
    from ashare.data.overseas_actions import segments

    T = 30
    for jump, shares, want in ((4.0, 1.4, 2), (4.0, 1.6, 1), (2.9, 0.1, 1)):
        close = np.full((T, 1), 5.0)
        close[15:] *= jump
        volume = np.full((T, 1), 100.0)
        volume[15] = 100.0 * shares
        found = []
        parts = segments(
            close,
            np.ones((T, 1), bool),
            np.ones((T, 1)),
            np.zeros((T, 1), bool),
            op.US_GAP_SPLIT,
            volume=volume,
            quiet=(op.US_QUIET_JUMP, op.US_QUIET_VOLUME, op.US_QUIET_DAYS),
            found=found,
        )
        assert len(parts) == want, (jump, shares)
        assert found == ([(0, 15)] if want == 2 else []), (jump, shares)
    # 窗口里第 10 天执行了 1:100 的合股（multiplier 0.01，之前的股数按旧单位大 100 倍）、第 15 天真实拉升
    # （新单位下平常的 6 倍）：基准换算成新单位后不拆；不换算时旧单位抬高中位数，会误拆。
    close = np.full((T, 1), 5.0)
    close[10:] *= 100
    close[15:] *= 4
    multiplier = np.ones((T, 1))
    multiplier[10] = 0.01
    volume = np.where(np.arange(T)[:, None] < 10, 100.0, 1.0)
    volume[15] = 6.0
    found = []
    parts = segments(
        close,
        np.ones((T, 1), bool),
        multiplier,
        np.zeros((T, 1), bool),
        op.US_GAP_SPLIT,
        volume=volume,
        quiet=(op.US_QUIET_JUMP, op.US_QUIET_VOLUME, op.US_QUIET_DAYS),
        found=found,
    )
    assert len(parts) == 1 and found == []


def test_us_verify_jump_confirmation_thresholds(tmp_path):
    """校验的跳变门槛逐个贴着边界核：上跳只看成交股数（成交额放大 100 倍也不算印证），下跳只看成交额
    （成交股数放大 100 倍也不算印证），印证门槛都是前 10 个可交易日中位数的 5 倍；上跳超过 3 倍而成交股数
    不超过中位数 1.5 倍的要报「应拆未拆」。

    在面板里改 VUP（第 14 天 ×10）、DRP（第 14 天 ÷3.5）那天的成交股数和成交额，倍数取 4.8 / 5.2、
    1.4 / 1.6 这样贴着门槛的两侧。"""
    store, _ = make(tmp_path)
    build_overseas(store, "all")
    exc = tmp_path / "exc.json"
    exc.write_text(json.dumps(US_EXCEPTIONS, ensure_ascii=False))
    path = _us_panel_path(store)
    meta = json.loads((path / "metadata.json").read_text())
    volume0 = np.load(path / "volume.npy")
    amount0 = np.load(path / "amount.npy")
    b = 14
    base = {}
    for code in ("VUP", "DRP"):
        j = meta["symbols"].index(code)
        base[code] = (
            j,
            float(np.median(volume0[b - 10 : b, j])),
            float(np.median(amount0[b - 10 : b, j])),
        )
    # (VUP 股数倍数, VUP 成交额倍数, DRP 股数倍数, DRP 成交额倍数) → (VUP 归属, VUP 应拆未拆, DRP 归属)
    cases = [
        ((1.4, 100.0, 100.0, 4.8), ("jump", True, "jump")),
        ((1.6, 100.0, 0.5, 5.2), ("jump", False, "confirmed")),
        ((4.8, 100.0, 100.0, 4.8), ("jump", False, "jump")),
        ((5.2, 0.5, 0.5, 5.2), ("confirmed", False, "confirmed")),
    ]
    for (vs, va, ds, da), (vup_want, vup_quiet, drp_want) in cases:
        volume, amount = volume0.copy(), amount0.copy()
        for code, (s, a) in (("VUP", (vs, va)), ("DRP", (ds, da))):
            j, base_volume, base_amount = base[code]
            volume[b, j] = s * base_volume
            amount[b, j] = a * base_amount
        np.save(path / "volume.npy", volume)
        np.save(path / "amount.npy", amount)
        checks = verify_overseas(store, exceptions_path=exc)["checks"]
        confirmed = {
            (e["ts_code"], e["trade_date"])
            for e in checks["price_jumps_confirmed"]["examples"]
        }
        jumps = checks["price_jumps"]
        listed = {
            (e["ts_code"], e["trade_date"])
            for e in jumps["examples"] + jumps["explained_examples"]
        }
        for code, want in (("VUP", vup_want), ("DRP", drp_want)):
            key = (code, US_DAYS[b])
            got = "confirmed" if key in confirmed else "jump" if key in listed else None
            assert got == want, (code, vs, va, ds, da, got)
        quiet = any(
            e.get("ts_code") == "VUP"
            and e.get("trade_date") == US_DAYS[b]
            and "漏记的合股" in e.get("problem", "")
            for e in checks["panel"]["examples"]
        )
        assert quiet == vup_quiet, (vs, checks["panel"]["examples"])


def test_us_new_jump_without_amount_must_be_registered(tmp_path):
    """面板里一只正常证券某天收盘 ×10（成交额不变）：没有交易印证的新跳变，不会被已有登记解释。
    价格跳变的例外只能逐条（代码、日期）登记，按类别整体豁免的条目直接拒绝。"""
    from ashare.data.overseas_verify import load_exceptions

    store, _ = make(tmp_path)
    build_overseas(store, "all")
    exc = tmp_path / "exc.json"
    exc.write_text(json.dumps(US_EXCEPTIONS, ensure_ascii=False))
    path = _us_panel_path(store)
    meta = json.loads((path / "metadata.json").read_text())
    close = np.load(path / "close.npy")
    close[5, meta["symbols"].index("AAA")] *= 10
    np.save(path / "close.npy", close)
    jumps = verify_overseas(store, exceptions_path=exc)["checks"]["price_jumps"]
    assert jumps["verdict"] == "fail"
    assert {(e["ts_code"], e["trade_date"]) for e in jumps["examples"]} == {
        ("AAA", US_DAYS[5]),
        ("AAA", US_DAYS[6]),
    }
    wide = dict(
        US_EXCEPTIONS,
        entries=[dict(market="us", check="price_jumps", kind="相邻", reason="U3")],
    )
    exc.write_text(json.dumps(wide, ensure_ascii=False))
    with pytest.raises(ValueError, match="逐条"):
        load_exceptions("us", exc)


def test_listing_us_orphan_streak_and_grace():
    """无因子连续计数遇到有因子的成交日清零；已过 20 天宽限（已不算上市）后的无因子成交日不再触发退市。"""
    from ashare.data.overseas_panel import _listing_us

    T = 60
    close = np.full((T, 1), 10.0)
    ones = np.ones((T, 1))
    # 3 个无因子成交日、1 个可交易日、再 3 个无因子成交日：连续数最多 3，不退市。
    trade = np.zeros((T, 1), bool)
    lone = np.zeros((T, 1), bool)
    trade[:10, 0] = trade[13, 0] = True
    lone[10:13, 0] = lone[14:17, 0] = True
    listed, delist, _, gone = _listing_us(
        trade, orphan=lone, close=close, multiplier=ones
    )
    assert gone == 0
    assert delist[:, 0].tolist() == [t == 13 + 21 for t in range(T)]
    # 最后可交易日第 0 天，满 20 天后第 21 天退市；第 30–34 天才有 5 个无因子成交日：那时已不算上市，
    # 不再记一次无因子退市。
    trade = np.zeros((T, 1), bool)
    lone = np.zeros((T, 1), bool)
    trade[0, 0] = True
    lone[30:35, 0] = True
    listed, delist, recovery, gone = _listing_us(
        trade, orphan=lone, close=close, multiplier=ones
    )
    assert gone == 0
    assert delist[:, 0].tolist() == [t == 21 for t in range(T)]
    assert listed[:, 0].tolist() == [t <= 20 for t in range(T)]
    assert np.isnan(recovery).all()


def test_cut_recovery_counts_split_on_last_day():
    """清算倍数包含窗口最后一天执行的拆并股：上一成交日收盘 10，b 当天 1 拆 2、收盘 4 →
    持有者价值 2 × 4 = 8，清算价 min(10, 8) = 8，回收率 0.8（记账价是上一成交日收盘 10）。"""
    from ashare.data.overseas_actions import cut_recovery

    close = np.array([[9.0], [10.0], [4.0]])
    tradable = np.array([[True], [True], [True]])
    multiplier = np.array([[1.0], [1.0], [2.0]])
    assert cut_recovery(close, tradable, multiplier, 0, 2) == pytest.approx(0.8)
