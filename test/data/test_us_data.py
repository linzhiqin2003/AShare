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
    reasons={"U1": "代码复用或源重复行", "U2": "同一代码几个交易所的因子序列口径不同"},
    cause_unknown=["U1"],
    entries=[
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
    assert meta["diagnostics"]["segments_split"] == 1
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
    assert "GAP#2" in p.symbols and meta["diagnostics"]["segments_split"] == 3
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
