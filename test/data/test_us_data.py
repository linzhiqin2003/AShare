"""美股原始库、面板与校验（合成数据）。"""

import asyncio
import json
from pathlib import Path

import numpy as np
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
    reasons={"U1": "代码复用或源重复行"},
    cause_unknown=["U1"],
    entries=[
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
    # AAA 4 拆 1；EEE 10 合 1；DDD 非整数比 → 分不出。
    assert col(p, "action_source", "AAA")[12] == 1 and col(p, "split", "AAA")[12] == 4.0
    assert col(p, "action_source", "EEE")[18] == 1 and col(p, "split", "EEE")[
        18
    ] == pytest.approx(0.1)
    assert col(p, "action_source", "DDD")[15] == 2 and col(p, "split", "DDD")[15] == 1.0
    close = col(p, "close", "DDD")
    assert col(p, "equivalent_cash", "DDD")[15] == pytest.approx(
        close[14] * (1 - 1 / 1.07), rel=1e-5
    )
    # 美股前收不除权。
    assert col(p, "previous_close", "AAA")[12] == pytest.approx(
        col(p, "close", "AAA")[11]
    )
    # BBB 第 7 天两行值不同 → 整组丢弃；FFF 第 9 天两行相同 → 保留一行。
    assert not col(p, "has_quote", "BBB")[7] and col(p, "has_quote", "BBB")[6]
    assert col(p, "has_quote", "FFF")[9]
    assert meta["diagnostics"]["daily_conflicting_keys"] == 1
    assert meta["diagnostics"]["daily_merged_duplicate_rows"] == 2
    # CCC 第 4 天后没有行情 → 推断退市。
    assert col(p, "is_listed", "CCC").tolist() == [i <= 4 for i in range(n)]
    assert col(p, "delist", "CCC").tolist() == [i == 5 for i in range(n)]
    assert not col(p, "delist", "AAA").any()
    assert col(p, "total_market_cap", "AAA")[0] == pytest.approx(
        col(p, "close", "AAA")[0] * 1e9
    )
    sec = us.securities().collect()
    assert sec.filter(sec["symbol"] == "BBB")["code_reused_in_basic"][0]
    assert us.field("close")["currency"] == "USD"
    assert us.fx("USDCNH").collect().height == 30
    assert us.fx("CNYUSD").collect()["rate"][0] == pytest.approx(0.915)


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
    assert report["checks"]["factor_vs_previous_close"]["checked"] == 3
    # 拆股日价格没有相应跳变 → fail。
    import polars as pl
    from test_hk_data import _rewrite

    _rewrite(
        store,
        "us_daily",
        US_DAYS[12],
        lambda t: t.with_columns(
            close=pl.when(pl.col("ts_code") == "AAA")
            .then(pl.col("close") * 4)
            .otherwise(pl.col("close"))
        ),
    )
    bad = verify_overseas(store, exceptions_path=path)
    assert bad["checks"]["factor_vs_previous_close"]["verdict"] == "fail"


def test_us_sample_panel(tmp_path):
    store, _ = make(tmp_path)
    build_overseas(store, "sample")
    s = ad.open(tmp_path / "root", market="us", universe="sample")
    p = s.panel(["split", "close"], start=US_DAYS[0], symbols=["AAA", "AAPL"])
    assert col(p, "split", "AAA")[12] == 4.0
    assert np.isnan(col(p, "close", "AAPL")).all()  # 夹具没有锚定大票的行情


def test_dedupe_merges_same_prices_and_drops_conflicts():
    import polars as pl

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


def test_factor_one_series_per_symbol():
    """同一代码两个交易所两条因子序列（口径不同）：只用最后挂牌的非场外交易所那条。"""
    import polars as pl

    from ashare.data.overseas_panel import _dedupe_factor

    rows = []
    for day, nys, nas in (("1", 0.3269, 1 / 3), ("2", 0.9808, 1.0), ("3", 0.9808, 1.0)):
        rows.append(
            dict(ts_code="W", trade_date=day, cum_adjfactor=nys, exchange="NYS")
        )
        rows.append(
            dict(
                ts_code="W",
                trade_date=day,
                cum_adjfactor=nas,
                exchange="NAS" if day != "1" else "NAS",
            )
        )
    rows.append(dict(ts_code="O", trade_date="1", cum_adjfactor=2.0, exchange="OTC"))
    rows.append(
        dict(ts_code="O", trade_date="1", cum_adjfactor=2.0000001, exchange="OTC")
    )
    rows.append(dict(ts_code="X", trade_date="1", cum_adjfactor=1.0, exchange="OTC"))
    rows.append(dict(ts_code="X", trade_date="2", cum_adjfactor=3.0, exchange="ASE"))
    frame = pl.DataFrame(rows).with_columns(pl.col("trade_date").cast(pl.String))
    # 两个交易所天数相同：取字母序靠前的 NAS；行的先后顺序不影响结果。
    frame = frame.sort("trade_date", "exchange", descending=[False, True])
    other = _dedupe_factor(frame.reverse(), {}).sort("ts_code", "trade_date")
    diag = {}
    out = _dedupe_factor(frame, diag).sort("ts_code", "trade_date")
    w = out.filter(pl.col("ts_code") == "W")["cum_adjfactor"].to_list()
    assert w == pytest.approx([1 / 3, 1.0, 1.0])
    assert out.filter(pl.col("ts_code") == "O").height == 1
    assert out.filter(pl.col("ts_code") == "X")["trade_date"].to_list() == ["2"]
    assert other.equals(out)
    assert (
        diag["factor_symbols_with_two_exchanges"] == 2
        and diag["factor_conflicting_keys"] == 0
    )
