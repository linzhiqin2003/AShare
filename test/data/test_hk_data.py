"""港股原始库、面板、读取接口与校验（合成数据），以及两地共用的公司行为识别。"""

import asyncio
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_hk_us_fixtures import DAYS, FakeOverseas

import ashare
import ashare.data as ad
from ashare.data import overseas_panel as op
from ashare.data.overseas_actions import classify_ratio, corporate_actions
from ashare.data.overseas_panel import build_overseas
from ashare.data.overseas_sync import select_sample, stratified_pick, sync_overseas
from ashare.data.overseas_verify import load_exceptions, verify_overseas
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

HK_EXCEPTIONS = dict(
    reasons={"R1": "代码复用", "R2": "日线四舍五入"},
    cause_unknown=[],
    entries=[
        dict(
            market="hk",
            check="quotes_vs_listing",
            ts_code="00003.HK",
            side="上市日前有行情",
            reason="R1",
        ),
        dict(
            market="hk",
            check="factor_price",
            ts_code="08001.HK",
            trade_date=DAYS[5],
            reason="R2",
        ),
        # 第 5 天收盘被四舍五入，第 6 天的前收（精确值）对不上它。
        dict(
            market="hk",
            check="factor_vs_previous_close",
            ts_code="08001.HK",
            trade_date=DAYS[6],
            reason="R2",
        ),
    ],
)


def make(tmp_path, source=None, *, start=DAYS[0], end=DAYS[-1], **kwargs):
    store = Store(tmp_path / "root", "hk")
    source = source or FakeOverseas()
    asyncio.run(
        sync_overseas(source, store, start, end, ["hk_calendar", "hk_securities"])
    )
    asyncio.run(select_sample(source, store, end))
    outcome = asyncio.run(sync_overseas(source, store, start, end, **kwargs))
    return store, source, outcome


def exceptions_file(tmp_path, value=HK_EXCEPTIONS):
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(value, ensure_ascii=False))
    return path


def col(panel, name, code):
    return np.asarray(panel[name])[:, list(panel.symbols).index(code)]


# ---------------------------------------------------------------- 同步


def test_sync_plan_resume_and_snapshot(tmp_path):
    store, source, first = make(tmp_path)
    days = {
        d
        for api, p in source.calls
        if api == "hk_daily" and "trade_date" in p
        for d in [p["trade_date"]]
    }
    assert days == set(DAYS)
    connect = [p for api, p in source.calls if api == "stock_hsgt"]
    assert {p["type"] for p in connect} == {"SH_HK", "SZ_HK"} and len(
        connect
    ) == 2 * len(DAYS)
    source.calls.clear()
    second = asyncio.run(sync_overseas(source, store, DAYS[0], DAYS[-1]))
    daily_days = sorted(
        p["trade_date"]
        for api, p in source.calls
        if api == "hk_daily" and "trade_date" in p
    )
    assert daily_days == DAYS[-5:]  # 只重抓最近 5 个交易日
    factor_days = sorted(
        p["trade_date"]
        for api, p in source.calls
        if api == "hk_adjfactor" and "trade_date" in p
    )
    assert factor_days == DAYS  # 前复权因子整段重抓
    snaps = {e["snapshot"] for e in store.entries("hk_adjfactor").values()}
    snaps |= {e["snapshot"] for e in store.entries("hk_adjfactor_hist").values()}
    assert snaps == {second["snapshot"]} and second["snapshot"] != first["snapshot"]
    assert not [v for v in store.manifest().values() if v["status"] != "complete"]


def test_failed_partition_recorded_and_retried(tmp_path):
    source = FakeOverseas(fail={("hk_daily", DAYS[3])})
    with pytest.raises(RuntimeError):
        make(tmp_path, source)
    store = Store(tmp_path / "root", "hk")
    assert store.manifest()[f"hk_daily/{DAYS[3]}"]["status"] == "failed"
    asyncio.run(sync_overseas(source, store, DAYS[0], DAYS[-1]))
    assert store.manifest()[f"hk_daily/{DAYS[3]}"]["status"] == "complete"


def test_store_markets_and_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="尚不支持市场"):
        Store(tmp_path, "jp")
    store = Store(tmp_path / "root", "hk")
    assert store.root == (tmp_path / "root" / "hk").resolve()
    with pytest.raises(ValueError, match="没有数据集"):
        asyncio.run(
            sync_overseas(FakeOverseas(), store, DAYS[0], DAYS[-1], ["us_daily"])
        )
    with pytest.raises(ValueError, match="只支持 hk、us"):
        asyncio.run(
            sync_overseas(FakeOverseas(), Store(tmp_path / "root"), DAYS[0], DAYS[-1])
        )


# ---------------------------------------------------------------- 面板


def test_hk_panel_events_listing_connect(tmp_path):
    store, _, _ = make(tmp_path)
    meta = build_overseas(store, "all")
    hk = ad.open(tmp_path / "root", market="hk")
    p = hk.panel(list(hk.fields()))
    assert list(p.dates) == DAYS and hk.version == meta["version"]
    # 00001.HK：第 10 天 5 拆 1，第 20 天派 0.5。
    src = col(p, "action_source", "00001.HK")
    assert src[10] == 1 and col(p, "split", "00001.HK")[10] == 5.0
    assert src[20] == 2 and col(p, "split", "00001.HK")[20] == 1.0
    assert abs(col(p, "equivalent_cash", "00001.HK")[20] - 0.5) < 1e-3
    assert (src[[i for i in range(30) if i not in (10, 20)]] == 0).all()
    adj = col(p, "adj_factor", "00001.HK")
    assert adj[0] == 1.0 and abs(adj[10] / adj[9] - 5.0) < 1e-4  # 源因子只留 6 位小数
    # 00002.HK 第 15 天退市。
    listed = col(p, "is_listed", "00002.HK")
    assert listed[:15].all() and not listed[15:].any()
    assert col(p, "delist", "00002.HK").tolist() == [i == 15 for i in range(30)]
    # 00003.HK 代码复用：第 0 天的行情归到旧公司 00003!.HK。
    assert col(p, "has_quote", "00003!.HK").tolist() == [i == 0 for i in range(30)]
    assert col(p, "has_quote", "00003.HK").tolist() == [i >= 3 for i in range(30)]
    assert col(p, "is_listed", "00003.HK").tolist() == [i >= 3 for i in range(30)]
    assert col(p, "is_listed", "00003!.HK").tolist() == [i < 1 for i in range(30)]
    assert meta["diagnostics"]["daily_rows_reassigned_to_old_code"] == 1
    # 08001.HK 第 5 天收盘价改用因子表的精确值。
    assert col(p, "close", "08001.HK")[5] == pytest.approx(round(0.045 * 1.005**6, 3))
    assert meta["diagnostics"]["close_from_factor_table"] == 1
    # 00005.HK 停牌日没有日线。
    quote = col(p, "has_quote", "00005.HK")
    assert (
        not quote[8:13].any()
        and quote[:8].all()
        and not col(p, "tradable", "00005.HK")[8:13].any()
    )
    # 港股通：名单第 20 天起才有；00005.HK 第 25 天起调出、上一交易日南向有持股 → 只能卖出。
    known = col(p, "connect_known", "00001.HK")
    assert known.tolist() == [i >= 20 for i in range(30)]
    assert col(p, "connect_eligible", "00001.HK").tolist() == [
        i >= 20 for i in range(30)
    ]
    assert col(p, "connect_eligible", "00005.HK").tolist() == [
        20 <= i < 25 for i in range(30)
    ]
    assert col(p, "connect_sell_only", "00005.HK").tolist() == [
        25 <= i <= 28 for i in range(30)
    ]
    # 南向持股次日才公布：面板错位一天（第 27 天的持股在第 28 天可见）。
    assert col(p, "connect_held", "00005.HK").tolist() == [
        1 <= i <= 28 for i in range(30)
    ]
    # 00005.HK 停牌期内第 9 天 10 合 1、第 11 天派息（因子比 1.02）：等效现金按合股后的价位。
    src5 = col(p, "action_source", "00005.HK")
    assert src5[9] == 1 and col(p, "split", "00005.HK")[9] == pytest.approx(0.1)
    assert src5[11] == 2
    close7 = round(60 * 1.005**8, 3)
    assert col(p, "equivalent_cash", "00005.HK")[11] == pytest.approx(
        close7 / 0.1 * (1 - 1 / 1.02), rel=1e-5
    )
    # 00003.HK 第 20 天因子比 1.37：分不出、大比值 → 按股数调整，不给现金。
    assert col(p, "action_source", "00003.HK")[20] == 4
    assert col(p, "split", "00003.HK")[20] == pytest.approx(1.37, rel=1e-5)
    assert col(p, "equivalent_cash", "00003.HK")[20] == 0
    assert not np.asarray(p["action_suspect"]).any()
    # 08009.HK 第 12 天 20 合 1：开盘、收盘都只有隐含价位的 0.385 倍，前收已除权 → 以前收为准执行。
    assert col(p, "action_source", "08009.HK")[12] == 1
    assert col(p, "split", "08009.HK")[12] == pytest.approx(0.05)
    assert col(p, "action_check", "08009.HK")[12] == 1
    held = 0.05 * col(p, "close", "08009.HK")[12] / col(p, "close", "08009.HK")[11]
    assert held == pytest.approx(0.385 * 1.005, rel=5e-3)  # 价格只留 3 位小数
    # 00009.HK：前收取整后差略超 0.5%，靠半个价位放宽由前收印证；00010.HK：没有上一个成交日。
    assert col(p, "action_check", "00009.HK")[14] == 1
    assert col(p, "action_source", "00009.HK")[14] == 2
    assert col(p, "action_check", "00010.HK")[2] == 4
    p3 = col(p, "close", "00010.HK")[3]
    assert col(p, "equivalent_cash", "00010.HK")[2] == pytest.approx(
        p3 * 0.03, rel=1e-4
    )
    # 停牌窗口（00005.HK）复牌那天的前收已按合计比值调整：也以前收印证。
    assert col(p, "action_check", "00005.HK")[9] == 1
    assert col(p, "action_check", "00005.HK")[11] == 1
    sec = hk.securities().collect()
    assert set(sec["primary_symbol"]) == set(p.symbols)  # 没有人民币柜台
    assert set(sec["factor_includes_cash"]) == {"含"}
    assert not col(p, "price_rounded", "08001.HK").any()  # 2026 年不在源的四舍五入期


def test_precise_close_only_within_half_cent(tmp_path):
    daily = pl.DataFrame(
        dict(
            ts_code=["A", "A", "B"], trade_date=["1", "2", "1"], close=[0.05, 0.30, 1.0]
        )
    )
    factor = pl.DataFrame(
        dict(
            ts_code=["A", "A", "B"],
            trade_date=["1", "2", "1"],
            close_price=[0.045, 0.31, None],
        )
    )
    diag = {}
    out = op._precise_close(daily, factor, diag)
    assert out["close"].to_list() == [0.045, 0.30, 1.0]
    assert diag == dict(close_from_factor_table=1, close_mismatch_kept=1)


def test_price_rounded_period(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "HK_ROUNDED", (DAYS[0], DAYS[4]))
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    p = ad.open(tmp_path / "root", market="hk").panel(["price_rounded"])
    assert col(p, "price_rounded", "08001.HK").tolist() == [i <= 4 for i in range(30)]
    assert not col(p, "price_rounded", "00001.HK").any()


def test_snapshot_mismatch_rejected_then_full_refresh_fixes(tmp_path):
    source = FakeOverseas(asof=DAYS[15])
    store, _, _ = make(tmp_path, source, end=DAYS[15])
    source.asof = DAYS[-1]
    # 只补新日子、不重抓旧因子：两段来自不同快照，第 20 天的派息让旧段整体缩放。
    asyncio.run(sync_overseas(source, store, DAYS[0], DAYS[-1], refresh=False))
    with pytest.raises(ValueError, match="不同的同步快照"):
        build_overseas(store, "all")
    asyncio.run(sync_overseas(source, store, DAYS[0], DAYS[-1]))
    build_overseas(store, "all")
    p = ad.open(tmp_path / "root", market="hk").panel(["action_source", "split"])
    assert col(p, "action_source", "00001.HK")[16] == 0
    assert col(p, "split", "00001.HK")[10] == 5.0


def test_build_version_reused_and_sample_universe(tmp_path):
    store, _, _ = make(tmp_path)
    a = build_overseas(store, "all")
    b = build_overseas(store, "all")
    assert a["version"] == b["version"]
    sample = build_overseas(store, "sample")
    s = ad.open(tmp_path / "root", market="hk", universe="sample")
    assert set(s.panel("close").symbols) == set(
        json.loads((store.root / "sample.json").read_text())["symbols"]
    )
    assert sample["sample"]["anchors"][0] == "00700.HK"
    assert col(s.panel(["split"], start=DAYS[0]), "split", "00001.HK")[10] == 5.0


# ---------------------------------------------------------------- 公司行为：手算例子与独立参考实现


def reference_classify(r):
    """直白的参考：比值 ≥ 10 只试整数，否则枚举 q=1..10 取误差最小的 p/q。"""
    if not math.isfinite(r) or r <= 0 or 1 - 1e-5 <= r <= 1 + 1e-5:
        return 0, 1.0
    tol = 2e-4
    if r < 1:
        tol = max(tol, 6e-5 / r)
    big = r if r > 1 else 1 / r
    best = None
    if 1.05 <= big <= 1000:
        qs = [1] if big >= 10 else (range(1, 3) if big >= 3 else range(1, 11))
        for q in qs:
            p = round(big * q)
            if p <= q:
                continue
            err = abs(p / q - big) / big
            if best is None or err < best[0]:
                best = (err, Fraction(p, q))
    if best and big >= 10:
        allowed = max(2e-3, big * tol) if r < 1 else 2e-3
        if abs(float(best[1]) - big) <= allowed:
            return 1, float(best[1]) if r > 1 else float(1 / best[1])
    elif best and best[0] <= tol:
        return 1, float(best[1]) if r > 1 else float(1 / best[1])
    return (2, 1.0) if r > 1 else (3, r)


def test_classify_ratio_matches_reference():
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(600):
        p, q = int(rng.integers(1, 60)), int(rng.integers(1, 11))
        if p == q:
            continue
        cases.append(p / q)
        cases.append(round(q / p, 4))  # 源只留 4 位的合股比
    cases += list(1 + rng.uniform(0.0005, 0.06, 400))  # 派息
    cases += list(rng.uniform(0.01, 30, 400))
    cases += [1.0, 0.9999, 1.000004, 0.999995, 5.0, 0.0833, 0.0333, 0.0286]
    cases += [1.1, 3.625, 1.07, 587 / 7, 20.875, 10.0, 0.05, 1 / 30]
    seen = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in cases:
        got = classify_ratio(float(r))
        want = reference_classify(float(r))
        assert got[0] == want[0], r
        assert got[1] == pytest.approx(want[1], rel=1e-12), r
        seen[got[0]] += 1
    assert all(seen[k] > 5 for k in (1, 2, 3))  # 每个分支都走到了
    assert classify_ratio(0.0833)[1] == pytest.approx(1 / 12)
    assert classify_ratio(1.008)[0] == 2
    # 复核里的并购伪事件：587/7 与 20.875 不再算整数比。
    assert classify_ratio(587 / 7)[0] == 2 and classify_ratio(20.875)[0] == 2


def test_random_ratio_false_split_rate():
    """均匀随机的 r 被当成整数比拆并股的比例（写进文档）。[10,100] 只认整数后应接近 0。"""
    rng = np.random.default_rng(1)
    rates = {}
    for lo, hi in ((1.05, 3), (3, 10), (10, 100)):
        r = rng.uniform(lo, hi, 20000)
        rates[(lo, hi)] = float(np.mean([classify_ratio(float(x))[0] == 1 for x in r]))
    assert rates[(10, 100)] < 0.005
    assert rates[(3, 10)] < 0.01
    print("误判为拆并股的比例", rates)


def _one(market, ratio, close, open_=None, tradable=None, pre=None):
    ratio = np.array(ratio, dtype=float)[:, None]
    close = np.array(close, dtype=float)[:, None]
    open_ = close if open_ is None else np.array(open_, dtype=float)[:, None]
    if tradable is None:
        tradable = np.isfinite(close)
    else:
        tradable = np.array(tradable, dtype=bool)[:, None]
    if pre is not None:
        pre = np.array(pre, dtype=float)[:, None]
    out = corporate_actions(market, ratio, close, open_, tradable, previous_close=pre)
    return {k: v[:, 0] for k, v in out.items()}


nan = math.nan


def test_hand_cash_after_consolidation_in_gap():
    """停牌期内先 10 合 1 再派息（复核的 01129.HK、01389.HK 形状），手算等效现金。

    第 0 天收盘 0.05；第 1 天（停牌）因子比 0.1；第 2 天（停牌）因子比 1.05；第 3 天复牌开盘 0.476。
    除权前参考价：0.05 / 0.1 = 0.5（合股后的价位），等效现金 = 0.5 × (1 − 1/1.05) = 0.0238095…
    旧实现用合股前的 0.05，小了 10 倍。
    """
    got = _one(
        "hk", [1, 0.1, 1.05, 1], [0.05, nan, nan, 0.476], [0.05, nan, nan, 0.476]
    )
    assert got["action_source"].tolist() == [0, 1, 2, 0]
    assert got["split"].tolist() == [1, 0.1, 1, 1]
    assert got["equivalent_cash"][2] == pytest.approx(0.5 * (1 - 1 / 1.05), rel=1e-12)
    assert got["equivalent_cash"][2] == pytest.approx(0.0238095238, rel=1e-9)
    # 后复权因子：第 3 天 = 0.1 × 1.05。
    assert got["adj_factor"].tolist() == pytest.approx([1, 0.1, 0.105, 0.105])


def test_hand_cash_same_window_twice():
    """一个窗口里两次派息：第二次的参考价先扣掉第一次。前收 10，r1 = 10/9.5，r2 = 9.5/9.0。"""
    r1, r2 = 10 / 9.5, 9.5 / 9.0
    got = _one("hk", [1, r1, r2, 1], [10, nan, nan, 9.0])
    assert got["equivalent_cash"][1] == pytest.approx(0.5, rel=1e-12)
    assert got["equivalent_cash"][2] == pytest.approx(0.5, rel=1e-12)


def test_large_ratio_is_shares_not_cash():
    """分不出时，r ≥ 1.10 按股数调整（不扣税），小比值按等效现金；美股小比值不执行。"""
    hk = _one("hk", [1, 1.37, 1, 1.05], [10, 10 / 1.37, 7.3, 7.3 / 1.05])
    assert hk["action_source"].tolist() == [0, 4, 0, 2]
    assert hk["split"][1] == 1.37 and hk["equivalent_cash"][1] == 0
    us = _one("us", [1, 1.37, 1, 1.05], [10, 10 / 1.37, 7.3, 7.3 / 1.05])
    assert us["action_source"].tolist() == [0, 4, 0, 5]
    assert us["equivalent_cash"].tolist() == [0, 0, 0, 0]
    assert us["adj_factor"][-1] == pytest.approx(1.37)  # 美股不含现金类调整


def test_unconfirmed_and_terminal_not_executed():
    # 因子 ×10 而价格不动 → 6；4 拆 1 价格跟着变 → 1。
    bad = _one("us", [1, 10, 1], [12.38, 12.30, 12.31])
    assert bad["action_source"].tolist() == [0, 6, 0] and bad["split"].tolist() == [
        1,
        1,
        1,
    ]
    assert bad["action_suspect"].tolist() == [False, True, False]
    assert bad["adj_factor"].tolist() == [1, 1, 1]
    good = _one("us", [1, 4, 1], [200, 50.5, 51])
    assert good["action_source"].tolist() == [0, 1, 0] and good["split"][1] == 4
    # 合股记在因子里、价格不动 → 6。
    assert _one("us", [1, 0.2], [10.81, 10.85])["action_source"].tolist() == [0, 6]
    # 最后一次成交之后的因子跳变（并购）→ 7；后面一行成交量为 0 的陈旧日线不算成交。
    m = _one("us", [1, 1, 80, 1], [58, 58, nan, 58], tradable=[1, 1, 0, 0])
    assert (
        m["action_source"].tolist() == [0, 0, 7, 0] and m["split"].tolist() == [1] * 4
    )
    # 价格印证的上限：2 拆 1 当天又涨 60%（h=1.6 > √2）→ 不执行。
    assert _one("us", [1, 2], [100, 80])["action_source"][1] == 6
    assert _one("us", [1, 2], [100, 55])["action_source"][1] == 1
    # 开盘价沿用旧价、收盘价才反映合股（港股小盘常见）：收盘能印证也算。
    assert _one("hk", [1, 0.1], [0.112, 1.14], [0.112, 0.135])["action_source"][1] == 1
    # 小窗口（< 1.25 倍）价格分不出，只排除明显矛盾。
    assert _one("hk", [1, 1.05], [10, 9.6])["action_source"][1] == 2
    assert _one("hk", [1, 1.05], [10, 16])["action_source"][1] == 6


def test_hk_pre_close_confirms_before_prices():
    """港股以交易所前收为第一印证（形状取自复核的一例 20 合 1，数值是同比例的合成数）。

    合股当天开盘、收盘都只有隐含价位的 0.385 倍，价格印证不了；前收已按 20 合 1 调整 → 执行。
    """
    before = 0.1
    after = before * 20 * 0.385
    good = _one(
        "hk", [1, 0.05], [before, after], [before, after], pre=[nan, before * 20]
    )
    assert good["action_source"].tolist() == [0, 1]
    assert good["split"][1] == pytest.approx(0.05) and good["action_check"][1] == 1
    # 持有者价值：股数 ×0.05、价格 ×7.7 → 0.385，照实承受下跌。
    assert 0.05 * after / before == pytest.approx(0.385)
    # 前收缺失时退回价格印证：这里价格印证不了 → 6。
    gone = _one("hk", [1, 0.05], [before, after], [before, after], pre=[nan, nan])
    assert gone["action_source"][1] == 6 and gone["action_check"][1] == 5
    # 前收对不上（没有除权，或缺口窗口里是前一天的按盘价）也退回价格印证：价格像合股 → 执行（2）；
    # 价格也对不上 → 6。
    plain = _one("hk", [1, 0.05], [before, 2.0], [before, 2.0], pre=[nan, before])
    assert plain["action_source"][1] == 1 and plain["action_check"][1] == 2
    both = _one("hk", [1, 0.05], [before, after], [before, after], pre=[nan, before])
    assert both["action_source"][1] == 6
    # 大比值窗口：前收与 1/R 相差 0.8%（在 1% 内）算前收印证；相差 1.5% 不算，改看价格。
    pc = before * 20
    near = _one("hk", [1, 0.05], [before, after], pre=[nan, pc * 1.008])
    assert near["action_check"][1] == 1
    far = _one("hk", [1, 0.05], [before, after], pre=[nan, pc * 1.015])
    assert far["action_source"][1] == 6
    # 小比值窗口仍是 0.5%：派息 r=1.02、前收差 0.8% 不算前收印证（价格也对不上 → 6）。
    small = _one("hk", [1, 1.02], [10, 16], pre=[nan, 10 / 1.02 * 1.008])
    assert small["action_source"][1] == 6
    assert (
        _one("hk", [1, 1.02], [10, 16], pre=[nan, 10 / 1.02 * 1.004])["action_check"][1]
        == 1
    )
    # 前收按价位取整：0.5–10 港元价位 0.01，隐含前收 0.526 写成 0.53，相对差 0.76% 但在半个价位内 → 印证。
    tick = _one("hk", [1, 2.0], [1.052, 0.2], [1.052, 0.2], pre=[nan, 0.53])
    assert tick["action_source"][1] == 1 and tick["action_check"][1] == 1
    # 小比值（派息）同样先看前收。
    div = _one("hk", [1, 1.02], [10, 9.3], pre=[nan, 10 / 1.02])
    assert div["action_source"][1] == 2 and div["action_check"][1] == 1


def test_open_or_close_confirmation_recorded():
    """价格印证：开盘、收盘任一个通过就算，开盘先试；只靠收盘通过的 action_check 记 3（日内前视）。"""
    # 开盘已除权、收盘当天又涨 70%：开盘印证（2）。
    by_open = _one("us", [1, 2], [100, 85], [100, 50])
    assert by_open["action_source"][1] == 1 and by_open["action_check"][1] == 2
    # 开盘沿用旧价、收盘才除权：只靠收盘（3）。
    by_close = _one("us", [1, 2], [100, 51], [100, 100])
    assert by_close["action_source"][1] == 1 and by_close["action_check"][1] == 3


def test_confirmation_cap_near_and_far():
    """R=10 的合股，持有者价值倍数 2.5：b−a ≤ 5 时上限 ln2 → 不执行；更长的窗口上限 ln3 → 执行。"""
    h = 2.5
    near = _one("us", [1, 1, 0.1], [10, nan, 10 / 0.1 * h], tradable=[1, 0, 1])
    assert near["action_source"][2] == 6
    rows = 8
    close = [10] + [nan] * (rows - 2) + [10 / 0.1 * h]
    ratio = [1] * (rows - 1) + [0.1]
    far = _one("us", ratio, close, tradable=[1] + [0] * (rows - 2) + [1])
    assert far["action_source"][-1] == 1 and far["action_check"][-1] == 2


def test_us_small_window_share_change_needs_price():
    """美股 R < 1.25 的窗口：整数比照常执行；非整数的股数调整价格要走一半以上，否则记 5 不执行。"""
    flat = _one("us", [1, 0.85, 1], [35, 35.2, 35.1])
    assert flat["action_source"][1] == 5 and flat["split"][1] == 1
    assert flat["adj_factor"][-1] == 1
    moved = _one("us", [1, 0.85, 1], [35, 35 / 0.85, 41])
    assert moved["action_source"][1] == 3 and moved["split"][1] == 0.85
    assert moved["action_check"][1] == 2
    up = _one("us", [1, 1.15, 1], [25, 25.1, 25.1])
    assert up["action_source"][1] == 5
    assert _one("us", [1, 1.15, 1], [25, 25 / 1.15, 22])["action_source"][1] == 4
    # 只认开盘：开盘没动、收盘才走到位 → 不执行（5）。
    close_only = _one("us", [1, 0.85, 1], [35, 35 / 0.85, 41], [35, 35.1, 41])
    assert close_only["action_source"][1] == 5
    # 6 拆 5（整数比）价格不动也照常执行（小窗口只排除明显矛盾）。
    assert _one("us", [1, 1.2, 1], [20, 20.1, 20.2])["action_source"][1] == 1
    # 港股不受这条影响（前收印证）。
    assert (
        _one("hk", [1, 0.85], [35, 35.2], pre=[nan, 35 / 0.85])["action_source"][1] == 3
    )


def test_segments_gap_threshold():
    """长缺口拆段（文档写的门槛是 60 个交易日）：90 天的缺口两侧价格不连续时拆开；50 天的不拆。"""
    from ashare.data.overseas_actions import segments

    T = 200
    for gap, want in ((90, 2), (50, 1)):
        close = np.full((T, 1), np.nan)
        close[:10] = 5.0
        close[10 + gap :] = 0.4  # 复牌后价格只剩 8%
        tradable = np.isfinite(close)
        parts = segments(
            close, tradable, np.ones((T, 1)), np.zeros((T, 1), bool), op.US_GAP_SPLIT
        )
        assert len(parts) == want, gap


REF_TICKS = [(0.25, 0.001), (0.5, 0.005), (10, 0.01), (20, 0.02), (100, 0.05)]


def reference_actions(market, ratio, close, open_, tradable, pre=None):
    """逐格写法的参考实现（不调 corporate_actions 的任何内部函数）。"""
    T, N = ratio.shape
    src = np.zeros((T, N), dtype=int)
    how = np.zeros((T, N), dtype=int)
    split = np.ones((T, N))
    cash = np.zeros((T, N))
    adj = np.full((T, N), np.nan)
    for j in range(N):
        events = [
            t
            for t in range(T)
            if math.isfinite(ratio[t, j]) and abs(ratio[t, j] - 1) > 1e-5
        ]
        done = set()
        for t in events:
            if t in done:
                continue
            a = next((u for u in range(t - 1, -1, -1) if tradable[u, j]), None)
            b = next((u for u in range(t, T) if tradable[u, j]), None)
            group = [
                u for u in events if (a is None or u > a) and (b is None or u <= b)
            ]
            done |= set(group)
            R = math.prod(ratio[u, j] for u in group)
            if b is None:
                for u in group:
                    src[u, j], how[u, j] = 7, 6
                continue
            small = max(R, 1 / R) < 1.25

            def by_price(limit, a=a, b=b, j=j, R=R, closing=True):
                # 先开盘、后收盘；只靠收盘通过的记 3。
                tries = (
                    ((2, open_[b, j]), (3, close[b, j]))
                    if closing
                    else ((2, open_[b, j]),)
                )
                for code, x in tries:
                    if x > 0 and abs(math.log(R * x / close[a, j])) <= limit:
                        return code
                return 0

            tight = 0
            if a is None:
                ok, ref = (4 if small else 0), close[b, j] * R
            else:
                ref = close[a, j]
                pc = pre[b, j] if (pre is not None and market == "hk") else math.nan
                near = False
                if pc > 0:
                    tick = next((k for top, k in REF_TICKS if pc <= top), 0.1)
                    tol = 0.005 if small else 0.01
                    near = abs(pc * R - ref) <= max(tol * ref, 0.5 * tick * R) + 1e-9
                if near:
                    ok = 1
                else:
                    if not small:
                        limit = min(
                            abs(math.log(R)) / 2, math.log(2 if b - a <= 5 else 3)
                        )
                    else:
                        limit = math.log(1.5)
                    ok = by_price(limit)
                if market == "us" and small:
                    tight = by_price(abs(math.log(R)) / 2, closing=False)
            for u in group:
                r = ratio[u, j]
                if not ok:
                    src[u, j], how[u, j] = 6, 5
                    continue
                how[u, j] = ok
                kind, value = reference_classify(r)
                needs = (
                    market == "us"
                    and small
                    and (kind == 3 or (kind == 2 and r >= 1.10))
                )
                if kind == 1:
                    src[u, j], split[u, j] = 1, value
                elif needs and not tight:
                    src[u, j] = 5
                elif kind == 3:
                    src[u, j], split[u, j] = 3, r
                    if needs:
                        how[u, j] = tight
                elif r >= 1.10:
                    src[u, j], split[u, j] = 4, r
                    if needs:
                        how[u, j] = tight
                elif market == "hk":
                    src[u, j], cash[u, j] = 2, ref - ref / r
                else:
                    src[u, j] = 5
                ref = ref / r
        rows = [t for t in range(T) if math.isfinite(ratio[t, j])]
        if rows:
            level = 1.0
            for t in range(rows[0], T):
                if src[t, j] in (1, 3, 4):
                    level *= split[t, j]
                elif src[t, j] == 2:
                    level *= ratio[t, j]
                adj[t, j] = level
    return dict(
        action_source=src,
        split=split,
        equivalent_cash=cash,
        adj_factor=adj,
        action_check=how,
    )


@pytest.mark.parametrize("market", ["hk", "us"])
def test_corporate_actions_match_reference(market):
    rng = np.random.default_rng(11 if market == "hk" else 12)
    T, N = 50, 40
    events = np.where(
        rng.random((T, N)) < 0.08,
        rng.choice([2.0, 0.1, 1.02, 0.97, 1.5, 1.37, 25.0, 1.07, 1.83], (T, N)),
        1.0,
    )
    close = 10 * np.cumprod(rng.uniform(0.97, 1.03, (T, N)), axis=0)
    # 一半事件价格跟着变（真实），一半不变（源错误）。
    follow = rng.random((T, N)) < 0.5
    close = close / np.cumprod(np.where(follow, events, 1.0), axis=0)
    tradable = rng.random((T, N)) > 0.25
    close[~tradable & (rng.random((T, N)) < 0.5)] = np.nan
    tradable[-10:, :8] = False  # 停止交易后因子还在变（并购、退市）
    tradable &= np.isfinite(close)
    open_ = close * rng.uniform(0.99, 1.01, (T, N))
    ratio = events.copy()
    ratio[0] = 1.0
    ratio[rng.random((T, N)) < 0.05] = np.nan
    # 港股前收：每个成交日一种，按交易所除权（上一成交日收盘 / 其间因子比之积）、没除权、缺失。
    pre = np.full((T, N), np.nan)
    kind = rng.integers(0, 3, (T, N))
    for j in range(N):
        last, product = None, 1.0
        for t in range(T):
            if math.isfinite(ratio[t, j]):
                product *= ratio[t, j]
            if not tradable[t, j]:
                continue
            if last is not None:
                pre[t, j] = (
                    close[last, j] / product
                    if kind[t, j] == 0
                    else close[last, j]
                    if kind[t, j] == 1
                    else math.nan
                )
            last, product = t, 1.0
    if market == "us":
        # 美股小窗口里的非整数股数调整：价格走一半以上才执行。
        events[rng.random((T, N)) < 0.02] = 0.85
        ratio = np.where(np.isfinite(ratio) & (events == 0.85), 0.85, ratio)
    got = corporate_actions(market, ratio, close, open_, tradable, previous_close=pre)
    want = reference_actions(market, ratio, close, open_, tradable, pre)
    for name in want:
        np.testing.assert_allclose(
            got[name].astype(float),
            want[name].astype(float),
            rtol=1e-12,
            equal_nan=True,
        )
    seen = set(np.unique(got["action_source"]).tolist())
    assert {1, 3, 4, 6, 7} <= seen and (2 in seen if market == "hk" else 5 in seen)
    checks = set(np.unique(got["action_check"]).tolist())
    assert ({1, 2, 5, 6} if market == "hk" else {2, 5, 6}) <= checks


def test_stratified_pick_deterministic():
    universe = pl.DataFrame(
        dict(
            ts_code=[f"S{i:03}" for i in range(100)],
            mv=[float(100 - i) for i in range(100)],
        )
    )
    a = stratified_pick(universe, size=17, strata=5, exclude=["S000"])
    b = stratified_pick(
        universe.sample(fraction=1.0, shuffle=True, seed=3),
        size=17,
        strata=5,
        exclude=["S000"],
    )
    assert a == b and len(a) == 17 and "S000" not in [c for c, _ in a]
    counts = {s: sum(1 for _, k in a if k == s) for s in range(5)}
    assert counts == {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}
    # 层号 0 是市值最大的一层。
    top = {c for c, k in a if k == 0}
    assert all(int(c[1:]) < 21 for c in top)


# ---------------------------------------------------------------- 读取接口


def test_reader_slicing_fx_and_errors(tmp_path):
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    hk = ad.open(tmp_path / "root", market="hk")
    p = hk.panel(
        ["close", "is_listed"],
        start=DAYS[2],
        end=DAYS[4],
        symbols=["00005.HK", "00001.HK"],
    )
    assert p.shape == (3, 2) and list(p.symbols) == ["00005.HK", "00001.HK"]
    full = hk.panel("close")
    np.testing.assert_array_equal(p["close"][:, 1], col(full, "close", "00001.HK")[2:5])
    with pytest.raises(ValueError):
        p["close"][0, 0] = 1.0
    assert hk.field("close")["currency"] == "HKD"
    with pytest.raises(KeyError):
        hk.field("upper_limit")
    with pytest.raises(KeyError):
        hk.panel("close", symbols=["NOPE"])
    with pytest.raises(ValueError):
        hk.panel("close", start="2025-01-01")
    fx = hk.fx("USDHKD").collect()
    assert fx["close"][0] == pytest.approx(7.801) and fx["pair"][0] == "USDHKD"
    assert hk.fx("CNYHKD").collect()["rate"][0] == pytest.approx(0.915)
    # 可用时刻：FXCM 标为 d 的那根 d+1 日纽约 17:00（冬令时 22:00 UTC）；中间价北京时间 9:15。
    assert str(fx["available_at"][0]) == "2026-01-06 22:00:00+00:00"
    mid = hk.fx("CNYHKD").collect()
    assert (
        str(mid["available_at"][0])
        == f"{mid['date'][0][:4]}-{mid['date'][0][4:6]}-{mid['date'][0][6:]} 01:15:00+00:00"
    )
    with pytest.raises(KeyError):
        hk.fx("EURUSD")
    sec = hk.securities().collect()
    assert sec.filter(pl.col("symbol") == "00001.HK")["board_lot_current"][0] == 500
    with pytest.raises(TypeError):
        ad.open(tmp_path / "root", universe="sample")
    with pytest.raises(FileNotFoundError):
        ad.open(tmp_path / "root", market="hk", universe="sample").panel("close")


def test_reader_finds_columns_by_name_not_sort_order(tmp_path):
    """按名字找列：列名没有排序（拆段列名带「#」，将来也可能有别的写法）时取到的仍是对的那一列。"""
    store, _, _ = make(tmp_path)
    meta = build_overseas(store, "all")
    hk = ad.open(tmp_path / "root", market="hk")
    full = hk.panel("close")
    before = {c: np.array(col(full, "close", c)) for c in full.symbols}
    path = store.root / "panel_all" / meta["version"]
    order = np.arange(len(meta["symbols"]))[::-1]
    for name in meta["fields"]:
        array = np.load(path / f"{name}.npy")
        np.save(path / f"{name}.npy", np.ascontiguousarray(array[:, order]))
    meta["symbols"] = [meta["symbols"][k] for k in order]
    (path / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False))
    hk = ad.open(tmp_path / "root", market="hk")
    for code in ("00001.HK", "00005.HK", "08009.HK"):
        got = hk.panel("close", symbols=[code])["close"][:, 0]
        np.testing.assert_array_equal(got, before[code])


# ---------------------------------------------------------------- 校验


def test_verify_clean_passes_with_registered_exceptions(tmp_path):
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    path = exceptions_file(tmp_path)
    report = verify_overseas(store, exceptions_path=path)
    assert report["verdict"] == "pass", {
        k: v for k, v in report["checks"].items() if v["verdict"] == "fail"
    }
    assert report["checks"]["quotes_vs_listing"]["verdict"] == "explained"
    assert report["checks"]["factor_price"]["verdict"] == "explained"
    assert report["checks"]["factor_vs_previous_close"]["checked"] >= 2
    assert report["known_exceptions"]["stale_count"] == 0
    # 清单的键对不上（代码写错）：不解释。
    wrong = dict(
        HK_EXCEPTIONS,
        entries=[dict(e, ts_code="00009.HK") for e in HK_EXCEPTIONS["entries"]],
    )
    mismatch = verify_overseas(store, exceptions_path=exceptions_file(tmp_path, wrong))
    assert mismatch["checks"]["factor_price"]["verdict"] == "fail"
    assert mismatch["known_exceptions"]["stale_count"] == 3
    # 日期区间：区间外的不符不解释。
    ranged = dict(
        HK_EXCEPTIONS,
        entries=[
            HK_EXCEPTIONS["entries"][0],
            dict(
                market="hk",
                check="factor_price",
                kind="半分以内",
                date_from=DAYS[6],
                date_to=DAYS[9],
                reason="R2",
            ),
        ],
    )
    assert (
        verify_overseas(store, exceptions_path=exceptions_file(tmp_path, ranged))[
            "checks"
        ]["factor_price"]["verdict"]
        == "fail"
    )
    ranged["entries"][1]["date_from"] = DAYS[5]
    assert (
        verify_overseas(store, exceptions_path=exceptions_file(tmp_path, ranged))[
            "checks"
        ]["factor_price"]["verdict"]
        == "explained"
    )
    # 没有清单：两处不符都判 fail。
    bare = verify_overseas(store, exceptions_path=tmp_path / "none.json")
    assert bare["verdict"] == "fail"
    assert {k for k, v in bare["checks"].items() if v["verdict"] == "fail"} == {
        "quotes_vs_listing",
        "factor_price",
        "factor_vs_previous_close",
    }


def _rewrite(store, name, partition, change):
    entry = store.manifest()[f"{name}/{partition}"]
    table = pl.read_parquet(store.root / entry["file"]).drop("_fetched_at")
    store.write(
        name,
        partition,
        entry["params"],
        FetchResult(change(table), "2026-02-15T00:00:00+00:00", 1),
    )
    if "snapshot" in entry:
        store.mark(f"{name}/{partition}", {"snapshot": entry["snapshot"]})


FAULTS = {
    "daily_ranges": lambda s: _rewrite(
        s,
        "hk_daily",
        DAYS[7],
        lambda t: t.with_columns(
            close=pl.when(pl.col("ts_code") == "00001.HK")
            .then(-1.0)
            .otherwise(pl.col("close"))
        ),
    ),
    "unique_keys": lambda s: _rewrite(
        s, "hk_daily", DAYS[7], lambda t: pl.concat([t, t.head(1)])
    ),
    "factor_vs_previous_close": lambda s: _rewrite(
        s,
        "hk_daily",
        DAYS[10],
        lambda t: t.with_columns(pre_close=pl.col("pre_close") * 1.2),
    ),
    "hist_vs_day": lambda s: _rewrite(
        s, "hk_daily_hist", "00005.HK", lambda t: t.with_columns(vol=pl.col("vol") + 1)
    ),
    "factor_snapshot": lambda s: s.mark(
        f"hk_adjfactor/{DAYS[4]}", {"snapshot": "other"}
    ),
    "manifest_files": lambda s: s.mark(f"hk_daily/{DAYS[6]}", {"file_hash": "0" * 64}),
    "partitions": lambda s: s.mark(f"hk_connect_hold/{DAYS[6]}", {"status": "failed"}),
    "calendar": lambda s: _rewrite(
        s,
        "hk_calendar",
        "all",
        lambda t: t.with_columns(
            pretrade_date=pl.when(pl.col("cal_date") == DAYS[9])
            .then(pl.lit(DAYS[2]))
            .otherwise(pl.col("pretrade_date"))
        ),
    ),
}


@pytest.mark.parametrize("check", sorted(FAULTS))
def test_verify_fails_on_each_fault(tmp_path, check):
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    FAULTS[check](store)
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    assert report["verdict"] == "fail"
    assert report["checks"][check]["verdict"] == "fail", report["checks"][check]


def test_verify_fails_when_panel_tampered(tmp_path):
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    current = (
        store.root
        / "panel_all"
        / json.loads((store.root / "panel_all/current.json").read_text())["version"]
    )
    close = np.load(current / "close.npy")
    close[np.isfinite(close)] *= 1.0001
    np.save(current / "close.npy", close)
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    assert report["checks"]["panel"]["verdict"] == "fail"


def test_exception_register_rules(tmp_path):
    bad = dict(
        HK_EXCEPTIONS,
        entries=[
            dict(market="hk", check="factor_price", ts_code="X", close=1.0, reason="R1")
        ],
    )
    with pytest.raises(ValueError, match="不允许的键"):
        load_exceptions("hk", exceptions_file(tmp_path, bad))
    unknown = dict(
        HK_EXCEPTIONS,
        entries=[dict(market="hk", check="factor_price", ts_code="X", reason="R9")],
    )
    with pytest.raises(ValueError, match="原因编号"):
        load_exceptions("hk", exceptions_file(tmp_path, unknown))
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    stale = dict(
        HK_EXCEPTIONS,
        entries=HK_EXCEPTIONS["entries"]
        + [
            dict(
                market="hk",
                check="factor_price",
                ts_code="00001.HK",
                trade_date=DAYS[1],
                reason="R2",
            )
        ],
    )
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path, stale))
    assert (
        report["verdict"] == "pass" and report["known_exceptions"]["stale_count"] == 1
    )


def test_real_exception_register_is_whitelisted():
    for market in ("hk", "us"):
        load_exceptions(market)  # 键集合、原因编号都合法


def test_cli_status_and_verify_exit_code(tmp_path):
    store, _, _ = make(tmp_path)
    root = tmp_path / "root"

    def run(*a):
        return subprocess.run(
            [sys.executable, "-m", "ashare.data", "--root", str(root), *a],
            capture_output=True,
            text=True,
            timeout=300,
        )

    status = json.loads(run("--market", "hk", "status").stdout)
    assert status["overseas"]["hk_daily"]["partitions"] == len(DAYS)
    assert run("--market", "hk", "build-overseas").returncode == 0
    failing = run("--market", "hk", "verify-overseas")
    assert failing.returncode == 1  # 正式清单里没有夹具的两处不符
    assert run("--market", "cn", "build-overseas").returncode != 0
    # A 股、期货子命令拒绝港美股市场（否则会把 A 股数据写进 <root>/hk）。
    for command in (
        "sync",
        "build-panel",
        "verify",
        "gc",
        "sync-futures",
        "verify-futures",
    ):
        refused = run("--market", "hk", command)
        assert refused.returncode == 2 and "只用于 A 股" in refused.stderr, command
    assert not (root / "hk" / "panel").exists()
    from ashare.data.reader import Data

    with pytest.raises(ValueError, match="只读 A 股"):
        Data(root, "us")


def test_hk_rmb_counter_currency_and_primary():
    """人民币柜台（8xxxx.HK、名称以 R 结尾）源写 HKD，证券表改为 CNY 并保留源值；primary_symbol 指向
    同一公司的港币柜台（0 + 后四位、名称去掉末尾 R 相同），找不到为空，其余指向自己。"""
    from ashare.data.overseas_panel import _securities

    codes = [
        "80175.HK",
        "89988.HK",
        "08001.HK",
        "80737.HK",
        "00700.HK",
        "00175.HK",
        "09988.HK",
        "86618.HK",
    ]
    names = [
        "吉利汽车-R",
        "阿里巴巴-WR",
        "某创业板R",
        "某公司",
        "腾讯控股",
        "吉利汽车",
        "阿里巴巴-W",
        "京东健康-R",
    ]
    raw = pl.DataFrame(
        dict(
            ts_code=codes,
            name=names,
            market=["主板"] * 8,
            list_status=["L"] * 8,
            list_date=["20230619"] * 8,
            delist_date=[None] * 8,
            trade_unit=[1000.0] * 8,
            curr_type=["HKD"] * 8,
            isin=[None] * 8,
        ),
        schema_overrides=dict(delist_date=pl.String, isin=pl.String),
    )

    class Stub:
        def read(self, name):
            return raw

    out = _securities(Stub(), "hk", None).sort("ts_code")
    got = dict(zip(out["ts_code"], out["currency"], strict=True))
    assert got == {
        "00175.HK": "HKD",
        "00700.HK": "HKD",
        "08001.HK": "HKD",
        "09988.HK": "HKD",
        "80175.HK": "CNY",
        "80737.HK": "HKD",
        "86618.HK": "CNY",
        "89988.HK": "CNY",
    }
    assert set(out["currency_source"]) == {"HKD"}
    primary = dict(zip(out["ts_code"], out["primary_symbol"], strict=True))
    assert primary["80175.HK"] == "00175.HK" and primary["89988.HK"] == "09988.HK"
    assert primary["86618.HK"] is None  # 证券表里没有港币柜台
    assert primary["00700.HK"] == "00700.HK" and primary["80737.HK"] == "80737.HK"


def test_price_rounded_tick_threshold():
    """价位 0.5 港元以下的最小价位小于 0.01：0.49、0.25 为真，0.5 为假；期外一律为假。"""
    dates = np.array(["20200701", "20200702", "20231229", "20240102"])
    close = np.array([[0.49, 0.5, 0.25, 0.3]] * 4)
    quote = np.isfinite(close)
    out = op.price_rounded(dates, quote, close)
    assert out.tolist() == [
        [False] * 4,
        [True, False, True, True],
        [True, False, True, True],
        [False] * 4,
    ]


def test_reused_code_boundary_day():
    """改记到旧代码只看「早于新公司上市日、且早于旧公司退市日」：上市日当天属于新公司。"""
    reused = {"00003.HK": ("20260108", "00003!.HK", "20260112")}
    frame = pl.DataFrame(
        dict(ts_code=["00003.HK"] * 3, trade_date=["20260107", "20260108", "20260109"])
    )
    out = op._assign_reused(frame, reused, {}, "daily")
    assert out["ts_code"].to_list() == ["00003!.HK", "00003.HK", "00003.HK"]


def test_version_changes_with_inputs_and_sample(tmp_path):
    """版本键：输入分区内容一变、样本名单一变（all 的证券表有 in_sample），都要换代。"""
    store, _, _ = make(tmp_path)
    first = build_overseas(store, "all")["version"]
    _rewrite(
        store,
        "hk_daily",
        DAYS[7],
        lambda t: t.with_columns(vol=pl.col("vol") + 1),
    )
    second = build_overseas(store, "all")["version"]
    assert second != first
    sample = json.loads((store.root / "sample.json").read_text())
    dropped = next(c for c in sample["symbols"] if c == "00001.HK")
    sample["symbols"] = [c for c in sample["symbols"] if c != dropped]
    (store.root / "sample.json").write_text(json.dumps(sample))
    third = build_overseas(store, "all")
    assert third["version"] != second
    sec = ad.open(tmp_path / "root", market="hk").securities().collect()
    assert not sec.filter(pl.col("symbol") == dropped)["in_sample"][0]


def test_verify_catches_previous_close_on_plain_day(tmp_path):
    """没有事件的日子前收错了（汇丰 ×0.95 那种）：原始库检查要抓到。"""
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    _rewrite(
        store,
        "hk_daily",
        DAYS[5],
        lambda t: t.with_columns(
            pre_close=pl.when(pl.col("ts_code") == "00001.HK")
            .then(pl.col("pre_close") * 0.95)
            .otherwise(pl.col("pre_close"))
        ),
    )
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    check = report["checks"]["factor_vs_previous_close"]
    assert check["verdict"] == "fail"
    assert check["examples"][0]["kind"] == "无事件日"


def test_calendar_marks_open_day_closed(tmp_path):
    """日历把开市日标成休市：构建不许静默丢掉这天的行情，校验不许崩溃、要出报告并判失败。"""
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    _rewrite(
        store,
        "hk_calendar",
        "all",
        lambda t: t.with_columns(
            is_open=pl.when(pl.col("cal_date") == DAYS[12])
            .then(0)
            .otherwise(pl.col("is_open"))
        ),
    )
    with pytest.raises(ValueError, match="不在交易日历的开市日里"):
        build_overseas(store, "all")
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    assert report["verdict"] == "fail"
    assert report["checks"]["rows_on_closed_days"]["verdict"] == "fail"
    assert report["checks"]["panel"]["verdict"] == "fail"
    assert (store.root / "reports" / "overseas_verify.md").exists()


def _tamper(field, change):
    def apply(path, meta):
        array = np.load(path / f"{field}.npy")
        change(array, meta)
        np.save(path / f"{field}.npy", array)

    return apply


def _cell(meta, code, day):
    return meta["dates"].index(DAYS[day]), meta["symbols"].index(code)


PANEL_FAULTS = {
    "equivalent_cash": _tamper(
        "equivalent_cash",
        lambda a, m: a.__setitem__(
            _cell(m, "00001.HK", 20), a[_cell(m, "00001.HK", 20)] * 1.5
        ),
    ),
    "split": _tamper(
        "split", lambda a, m: a.__setitem__(_cell(m, "00001.HK", 10), 4.0)
    ),
    "action_source": _tamper(
        "action_source", lambda a, m: a.__setitem__(_cell(m, "00001.HK", 20), 0)
    ),
    "is_listed": _tamper(
        "is_listed", lambda a, m: a.__setitem__(_cell(m, "00002.HK", 16), True)
    ),
    "delist": _tamper(
        "delist", lambda a, m: a.__setitem__(_cell(m, "00002.HK", 15), False)
    ),
    "connect_eligible": _tamper(
        "connect_eligible", lambda a, m: a.__setitem__(_cell(m, "00001.HK", 3), True)
    ),
    "connect_held": _tamper(
        "connect_held", lambda a, m: a.__setitem__(_cell(m, "00005.HK", 0), True)
    ),
    "connect_sell_only": _tamper(
        "connect_sell_only", lambda a, m: a.__setitem__(_cell(m, "00005.HK", 29), True)
    ),
    "price_rounded": _tamper(
        "price_rounded", lambda a, m: a.__setitem__(_cell(m, "08001.HK", 3), True)
    ),
    "adj_factor": _tamper(
        "adj_factor",
        lambda a, m: a.__setitem__((slice(None), m["symbols"].index("00005.HK")), 1.0),
    ),
    "action_suspect": _tamper(
        "action_suspect", lambda a, m: a.__setitem__(_cell(m, "00001.HK", 4), True)
    ),
}


@pytest.mark.parametrize("field", sorted(PANEL_FAULTS))
def test_verify_panel_catches_each_field(tmp_path, field):
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    path = (
        store.root
        / "panel_all"
        / json.loads((store.root / "panel_all/current.json").read_text())["version"]
    )
    meta = json.loads((path / "metadata.json").read_text())
    PANEL_FAULTS[field](path, meta)
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    panel = report["checks"]["panel"]
    assert panel["verdict"] == "fail"
    assert field in {e.get("field") for e in panel["examples"]}, panel["examples"]


def _drop_event(path, meta, code, day):
    """把一次执行了的事件改成「没有印证、不执行」，并把相关字段（含 adj_factor）一起改得自洽。"""
    i, j = _cell(meta, code, day)
    for name, value in (
        ("action_source", 6),
        ("action_suspect", True),
        ("action_check", 5),
        ("split", 1.0),
        ("equivalent_cash", 0.0),
    ):
        array = np.load(path / f"{name}.npy")
        array[i, j] = value
        np.save(path / f"{name}.npy", array)
    adj = np.load(path / "adj_factor.npy")
    adj[:, j] = np.where(np.isfinite(adj[:, j]), 1.0, np.nan)
    np.save(path / "adj_factor.npy", adj)


def _scale_cash(code, day):
    return _tamper(
        "equivalent_cash",
        lambda a, m: a.__setitem__(_cell(m, code, day), a[_cell(m, code, day)] * 1.5),
    )


# (名字, 应报的字段, 篡改)
HK_COMBINED_FAULTS = [
    # 停牌缺口之后的等效现金 ×1.5（00005.HK 第 11 天，复牌前）。
    ("cash_after_gap", "equivalent_cash", _scale_cash("00005.HK", 11)),
    # 没有上一个成交日的窗口（00010.HK 第 2 天）等效现金 ×1.5。
    ("cash_no_prior_trade", "equivalent_cash", _scale_cash("00010.HK", 2)),
    # 前收印证过的 20 合 1 被标成可疑、连 adj_factor 一起改过：只能靠前收查出来。
    (
        "dropped_consolidation",
        "action_source",
        lambda p, m: _drop_event(p, m, "08009.HK", 12),
    ),
    # 只靠半个价位放宽才由前收印证的派息（00009.HK 第 14 天）改成不执行。
    (
        "dropped_tick_only",
        "action_source",
        lambda p, m: _drop_event(p, m, "00009.HK", 14),
    ),
    # 前收存在却记成「只靠收盘印证」。
    (
        "check_close_only",
        "action_check",
        _tamper(
            "action_check", lambda a, m: a.__setitem__(_cell(m, "00001.HK", 10), 3)
        ),
    ),
]


@pytest.mark.parametrize(
    "fault", HK_COMBINED_FAULTS, ids=[x[0] for x in HK_COMBINED_FAULTS]
)
def test_verify_panel_catches_gap_cash_and_dropped_events(tmp_path, fault):
    _, field, apply = fault
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    path = (
        store.root
        / "panel_all"
        / json.loads((store.root / "panel_all/current.json").read_text())["version"]
    )
    meta = json.loads((path / "metadata.json").read_text())
    apply(path, meta)
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    panel = report["checks"]["panel"]
    assert panel["verdict"] == "fail"
    assert field in {e.get("field") for e in panel["examples"]}, panel["examples"]


@pytest.mark.parametrize(("scale", "executed"), [(1.008, True), (1.015, False)])
def test_verify_large_ratio_pre_close_tolerance_matches_build(
    tmp_path, scale, executed
):
    """大比值窗口的前收容差（1%）构建与校验一致：08009.HK 的 20 合 1 前收偏 0.8% 时构建以前收印证执行，
    校验不得把它判成「前收印证不了却记 1」；偏 1.5% 时两边都不认前收（价格也印证不了 → 6）。"""
    store, _, _ = make(tmp_path)
    _rewrite(
        store,
        "hk_daily",
        DAYS[12],
        lambda t: t.with_columns(
            pre_close=pl.when(pl.col("ts_code") == "08009.HK")
            .then(pl.col("pre_close") * scale)
            .otherwise(pl.col("pre_close"))
        ),
    )
    build_overseas(store, "all")
    p = ad.open(tmp_path / "root", market="hk").panel(["action_source", "action_check"])
    if executed:
        assert col(p, "action_source", "08009.HK")[12] == 1
        assert col(p, "action_check", "08009.HK")[12] == 1
    else:
        assert col(p, "action_source", "08009.HK")[12] == 6
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    panel = report["checks"]["panel"]
    assert panel["verdict"] == "pass", panel["examples"]


def test_verify_reports_instead_of_crashing_on_corrupt_panel(tmp_path):
    """面板文件损坏（读不出来）时，校验记为失败并照常写报告，不抛异常。"""
    store, _, _ = make(tmp_path)
    build_overseas(store, "all")
    path = (
        store.root
        / "panel_all"
        / json.loads((store.root / "panel_all/current.json").read_text())["version"]
    )
    (path / "adj_factor.npy").write_bytes(b"not a numpy file")
    report = verify_overseas(store, exceptions_path=exceptions_file(tmp_path))
    assert report["checks"]["panel"]["verdict"] == "fail"
    assert "面板核对出错" in report["checks"]["panel"]["examples"][0]["problem"]
