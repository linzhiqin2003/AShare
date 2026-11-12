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
from ashare.data.overseas_panel import (
    actions_from_factor,
    build_overseas,
    classify_ratio,
)
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
    assert col(p, "connect_held", "00005.HK").tolist() == [i <= 27 for i in range(30)]
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


# ---------------------------------------------------------------- 公司行为识别：独立参考实现对拍


def reference_classify(r, previous=None, current=None):
    """直白的参考：枚举 q=1..10，取误差最小的 p/q。"""
    if (
        not (isinstance(r, float) and math.isfinite(r))
        or r <= 0
        or 1 - 1e-5 <= r <= 1 + 1e-5
    ):
        return 0, 1.0
    tol = 2e-4
    if previous and current:
        tol = max(tol, 1e-6 / min(previous, current))
    if r < 1:
        tol = max(tol, 6e-5 / r)
    big = r if r > 1 else 1 / r
    best = None
    if big >= 1.05:
        for q in range(1, 11):
            p = round(big * q)
            if p <= q or p > 1000:
                continue
            err = abs(p / q - big) / big
            if best is None or err < best[0]:
                best = (err, Fraction(p, q))
    if best and best[0] <= tol:
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
    cases += [
        1.0,
        0.9999,
        1.000004,
        0.999995,
        5.0,
        0.0833,
        0.0333,
        0.0286,
        1.1,
        3.625,
        1.07,
    ]
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


def reference_actions(factor, close):
    T, N = factor.shape
    out = {
        k: [[None] * N for _ in range(T)]
        for k in ("adj", "ratio", "split", "cash", "src")
    }
    for j in range(N):
        points = [
            (t, factor[t, j])
            for t in range(T)
            if math.isfinite(factor[t, j]) and factor[t, j] > 0
        ]
        for t in range(T):
            out["split"][t][j], out["cash"][t][j], out["src"][t][j] = 1.0, 0.0, 0
        for k, (t, value) in enumerate(points):
            out["adj"][t][j] = value / points[0][1]
            if k == 0:
                out["ratio"][t][j] = 1.0
                continue
            s, before = points[k - 1]
            r = value / before
            out["ratio"][t][j] = r
            kind, split = reference_classify(r, before, value)
            out["src"][t][j] = kind
            if kind in (1, 3):
                out["split"][t][j] = split
            if kind == 2:
                prices = [close[u, j] for u in range(t) if math.isfinite(close[u, j])]
                out["cash"][t][j] = prices[-1] * (1 - 1 / r) if prices else math.nan
    return out


def test_actions_from_factor_matches_reference():
    rng = np.random.default_rng(11)
    T, N = 40, 12
    factor = np.cumprod(
        np.where(
            rng.random((T, N)) < 0.08,
            rng.choice([2.0, 0.1, 1.02, 0.97, 1.5], (T, N)),
            1.0,
        ),
        axis=0,
    )
    factor /= factor[-1]
    factor = np.round(factor, 6)
    factor[rng.random((T, N)) < 0.15] = np.nan
    factor[:, 3] = np.nan  # 整列没有因子
    close = rng.uniform(1, 20, (T, N))
    close[rng.random((T, N)) < 0.2] = np.nan
    got = actions_from_factor(factor, close)
    want = reference_actions(factor, close)
    for mine, theirs in (
        ("adj_factor", "adj"),
        ("action_ratio", "ratio"),
        ("split", "split"),
        ("equivalent_cash", "cash"),
        ("action_source", "src"),
    ):
        expected = np.array(
            [[np.nan if v is None else v for v in row] for row in want[theirs]],
            dtype=float,
        )
        np.testing.assert_allclose(
            got[mine].astype(float), expected, rtol=1e-12, equal_nan=True
        )
    assert (got["action_source"] == 1).any() and (got["action_source"] == 2).any()


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
    with pytest.raises(KeyError):
        hk.fx("EURUSD")
    sec = hk.securities().collect()
    assert sec.filter(pl.col("symbol") == "00001.HK")["board_lot_current"][0] == 500
    with pytest.raises(TypeError):
        ad.open(tmp_path / "root", universe="sample")
    with pytest.raises(FileNotFoundError):
        ad.open(tmp_path / "root", market="hk", universe="sample").panel("close")


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
    assert mismatch["known_exceptions"]["stale_count"] == 2
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


def test_hk_rmb_counter_currency():
    """人民币柜台（8xxxx.HK、名称以 R 结尾）源写 HKD，证券表改为 CNY 并保留源值。"""
    import polars as pl

    from ashare.data.overseas_panel import _securities

    raw = pl.DataFrame(
        dict(
            ts_code=["80175.HK", "89988.HK", "08001.HK", "80737.HK", "00700.HK"],
            name=["吉利汽车-R", "阿里巴巴-WR", "某创业板R", "某公司", "腾讯控股"],
            market=["主板"] * 5,
            list_status=["L"] * 5,
            list_date=["20230619"] * 5,
            delist_date=[None] * 5,
            trade_unit=[1000.0] * 5,
            curr_type=["HKD"] * 5,
            isin=[None] * 5,
        ),
        schema_overrides=dict(delist_date=pl.String, isin=pl.String),
    )

    class Stub:
        def read(self, name):
            return raw

    daily = pl.DataFrame(dict(ts_code=["00700.HK"], trade_date=["20260105"]))
    out = _securities(Stub(), "hk", daily, None).sort("ts_code")
    got = dict(zip(out["ts_code"], out["currency"], strict=True))
    assert got == {
        "00700.HK": "HKD",
        "08001.HK": "HKD",
        "80175.HK": "CNY",
        "80737.HK": "HKD",
        "89988.HK": "CNY",
    }
    assert set(out["currency_source"]) == {"HKD"}
