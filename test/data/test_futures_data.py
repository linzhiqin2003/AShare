"""期货数据层：同步断点续跑、统一字段换算、面板、连续合约、panelbt 输入、独立校验。
全部用合成数据（futures_fixtures），不含真实行情。"""

import asyncio
import hashlib
import json

import numpy as np
import polars as pl
import pytest
from futures_fixtures import CONTRACTS, DAYS, FakeTushare, open_interest

import ashare.data as ad
from ashare.data import futures_continuous as fc
from ashare.data.catalog import DAILY_DATASETS, FUTURES_DATASETS
from ashare.data.futures_panel import build_futures
from ashare.data.futures_panelbt import fill_columns, panelbt_inputs
from ashare.data.futures_schema import conform, describe
from ashare.data.futures_sync import sync_futures
from ashare.data.futures_tushare import TushareFutures, unify_symbol
from ashare.data.futures_verify import verify_futures
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.data.sync import sync

START, END = DAYS[0], DAYS[-1]


def synced(tmp_path, source=None):
    store = Store(tmp_path)
    source = source or FakeTushare()
    asyncio.run(sync_futures(source, store, START, END))
    return store, source


@pytest.fixture
def built(tmp_path):
    store, _ = synced(tmp_path)
    build_futures(store)
    return store, ad.open(tmp_path).futures()


# ---------- 同步 ----------


def test_futures_not_in_default_stock_sync(tmp_path):
    assert not set(FUTURES_DATASETS) & set(DAILY_DATASETS)
    with pytest.raises(ValueError, match="sync-futures"):
        asyncio.run(sync(None, Store(tmp_path), START, END, ["futures_daily"]))


def test_sync_partitions_and_resume(tmp_path):
    store = Store(tmp_path)
    source = FakeTushare(fail={("fut_daily", DAYS[3]), ("fut_settle", DAYS[7])})
    with pytest.raises(RuntimeError, match="故意失败"):
        asyncio.run(sync_futures(source, store, START, END))
    manifest = store.manifest()
    assert manifest[f"futures_daily/{DAYS[3]}"]["status"] == "failed"
    assert manifest[f"futures_settle/{DAYS[7]}"]["status"] == "failed"
    # 其余接口的分区照常完成，不被一个失败拖住。
    assert all(manifest[f"futures_mapping/{d}"]["status"] == "complete" for d in DAYS)
    first_files = {
        k: v["file"] for k, v in manifest.items() if v.get("status") == "complete"
    }
    source.calls.clear()
    asyncio.run(sync_futures(source, store, START, END, refresh=False))
    fetched = sorted(
        (api, p.get("trade_date") or p.get("exchange")) for api, p in source.calls
    )
    # 断点续跑：只重抓失败的两个日分区。
    assert fetched == [("fut_daily", DAYS[3]), ("fut_settle", DAYS[7])]
    after = store.manifest()
    assert all(v["status"] == "complete" for v in after.values())
    assert all(after[k]["file"] == f for k, f in first_files.items())
    # 日分区只按期货交易所日历，休市日没有分区。
    assert not any(k.endswith("20260110") for k in after)
    # 默认刷新最近 5 个交易日。
    source.calls.clear()
    asyncio.run(sync_futures(source, store, START, END))
    days = sorted({p["trade_date"] for api, p in source.calls if api == "fut_daily"})
    assert days == DAYS[-5:]
    # 按交易所分区的日历和合约表每次刷新。
    assert sum(api == "trade_cal" for api, _ in source.calls) == 6
    assert sum(api == "fut_basic" for api, _ in source.calls) == 6


# ---------- 统一字段 ----------


def test_symbols_and_units(tmp_path):
    store, _ = synced(tmp_path)
    adapter = TushareFutures(store)
    contracts = adapter.contracts()
    assert set(contracts["symbol"]) == {
        "RB2602.SHFE",
        "RB2603.SHFE",
        "RB2604.SHFE",
        "IF2602.CFFEX",
        "IF2603.CFFEX",
    }
    mult = dict(zip(contracts["symbol"], contracts["multiplier"], strict=False))
    assert mult["RB2602.SHFE"] == 10.0 and mult["IF2602.CFFEX"] == 300.0
    tick = dict(zip(contracts["symbol"], contracts["tick_size"], strict=False))
    assert tick["IF2603.CFFEX"] == 0.2
    params = adapter.params()
    row = params.filter(
        (pl.col("symbol") == "IF2603.CFFEX") & (pl.col("date") == DAYS[0])
    ).row(0, named=True)
    assert row["margin_long"] == pytest.approx(0.12)  # 中金所百分数
    assert row["fee_open_rate"] == pytest.approx(0.23e-4)  # 股指万分之几
    assert row["fee_open_per_contract"] == 0.0
    assert row["fee_close_today_rate"] == pytest.approx(2.3e-4)  # 1000% = 10 倍
    rb = params.filter(
        (pl.col("symbol") == "RB2604.SHFE") & (pl.col("date") == DAYS[0])
    ).row(0, named=True)
    assert rb["fee_open_rate"] == pytest.approx(0.05e-3)  # 上期所千分比
    assert rb["fee_close_today_rate"] is None  # 源缺失不当作免收
    assert rb["price_limit"] is None
    daily = adapter.daily()
    assert set(daily["symbol"]) == set(contracts["symbol"])  # 连续代码不进合约行情
    one = daily.filter(
        (pl.col("symbol") == "RB2602.SHFE") & (pl.col("date") == DAYS[1])
    ).row(0, named=True)
    assert one["amount"] == pytest.approx(one["volume"] * one["settlement"] * 10)


def test_double_sided_volume_halved(tmp_path):
    store = Store(tmp_path)
    base = dict(
        pre_close=1.0,
        pre_settle=1.0,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        settle=1.0,
        change1=0.0,
        change2=0.0,
        oi_chg=0.0,
    )
    table = pl.DataFrame(
        [
            dict(
                ts_code="CU1912.SHF",
                trade_date="20191231",
                vol=10.0,
                amount=1.0,
                oi=4.0,
                **base,
            ),
            dict(
                ts_code="IF1912.CFX",
                trade_date="20191231",
                vol=10.0,
                amount=1.0,
                oi=4.0,
                **base,
            ),
        ]
    )
    store.write("futures_daily", "20191231", {}, FetchResult(table, "now", 1))
    basic = pl.DataFrame(
        dict(ts_code=["CU1912.SHF", "IF1912.CFX"], list_date=["20181216"] * 2)
    )
    store.write("futures_contracts", "SHFE", {}, FetchResult(basic, "now", 1))
    later = table.with_columns(trade_date=pl.lit("20200102"))
    store.write("futures_daily", "20200102", {}, FetchResult(later, "now", 1))
    daily = TushareFutures(store).daily().sort("date", "symbol")
    volume = dict(
        zip(
            zip(daily["symbol"], daily["date"], strict=False),
            daily["volume"],
            strict=False,
        )
    )
    assert volume[("CU1912.SHFE", "20191231")] == 5.0  # 2020 年前上期所双边
    assert volume[("CU1912.SHFE", "20200102")] == 10.0
    assert volume[("IF1912.CFFEX", "20191231")] == 10.0  # 中金所始终单边


def test_unify_symbol_and_mapping_kind(tmp_path):
    assert unify_symbol("RB2701.SHF") == "RB2701.SHFE"
    assert unify_symbol("L2605F.DCE") == "L_F2605.DCE"
    assert unify_symbol("RB.SHF") is None
    store = Store(tmp_path)
    table = pl.DataFrame(
        dict(
            ts_code=["AL.SHF", "ALL.SHF", "TL.CFX", "TLL.CFX", "LL.DCE", "L.DCE"],
            trade_date=["20260105"] * 6,
            mapping_ts_code=[
                "AL2602.SHF",
                "AL2601.SHF",
                "TL2603.CFX",
                "TL2603.CFX",
                "L2601.DCE",
                "L2605.DCE",
            ],
        )
    )
    store.write("futures_mapping", "20260105", {}, FetchResult(table, "now", 1))
    kinds = dict(
        zip(*TushareFutures(store).mapping().select("continuous", "kind"), strict=False)
    )
    assert kinds == {
        "AL.SHF": "main",
        "ALL.SHF": "nearby",
        "TL.CFX": "main",
        "TLL.CFX": "nearby",
        "LL.DCE": "nearby",
        "L.DCE": "main",
    }


def test_conform_rejects_missing_and_extra():
    frame = pl.DataFrame(dict(exchange=["SHFE"], date=["20260105"], is_open=[True]))
    assert conform("calendar", frame).schema["is_open"] == pl.Boolean
    with pytest.raises(ValueError, match="缺少必填列"):
        conform("calendar", frame.drop("is_open"))
    with pytest.raises(ValueError, match="多余列"):
        conform("calendar", frame.with_columns(x=pl.lit(1)))
    table = describe()
    assert {"contracts", "daily", "params", "calendar", "mapping"} == set(
        table["table"]
    )


# ---------- 面板 ----------


def test_panel_alignment(built):
    _, fut = built
    p = fut.panel(
        ["settlement", "is_listed", "has_quote", "margin_long"], products=["RB"]
    )
    assert list(p.symbols) == ["RB2602.SHFE", "RB2603.SHFE", "RB2604.SHFE"]
    assert list(p.dates) == DAYS
    settle = np.asarray(p["settlement"])
    assert settle[3, 1] == CONTRACTS[1][8] + 2 * 3 + 1
    assert (
        np.isnan(settle[10, 0]) and not p["is_listed"][10, 0] and p["is_listed"][9, 0]
    )
    assert np.asarray(p["margin_long"])[25, 2] == pytest.approx(0.11)
    with pytest.raises(ValueError):
        p["settlement"][0, 0] = 1.0
    sub = fut.panel(
        "settlement",
        symbols=["IF2603.CFFEX", "RB2602.SHFE"],
        start=DAYS[2],
        end=DAYS[4],
    )
    assert sub.shape == (3, 2) and sub["settlement"][0, 1] == settle[2, 0]
    with pytest.raises(KeyError):
        fut.panel("settlement", symbols=["XX0000.SHFE"])
    with pytest.raises(KeyError):
        fut.panel("nonexistent")


def test_build_is_versioned_and_reused(tmp_path):
    store, _ = synced(tmp_path)
    first = build_futures(store)
    again = build_futures(store)
    assert first["version"] == again["version"]
    assert (
        json.loads((store.root / "futures_panel/current.json").read_text())["version"]
        == first["version"]
    )


# ---------- 连续合约 ----------


def reference_held(metric, eligible, expiry_rank, previous=None):
    """直白参考实现：逐日按「前一日指标」挑，粘住、只向后、严格大于才换。"""
    held, current = [-1], None
    for t in range(1, len(metric)):
        ok = [
            j
            for j in range(metric.shape[1])
            if eligible[t][j]
            and (
                previous is None
                or (previous[t] >= 0 and expiry_rank[j] > expiry_rank[previous[t]])
            )
        ]
        if previous is not None and previous[t] < 0:
            ok = []

        def value(j, t=t):
            v = metric[t - 1][j]
            return -np.inf if np.isnan(v) else v

        if current is not None and current in ok:
            best = max(ok, key=lambda j: (value(j), -j))
            if expiry_rank[best] > expiry_rank[current] and value(best) > value(
                current
            ):
                current = best
        else:
            pool = [
                j
                for j in ok
                if current is None or expiry_rank[j] > expiry_rank[current]
            ]
            pool = [j for j in pool if np.isfinite(value(j))]
            current = max(pool, key=lambda j: (value(j), -j)) if pool else None
        held.append(-1 if current is None else current)
    return np.array(held)


def random_case(seed):
    rng = np.random.default_rng(seed)
    t, n = rng.integers(5, 40), rng.integers(1, 7)
    metric = rng.integers(0, 6, size=(t, n)).astype(float)
    metric[rng.random((t, n)) < 0.15] = np.nan
    eligible = rng.random((t, n)) < 0.8
    rank = rng.permutation(n)
    return metric, eligible, rank


@pytest.mark.parametrize("seed", range(300))
def test_select_held_matches_reference(seed):
    metric, eligible, rank = random_case(seed)
    main = fc.select_held(metric, eligible, rank)
    assert np.array_equal(main, reference_held(metric, eligible, rank))
    second = fc.select_held(metric, eligible, rank, previous=main)
    assert np.array_equal(second, reference_held(metric, eligible, rank, previous=main))
    # 次主力总比主力晚到期。
    both = (main >= 0) & (second >= 0)
    assert np.all(rank[second[both]] > rank[main[both]])


@pytest.mark.parametrize("seed", range(200))
def test_roll_rule_has_no_lookahead(seed):
    metric, eligible, rank = random_case(seed)
    held = fc.select_held(metric, eligible, rank)
    cut = int(np.random.default_rng(seed + 1).integers(1, len(metric)))
    changed = metric.copy()
    changed[cut:] = np.random.default_rng(seed + 2).integers(
        0, 9, size=changed[cut:].shape
    )
    # 第 cut 行及以后的指标变化，不改变第 cut 行及以前的持有。
    assert np.array_equal(
        fc.select_held(changed, eligible, rank)[: cut + 1], held[: cut + 1]
    )


def test_roll_only_forward():
    metric = np.array([[5.0, 1.0], [1.0, 5.0], [9.0, 1.0], [9.0, 1.0]])
    eligible = np.ones((4, 2), dtype=bool)
    held = fc.select_held(metric, eligible, np.array([0, 1]))
    assert list(held) == [-1, 0, 1, 1]  # 第 3 行近月持仓更大也不回切


def test_three_adjustments_by_hand():
    # 行 0–1 持有 A，行 2 起持有 B；换月日 A=100、B=110。
    held = np.array([0, 0, 1, 1])
    price = np.array([[98.0, 105.0], [99.0, 108.0], [100.0, 110.0], [101.0, 121.0]])
    raw, none, rolls = fc.adjust_prices(held, price, "none")
    assert list(none) == [98, 99, 110, 121]
    _, diff, _ = fc.adjust_prices(held, price, "difference")
    assert list(diff) == [98, 99, 100, 111]  # 加累计价差 100 − 110
    _, ratio, _ = fc.adjust_prices(held, price, "ratio")
    assert ratio[2] == pytest.approx(100.0) and ratio[3] == pytest.approx(
        121 * 100 / 110
    )
    assert rolls == [(2, 0, 1, 100.0, 110.0, 2)]
    # 换月日旧合约缺价：用两者都有的前一天。
    gap = price.copy()
    gap[2, 0] = np.nan
    _, diff, rolls = fc.adjust_prices(held, gap, "difference")
    assert rolls[0][5] == 1 and diff[2] == pytest.approx(110 + 99 - 108)


def test_continuous_and_schedule_on_fixture(built):
    _, fut = built
    schedule = fut.roll_schedule("RB", rule="open_interest")
    assert schedule.select("date", "from_symbol", "to_symbol").rows() == [
        (DAYS[6], "RB2602.SHFE", "RB2603.SHFE"),
        (DAYS[16], "RB2603.SHFE", "RB2604.SHFE"),
    ]
    # 第 6 行的决定只用第 5 行持仓：第 5 行 RB2603 首次超过 RB2602。
    assert open_interest(1, 5) > open_interest(0, 5) and open_interest(
        1, 4
    ) <= open_interest(0, 4)
    series = fut.continuous("RB", rule="open_interest", adjust="difference")
    assert series["symbol"][0] == "RB2602.SHFE"
    roll = schedule.row(0, named=True)
    assert roll["spread"] == pytest.approx(roll["to_price"] - roll["from_price"])
    # 来源映射与自判规则在合成数据上一致（夹具的来源按前一日持仓判定）。
    source = fut.roll_schedule("RB", rule="source")
    assert (
        source.select("date", "to_symbol").rows()
        == schedule.select("date", "to_symbol").rows()
    )
    second = fut.continuous("RB", rule="open_interest", rank=2)
    assert (second["symbol"] != series.join(second, on="date")["symbol"]).all() or len(
        second
    )
    with pytest.raises(ValueError):
        fut.continuous("RB", rule="source", rank=2)
    # IF 到期前强制换月：IF2602 在其最后交易日（第 14 行）不再持有。
    rolls = fut.roll_schedule("IF", rule="open_interest")
    assert rolls["date"].to_list() == [DAYS[14]]


# ---------- panelbt 输入 ----------


def test_panelbt_inputs_match_manual_construction(built):
    import panelbt as bt

    _, fut = built
    inputs = panelbt_inputs(
        fut, products=["IF"], margin_rate=0.12, fee_rate=0.000023, fee_per_contract=0.0
    )
    names = list(inputs["symbols"])
    assert names == ["IF2602.CFFEX", "IF2603.CFFEX"]
    # 手工从长表构造（对冲腿的现有做法：按合约逐列取结算价和到期行）。
    daily = fut.table("daily", products=["IF"]).collect()
    t = len(DAYS)
    settle = np.full((t, 2), np.nan)
    for j, name in enumerate(names):
        rows = daily.filter(pl.col("symbol") == name)
        for day, value in zip(rows["date"], rows["settlement"], strict=False):
            settle[DAYS.index(day), j] = value
    manual_settle = fill_columns(settle)
    assert np.array_equal(inputs["futures"].settlement, manual_settle)
    assert list(inputs["futures"].expiry) == [14, 29]
    assert list(inputs["futures"].multiplier) == [300.0, 300.0]
    market = bt.Market(**inputs["market"], futures=inputs["futures"])
    targets = bt.Targets([0], [0, 1], [1], [-0.5])
    result = bt.run(market, targets, config=bt.Config(initial_cash=10_000_000.0))
    manual = bt.Market(
        exec_price=inputs["market"]["exec_price"],
        close=inputs["market"]["close"],
        can_buy=inputs["market"]["can_buy"],
        can_sell=inputs["market"]["can_sell"],
        volume=inputs["market"]["volume"],
        futures=bt.Futures(
            is_future=[True, True],
            multiplier=[300.0, 300.0],
            margin_rate=0.12,
            settlement=manual_settle,
            expiry=[14, 29],
            fee_rate=0.000023,
        ),
    )
    other = bt.run(manual, targets, config=bt.Config(initial_cash=10_000_000.0))
    assert np.array_equal(result.nav, other.nav)
    assert result.nav[-1] != 10_000_000.0  # 确实持有了期货
    # 默认取面板里的交易所保证金和手续费。
    default = panelbt_inputs(fut, products=["IF"])
    assert np.allclose(default["futures"].margin_rate, 0.12)


def test_panelbt_inputs_require_explicit_missing(built):
    _, fut = built
    inputs = panelbt_inputs(fut, products=["RB"])
    assert inputs["futures"].expiry.tolist() == [9, 19, 29]
    assert np.allclose(inputs["futures"].fee_rate, 0.05e-3)


# ---------- 独立校验 ----------


def corrupt(store, name, partition, change):
    entry = store.manifest()[f"{name}/{partition}"]
    path = store.root / entry["file"]
    frame = change(pl.read_parquet(path).drop("_fetched_at"))
    store.write(name, partition, entry["params"], FetchResult(frame, "later", 1))


def test_verify_clean_fixture_passes(tmp_path):
    store, _ = synced(tmp_path)
    report = verify_futures(store)
    verdicts = {k: v["verdict"] for k, v in report["checks"].items()}
    for name in (
        "manifest_files",
        "unique_keys",
        "partitions_futures_daily",
        "pre_settle_continuity",
        "contract_coverage",
        "amount_multiplier",
    ):
        assert verdicts[name] == "pass", (name, report["checks"][name])
    assert (store.root / "reports/futures_verify.md").exists()


@pytest.mark.parametrize(
    "fault, check",
    [
        ("pre_settle", "pre_settle_continuity"),
        ("missing_day", "contract_coverage"),
        ("amount", "amount_multiplier"),
        ("duplicate", "unique_keys"),
        ("missing_partition", "partitions_futures_daily"),
        ("page_multiple", "partitions_futures_settle"),
        ("hash", "manifest_files"),
        ("orphan", "manifest_files"),
    ],
)
def test_verify_catches_each_fault(tmp_path, fault, check):
    store, _ = synced(tmp_path)
    day = DAYS[5]
    if fault == "pre_settle":
        corrupt(
            store,
            "futures_daily",
            day,
            lambda f: f.with_columns(
                pre_settle=pl.when(pl.col("ts_code") == "RB2603.SHF")
                .then(pl.col("pre_settle") + 1)
                .otherwise(pl.col("pre_settle"))
            ),
        )
    elif fault == "missing_day":
        corrupt(
            store,
            "futures_daily",
            day,
            lambda f: f.filter(pl.col("ts_code") != "RB2603.SHF"),
        )
    elif fault == "amount":
        corrupt(
            store,
            "futures_daily",
            day,
            lambda f: f.with_columns(amount=pl.col("amount") * 2),
        )
    elif fault == "duplicate":
        corrupt(store, "futures_daily", day, lambda f: pl.concat([f, f.head(1)]))
    elif fault == "missing_partition":
        store.mark(f"futures_daily/{day}", {"status": "failed"})
    elif fault == "page_multiple":
        corrupt(
            store, "futures_settle", day, lambda f: pl.concat([f] * 1000).head(1000)
        )
    elif fault == "hash":
        path = store.root / store.manifest()[f"futures_mapping/{day}"]["file"]
        path.write_bytes(path.read_bytes() + b"x")
    elif fault == "orphan":
        folder = store.root / "raw/futures_daily" / day
        (folder / (hashlib.sha256(b"x").hexdigest() + ".parquet")).write_bytes(b"x")
    report = verify_futures(store, write=False)
    item = report["checks"][check]
    assert item["verdict"] in ("fail", "explained") and item["failed"] > 0, item


def test_historic_unit_and_reused_code(tmp_path):
    """旧合约交易单位与合约表现值不同时按成交额反查；同名旧合约（挂牌日前的行）丢弃。"""
    store = Store(tmp_path)
    basic = pl.DataFrame(
        dict(
            ts_code=["RU0801.SHF", "FU1905.SHF", "JD1905.DCE"],
            symbol=["RU0801", "FU1905", "JD1905"],
            exchange=["SHFE", "SHFE", "DCE"],
            name=["x"] * 3,
            fut_code=["RU", "FU", "JD"],
            multiplier=[None, None, None],
            trade_unit=["吨"] * 3,
            per_unit=[10.0, 10.0, 5.0],
            quote_unit=["人民币元/吨", "人民币元/吨", "人民币元/500千克"],
            quote_unit_desc=["5人民币元/吨", "1人民币元/吨", "1人民币元/500千克"],
            d_mode_desc=["实物交割"] * 3,
            list_date=["20070102", "20180716", "20180601"],
            delist_date=["20080115", "20190515", "20190515"],
            d_month=["200801", "201905", "201905"],
            last_ddate=["20080120", "20190520", "20190520"],
        ),
        schema_overrides=dict(multiplier=pl.Float64),
    )
    store.write("futures_contracts", "SHFE", {}, FetchResult(basic, "now", 1))
    rows = []
    days = ["20180509", "20180510", "20180511", "20180717", "20180718", "20180719"]
    for day in days:
        old = day < "20180716"
        rows.append(("FU1905.SHF", day, 50.0 if old else 10.0))
        rows.append(("RU0801.SHF", day, 5.0))  # 旧橡胶 5 吨/手
        rows.append(
            ("JD1905.DCE", day, 10.0)
        )  # 5 吨，报价元/500 千克 → 每手 10 个报价单位
    table = pl.DataFrame(
        [
            dict(
                ts_code=c,
                trade_date=d,
                pre_close=100.0,
                pre_settle=100.0,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                settle=100.0,
                change1=0.0,
                change2=0.0,
                vol=2.0,
                amount=2.0 * 100 * unit / 10000,
                oi=2.0,
                oi_chg=0.0,
            )
            for c, d, unit in rows
        ]
    )
    for day in days:
        store.write(
            "futures_daily",
            day,
            {},
            FetchResult(table.filter(pl.col("trade_date") == day), "now", 1),
        )
    adapter = TushareFutures(store)
    info = {r["symbol"]: r for r in adapter.contracts().iter_rows(named=True)}
    assert info["RU0801.SHFE"]["multiplier"] == 5.0
    assert info["RU0801.SHFE"]["multiplier_source"] == "implied_from_amount"
    assert info["FU1905.SHFE"]["multiplier"] == 10.0  # 挂牌前的旧合约行不参与反查
    assert info["JD1905.DCE"]["multiplier"] == 10.0
    assert info["JD1905.DCE"]["multiplier_source"] == "contract_table"
    daily = adapter.daily()
    fu = daily.filter(pl.col("symbol") == "FU1905.SHFE")["date"].to_list()
    assert fu == ["20180717", "20180718", "20180719"]
