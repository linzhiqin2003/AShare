"""港股、美股「交易日 × 证券」面板的构建（``build_overseas``）。

两个证券范围（universe）各一套代次：

- ``all``：按交易日分区的全部证券（开发库为近 3 年），列是区间内出现过日线的全部代码。
- ``sample``：2010 年起的样本证券（``sample.json``），日线与因子取按证券分区的 ``*_hist``。

存储与 A 股面板相同：``<root>/<市场>/panel_<universe>/<版本>/<字段>.npy``（T×N，C 连续），
``metadata.json``、``securities.parquet``、``calendar.parquet``，``current.json`` 指向当前代。
版本由输入分区的内容哈希、区间、范围、样本名单和构建代码决定，同样输入重建得到同一版本。

公司行为：两地都没有分红送转的事件明细接口，只有前复权累计因子（``cum_adjfactor``，最新日为 1）。
本模块只用相邻两日的因子比 r=F[t]/F[t-1]：整数比识别为拆并股，其余记「分不出现金与股数」的
等效调整（规则见 ``overseas_fields.ACTION_RULE``）。因子是快照，区间内所有因子分区必须来自同一次
同步（``snapshot`` 唯一），否则拒绝构建。
"""

import hashlib
import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl

from .overseas_fields import ACTION_RULE, describe, fields_for
from .overseas_sync import read_sample, spec_for
from .store import atomic_json, lock
from .sync import compact

FORMAT_VERSION = 1
CODE_FILES = ("overseas_fields.py", "overseas_panel.py")
SPLIT_MIN = 1.05
# 因子只留 6 位小数：相邻两日之比与 1 相差不到 1e-5 的视为舍入噪声，不算事件。
RATIO_NOISE = 1e-5
SPLIT_MAX_DEN = 10
SPLIT_MAX_NUM = 1000
SPLIT_REL_TOL = 2e-4
SPLIT_ABS_TOL = 6e-5
DELIST_GRACE = 20
# 2026-11-13 实测：hk_daily 在这段日期把全部价格四舍五入到 0.01（低价股失真），之外与因子表一致。
HK_ROUNDED = ("20200702", "20231229")
HK_FINE_TICK_BELOW = 0.5


def code_version():
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def classify_ratio(r, previous=None, current=None):
    """因子比 → (action_source, split)。0 无事件、1 拆并股、2 分不出、3 下调且非整数比。

    ``previous``、``current`` 是两日的因子值，用来放宽源 6 位小数舍入带来的误差。
    """
    if not np.isfinite(r) or r <= 0:
        return 0, 1.0
    if abs(r - 1.0) <= RATIO_NOISE:
        return 0, 1.0
    tolerance = SPLIT_REL_TOL
    if previous is not None and current is not None:
        smallest = min(abs(previous), abs(current))
        if smallest > 0:
            tolerance = max(tolerance, 1e-6 / smallest)
    big = max(r, 1 / r)
    if r < 1:
        # 港股合股的因子比在源里只保留 4 位小数（1/12 写成 0.0833），按绝对误差放宽。
        tolerance = max(tolerance, SPLIT_ABS_TOL / r)
    if big >= SPLIT_MIN:
        fraction = Fraction(big).limit_denominator(SPLIT_MAX_DEN)
        if (
            fraction.numerator <= SPLIT_MAX_NUM
            and fraction > 1
            and abs(float(fraction) - big) / big <= tolerance
        ):
            value = float(fraction) if r > 1 else float(1 / fraction)
            return 1, value
    if r > 1:
        return 2, 1.0
    return 3, float(r)


def _inputs(store, names, start, end):
    """区间用到的完成分区 → 内容哈希；日分区只算区间内的。"""
    from .catalog import OVERSEAS_DATASETS

    out = {}
    for name in names:
        by_day = OVERSEAS_DATASETS[name].partition == "day"
        for key, entry in store.entries(name).items():
            if entry.get("status") != "complete":
                continue
            if by_day and not (start <= entry["partition"] <= end):
                continue
            out[key] = entry["content_hash"]
    return out


CORE_DAILY = ("open", "high", "low", "close", "pre_close", "vol", "amount")


def _dedupe(frame, keys, diagnostics, name, core):
    """同一主键多行（美股同一代码在两个交易所各一行等）。

    核心列（价格、量额或因子）取值相同的行合并成一行，其余列取第一个非空值（估值字段常常只有
    一行有值）；核心列取值不同的整组丢弃并计数，不猜哪一行是对的。
    """
    core = [c for c in core if c in frame.columns]
    # 排序后再取「第一个非空」，结果与原始分区的读取顺序无关。
    frame = frame.sort(frame.columns, nulls_last=True)
    distinct = frame.unique(subset=[*keys, *core], keep="first")
    conflict = distinct.filter(pl.struct(keys).is_duplicated()).select(keys).unique()
    diagnostics[f"{name}_conflicting_keys"] = len(conflict)
    diagnostics[f"{name}_conflicting_examples"] = (
        conflict.sort(keys).head(20).to_dicts()
    )
    clean = frame.join(conflict, on=keys, how="anti")
    repeated = clean.filter(pl.struct(keys).is_duplicated())
    diagnostics[f"{name}_merged_duplicate_rows"] = len(repeated)
    if not len(repeated):
        return clean
    others = [c for c in clean.columns if c not in keys]
    merged = repeated.group_by(keys, maintain_order=True).agg(
        [pl.col(c).drop_nulls().first() for c in others]
    )
    single = clean.filter(~pl.struct(keys).is_duplicated())
    return pl.concat([single, merged.select(clean.columns)], how="vertical_relaxed")


def _dedupe_factor(factor, diagnostics):
    """因子表同一代码同一天多行时，每只证券只用一条因子序列。

    美股同一代码常同时有两个交易所的因子历史（转板、场外份额），两条序列的口径可以不同
    （实测 WMT 的纽交所序列含现金分红、纳斯达克序列只含拆股），逐日混取会在两条序列之间来回跳，
    丢掉冲突日又会让跨缺口的比值把两条序列接在一起。所以每只证券先选定一个交易所：区间内有因子
    的天数最多的非场外（OTC）交易所（同样多取代码字母序靠前的），全是场外才用场外；只取这个交易所的行。同一交易所同一天仍有
    多行时，相对差不超过 1e-6 的合并，其余丢弃并计数。
    """
    keys = ["ts_code", "trade_date"]
    if "exchange" not in factor.columns or factor["exchange"].null_count() == len(
        factor
    ):
        factor = factor.with_columns(pl.lit("").alias("exchange"))
    factor = factor.select(*keys, "cum_adjfactor", pl.col("exchange").fill_null(""))
    counts = (
        factor.group_by("ts_code", "exchange")
        .agg(pl.col("trade_date").n_unique().alias("_days"))
        .with_columns(_otc=pl.col("exchange") == "OTC")
        .sort(
            ["ts_code", "_otc", "_days", "exchange"],
            descending=[False, False, True, False],
        )
    )
    primary = counts.unique("ts_code", keep="first", maintain_order=True).select(
        "ts_code", pl.col("exchange").alias("_primary")
    )
    chosen = factor.join(primary, on="ts_code").filter(
        pl.col("exchange") == pl.col("_primary")
    )
    diagnostics["factor_rows_other_exchange"] = len(factor) - len(chosen)
    diagnostics["factor_symbols_with_two_exchanges"] = (
        factor.group_by("ts_code")
        .agg(pl.col("exchange").n_unique())
        .filter(pl.col("exchange") > 1)
        .height
    )
    groups = chosen.group_by(keys).agg(
        pl.len().alias("_n"),
        pl.col("cum_adjfactor").min().alias("_lo"),
        pl.col("cum_adjfactor").max().alias("_hi"),
    )
    same = (pl.col("_n") == 1) | (
        (pl.col("_hi") - pl.col("_lo")).abs() <= 1e-6 * pl.col("_hi").abs()
    )
    dropped = groups.filter(~same).select(keys)
    diagnostics["factor_conflicting_keys"] = len(dropped)
    diagnostics["factor_conflicting_examples"] = dropped.sort(keys).head(20).to_dicts()
    kept = (
        chosen.join(dropped, on=keys, how="anti")
        .sort([*keys, "cum_adjfactor"])
        .unique(keys, keep="first", maintain_order=True)
    )
    return kept.select(*keys, "cum_adjfactor")


def _reused_codes(securities):
    """港股证券表用 ``代码!.HK`` 记已退市的旧公司，日线却把新旧公司都写成 ``代码.HK``。

    返回 {代码.HK: (新公司上市日, 旧公司代码!.HK, 旧公司退市日)}，只收新旧两条记录都在的代码。
    """
    old = securities.filter(pl.col("ts_code").str.contains("!", literal=True))
    out = {}
    current = {
        r["ts_code"]: r
        for r in securities.unique("ts_code", keep="first").iter_rows(named=True)
    }
    for row in old.iter_rows(named=True):
        bare = row["ts_code"].replace("!", "")
        new = current.get(bare)
        if new and new.get("list_date") and row.get("delist_date"):
            out[bare] = (new["list_date"], row["ts_code"], row["delist_date"])
    return out


def _assign_reused(frame, reused, diagnostics, name):
    """新公司上市日之前、旧公司退市日之前的行改记到旧公司代码（``代码!.HK``）下。"""
    if not reused or not len(frame):
        diagnostics[f"{name}_rows_reassigned_to_old_code"] = 0
        return frame
    table = pl.DataFrame(
        dict(
            ts_code=list(reused),
            _new_listing=[v[0] for v in reused.values()],
            _old_code=[v[1] for v in reused.values()],
            _old_delist=[v[2] for v in reused.values()],
        )
    )
    joined = frame.join(table, on="ts_code", how="left")
    move = (
        pl.col("_old_code").is_not_null()
        & (pl.col("trade_date") < pl.col("_new_listing"))
        & (pl.col("trade_date") < pl.col("_old_delist"))
    )
    diagnostics[f"{name}_rows_reassigned_to_old_code"] = joined.filter(move).height
    return joined.with_columns(
        ts_code=pl.when(move).then(pl.col("_old_code")).otherwise(pl.col("ts_code"))
    ).drop("_new_listing", "_old_code", "_old_delist")


def _precise_close(daily, factor, diagnostics):
    """港股：源日线在 HK_ROUNDED 期间把价格四舍五入到 0.01，因子表的 close_price 保留了 0.001。

    两者相差不超过半分时收盘价改用 close_price；相差更大的保留日线原值并计数。开高低价和前收
    没有精确来源，原样保留，由 ``price_rounded`` 标记。
    """
    precise = factor.select("ts_code", "trade_date", "close_price").unique(
        ["ts_code", "trade_date"], keep="none"
    )
    joined = daily.join(precise, on=["ts_code", "trade_date"], how="left")
    differ = pl.col("close_price").is_not_null() & (
        pl.col("close") != pl.col("close_price")
    )
    near = differ & ((pl.col("close") - pl.col("close_price")).abs() <= 0.0051)
    diagnostics["close_from_factor_table"] = joined.filter(near).height
    diagnostics["close_mismatch_kept"] = joined.filter(differ & ~near).height
    return joined.with_columns(
        close=pl.when(near).then(pl.col("close_price")).otherwise(pl.col("close"))
    ).drop("close_price")


def _coordinates(frame, dates, symbols):
    """长表每行在「交易日 × 证券」矩阵里的 (行, 列)；不在范围内的行丢弃。"""
    day = pl.DataFrame(
        {"trade_date": [str(d) for d in dates], "_row": np.arange(len(dates))}
    )
    code = pl.DataFrame(
        {"ts_code": [str(c) for c in symbols], "_col": np.arange(len(symbols))}
    )
    joined = (
        frame.select("ts_code", "trade_date")
        .with_row_index("_i")
        .join(day, on="trade_date", how="inner")
        .join(code, on="ts_code", how="inner")
    )
    if joined.select(pl.struct("_row", "_col").is_duplicated().any()).item():
        raise ValueError("长表有重复主键，去重后才能散布到面板")
    return (
        joined["_i"].to_numpy(),
        joined["_row"].to_numpy(),
        joined["_col"].to_numpy(),
    )


def _fill(frame, coords, column, shape):
    index, rows, cols = coords
    out = np.full(shape, np.nan)
    values = frame[column].cast(pl.Float64).to_numpy()
    out[rows, cols] = values[index]
    return out


def _scatter(frame, dates, symbols, column, dtype=np.float64, fill=np.nan):
    """布尔标记用：有行即为真（名单、持股）。"""
    out = np.full((len(dates), len(symbols)), fill, dtype=dtype)
    _, rows, cols = _coordinates(
        frame.unique(["ts_code", "trade_date"]), dates, symbols
    )
    out[rows, cols] = True
    return out


def actions_from_factor(factor, close):
    """前复权因子矩阵（NaN 为缺）→ 后复权因子与逐日公司行为字段。

    纯 numpy：每列沿时间找上一个有因子的格，r=F[t]/F[prev]。``close`` 用于等效现金（取上一个
    有收盘价的格）。返回 dict：adj_factor、action_ratio、split、equivalent_cash、action_source。
    """
    T, N = factor.shape
    adj = np.full((T, N), np.nan)
    ratio = np.full((T, N), np.nan)
    split = np.ones((T, N))
    cash = np.zeros((T, N))
    source = np.zeros((T, N), dtype=np.uint8)
    for j in range(N):
        column = factor[:, j]
        valid = np.flatnonzero(np.isfinite(column) & (column > 0))
        if not len(valid):
            continue
        base = column[valid[0]]
        adj[valid, j] = column[valid] / base
        ratio[valid[0], j] = 1.0
        prices = close[:, j]
        last_price = np.nan
        p = 0
        for k in range(1, len(valid)):
            t, s = valid[k], valid[k - 1]
            while p < t:
                if np.isfinite(prices[p]):
                    last_price = prices[p]
                p += 1
            r = column[t] / column[s]
            ratio[t, j] = r
            kind, value = classify_ratio(r, column[s], column[t])
            source[t, j] = kind
            if kind in (1, 3):
                split[t, j] = value
            elif kind == 2:
                cash[t, j] = (
                    last_price * (1 - 1 / r) if np.isfinite(last_price) else np.nan
                )
    return dict(
        adj_factor=adj,
        action_ratio=ratio,
        split=split,
        equivalent_cash=cash,
        action_source=source,
    )


def _snapshot_check(store, names, keys):
    snapshots = {
        store.manifest()[k].get("snapshot")
        for k in keys
        if store.manifest()[k]["dataset"] in names
    }
    if len(snapshots) > 1 or None in snapshots:
        raise ValueError(
            f"复权因子分区来自不同的同步快照 {sorted(map(str, snapshots))[:5]}；"
            "前复权因子每次除权都改写历史，请整段重抓（sync-overseas 默认会）"
        )
    return next(iter(snapshots), None)


def build_overseas(store, universe="all", start=None, end=None, *, keep=1):
    """构建港股或美股面板；返回元数据。``universe`` 为 ``all`` 或 ``sample``。"""
    market = store.market
    m = spec_for(market)
    if universe not in ("all", "sample"):
        raise ValueError("universe 只能是 all 或 sample")
    started = time.perf_counter()
    store = store.snapshot()
    base = store.root / f"panel_{universe}"
    with lock(store.root / f".build_{universe}.lock"):
        calendar_table = store.read(m["calendar"]).sort("cal_date")
        open_days = np.array(
            sorted(
                set(
                    calendar_table.filter(pl.col("is_open").cast(pl.Int64) == 1)[
                        "cal_date"
                    ].to_list()
                )
            ),
            dtype="U8",
        )
        sample = read_sample(store)
        if universe == "all":
            daily_name, factor_name = m["daily"], m["factor"]
            entries = [
                e
                for e in store.entries(daily_name).values()
                if e["status"] == "complete" and e["rows"]
            ]
            if not entries:
                raise FileNotFoundError(
                    f"缺少 {daily_name}；请运行 python -m ashare.data --market {market} sync-overseas"
                )
            first = min(e["partition"] for e in entries)
            last = max(e["partition"] for e in entries)
        else:
            if sample is None:
                raise FileNotFoundError(
                    "缺少 sample.json；先运行 select-sample-overseas"
                )
            daily_name, factor_name = m["daily_hist"], m["factor_hist"]
            first = sample.get("start", "20100101")
            entries = [
                e
                for e in store.entries(daily_name).values()
                if e["status"] == "complete"
            ]
            if not entries:
                raise FileNotFoundError(f"缺少 {daily_name}；请先同步样本历史")
            last = max(e["params"]["end_date"] for e in entries)
        start = compact(start) if start else first
        end = compact(end) if end else last
        dates = open_days[(open_days >= start) & (open_days <= end)]
        if not len(dates):
            raise ValueError("构建范围没有交易日")
        start, end = str(dates[0]), str(dates[-1])
        used = [m["calendar"], m["securities"], daily_name, factor_name]
        if market == "hk":
            used += list(m["extra"])
        inputs = _inputs(store, used, start, end)
        factor_keys = [
            k for k in inputs if store.manifest()[k]["dataset"] == factor_name
        ]
        snapshot = _snapshot_check(store, {factor_name}, factor_keys)
        code = code_version()
        version = hashlib.sha256(
            json.dumps(
                dict(
                    inputs=inputs,
                    start=start,
                    end=end,
                    universe=universe,
                    sample=sample["symbols"]
                    if (sample and universe == "sample")
                    else None,
                    code=code,
                    format=FORMAT_VERSION,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        target = base / version
        if target.exists():
            atomic_json(base / "current.json", {"version": version})
            return json.loads((target / "metadata.json").read_text())
        diagnostics = {}
        # 只读要用的列和区间内的分区：美股近 3 年日线约 900 万行。
        wanted = [*CORE_DAILY, "ts_code", "trade_date"] + (
            ["vwap", "total_mv"] if market == "us" else []
        )
        daily = (
            store.scan(daily_name, start, end)
            .select(wanted)
            .filter(pl.col("trade_date").is_between(pl.lit(start), pl.lit(end)))
            .collect()
        )
        if universe == "sample":
            daily = daily.filter(pl.col("ts_code").is_in(sample["symbols"]))
        daily = _dedupe(
            daily, ["ts_code", "trade_date"], diagnostics, "daily", CORE_DAILY
        )
        factor = (
            store.scan(factor_name, start, end)
            .select(
                ["ts_code", "trade_date", "cum_adjfactor", "close_price"]
                + (["exchange"] if market == "us" else [])
            )
            .filter(pl.col("trade_date").is_between(pl.lit(start), pl.lit(end)))
            .collect()
        )
        exchanges = (
            factor.sort("trade_date").group_by("ts_code").agg(pl.col("exchange").last())
            if market == "us"
            else None
        )
        if market == "hk":
            reused = _reused_codes(store.read(m["securities"]))
            daily = _assign_reused(daily, reused, diagnostics, "daily")
            factor = _assign_reused(factor, reused, diagnostics, "factor")
            daily = _precise_close(daily, factor, diagnostics)
        factor = _dedupe_factor(factor, diagnostics)
        securities = _securities(store, market, daily, sample, exchanges)
        if universe == "sample":
            symbols = np.array(sorted(sample["symbols"]))
        else:
            symbols = np.array(sorted(set(daily["ts_code"].to_list())))
        diagnostics["factor_rows_outside_daily_symbols"] = factor.filter(
            ~pl.col("ts_code").is_in(list(symbols))
        ).height
        specs = fields_for(market)
        staging = base / f".building-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        written = set()

        def put(name, array):
            # 逐字段落盘后释放，美股近 3 年约 2.8 万列，整体放在内存里要 6 GB。
            np.save(
                staging / f"{name}.npy",
                np.ascontiguousarray(array, dtype=specs[name].dtype),
            )
            written.add(name)

        try:
            coords = _coordinates(daily, dates, symbols)
            close = _fill(daily, coords, "close", (len(dates), len(symbols)))
            volume = _fill(daily, coords, "vol", close.shape)
            amount = _fill(daily, coords, "amount", close.shape)
            # 源在没有成交的日子会给收盘价 0、开高低价为空（美股场外证券常见）：价格非正一律视为缺失。
            close[close <= 0] = np.nan
            for name, raw in (
                ("open", "open"),
                ("high", "high"),
                ("low", "low"),
                ("previous_close", "pre_close"),
            ):
                values = _fill(daily, coords, raw, close.shape)
                values[values <= 0] = np.nan
                put(name, values)
            if market == "us":
                vwap = _fill(daily, coords, "vwap", close.shape)
                put("total_market_cap", _fill(daily, coords, "total_mv", close.shape))
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = np.where(volume > 0, amount / volume, np.nan)
            vwap[vwap <= 0] = np.nan
            put("vwap", vwap)
            del vwap
            has_quote = np.isfinite(close)
            put("close", close)
            put("volume", volume)
            put("amount", amount)
            put("has_quote", has_quote)
            put("tradable", has_quote & (volume > 0))
            del volume, amount
            factor_coords = _coordinates(factor, dates, symbols)
            raw_factor = _fill(factor, factor_coords, "cum_adjfactor", close.shape)
            actions = actions_from_factor(raw_factor, close)
            del raw_factor
            diagnostics["actions"] = {
                str(k): int((actions["action_source"] == k).sum()) for k in range(4)
            }
            for name, array in actions.items():
                put(name, array)
            del actions
            listed, delist = _listing(market, securities, dates, symbols, has_quote)
            diagnostics["quotes_outside_listing"] = int((has_quote & ~listed).sum())
            put("is_listed", listed)
            put("delist", delist)
            del listed, delist
            if market == "hk":
                for name, array in _connect(
                    store, m, dates, symbols, diagnostics
                ).items():
                    put(name, array)
                in_period = (dates >= HK_ROUNDED[0]) & (dates <= HK_ROUNDED[1])
                put(
                    "price_rounded",
                    in_period[:, None] & has_quote & (close < HK_FINE_TICK_BELOW),
                )
            del close, has_quote
            assert written == set(specs), written ^ set(specs)
            metadata = dict(
                version=version,
                market=market,
                universe=universe,
                currency=m["currency"],
                inputs=inputs,
                factor_snapshot=snapshot,
                code_version=code,
                format=FORMAT_VERSION,
                built_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                shape=[len(dates), len(symbols)],
                dates=[str(d) for d in dates],
                symbols=[str(s) for s in symbols],
                fields=describe(market),
                diagnostics=diagnostics,
                action_rule=ACTION_RULE,
                sample=sample if universe == "sample" else None,
            )
            securities.filter(pl.col("ts_code").is_in(list(symbols))).write_parquet(
                staging / "securities.parquet"
            )
            calendar_table.write_parquet(staging / "calendar.parquet")
            metadata["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(staging / "metadata.json", metadata)
            staging.rename(target)
            atomic_json(base / "current.json", {"version": version})
            _collect(base, version, keep)
            return metadata
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _collect(base, current, keep):
    """只保留当前代和最近 keep-1 个旧代。"""
    others = sorted(
        (
            p
            for p in base.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != current
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in others[max(0, keep - 1) :]:
        shutil.rmtree(path)


def _securities(store, market, daily, sample, exchanges=None):
    """统一的证券表：代码、名称、板块/类别、上市退市日、每手（港股，当前值）、币种、首末行情日。"""
    m = spec_for(market)
    quotes = daily.group_by("ts_code").agg(
        pl.col("trade_date").min().alias("first_quote"),
        pl.col("trade_date").max().alias("last_quote"),
    )
    raw = store.read(m["securities"])
    if market == "hk":
        table = raw.unique("ts_code", keep="first").select(
            "ts_code",
            "name",
            pl.col("market").alias("board"),
            "list_status",
            pl.col("list_date").alias("listing_date"),
            pl.col("delist_date").alias("delisting_date"),
            pl.col("trade_unit").cast(pl.Float64).alias("board_lot_current"),
            # 人民币柜台（8xxxx.HK、名称以 R 结尾）源也写 HKD，按代码和名称改为 CNY。
            pl.when(
                pl.col("ts_code").str.contains(r"^8\d{4}\.HK$")
                & pl.col("name").str.ends_with("R")
            )
            .then(pl.lit("CNY"))
            .otherwise(pl.col("curr_type"))
            .alias("currency"),
            pl.col("curr_type").alias("currency_source"),
            "isin",
        )
    else:
        counts = raw.group_by("ts_code").len().rename({"len": "basic_rows"})
        table = (
            raw.sort("ts_code", "list_date", nulls_last=True)
            .unique("ts_code", keep="first", maintain_order=True)
            .select(
                "ts_code",
                "name",
                "enname",
                pl.col("classify").alias("kind"),
                pl.col("list_date").alias("listing_date"),
                pl.col("delist_date").alias("delisting_date"),
            )
            .join(counts, on="ts_code")
            .with_columns(
                (pl.col("basic_rows") > 1).alias("code_reused_in_basic"),
                pl.lit("USD").alias("currency"),
            )
            .drop("basic_rows")
        )
        if exchanges is not None:
            # 交易所取区间内因子表最后一次出现的值（源证券表没有交易所）。
            table = table.join(exchanges, on="ts_code", how="full", coalesce=True)
        else:
            table = table.with_columns(pl.lit(None, pl.String).alias("exchange"))
    table = table.join(quotes, on="ts_code", how="full", coalesce=True)
    if sample is not None:
        table = table.with_columns(
            pl.col("ts_code").is_in(sample["symbols"]).alias("in_sample")
        )
    return table.sort("ts_code")


def _listing(market, securities, dates, symbols, has_quote):
    T, N = len(dates), len(symbols)
    listed = np.zeros((T, N), dtype=bool)
    delist = np.zeros((T, N), dtype=bool)
    info = {r["ts_code"]: r for r in securities.iter_rows(named=True)}
    for j, code in enumerate(symbols):
        row = info.get(code, {})
        if market == "hk" and row.get("listing_date"):
            lo = int(np.searchsorted(dates, row["listing_date"]))
            out = row.get("delisting_date")
            hi = int(np.searchsorted(dates, out)) if out else T
            listed[lo:hi, j] = True
            if out and hi < T:
                delist[hi, j] = True
        else:
            quoted = np.flatnonzero(has_quote[:, j])
            if not len(quoted):
                continue
            listed[quoted[0] : quoted[-1] + 1, j] = True
            if T - 1 - quoted[-1] > DELIST_GRACE:
                delist[quoted[-1] + 1, j] = True
    return listed, delist


def _connect(store, m, dates, symbols, diagnostics):
    """港股通：当天名单（沪、深并集）、只能卖出（南向有持股但不在名单）、名单是否已知、南向持股。"""
    shape = (len(dates), len(symbols))
    frames = []
    known_days = None
    for name in ("hk_connect_sh", "hk_connect_sz"):
        complete = {
            e["partition"]
            for e in store.entries(name).values()
            if e["status"] == "complete" and e["rows"]
        }
        known_days = complete if known_days is None else known_days & complete
        try:
            frames.append(store.read(name, str(dates[0]), str(dates[-1])))
        except FileNotFoundError:
            pass
    listing = (
        pl.concat(frames, how="diagonal_relaxed").select("ts_code", "trade_date")
        if frames
        else pl.DataFrame(schema={"ts_code": pl.String, "trade_date": pl.String})
    )
    eligible = (
        _scatter(listing, dates, symbols, "ts_code", dtype=bool, fill=False)
        if len(listing)
        else np.zeros(shape, dtype=bool)
    )
    known = np.zeros(shape, dtype=bool)
    known[np.isin(dates, sorted(known_days or []))] = True
    try:
        hold = store.read("hk_connect_hold", str(dates[0]), str(dates[-1]))
        hold = hold.filter(pl.col("vol").cast(pl.Float64) > 0).select(
            "ts_code", "trade_date"
        )
    except FileNotFoundError:
        hold = pl.DataFrame(schema={"ts_code": pl.String, "trade_date": pl.String})
    held = (
        _scatter(hold, dates, symbols, "ts_code", dtype=bool, fill=False)
        if len(hold)
        else np.zeros(shape, dtype=bool)
    )
    eligible &= known
    # 南向持股是日终状态、次一交易日公布：判断当天能否卖出用上一交易日的持股。
    held_before = np.zeros(shape, dtype=bool)
    held_before[1:] = held[:-1]
    diagnostics["connect"] = dict(
        known_days=int(known[:, 0].sum()) if shape[1] else 0,
        listed_codes_not_in_panel=listing.filter(
            ~pl.col("ts_code").is_in(list(symbols))
        )["ts_code"].n_unique(),
    )
    return dict(
        connect_eligible=eligible,
        connect_sell_only=known & held_before & ~eligible,
        connect_known=known,
        connect_held=held,
    )


__all__ = ["actions_from_factor", "build_overseas", "classify_ratio"]
