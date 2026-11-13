"""港股、美股「交易日 × 证券」面板的构建（``build_overseas``）。

两个证券范围（universe）各一套代次：

- ``all``：按交易日分区的全部证券（开发库为近 3 年），列是区间内出现过日线的全部代码。
- ``sample``：2010 年起的样本证券（``sample.json``），日线与因子取按证券分区的 ``*_hist``。

存储与 A 股面板相同：``<root>/<市场>/panel_<universe>/<版本>/<字段>.npy``（T×N，C 连续），
``metadata.json``、``securities.parquet``、``calendar.parquet``，``current.json`` 指向当前代。
版本由输入分区的内容哈希、区间、范围、样本名单和构建代码决定，同样输入重建得到同一版本。

公司行为：两地都没有分红送转的事件明细接口，只有前复权累计因子。事件由 ``overseas_actions`` 从相邻
因子比推出并用价格印证，印证不了的不执行、逐格标记。因子是快照，区间内所有因子分区必须来自同一次
同步（``snapshot`` 唯一），否则拒绝构建。

美股只放纽交所、纳斯达克、美交所的普通股与存托凭证（因子表 ``exchange`` 判断，含已退市的；按证券表
类别与代码写法去掉优先股、权证、单位、权利），场外的原始分区留着但不进面板。长缺口两侧价格不连续的
列拆成两段（后段记作 ``代码#2``）。
"""

import hashlib
import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .overseas_actions import (
    CHECKS,
    RULE,
    SOURCES,
    SUSPECT,
    classify_ratio,
    corporate_actions,
    cut_recovery,
    segments,
)
from .overseas_fields import describe, fields_for
from .overseas_sync import US_MAJOR_EXCHANGES, read_sample, spec_for
from .store import atomic_json, lock
from .sync import compact

FORMAT_VERSION = 2
# 版本键哈希的代码：影响面板输出的全部模块（同步模块决定数据集名、币种和样本名单的读法）。
CODE_FILES = (
    "overseas_actions.py",
    "overseas_fields.py",
    "overseas_panel.py",
    "overseas_sync.py",
)
DELIST_GRACE = 20
# 美股：当天有成交却没有三大交易所因子（多为转到场外）的日子不可交易；连续这么多个这样的成交日后退市。
# 开发库近 3 年：因子覆盖期内这样的成交格偶发、连续不超过 4 个成交日后因子恢复（36 次 1 天、
# 2 次 4 天），再长的（22 天起）都是转到场外后的长段；取 5，偶发缺失不会退市。
US_ORPHAN_DAYS = 5
US_GAP_SPLIT = 60
# 美股疑似漏记的合股（见 overseas_actions.segments 的 quiet）：没有执行了的事件、相邻两次成交价格上跳超过
# 3 倍、当天成交股数不超过前 10 个成交日中位数的 1.5 倍、成交额不超过 5 倍时在跳变日拆段。开发库近 3 年
# 命中 12 处（低价股的 ×4 至 ×67），真实拉升的成交额是前 10 日中位数的几十到几百倍，碰不到 5 倍。
US_QUIET_JUMP = 3.0
US_QUIET_VOLUME = 1.5
US_QUIET_AMOUNT = 5.0
US_QUIET_DAYS = 10
# 2026-11-13 实测：hk_daily 在这段日期把全部价格四舍五入到 0.01（低价股失真），之外与因子表一致。
HK_ROUNDED = ("20200702", "20231229")
HK_FINE_TICK_BELOW = 0.5
US_KINDS = ("EQ", "ADR", "GDR")
# 截断不变性：以 end=D 构建与全量构建相比，D 及以前的格逐位相同，只有下面这些字段可以不同，而且只在
# 「这一列在 D 及以前最后一次成交之后」的格（停牌或退市后、下一个成交日还没到的那段）。test 里有测试守住。
LOOKAHEAD_FIELDS = {
    "action_source": "事件之后的下一个成交日在 D 之后：截到 D 时只能记 7（最后成交日之后，不执行）；"
    "全量构建要看那天的前收或价格才定执行还是 6",
    "action_suspect": "随 action_source（6、7）",
    "action_check": "随 action_source：截到 D 时是 6（最后成交日之后）",
    "split": "同上：执行与否要看下一个成交日",
    "equivalent_cash": "同上（港股等效现金）",
    "adj_factor": "从那个事件起累乘的倍数随执行与否变化",
    "delist": "美股退市要知道以后再没有成交：截到 D 时最后一次成交后满 20 个交易日就记退市，"
    "全量构建里之后又有成交则不记",
}
# 同一天有几条交易所序列时的优先顺序（只用当天可知的信息：序列开始日、交易所）。
US_EXCHANGE_PRIORITY = ("NYS", "NAS", "ASE")


def code_version():
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode())
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


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


def _choose_series(factor, allowed, diagnostics):
    """每只证券每天取一条因子序列，并算出当天的因子比 ``ratio``。

    美股同一代码常有几个交易所的因子序列（转板、源回填、口径不同：实测 WMT 的纽交所序列含现金分红、
    纳斯达克序列只含拆股）。规则：只用 ``allowed`` 里的交易所；某天有几条序列时，取开始日最早的那条
    （先有的序列；开始日不晚于当天，当天可知），一样早按交易所优先顺序 ``US_EXCHANGE_PRIORITY``（纽交所、
    纳斯达克、美交所）。不看序列在哪天结束、有多少天：那要用到以后的数据。转板时旧交易所的序列先有、
    先用，它结束后换到新交易所的序列。因子比只在同一条序列内部算：r = 当天值 / 该序列上一行的值，而且该序列上一行必须正好是这只证券
    上一个取用的日子；换序列的那天（拼接处）记 1，不算事件。同一序列同一天多行相对差不超过 1e-6 的
    合并，其余整组丢弃并计数。港股没有交易所列，只有一条序列。
    """
    keys = ["ts_code", "trade_date"]
    if "exchange" not in factor.columns:
        factor = factor.with_columns(pl.lit("").alias("exchange"))
    factor = factor.select(
        *keys, "cum_adjfactor", pl.col("exchange").fill_null("")
    ).filter(pl.col("cum_adjfactor") > 0)
    if allowed is not None:
        factor = factor.filter(pl.col("exchange").is_in(list(allowed)))
    series = ["ts_code", "exchange", "trade_date"]
    groups = factor.group_by(series).agg(
        pl.len().alias("_n"),
        pl.col("cum_adjfactor").min().alias("_lo"),
        pl.col("cum_adjfactor").max().alias("_hi"),
    )
    bad = groups.filter(
        (pl.col("_n") > 1) & ((pl.col("_hi") - pl.col("_lo")) > 1e-6 * pl.col("_hi"))
    ).select(series)
    diagnostics["factor_conflicting_keys"] = len(bad)
    diagnostics["factor_conflicting_examples"] = bad.sort(series).head(20).to_dicts()
    factor = (
        factor.join(bad, on=series, how="anti")
        .group_by(series)
        .agg(pl.col("cum_adjfactor").min())
        .sort(series)
        .with_columns(
            _prev=pl.col("trade_date").shift(1).over("ts_code", "exchange"),
            _r=(pl.col("cum_adjfactor") / pl.col("cum_adjfactor").shift(1)).over(
                "ts_code", "exchange"
            ),
        )
    )
    span = factor.group_by("ts_code", "exchange").agg(
        pl.col("trade_date").min().alias("_start")
    )
    rank = {e: k for k, e in enumerate(US_EXCHANGE_PRIORITY)}
    span = span.with_columns(
        _rank=pl.col("exchange").replace_strict(
            rank, default=len(rank), return_dtype=pl.Int64
        )
    )
    chosen = (
        factor.join(span, on=["ts_code", "exchange"])
        .sort(["ts_code", "trade_date", "_start", "_rank", "exchange"])
        .unique(keys, keep="first", maintain_order=True)
        .sort(keys)
        .with_columns(_prev_used=pl.col("trade_date").shift(1).over("ts_code"))
    )
    splice = pl.col("_prev_used").is_not_null() & (
        pl.col("_prev").is_null() | (pl.col("_prev") != pl.col("_prev_used"))
    )
    chosen = chosen.with_columns(
        ratio=pl.when(pl.col("_prev_used").is_null() | splice)
        .then(1.0)
        .otherwise(pl.col("_r"))
    )
    diagnostics["factor_series_splices"] = chosen.filter(splice).height
    diagnostics["factor_symbols_with_two_series"] = (
        span.group_by("ts_code").len().filter(pl.col("len") > 1).height
    )
    return chosen.select(*keys, "cum_adjfactor", "exchange", "ratio")


def _cash_in_factor(factor_all, chosen):
    """美股：每只证券的因子是否含现金分红（推断）。

    小上调（1 < r < 1.10）几乎只来自现金分红。用到的每条序列都有 → 含；有的有有的没有 → 部分；
    用到的都没有、但同一代码的其他序列（含场外）有 → 不含；都没有 → 未知（不派息与源不含分红分不开）。
    """
    f = (
        factor_all.select(
            "ts_code", "trade_date", "cum_adjfactor", pl.col("exchange").fill_null("")
        )
        .filter(pl.col("cum_adjfactor") > 0)
        .unique(["ts_code", "exchange", "trade_date"], keep="first")
        .sort("ts_code", "exchange", "trade_date")
        .with_columns(
            _r=(pl.col("cum_adjfactor") / pl.col("cum_adjfactor").shift(1)).over(
                "ts_code", "exchange"
            )
        )
    )
    small = (pl.col("_r") > 1 + 1e-5) & (pl.col("_r") < 1.10)
    per = f.group_by("ts_code", "exchange").agg(small.sum().alias("_cash"))
    used = (
        chosen.select("ts_code", "exchange").unique().with_columns(_used=pl.lit(True))
    )
    per = per.join(used, on=["ts_code", "exchange"], how="left").with_columns(
        pl.col("_used").fill_null(False)
    )
    agg = per.group_by("ts_code").agg(
        (pl.col("_used") & (pl.col("_cash") > 0)).sum().alias("_uy"),
        (pl.col("_used") & (pl.col("_cash") == 0)).sum().alias("_un"),
        (~pl.col("_used") & (pl.col("_cash") > 0)).sum().alias("_oy"),
    )
    return agg.select(
        "ts_code",
        pl.when((pl.col("_uy") > 0) & (pl.col("_un") == 0))
        .then(pl.lit("含"))
        .when((pl.col("_uy") > 0) & (pl.col("_un") > 0))
        .then(pl.lit("部分"))
        .when(pl.col("_oy") > 0)
        .then(pl.lit("不含"))
        .otherwise(pl.lit("未知"))
        .alias("factor_includes_cash"),
    )


def us_code_kept(code, kind):
    """美股面板收哪些代码：证券表类别为普通股、ADR、GDR 的都收，优先股不收；没有类别的（多为已退市
    或基金）只收 1–5 位纯字母、且不是第 5 位为 W/U/R（纳斯达克的权证、单位、权利）的代码。"""
    if kind in US_KINDS:
        return True
    if kind == "PF":
        return False
    if not code.isascii() or not code.isalpha() or not code.isupper() or len(code) > 5:
        return False
    return not (len(code) == 5 and code[-1] in "WUR")


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
                    # 两套代次都用样本名单（all 的证券表有 in_sample 列），整份内容进版本。
                    sample=hashlib.sha256(
                        json.dumps(sample, sort_keys=True).encode()
                    ).hexdigest()
                    if sample
                    else None,
                    code=code,
                    # 模块常量也进版本：测试或调用方改了它们不会复用按旧参数建的代次。
                    params=[
                        US_GAP_SPLIT,
                        DELIST_GRACE,
                        US_ORPHAN_DAYS,
                        [
                            US_QUIET_JUMP,
                            US_QUIET_VOLUME,
                            US_QUIET_AMOUNT,
                            US_QUIET_DAYS,
                        ],
                        list(HK_ROUNDED),
                        HK_FINE_TICK_BELOW,
                    ],
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
        _calendar_check(daily, dates, "日线")
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
        # 因子表在休市日偶有行（美股 2014-05-26、2014-09-01，校验已登记）：不是行情，丢弃并计数。
        on_days = pl.col("trade_date").is_in([str(x) for x in dates])
        diagnostics["factor_rows_on_closed_days"] = factor.filter(~on_days).height
        factor = factor.filter(on_days)
        exchanges = cash_info = None
        if market == "hk":
            reused = _reused_codes(store.read(m["securities"]))
            daily = _assign_reused(daily, reused, diagnostics, "daily")
            factor = _assign_reused(factor, reused, diagnostics, "factor")
            daily = _precise_close(daily, factor, diagnostics)
            chosen = _choose_series(factor, None, diagnostics)
        else:
            chosen = _choose_series(factor, US_MAJOR_EXCHANGES, diagnostics)
            daily, chosen = _us_scope(store, m, daily, chosen, diagnostics)
            cash_info = _cash_in_factor(
                factor.filter(
                    pl.col("ts_code").is_in(chosen["ts_code"].unique().implode())
                ),
                chosen,
            )
            exchanges = (
                chosen.sort("trade_date")
                .group_by("ts_code")
                .agg(
                    pl.col("exchange").last(),
                    pl.col("exchange").unique().sort().str.join(",").alias("exchanges"),
                )
            )
        del factor
        if universe == "sample":
            present = set(daily["ts_code"].to_list()) if market == "us" else None
            symbols = np.array(
                sorted(c for c in sample["symbols"] if present is None or c in present)
            )
            diagnostics["sample_symbols_not_in_panel"] = sorted(
                set(sample["symbols"]) - set(symbols.tolist())
            )
        else:
            symbols = np.array(sorted(set(daily["ts_code"].to_list())))
        diagnostics["factor_rows_outside_daily_symbols"] = chosen.filter(
            ~pl.col("ts_code").is_in(list(symbols))
        ).height
        specs = fields_for(market)
        staging = base / f".building-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        written = set()
        shape = (len(dates), len(symbols))

        try:
            coords = _coordinates(daily, dates, symbols)
            close = _fill(daily, coords, "close", shape)
            # 源在没有成交的日子会给收盘价 0、开高低价为空（美股场外证券常见）：价格非正一律视为缺失。
            close[close <= 0] = np.nan
            open_ = _fill(daily, coords, "open", shape)
            open_[open_ <= 0] = np.nan
            volume = _fill(daily, coords, "vol", shape)
            has_quote = np.isfinite(close)
            tradable = has_quote & (volume > 0)
            ratio = _fill(chosen, _coordinates(chosen, dates, symbols), "ratio", shape)
            orphan = np.zeros(shape, dtype=bool)
            if market == "us":
                # 当天有成交却没有三大交易所因子：已不在三大交易所（多为转到场外，报价常有错价），当天就能
                # 判断。这些日子不可交易（行情保留在 has_quote、close 里），不算成交。
                orphan = tradable & ~np.isfinite(ratio)
                tradable = tradable & ~orphan
                diagnostics["trades_without_major_factor"] = int(orphan.sum())
            pre_close = _fill(daily, coords, "pre_close", shape)
            pre_close[pre_close <= 0] = np.nan
            coarse = (dates >= HK_ROUNDED[0]) & (dates <= HK_ROUNDED[1])
            actions = corporate_actions(
                market,
                ratio,
                close,
                open_,
                tradable,
                previous_close=pre_close if market == "hk" else None,
                coarse=coarse,
            )
            del ratio
            if market == "us":
                amount = _fill(daily, coords, "amount", shape)
                # 成交额缺失（2016 年前多为空）时按收盘价 × 成交量估。
                dollar = np.where(
                    np.isfinite(amount) & (amount > 0), amount, close * volume
                )
                del amount
                quiet_cuts = []
                parts = segments(
                    close,
                    tradable,
                    actions["multiplier"],
                    actions["action_suspect"],
                    US_GAP_SPLIT,
                    volume=volume,
                    amount=dollar,
                    quiet=(
                        US_QUIET_JUMP,
                        US_QUIET_VOLUME,
                        US_QUIET_AMOUNT,
                        US_QUIET_DAYS,
                    ),
                    found=quiet_cuts,
                )
                del dollar
                diagnostics["quiet_jump_cuts"] = [
                    [str(symbols[j]), str(dates[b])] for j, b in quiet_cuts
                ]
            else:
                parts = [(j, 0, len(dates)) for j in range(len(symbols))]
            colmap = np.array([p[0] for p in parts], dtype=np.int64)
            rows = np.arange(len(dates))[:, None]
            keep_cells = (rows >= np.array([p[1] for p in parts])) & (
                rows < np.array([p[2] for p in parts])
            )
            split_any = len(parts) != len(symbols)
            names, seen = [], {}
            for j, _, _ in parts:
                seen[j] = seen.get(j, 0) + 1
                symbol = str(symbols[j])
                names.append(symbol if seen[j] == 1 else f"{symbol}#{seen[j]}")
            diagnostics["segments_split"] = len(parts) - len(symbols)
            # 拆段处旧列在复牌日（新列第一行）退市，按偏悲观的清算价了结（见 cut_recovery）。
            cuts = {}
            for k, (j, _, end) in enumerate(parts):
                if end < len(dates):
                    cuts[k] = (
                        end,
                        cut_recovery(close, tradable, actions["multiplier"], j, end),
                    )

            def seg(array, fill):
                if not split_any:
                    return array
                out = array[:, colmap]
                out[~keep_cells] = fill
                return out

            def put(name, array, fill=np.nan):
                # 逐字段落盘后释放：美股近 3 年约 9000 列，所有字段一起放在内存里要几 GB。
                np.save(
                    staging / f"{name}.npy",
                    np.ascontiguousarray(seg(array, fill), dtype=specs[name].dtype),
                )
                written.add(name)

            fills = dict(
                split=1.0,
                equivalent_cash=0.0,
                action_source=0,
                action_suspect=False,
                action_check=0,
            )
            for name in (
                "adj_factor",
                "action_ratio",
                "split",
                "equivalent_cash",
                "action_source",
                "action_suspect",
                "action_check",
            ):
                put(name, actions[name], fills.get(name, np.nan))
            source = seg(actions["action_source"], 0)
            diagnostics["actions"] = {str(k): int((source == k).sum()) for k in SOURCES}
            check = seg(actions["action_check"], 0)
            diagnostics["action_checks"] = {
                str(k): int((check == k).sum()) for k in CHECKS
            }
            # 只靠当日收盘价才印证、而且执行了的事件格（日内前视）。
            diagnostics["executed_close_only"] = int(
                ((check == 3) & np.isin(source, (1, 2, 3, 4))).sum()
            )
            del check
            multiplier = actions["multiplier"]
            put("close", close)
            put("open", open_)
            put("volume", volume)
            put("has_quote", has_quote, False)
            put("tradable", tradable, False)
            del volume
            del actions
            put("previous_close", pre_close)
            del pre_close
            for name in ("high", "low"):
                values = _fill(daily, coords, name, shape)
                values[values <= 0] = np.nan
                put(name, values)
            amount = _fill(daily, coords, "amount", shape)
            put("amount", amount)
            if market == "us":
                vwap = _fill(daily, coords, "vwap", shape)
                put("total_market_cap", _fill(daily, coords, "total_mv", shape))
            else:
                volume = _fill(daily, coords, "vol", shape)
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = np.where(volume > 0, amount / volume, np.nan)
                del volume
            vwap[vwap <= 0] = np.nan
            put("vwap", vwap)
            del vwap, amount
            final_quote = seg(has_quote, False)
            final_trade = seg(tradable, False)
            final_orphan = seg(orphan, False)
            final_close = seg(close, np.nan)
            securities = _securities(store, market, sample, exchanges, cash_info)
            table = _column_table(
                names, parts, symbols, dates, final_quote, source, securities
            )
            recovery = np.full((len(dates), len(names)), np.nan)
            if market == "hk":
                listed, delist = _listing_hk(table, dates)
            else:
                listed, delist, recovery, gone = _listing_us(
                    final_trade,
                    cuts=cuts,
                    orphan=final_orphan,
                    close=final_close,
                    multiplier=seg(multiplier, 1.0),
                )
                diagnostics["orphan_delists"] = gone
            diagnostics["cut_delists"] = len(cuts)
            diagnostics["quotes_outside_listing"] = int((final_quote & ~listed).sum())
            np.save(staging / "is_listed.npy", listed)
            np.save(staging / "delist.npy", delist)
            written |= {"is_listed", "delist"}
            if "delist_recovery" in specs:
                np.save(staging / "delist_recovery.npy", recovery)
                written.add("delist_recovery")
            del listed, delist, final_trade, final_orphan, final_close, source, recovery
            if market == "hk":
                for name, array in _connect(
                    store, m, dates, symbols, diagnostics
                ).items():
                    put(name, array, False)
                put("price_rounded", price_rounded(dates, has_quote, close), False)
            del close, has_quote, open_, final_quote
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
                shape=[len(dates), len(names)],
                dates=[str(d) for d in dates],
                symbols=names,
                fields=describe(market),
                diagnostics=diagnostics,
                action_rule=RULE,
                action_sources={str(k): v for k, v in SOURCES.items()},
                action_checks={str(k): v for k, v in CHECKS.items()},
                sample=sample if universe == "sample" else None,
            )
            table.write_parquet(staging / "securities.parquet")
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


def price_rounded(dates, has_quote, close):
    """港股源日线四舍五入期内、收盘价低于 0.5 港元（最小价位小于 0.01）的格。"""
    in_period = (dates >= HK_ROUNDED[0]) & (dates <= HK_ROUNDED[1])
    with np.errstate(invalid="ignore"):
        return in_period[:, None] & has_quote & (close < HK_FINE_TICK_BELOW)


def _calendar_check(frame, dates, what):
    """区间内的行必须落在交易日历的开市日上：日历把开市日标成休市时不许静默丢掉整天的行情。"""
    stray = (
        frame.select("trade_date")
        .unique()
        .filter(~pl.col("trade_date").is_in([str(d) for d in dates]))
        .sort("trade_date")["trade_date"]
        .to_list()
    )
    if stray:
        raise ValueError(
            f"{what}有 {len(stray)} 个日期不在交易日历的开市日里（{', '.join(stray[:5])}）；"
            "日历或行情有误，拒绝构建"
        )


def _us_scope(store, m, daily, chosen, diagnostics):
    """美股面板只放三大交易所的普通股与存托凭证（见 ``us_code_kept``）。

    行情从第一次有三大交易所因子的那天起取（当天可知），之后一直取到日线自身的范围为止：不用因子序列的
    结束日截断（那要用到以后的数据）。因子结束后的行情（多为转到场外）没有因子，不会生成公司行为。
    """
    basic = store.read(m["securities"])
    kinds = (
        basic.sort("ts_code", "classify", nulls_last=True)
        .group_by("ts_code", maintain_order=True)
        .agg(pl.col("classify").drop_nulls().first().alias("kind"))
    )
    cover = (
        chosen.group_by("ts_code")
        .agg(pl.col("trade_date").min().alias("_lo"))
        .join(kinds, on="ts_code", how="left")
    )
    kept = cover.filter(
        pl.struct("ts_code", "kind").map_elements(
            lambda r: us_code_kept(r["ts_code"], r["kind"]), return_dtype=pl.Boolean
        )
    )
    codes = set(daily["ts_code"].unique().to_list())
    diagnostics["us_scope"] = dict(
        daily_codes=len(codes),
        without_major_exchange_factor=len(codes - set(cover["ts_code"].to_list())),
        dropped_by_kind=cover.height - kept.height,
        kept=len(codes & set(kept["ts_code"].to_list())),
    )
    before = daily.height
    daily = (
        daily.join(kept.select("ts_code", "_lo"), on="ts_code")
        .filter(pl.col("trade_date") >= pl.col("_lo"))
        .drop("_lo")
    )
    diagnostics["us_scope"]["daily_rows_dropped"] = before - daily.height
    chosen = chosen.filter(pl.col("ts_code").is_in(kept["ts_code"].implode()))
    return daily, chosen


def _column_table(names, parts, symbols, dates, quote, source, securities):
    """面板每一列一行：拆段后的代码、所属证券、首末行情日、可疑公司行为的格数与日期。"""
    rows = []
    for k, (name, (j, _, _)) in enumerate(zip(names, parts, strict=True)):
        q = np.flatnonzero(quote[:, k])
        bad = np.flatnonzero(np.isin(source[:, k], SUSPECT))
        rows.append(
            dict(
                symbol=name,
                ts_code=str(symbols[j]),
                segment=int(name.split("#")[1]) if "#" in name else 1,
                first_quote=str(dates[q[0]]) if len(q) else None,
                last_quote=str(dates[q[-1]]) if len(q) else None,
                action_suspect=bool(len(bad)),
                action_suspect_dates=",".join(str(dates[t]) for t in bad[:20]) or None,
            )
        )
    table = pl.DataFrame(
        rows,
        schema_overrides=dict(
            first_quote=pl.String, last_quote=pl.String, action_suspect_dates=pl.String
        ),
    )
    return (
        table.join(securities, on="ts_code", how="left")
        .drop("ts_code")
        .rename({"symbol": "ts_code"})
        .select("ts_code", pl.exclude("ts_code"))
    )


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


def _securities(store, market, sample, exchanges=None, cash_info=None):
    """证券的静态信息（每只证券一行）：名称、板块或类别、上市退市日、每手（港股，当前值）、币种。"""
    m = spec_for(market)
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
            pl.when(_rmb_counter())
            .then(pl.lit("CNY"))
            .otherwise(pl.col("curr_type"))
            .alias("currency"),
            pl.col("curr_type").alias("currency_source"),
            "isin",
        )
        # 人民币柜台与港币柜台是同一家公司：primary_symbol 指向港币柜台（0 + 后四位，名称去掉末尾的
        # R 和随之空出的连字符后相同才认：腾讯控股-R ↔ 腾讯控股、美团-WR ↔ 美团-W），其余证券指向自己。
        # 截面研究按它去重；对不上的人民币柜台为空。
        names = dict(zip(table["ts_code"], table["name"], strict=True))

        def primary(code, name, rmb):
            if not rmb:
                return code
            other = "0" + code[1:]
            return other if names.get(other) == name[:-1].removesuffix("-") else None

        table = table.with_columns(
            pl.struct("ts_code", "name", _rmb_counter().alias("_rmb"))
            .map_elements(
                lambda r: primary(r["ts_code"], r["name"] or "", r["_rmb"]),
                return_dtype=pl.String,
            )
            .alias("primary_symbol"),
            pl.lit("含").alias("factor_includes_cash"),
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
            # 交易所取区间内取用的因子序列最后一天的交易所；exchanges 列出用到过的全部交易所。
            table = table.join(exchanges, on="ts_code", how="full", coalesce=True)
        if cash_info is not None:
            table = table.join(cash_info, on="ts_code", how="left")
    if sample is not None:
        table = table.with_columns(
            pl.col("ts_code").is_in(sample["symbols"]).alias("in_sample")
        )
    return table.sort("ts_code")


def _rmb_counter():
    return pl.col("ts_code").str.contains(r"^8\d{4}\.HK$") & pl.col(
        "name"
    ).str.ends_with("R")


def _listing_hk(table, dates):
    """港股：证券表上市日（含）至退市日（不含）；退市日（或其后首个交易日）delist 为真。"""
    T, N = len(dates), table.height
    listed = np.zeros((T, N), dtype=bool)
    delist = np.zeros((T, N), dtype=bool)
    for j, row in enumerate(table.iter_rows(named=True)):
        if not row.get("listing_date"):
            continue
        lo = int(np.searchsorted(dates, row["listing_date"]))
        out = row.get("delisting_date")
        hi = int(np.searchsorted(dates, out)) if out else T
        listed[lo:hi, j] = True
        if out and hi < T:
            delist[hi, j] = True
    return listed, delist


def _listing_us(
    tradable,
    grace=DELIST_GRACE,
    cuts=None,
    orphan=None,
    close=None,
    multiplier=None,
    days=None,
):
    """美股没有可靠的上市退市日，按成交推断，只用当时可知的信息：

    第一次成交起为上市；连续 ``grace`` 个交易日没有成交后的下一天起不算上市（之前那 grace 天还不知道
    停多久，仍为真），再有成交又算上市。最后一次成交后满 grace 个交易日的下一天 delist 为真（之后再
    没有成交时；引擎按那时的最后价格了结）。

    ``orphan``：当天有成交却没有三大交易所因子的格（不可交易，不算成交）。还在上市时连续 ``days``
    （默认 ``US_ORPHAN_DAYS``）个这样的成交日，第 ``days`` 个那天退市：从那天起不算上市，直到再有
    （带因子的）成交；按与拆段相同的偏悲观口径清算，回收率见 ``cut_recovery``（清算价 =
    min(最后一个可交易日收盘, 当天收盘 × 其间执行了的倍数)）。

    ``cuts``：{列: (新列开始行, 回收率)}，拆段处的旧列：从新列开始那天起不算上市（新旧两列不重叠），
    delist 放在那一天，不再按「满 grace 天」另记；最后一次成交之后已因无因子退市的不再记。

    返回 (is_listed, delist, delist_recovery, 无因子退市次数)。
    """
    days = US_ORPHAN_DAYS if days is None else days
    T, N = tradable.shape
    listed = np.zeros((T, N), dtype=bool)
    delist = np.zeros((T, N), dtype=bool)
    recovery = np.full((T, N), np.nan)
    rows = np.arange(T)
    gone = 0
    for j in range(N):
        trades = np.flatnonzero(tradable[:, j])
        if not len(trades):
            continue
        last = np.maximum.accumulate(np.where(tradable[:, j], rows, -1))
        listed[:, j] = (last >= 0) & (rows - last <= grace)
        downs = []
        if orphan is not None and orphan[:, j].any():
            run, down = 0, None
            for t in np.flatnonzero(tradable[:, j] | orphan[:, j]):
                if tradable[t, j]:
                    if down is not None:
                        listed[down:t, j] = False
                    run, down = 0, None
                    continue
                run += 1
                if (
                    run == days
                    and down is None
                    and 0 <= last[t]
                    and t - last[t] <= grace
                ):
                    down = int(t)
                    downs.append(down)
                    delist[t, j] = True
                    recovery[t, j] = cut_recovery(close, tradable, multiplier, j, t)
                    gone += 1
            if down is not None:
                listed[down:, j] = False
        final_down = bool(downs) and downs[-1] > trades[-1]
        if cuts and j in cuts:
            cut, value = cuts[j]
            listed[cut:, j] = False
            if not final_down:
                delist[cut, j] = True
                recovery[cut, j] = value
            continue
        end = trades[-1] + grace + 1
        if end < T and not final_down:
            delist[end, j] = True
    return listed, delist, recovery, gone


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
    # 南向持股是日终状态、次一交易日公布：t 日只知道 t−1 日终的持股，面板整体错位一天。
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
        connect_held=held_before,
    )


__all__ = ["LOOKAHEAD_FIELDS", "build_overseas", "classify_ratio", "us_code_kept"]
