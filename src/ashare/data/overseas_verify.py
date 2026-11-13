"""港股、美股原始库与面板的独立校验（``verify_overseas``）。

与同步、面板构建不共用代码：直接按清单读原始 Parquet，按自己的写法重算。每项判 ``pass`` /
``explained`` / ``fail`` / ``info``；「已解释」只认已知例外清单
（``overseas_known_exceptions.json``，逐条写市场、检查项、匹配键和原因编号，不放由行情算出的
数值），清单外的不符一律 ``fail``，有一项 ``fail`` 总结论就是 ``fail``。面板部分不允许例外。

最短例子::

    from ashare.data.store import Store
    from ashare.data.overseas_verify import verify_overseas

    report = verify_overseas(Store("/path/to/root", "hk"))
    report["verdict"], report["checks"]["daily_ranges"]
"""

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .catalog import OVERSEAS_DATASETS
from .store import atomic_json

EXCEPTIONS_PATH = Path(__file__).with_name("overseas_known_exceptions.json")
# 与 overseas_sync 的声明各写一份：校验不信任采集代码。
NAMES = {
    "hk": dict(
        calendar="hk_calendar",
        securities="hk_securities",
        daily="hk_daily",
        factor="hk_adjfactor",
        daily_hist="hk_daily_hist",
        factor_hist="hk_adjfactor_hist",
        extra=("hk_connect_sh", "hk_connect_sz", "hk_connect_hold"),
    ),
    "us": dict(
        calendar="us_calendar",
        securities="us_securities",
        daily="us_daily",
        factor="us_adjfactor",
        daily_hist="us_daily_hist",
        factor_hist="us_adjfactor_hist",
        extra=(),
    ),
}
# 每项检查允许的匹配键（清单按白名单核对，不允许放数值）。
EXCEPTION_KEYS = {
    "partitions": {"dataset", "partition"},
    "unique_keys": {"dataset", "ts_code", "trade_date", "kind", "date_from", "date_to"},
    "daily_ranges": {
        "dataset",
        "ts_code",
        "trade_date",
        "rule",
        "date_from",
        "date_to",
    },
    "factor_price": {
        "ts_code",
        "trade_date",
        "kind",
        "exchange",
        "date_from",
        "date_to",
    },
    "factor_vs_previous_close": {
        "ts_code",
        "trade_date",
        "exchange",
        "kind",
        "date_from",
        "date_to",
    },
    "quotes_vs_listing": {"ts_code", "side"},
    "price_jumps": {"ts_code", "trade_date", "kind"},
    "hist_vs_day": {"dataset", "ts_code", "trade_date"},
    "calendar": {"cal_date"},
    "rows_on_closed_days": {"dataset", "ts_code", "trade_date", "date_from", "date_to"},
}
EXAMPLES = 20
FACTOR_TOL = 5e-3
US_MAJOR = ("NYS", "NAS", "ASE")
HK_ROUNDED = ("20200702", "20231229")
GRACE = 20
# 美股：连续这么多个「有成交、没有三大交易所因子」的成交日后退市（见 HKUS_DATA.md 第三节）。
US_ORPHAN = 5
# 美股价格跳变：放大到前 10 个可交易日中位数这么多倍的，算交易印证的真实行情，不需要登记；其余逐条登记
# （代码、日期）。上跳看成交股数：成交额 ≈ 价格 × 股数，漏记的 N 合 1 让股数 ÷N 而成交额照样随活跃度
# 放大，用成交额会把「活跃的一天 + 漏记的合股」算成真实行情。下跳看成交额（缺失时用收盘价 × 成交量）：
# 漏记的 1 拆 N 让股数 ×N 而成交额不变。两处都取 5 倍，同一个「交易明显放大」的标准。
JUMP_SHARES_CONFIRM = 5.0
JUMP_AMOUNT_CONFIRM = 5.0
JUMP_BASE_DAYS = 10
# 疑似漏记的合股（构建侧应已拆段）：上跳超过 3 倍、成交股数不超过前 10 个可交易日中位数的 1.5 倍。
QUIET_JUMP, QUIET_VOLUME = 3.0, 1.5
# 必须逐条登记、不许按类别整体豁免的检查项。
EXCEPTION_REQUIRED = {"price_jumps": {"ts_code", "trade_date"}}
RATIO_NOISE = 1e-5
# 港交所价位表（2025-08-04 前的旧表，较粗的一档；只用来放宽前收取整的容差）。
HK_TICKS = (
    (0.25, 0.001),
    (0.5, 0.005),
    (10, 0.01),
    (20, 0.02),
    (100, 0.05),
    (200, 0.1),
    (500, 0.2),
    (1000, 0.5),
    (2000, 1.0),
    (5000, 2.0),
    (math.inf, 5.0),
)


def hk_tick(price):
    for upper, tick in HK_TICKS:
        if price <= upper:
            return tick
    return 5.0


def _now():
    return datetime.now(UTC).isoformat()


def load_exceptions(market, path=EXCEPTIONS_PATH):
    if not Path(path).exists():
        return dict(entries=[], reasons={}, cause_unknown=[])
    value = json.loads(Path(path).read_text())
    entries = [e for e in value.get("entries", []) if e.get("market") == market]
    for e in entries:
        allowed = EXCEPTION_KEYS.get(e["check"])
        if allowed is None:
            raise ValueError(f"已知例外里有未知检查项 {e['check']}")
        keys = set(e) - {"market", "check", "reason"}
        if not keys <= allowed:
            raise ValueError(f"已知例外 {e} 含不允许的键 {sorted(keys - allowed)}")
        required = EXCEPTION_REQUIRED.get(e["check"], set())
        if not required <= keys:
            raise ValueError(f"已知例外 {e} 必须逐条写明 {sorted(required - keys)}")
        if e.get("reason") not in value.get("reasons", {}):
            raise ValueError(f"已知例外 {e} 的原因编号不存在")
    return dict(
        entries=entries,
        reasons=value.get("reasons", {}),
        cause_unknown=value.get("cause_unknown", []),
    )


class Checks:
    def __init__(self, market, exceptions):
        self.market = market
        self.exceptions = exceptions
        self.used = set()
        self.results = {}

    def _match(self, check, record):
        for i, e in enumerate(self.exceptions["entries"]):
            if e["check"] != check:
                continue
            keys = set(e) - {"market", "check", "reason", "date_from", "date_to"}
            day = record.get("trade_date") or ""
            if "date_from" in e and not day >= e["date_from"]:
                continue
            if "date_to" in e and not day <= e["date_to"]:
                continue
            if all(str(record.get(k)) == str(e[k]) for k in keys):
                self.used.add(i)
                return e["reason"]
        return None

    def add(self, name, checked, problems, *, info=False, exceptions=True, note=""):
        """problems：不符记录的列表（dict），按已知例外分为已解释与失败。"""
        failed, explained = [], []
        for record in problems:
            reason = self._match(name, record) if exceptions and not info else None
            (explained if reason else failed).append(
                {**record, **({"reason": reason} if reason else {})}
            )
        verdict = (
            "info"
            if info
            else "fail"
            if failed
            else "explained"
            if explained
            else "pass"
        )
        self.results[name] = dict(
            verdict=verdict,
            checked=int(checked),
            failed=0 if info else len(failed),
            explained=len(explained),
            examples=(failed if not info else problems)[:EXAMPLES],
            explained_examples=explained[:EXAMPLES],
            note=note,
        )
        if info:
            self.results[name]["count"] = len(problems)


def _complete(store, name):
    return {
        e["partition"]: e
        for e in store.manifest().values()
        if e.get("dataset") == name and e.get("status") == "complete" and e.get("file")
    }


def _read(store, entries, codes=None):
    """读一批分区。``codes`` 给出时每个文件先过滤代码再读进内存（美股日线大多是场外代码）。"""
    files = [store.root / e["file"] for e in entries if e.get("rows")]
    if not files:
        return pl.DataFrame()
    if codes is None:
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    wanted = pl.Series(sorted(codes), dtype=pl.String)
    frames = [
        pl.scan_parquet(f).filter(pl.col("ts_code").is_in(wanted.implode())).collect()
        for f in files
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _open_days(calendar):
    return sorted(
        calendar.filter(pl.col("is_open").cast(pl.Int64) == 1)["cal_date"].to_list()
    )


def verify_overseas(
    store, *, universe="all", exceptions_path=EXCEPTIONS_PATH, output=None
):
    market = store.market
    names = NAMES[market]
    started = time.perf_counter()
    checks = Checks(market, load_exceptions(market, exceptions_path))
    manifest = store.manifest()

    # 1. 清单对文件：本市场的每个分区完成、文件存在、哈希一致。
    problems = []
    for key, entry in sorted(manifest.items()):
        if entry.get("dataset") not in OVERSEAS_DATASETS:
            problems.append(dict(key=key, problem="非港美股数据集"))
            continue
        if entry.get("status") != "complete":
            problems.append(dict(key=key, problem=f"status={entry.get('status')}"))
            continue
        path = store.root / entry["file"]
        if not path.is_file():
            problems.append(dict(key=key, problem="文件缺失"))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_hash"]:
            problems.append(dict(key=key, problem="文件哈希不符"))
    checks.add("manifest_files", len(manifest), problems, exceptions=False)

    calendar = _read(store, _complete(store, names["calendar"]).values())
    days = _open_days(calendar)

    # 2. 日历自洽：每个日子的 pretrade_date 是上一个开市日。
    problems = []
    ordered = calendar.sort("cal_date")
    previous_open = None
    # 只核对开市日：休市日的 pretrade_date 在源的早年数据里口径不一（周六写成周四）。
    for row in ordered.iter_rows(named=True):
        if int(row["is_open"]) != 1:
            continue
        if previous_open and row["pretrade_date"] != previous_open:
            problems.append(
                dict(cal_date=row["cal_date"], rule="pretrade_date 不是上一开市日")
            )
        previous_open = row["cal_date"]
    checks.add("calendar", len(ordered), problems)

    # 3. 日分区覆盖：区间内每个开市日都有完成分区；空分区、整页行数（疑似截断）逐条报。
    problems = []
    checked = 0
    for name in (names["daily"], names["factor"], *names["extra"]):
        parts = _complete(store, name)
        if not parts:
            continue
        spec = OVERSEAS_DATASETS[name]
        lo, hi = min(parts), max(parts)
        lo = max(lo, spec.coverage_start or lo)
        for day in days:
            if not lo <= day <= hi:
                continue
            checked += 1
            entry = parts.get(day)
            if entry is None:
                problems.append(dict(dataset=name, partition=day, problem="缺分区"))
            elif not entry["rows"] and not spec.empty_allowed:
                problems.append(dict(dataset=name, partition=day, problem="空分区"))
            elif (
                entry["rows"]
                and entry["rows"] % spec.limit == 0
                and entry.get("calls", 0) <= entry["rows"] // spec.limit
            ):
                # 恰好整页且没有再翻一页：可能被截断。翻到空页为止的（calls 多一次）不算。
                problems.append(
                    dict(
                        dataset=name,
                        partition=day,
                        problem="行数是页大小整数倍且没有再翻页",
                    )
                )
        extra_days = sorted(set(parts) - set(days))
        for day in extra_days:
            problems.append(dict(dataset=name, partition=day, problem="休市日有分区"))
    checks.add("partitions", checked, problems)

    daily_parts = _complete(store, names["daily"])
    factor_parts = _complete(store, names["factor"])
    factor = _read(store, factor_parts.values())
    securities = _read(store, _complete(store, names["securities"]).values())
    scope = None
    if market == "us" and len(factor):
        # 美股面板只放三大交易所的普通股与存托凭证：原始库检查也只看这些代码、这些交易所的因子。
        # 日线按这批代码过滤后再读进内存。
        scope = _us_scope_codes(factor, securities)
        daily = _read(store, daily_parts.values(), scope)
        factor = factor.filter(
            pl.col("ts_code").is_in(sorted(scope)) & pl.col("exchange").is_in(US_MAJOR)
        )
    else:
        daily = _read(store, daily_parts.values())

    # 4. 主键唯一（同一天同一代码多行：美股代码复用、源重复行）。
    #    kind：「同值重复」= 价格量额（因子表为因子，相对差 1e-6 内）相同、只差交易所或估值字段；
    #    「取值冲突」= 同一代码同一天两套不同的数。
    problems = []
    for name, frame, core in (
        (
            names["daily"],
            daily,
            ["open", "high", "low", "close", "pre_close", "vol", "amount"],
        ),
        (names["factor"], factor, ["cum_adjfactor"]),
    ):
        if not len(frame):
            continue
        if name == names["factor"]:
            counts = frame.group_by("ts_code", "trade_date").agg(
                pl.len().alias("len"),
                (
                    (
                        pl.col("cum_adjfactor").max() - pl.col("cum_adjfactor").min()
                    ).abs()
                    <= 1e-6 * pl.col("cum_adjfactor").max().abs()
                ).alias("same"),
            )
        else:
            counts = frame.group_by("ts_code", "trade_date").agg(
                pl.len().alias("len"),
                (pl.struct(core).n_unique() == 1).alias("same"),
            )
        for row in (
            counts.filter(pl.col("len") > 1)
            .sort("trade_date", "ts_code")
            .iter_rows(named=True)
        ):
            problems.append(
                dict(
                    dataset=name,
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    kind="同值重复" if row["same"] else "取值冲突",
                )
            )
    checks.add("unique_keys", len(daily) + len(factor), problems)

    # 5. 日线取值范围：价格为正；成交量、额非负；最高 ≥ 最低。
    #    港股开收盘价来自竞价、最高最低只统计连续交易（源文档说明），开收盘在高低之外只计信息。
    problems, crossing = [], []
    for row in (
        daily.filter(
            (pl.col("close") <= 0)
            | (pl.col("open") <= 0)
            | (pl.col("high") < pl.col("low"))
            | (pl.col("vol") < 0)
            | (pl.col("amount") < 0)
        )
        .sort("trade_date", "ts_code")
        .iter_rows(named=True)
        if len(daily)
        else []
    ):
        rule = (
            "价格非正"
            if any(row[c] is not None and row[c] <= 0 for c in ("close", "open"))
            else "最高低于最低"
            if row["high"] is not None
            and row["low"] is not None
            and row["high"] < row["low"]
            else "量额为负"
        )
        problems.append(
            dict(
                dataset=names["daily"],
                ts_code=row["ts_code"],
                trade_date=row["trade_date"],
                rule=rule,
            )
        )
    if len(daily):
        outside = daily.filter(
            (pl.col("open") > pl.col("high"))
            | (pl.col("open") < pl.col("low"))
            | (pl.col("close") > pl.col("high"))
            | (pl.col("close") < pl.col("low"))
        )
        crossing = outside.select("ts_code", "trade_date").head(EXAMPLES).to_dicts()
        crossing_count = len(outside)
    else:
        crossing_count = 0
    checks.add("daily_ranges", len(daily), problems)
    checks.add(
        "open_close_outside_high_low",
        len(daily),
        [{}] * crossing_count,
        info=True,
        note=f"样例 {crossing}",
    )

    # 6. 因子快照唯一。
    snaps = defaultdict(int)
    for name in (names["factor"], names["factor_hist"]):
        for e in _complete(store, name).values():
            snaps[(name, e.get("snapshot"))] += 1
    problems = []
    by_name = defaultdict(set)
    for (name, snap), _ in snaps.items():
        by_name[name].add(snap)
    for name, values in by_name.items():
        if len(values) > 1 or None in values:
            problems.append(dict(dataset=name, snapshots=sorted(map(str, values))))
    checks.add("factor_snapshot", sum(snaps.values()), problems, exceptions=False)

    # 7. 因子表 close_price 与日线收盘价一致（同一源、同一天；美股每条交易所序列分别核）。
    problems = []
    joined = pl.DataFrame()
    if len(daily) and len(factor):
        d1 = daily.unique(["ts_code", "trade_date"], keep="none").select(
            "ts_code", "trade_date", "close", "pre_close"
        )
        if "exchange" not in factor.columns:
            factor = factor.with_columns(pl.lit(None, pl.String).alias("exchange"))
        f1 = factor.unique(["ts_code", "exchange", "trade_date"], keep="none").select(
            "ts_code", "trade_date", "cum_adjfactor", "close_price", "exchange"
        )
        joined = d1.join(f1, on=["ts_code", "trade_date"], how="inner")
        gap = (pl.col("close") - pl.col("close_price")).abs()
        bad = joined.filter(
            pl.col("close_price").is_not_null()
            & (gap > 1e-6 + 1e-6 * pl.col("close").abs())
        ).with_columns(within=gap <= 0.0051)
        for row in bad.sort("trade_date", "ts_code").iter_rows(named=True):
            problems.append(
                dict(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    exchange=row["exchange"],
                    kind="半分以内" if row["within"] else "超过半分",
                )
            )
    checks.add("factor_price", len(joined), problems)

    # 8. 因子与前收（港股，每对相邻交易日都核）：源的前收已除权，F[t]/F[t−1] 应等于
    #    close[t−1]/pre_close[t]（没有事件的日子即前收 = 上一日收盘）。允许 0.5% 或一个价位的差。
    #    美股前收不除权，只能看价格跳变：因子与价格矛盾的事件列为信息，面板必须不执行（见 panel）。
    problems = []
    events = 0
    if len(joined):
        series = joined.sort("ts_code", "exchange", "trade_date")
        codes = series["ts_code"].to_list()
        venues = series["exchange"].to_list()
        dates = series["trade_date"].to_list()
        closes = series["close"].to_numpy()
        pres = series["pre_close"].to_numpy()
        facs = series["cum_adjfactor"].to_numpy()
        previous_day = {days[k]: days[k - 1] for k in range(1, len(days))}
        for i in range(1, len(codes)):
            if codes[i] != codes[i - 1] or venues[i] != venues[i - 1]:
                continue
            # 只核对相邻交易日：中间缺日线（停牌、无成交）时除权日落在缺口里，无从对照。
            if previous_day.get(dates[i]) != dates[i - 1]:
                continue
            a, b = facs[i - 1], facs[i]
            if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
                continue
            r = b / a
            moved = abs(r - 1) > RATIO_NOISE
            if market == "hk":
                events += 1
                if not (pres[i] and closes[i - 1]):
                    continue
                implied = closes[i - 1] / (r if moved else 1.0)
                within = (
                    abs(implied / pres[i] - 1) <= FACTOR_TOL
                    or abs(implied - pres[i]) <= hk_tick(pres[i]) + 1e-9
                )
                if not within:
                    problems.append(
                        dict(
                            ts_code=codes[i],
                            trade_date=dates[i],
                            exchange=venues[i],
                            kind="因子变化日" if moved else "无事件日",
                        )
                    )
            elif moved:
                events += 1
                move = closes[i] / closes[i - 1] * r if closes[i - 1] else math.nan
                if not (math.isfinite(move) and 1 / 3 <= move <= 3):
                    problems.append(
                        dict(ts_code=codes[i], trade_date=dates[i], exchange=venues[i])
                    )
    checks.add(
        "factor_vs_previous_close",
        events,
        problems,
        info=market == "us",
        note=(
            "港股：相邻两个交易日都有日线的每一对（含没有事件的日子）"
            if market == "hk"
            else "美股：相邻交易日上因子比与价格跳变矛盾的事件（源错误）；面板不许执行它们，由 panel 检查"
        ),
    )

    # 9. 大幅跳价但因子未变（可能缺拆并股、代码复用或源错误），只计信息。
    jumps = []
    if len(joined):
        s = joined.sort("ts_code", "exchange", "trade_date").with_columns(
            prev_close=pl.col("close").shift(1).over("ts_code", "exchange"),
            prev_factor=pl.col("cum_adjfactor").shift(1).over("ts_code", "exchange"),
        )
        jumps = (
            s.filter(
                (pl.col("prev_factor") == pl.col("cum_adjfactor"))
                & (
                    (pl.col("close") / pl.col("prev_close") > 3)
                    | (pl.col("close") / pl.col("prev_close") < 1 / 3)
                )
            )
            .select("ts_code", "trade_date")
            .unique()
            .sort("trade_date", "ts_code")
            .to_dicts()
        )
    checks.add(
        "price_jump_without_factor",
        len(joined),
        jumps,
        info=True,
        note="单日涨跌超过 3 倍且因子不变",
    )

    # 10. 上市期与行情：港股上市日前、退市日后不应有日线；美股证券表几乎没有日期，只计信息。
    problems = []
    if market == "hk" and len(daily):
        span = daily.group_by("ts_code").agg(
            pl.col("trade_date").min().alias("first"),
            pl.col("trade_date").max().alias("last"),
        )
        info = securities.unique("ts_code", keep="first").select(
            "ts_code", "list_date", "delist_date"
        )
        both = span.join(info, on="ts_code", how="left")
        for row in both.sort("ts_code").iter_rows(named=True):
            if row["list_date"] is None and row["delist_date"] is None:
                problems.append(dict(ts_code=row["ts_code"], side="不在证券表"))
                continue
            if row["list_date"] and row["first"] < row["list_date"]:
                problems.append(dict(ts_code=row["ts_code"], side="上市日前有行情"))
            if row["delist_date"] and row["last"] >= row["delist_date"]:
                problems.append(dict(ts_code=row["ts_code"], side="退市日及以后有行情"))
        checks.add("quotes_vs_listing", len(both), problems)
    else:
        known = (
            securities.filter(pl.col("delist_date").is_not_null())
            if len(securities)
            else securities
        )
        checks.add(
            "quotes_vs_listing",
            len(securities),
            [{}] * len(known),
            info=True,
            note="美股证券表的退市日几乎全空，上市退市由行情推断",
        )

    # 11. 代码复用与改名的线索（信息）：证券表同一代码多行、日线长缺口后重新出现。
    reused = []
    if len(securities):
        dup = securities.group_by("ts_code").len().filter(pl.col("len") > 1)
        reused = dup["ts_code"].to_list()
    gaps = []
    if len(daily):
        index = {d: i for i, d in enumerate(days)}
        s = daily.select("ts_code", "trade_date").unique().sort("ts_code", "trade_date")
        s = s.with_columns(pos=pl.col("trade_date").replace_strict(index, default=None))
        s = s.with_columns(gap=pl.col("pos") - pl.col("pos").shift(1).over("ts_code"))
        gaps = (
            s.filter(pl.col("gap") > 60)
            .select("ts_code", "trade_date", "gap")
            .to_dicts()
        )
    checks.add(
        "code_reuse_signals",
        len(securities),
        [dict(ts_code=c, signal="证券表多行") for c in reused]
        + [dict(**g, signal="缺口超过 60 个交易日后重新有行情") for g in gaps],
        info=True,
    )
    if market == "hk":
        lots = securities["trade_unit"] if len(securities) else pl.Series([])
        checks.add(
            "board_lot_available",
            len(securities),
            [{}] * int(lots.is_null().sum()),
            info=True,
            note="证券表 trade_unit 只是当前值；为空的个数见 count",
        )

    # 11b. 休市日有行（日线、因子，含样本历史）：面板只取开市日，这些行被丢弃。
    problems = []
    open_set = set(days)
    checked = 0
    for name in (
        names["daily"],
        names["factor"],
        names["daily_hist"],
        names["factor_hist"],
    ):
        frame = (
            daily
            if name == names["daily"]
            else factor
            if name == names["factor"]
            else _read(store, _complete(store, name).values())
        )
        if not len(frame):
            continue
        checked += len(frame)
        closed = frame.filter(~pl.col("trade_date").is_in(sorted(open_set)))
        for row in (
            closed.select("ts_code", "trade_date")
            .unique()
            .sort("trade_date", "ts_code")
            .iter_rows(named=True)
        ):
            problems.append(
                dict(dataset=name, ts_code=row["ts_code"], trade_date=row["trade_date"])
            )
    checks.add("rows_on_closed_days", checked, problems)

    # 12. 样本历史与按日分区在重叠日上逐值相同（同一源同一快照）。
    problems = []
    checked = 0
    for day_name, hist_name, columns in (
        (names["daily"], names["daily_hist"], ("open", "high", "low", "close", "vol")),
        (names["factor"], names["factor_hist"], ("cum_adjfactor",)),
    ):
        hist = _read(store, _complete(store, hist_name).values())
        day_frame = daily if day_name == names["daily"] else factor
        if not len(hist) or not len(day_frame):
            continue
        h = hist.unique(["ts_code", "trade_date"], keep="none")
        d = day_frame.unique(["ts_code", "trade_date"], keep="none")
        both = h.join(d, on=["ts_code", "trade_date"], suffix="_day")
        checked += len(both)
        mask = pl.lit(False)
        for c in columns:
            mask = mask | ~(
                (pl.col(c).is_null() & pl.col(c + "_day").is_null())
                | (
                    (pl.col(c) - pl.col(c + "_day")).abs()
                    <= 1e-9 * pl.col(c).abs() + 1e-9
                )
            )
        for row in (
            both.filter(mask).sort("trade_date", "ts_code").iter_rows(named=True)
        ):
            problems.append(
                dict(
                    dataset=hist_name,
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                )
            )
    checks.add("hist_vs_day", checked, problems)

    # 13. 面板（不允许例外）：抽查与原始库逐值一致；公司行为、上市、港股通字段按定义独立核对。
    try:
        panel_problems, panel_checked, (jumps, confirmed) = _audit_panel(
            store, universe, daily, factor, names, securities, days, scope
        )
    except Exception as error:  # 坏数据不许让校验崩溃：记为失败并出报告。
        panel_problems = [
            dict(problem=f"面板核对出错：{type(error).__name__}: {error}")
        ]
        panel_checked, jumps, confirmed = 0, [], []
    checks.add("panel", panel_checked, panel_problems, exceptions=False)
    if market == "us":
        # 可登记例外：真实的暴涨暴跌（只登记代码与日期，不写数值）。
        checks.add(
            "price_jumps",
            panel_checked,
            jumps,
            note="相邻可交易日价格跳出 [1/3, 3]、没有执行了的公司行为、交易没有明显放大（上跳看成交股数、"
            "下跳看成交额，不到前 10 个可交易日中位数的 5 倍）：逐条登记",
        )
        checks.add(
            "price_jumps_confirmed",
            panel_checked,
            confirmed,
            info=True,
            note="价格跳出 [1/3, 3]、交易明显放大（上跳成交股数、下跳成交额达到前 10 个可交易日中位数 5 倍）："
            "交易印证的真实行情",
        )

    stale = [
        e for i, e in enumerate(checks.exceptions["entries"]) if i not in checks.used
    ]
    verdicts = [c["verdict"] for c in checks.results.values()]
    report = dict(
        market=market,
        universe=universe,
        generated_at=_now(),
        elapsed_seconds=time.perf_counter() - started,
        verdict="fail" if "fail" in verdicts else "pass",
        checks=checks.results,
        known_exceptions=dict(
            entries=len(checks.exceptions["entries"]),
            used=len(checks.used),
            stale=stale[:EXAMPLES],
            stale_count=len(stale),
            cause_unknown=sum(
                1
                for i in checks.used
                if checks.exceptions["entries"][i]["reason"]
                in checks.exceptions["cause_unknown"]
            ),
        ),
    )
    out = Path(output) if output else store.root / "reports"
    atomic_json(out / "overseas_verify.json", report)
    (out / "overseas_verify.md").write_text(_markdown(report))
    return report


def _us_scope_codes(factor, securities):
    """美股面板应有的代码（按自己的写法重写一遍面板范围的定义）：在纽交所、纳斯达克、美交所有过因子；
    证券表任一行类别是普通股、ADR、GDR 的收；只标优先股的不收；没有类别的只收 1–5 位大写字母、
    且不是第 5 位为 W、U、R 的代码。"""
    major = set(
        factor.filter(pl.col("exchange").is_in(US_MAJOR))["ts_code"].unique().to_list()
    )
    kinds = defaultdict(set)
    if len(securities):
        for code, kind in securities.select("ts_code", "classify").iter_rows():
            if kind:
                kinds[code].add(kind)
    out = set()
    for code in major:
        k = kinds.get(code, set())
        if k & {"EQ", "ADR", "GDR"}:
            out.add(code)
        elif "PF" in k:
            continue
        elif re.fullmatch(r"[A-Z]{1,5}", code) and not (
            len(code) == 5 and code[-1] in "WUR"
        ):
            out.add(code)
    return out


def _audit_panel(store, universe, daily, factor, names, securities, days, scope):
    """面板核对（不允许例外）。除了抽查价格成交量，公司行为、上市、港股通字段都按字段定义独立核对，
    不调用构建代码：持有者价值不变式、因子账目、字段之间的约束、上市退市规则、名单与持股。"""
    current = store.root / f"panel_{universe}" / "current.json"
    if not current.exists():
        return [dict(problem="面板未构建")], 0, ([], [])
    path = current.parent / json.loads(current.read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    market = meta["market"]
    problems = []
    dates = meta["dates"]
    symbols = meta["symbols"]
    for name, spec in meta["fields"].items():
        file = path / f"{name}.npy"
        if not file.exists():
            problems.append(dict(field=name, problem="缺文件"))
            continue
        array = np.load(file, mmap_mode="r")
        if (
            array.shape != (len(dates), len(symbols))
            or str(array.dtype) != spec["dtype"]
        ):
            problems.append(
                dict(field=name, problem=f"形状或类型不符 {array.shape} {array.dtype}")
            )
    if problems:
        return problems, len(meta["fields"]), ([], [])

    def load(name):
        return np.asarray(np.load(path / f"{name}.npy", mmap_mode="r"))

    def flag(field, problem, mask, limit=200):
        for i, j in np.argwhere(mask)[:limit]:
            problems.append(
                dict(
                    field=field,
                    problem=problem,
                    ts_code=symbols[j],
                    trade_date=dates[i],
                )
            )
        extra = int(mask.sum()) - limit
        if extra > 0:
            problems.append(dict(field=field, problem=f"{problem}（另有 {extra} 格）"))

    T, N = len(dates), len(symbols)
    jumps, confirmed = [], []
    source_daily, source_factor = daily, factor
    if universe == "sample":
        source_daily = _read(store, _complete(store, names["daily_hist"]).values())
        source_factor = _read(store, _complete(store, names["factor_hist"]).values())
        if market == "us" and len(source_factor):
            source_factor = source_factor.filter(pl.col("exchange").is_in(US_MAJOR))
    base = [s.split("#")[0] for s in symbols]
    open_days = set(days)
    stray = sorted(set(dates) - open_days)
    if stray:
        problems.append(dict(field="dates", problem=f"面板日期不是开市日 {stray[:5]}"))
    missing_days = [
        d for d in days if dates[0] <= d <= dates[-1] and d not in set(dates)
    ]
    if missing_days:
        problems.append(
            dict(field="dates", problem=f"开市日不在面板里 {missing_days[:5]}")
        )

    # ---- 范围：美股每列都应在三大交易所普通股与存托凭证里；没有代码重复（拆段的用 #2 区分）。
    if len(set(symbols)) != N:
        problems.append(dict(field="symbols", problem="代码重复"))
    if market == "us":
        wanted = _us_scope_codes(source_factor, securities)
        outside = sorted(set(base) - wanted)
        if outside:
            problems.append(
                dict(field="symbols", problem=f"不在美股范围内 {outside[:10]}")
            )

    lo, hi = dates[0], dates[-1]
    d = (
        source_daily.filter(pl.col("trade_date").is_between(pl.lit(lo), pl.lit(hi)))
        .filter(pl.col("ts_code").is_in(sorted(set(base))))
        .unique(["ts_code", "trade_date"], keep="none")
        if len(source_daily)
        else source_daily
    )
    f = (
        source_factor.filter(pl.col("trade_date").is_between(pl.lit(lo), pl.lit(hi)))
        if len(source_factor)
        else source_factor
    )
    if market == "hk":
        # 港股代码复用：证券表的「代码!.HK」是已退市的旧公司，日线却记在「代码.HK」下；
        # 新公司上市日前、旧公司退市日前的行属于旧公司。
        owners = {}
        rows = {r["ts_code"]: r for r in securities.iter_rows(named=True)}
        for code, row in rows.items():
            if "!" in code and code.replace("!", "") in rows:
                new = rows[code.replace("!", "")]
                if new["list_date"] and row["delist_date"]:
                    owners[code.replace("!", "")] = (
                        min(new["list_date"], row["delist_date"]),
                        code,
                    )

        def remap(frame):
            if not owners or not len(frame):
                return frame
            cut = pl.col("ts_code").replace_strict(
                {k: v[0] for k, v in owners.items()},
                default=None,
                return_dtype=pl.String,
            )
            old = pl.col("ts_code").replace_strict(
                {k: v[1] for k, v in owners.items()},
                default=None,
                return_dtype=pl.String,
            )
            return frame.with_columns(
                ts_code=pl.when(cut.is_not_null() & (pl.col("trade_date") < cut))
                .then(old)
                .otherwise(pl.col("ts_code"))
            )

        d = remap(d).filter(pl.col("ts_code").is_in(symbols))
        f = remap(f)
        # 港股收盘价：日线与因子表 close_price 相差不超过半分时以后者为准（源日线四舍五入期）。
        precise = f.select("ts_code", "trade_date", "close_price").unique(
            ["ts_code", "trade_date"], keep="none"
        )
        d = d.join(precise, on=["ts_code", "trade_date"], how="left").with_columns(
            close=pl.when(
                pl.col("close_price").is_not_null()
                & ((pl.col("close") - pl.col("close_price")).abs() <= 0.0051)
            )
            .then(pl.col("close_price"))
            .otherwise(pl.col("close"))
        )
    elif len(f) and len(d):
        # 美股：从第一次有三大交易所因子的那天起（之后不按因子结束日截断）。
        cover = f.group_by("ts_code").agg(pl.col("trade_date").min().alias("_lo"))
        d = d.join(cover, on="ts_code", how="left").filter(
            pl.col("trade_date") >= pl.col("_lo")
        )

    close = load("close")
    volume = load("volume")
    quote = load("has_quote")
    tradable = load("tradable")
    previous = load("previous_close")
    row_of = {x: i for i, x in enumerate(dates)}
    columns = defaultdict(list)
    for j, code in enumerate(base):
        columns[code].append(j)
    starts = {}
    for js in columns.values():
        for k, j in enumerate(js):
            q = np.flatnonzero(quote[:, j])
            starts[j] = 0 if k == 0 or not len(q) else int(q[0])

    def column(code, i):
        js = columns.get(code)
        if not js:
            return None
        return [j for j in js if starts[j] <= i][-1]

    # ---- 抽 2000 行逐值核对收盘价、成交量、has_quote（拆段的按日期找到所在的那一列）。
    rng = np.random.default_rng(20261113)
    take = d[rng.choice(len(d), size=min(2000, len(d)), replace=False)] if len(d) else d
    checked = len(take)
    for row in take.iter_rows(named=True):
        i = row_of.get(row["trade_date"])
        j = column(row["ts_code"], i) if i is not None else None
        if i is None or j is None:
            problems.append(
                dict(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    problem="原始行在面板里找不到",
                )
            )
            continue
        want_close = (
            row["close"] if row["close"] is not None and row["close"] > 0 else math.nan
        )
        if not (
            close[i, j] == want_close
            or (math.isnan(close[i, j]) and math.isnan(want_close))
        ):
            problems.append(
                dict(ts_code=symbols[j], trade_date=row["trade_date"], field="close")
            )
        want_pre = (
            row["pre_close"]
            if row.get("pre_close") is not None and row["pre_close"] > 0
            else math.nan
        )
        if not (
            previous[i, j] == want_pre
            or (math.isnan(previous[i, j]) and math.isnan(want_pre))
        ):
            problems.append(
                dict(
                    ts_code=symbols[j],
                    trade_date=row["trade_date"],
                    field="previous_close",
                )
            )
        if row["vol"] is not None and volume[i, j] != row["vol"]:
            problems.append(
                dict(ts_code=symbols[j], trade_date=row["trade_date"], field="volume")
            )
        if bool(quote[i, j]) != math.isfinite(want_close):
            problems.append(
                dict(
                    ts_code=symbols[j], trade_date=row["trade_date"], field="has_quote"
                )
            )
    # 美股：当天三大交易所有因子行（同一序列同一天取值冲突的不算）才可交易。
    raw_trade = quote & (volume > 0)
    has_f = np.ones((T, N), bool)
    if market == "us":
        has_f = np.zeros((T, N), bool)
        if len(f):
            g = (
                f.filter(pl.col("cum_adjfactor") > 0)
                .group_by("ts_code", "exchange", "trade_date")
                .agg(
                    pl.col("cum_adjfactor").min().alias("lo"),
                    pl.col("cum_adjfactor").max().alias("hi"),
                )
                .filter((pl.col("hi") - pl.col("lo")) <= 1e-6 * pl.col("hi"))
                .select("ts_code", "trade_date")
                .unique()
            )
            where = pl.DataFrame(
                dict(
                    ts_code=base,
                    _col=list(range(N)),
                    _start=[starts[j] for j in range(N)],
                )
            )
            when = pl.DataFrame(dict(trade_date=list(dates), _row=list(range(T))))
            hit = (
                g.join(when, on="trade_date")
                .join(where, on="ts_code")
                .filter(pl.col("_start") <= pl.col("_row"))
                .group_by("ts_code", "_row")
                .agg(pl.col("_col").max())
            )
            has_f[hit["_row"].to_numpy(), hit["_col"].to_numpy()] = True
    flag(
        "tradable",
        "不等于 has_quote 且成交量 > 0（美股另要当天有三大交易所因子）",
        tradable != (raw_trade & has_f),
    )

    # ---- 公司行为字段之间的约束。
    src = load("action_source")
    split = load("split")
    cash = load("equivalent_cash")
    ratio = load("action_ratio")
    adj = load("adj_factor")
    suspect = load("action_suspect")
    how = load("action_check")
    checked += T * N
    flag("action_source", "枚举越界", src > 7)
    flag("action_check", "枚举越界", how > 6)
    flag("action_check", "有事件 ⇔ 有印证方式 不成立", (how == 0) != (src == 0))
    flag("action_check", "6 类没有记成「没有印证」", (src == 6) != (how == 5))
    flag("action_check", "7 类没有记成「最后成交日之后」", (src == 7) != (how == 6))
    flag(
        "action_check",
        "美股出现前收印证",
        (how == 1) if market == "us" else np.zeros_like(how, bool),
    )
    flag(
        "action_suspect",
        "不等于 action_source 为 6、7",
        suspect != np.isin(src, (6, 7)),
    )
    shares = np.isin(src, (1, 3, 4))
    flag("split", "非拆并股的格 split 不为 1", ~shares & (split != 1))
    flag("equivalent_cash", "非等效现金的格不为 0", (src != 2) & (cash != 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        flag("equivalent_cash", "等效现金不是正数", (src == 2) & ~(cash > 0))
        flag(
            "split",
            "3、4 类 split 不等于因子比",
            np.isin(src, (3, 4)) & (split != ratio),
        )
        flag(
            "split",
            "拆并股的 split 与因子比相差超过 1%",
            (src == 1) & ~(np.abs(np.log(split / ratio)) <= 0.01),
        )
        flag("action_source", "4 类因子比小于 1.10", (src == 4) & ~(ratio >= 1.10))
        flag(
            "action_source",
            "2 类因子比不在 (1, 1.10)",
            (src == 2) & ~((ratio > 1) & (ratio < 1.10)),
        )
        # 5：美股现金类（1 < r < 1.10），或小窗口（窗口合计比值在 1/1.25 与 1.25 之间）里价格没印证的
        # 非整数股数调整。
        wide = np.argwhere((src == 5) & ~((ratio > 1) & (ratio < 1.10)))
        bad5 = np.zeros((T, N), bool)
        for t, j in wide:
            trades = np.flatnonzero(tradable[:, j])
            k = int(np.searchsorted(trades, t))
            lo = trades[k - 1] + 1 if k > 0 else 0
            hi = trades[k] if k < len(trades) else T - 1
            events = [u for u in range(lo, hi + 1) if src[u, j] > 0]
            whole = float(np.prod(ratio[events, j]))
            bad5[t, j] = not (1 / 1.25 < whole < 1.25)
        flag("action_source", "5 类不是现金类，所在窗口的合计比值也不小", bad5)
        flag(
            "action_source",
            "港股出现 5 类",
            (src == 5) if market == "hk" else np.zeros_like(src, bool),
        )
        flag(
            "action_source",
            "因子比变了却没有记事件",
            (src == 0) & np.isfinite(ratio) & (np.abs(ratio - 1) > RATIO_NOISE),
        )
    flag(
        "action_source",
        "美股出现等效现金",
        (src == 2) if market == "us" else np.zeros_like(src, bool),
    )
    # 后复权因子只累乘执行了的事件。
    step = np.where(shares, split, np.where(src == 2, ratio, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.abs(adj[1:] / adj[:-1] / step[1:] - 1)
    flag(
        "adj_factor",
        "相邻两日之比不等于执行了的事件",
        np.vstack([np.zeros((1, N), bool), np.isfinite(adj[:-1]) & (drift > 1e-9)]),
    )

    # ---- 持有者价值不变式：执行了事件的窗口（上一个成交日 → 下一个成交日），按引擎记账
    #      股数 ×split、现金按当时股数计，价值倍数应在 [1/3, 3]；最后一次成交之后不许有执行的事件。
    #      美股另核：窗口里有不执行的可疑事件、价格又跳出 [1/3, 3] 的，必须已拆成两列（不在同一列里）。
    executed = np.isin(src, (1, 2, 3, 4))
    opens = load("open")
    for j in np.flatnonzero(executed.any(axis=0)):
        trades = np.flatnonzero(tradable[:, j])
        done = set()
        for t in np.flatnonzero(executed[:, j]):
            k = int(np.searchsorted(trades, t))
            if k == len(trades):
                problems.append(
                    dict(
                        field="action_source",
                        problem="最后一次成交之后执行了事件",
                        ts_code=symbols[j],
                        trade_date=dates[t],
                    )
                )
                continue
            if k == 0 or (k, j) in done:
                continue
            done.add((k, j))
            a, b = trades[k - 1], trades[k]
            s, c = 1.0, 0.0
            for u in range(a + 1, b + 1):
                c += s * cash[u, j]
                s *= split[u, j]
            # 除权后价格取开盘或收盘（小盘股开盘价常沿用旧价），两个都超出才算不符。
            values = [
                (s * p + c) / close[a, j] for p in (opens[b, j], close[b, j]) if p > 0
            ]
            if not any(1 / 3 <= h <= 3 for h in values):
                problems.append(
                    dict(
                        field="split",
                        problem=f"持有者价值倍数 {values[-1]:.3g} 超出 [1/3, 3]",
                        ts_code=symbols[j],
                        trade_date=dates[t],
                    )
                )
    if market == "us":
        for j in np.flatnonzero(suspect.any(axis=0)):
            trades = np.flatnonzero(tradable[:, j])
            for t in np.flatnonzero(suspect[:, j]):
                k = int(np.searchsorted(trades, t))
                if 0 < k < len(trades):
                    a, b = trades[k - 1], trades[k]
                    h = (
                        float(
                            np.prod(
                                np.where(
                                    executed[a + 1 : b + 1, j],
                                    step[a + 1 : b + 1, j],
                                    1.0,
                                )
                            )
                        )
                        * close[b, j]
                        / close[a, j]
                    )
                    if not (1 / 3 <= h <= 3):
                        problems.append(
                            dict(
                                field="split",
                                problem=f"可疑事件处价格跳变 {h:.3g} 没有拆列",
                                ts_code=symbols[j],
                                trade_date=dates[t],
                            )
                        )
    # 港股等效现金（含停牌缺口之后）：窗口两端口径，参考价从上一个成交日的收盘起，窗口里每个事件之后
    # 除以它的因子比；等效现金 = 当时参考价 × (1 − 1/因子比)。
    # 没有上一个成交日的窗口，参考价 = 下一个成交日的价格（收盘，没有收盘用开盘）× R。
    # 港股前收：窗口合计比值 R，|前收 × R − 上一成交日收盘| 不超过上一收盘的 0.5%（大比值窗口 1%）或
    # 前收半个价位折回去（源日线四舍五入期按 0.01）的，就是前收能印证：必须执行且记 1；记 1 的必须能印证；
    # 记 2、3（价格印证）的必须印证不了。
    if market == "hk":
        bad = np.zeros((T, N), bool)
        pre_bad = np.zeros((T, N), bool)
        pre_miss = np.zeros((T, N), bool)
        for j in np.flatnonzero((src > 0).any(axis=0)):
            trades = np.flatnonzero(tradable[:, j])
            windows = defaultdict(list)
            for t in np.flatnonzero(src[:, j] > 0):
                k = int(np.searchsorted(trades, t))
                if k < len(trades):
                    windows[(trades[k - 1] if k else None, trades[k])].append(t)
            for (a, b), rows in windows.items():
                R = float(np.prod(ratio[rows, j]))
                pc = previous[b, j]
                if a is not None and math.isfinite(pc) and pc > 0:
                    tick = hk_tick(pc)
                    if any(HK_ROUNDED[0] <= dates[x] <= HK_ROUNDED[1] for x in (a, b)):
                        tick = max(tick, 0.01)  # 源日线四舍五入期，前收只到 0.01
                    share = 0.01 if max(R, 1 / R) >= 1.25 else 5e-3
                    room = max(share * close[a, j], 0.5 * tick * R) + 1e-9
                    near = abs(pc * R - close[a, j]) <= room
                    for t in rows:
                        if how[t, j] in (2, 3) and near:
                            pre_bad[t, j] = True
                        if how[t, j] == 5 and near:
                            pre_miss[t, j] = True
                        if how[t, j] == 1 and not near:
                            pre_bad[t, j] = True
                if not np.isin(src[rows, j], (1, 2, 3, 4)).all():
                    continue
                if a is None:
                    after = close[b, j] if math.isfinite(close[b, j]) else opens[b, j]
                    ref = after * R
                else:
                    ref = close[a, j]
                for t in rows:
                    if src[t, j] == 2:
                        want = ref - ref / ratio[t, j]
                        bad[t, j] = not abs(cash[t, j] - want) <= 1e-9 * ref
                    ref = ref / ratio[t, j]
        flag(
            "equivalent_cash",
            "不等于窗口参考价（上一成交日收盘，逐个事件除以因子比）×(1 − 1/因子比)",
            bad,
        )
        flag("action_source", "交易所前收印证了因子却没有执行", pre_miss)
        flag(
            "action_check",
            "前收能印证却记成价格印证，或记成前收印证而前收对不上",
            pre_bad,
        )

    # ---- 因子账目：面板记了事件的格，因子比必须是源某条序列当天的比值；港股（只有一条序列）源因子
    #      每一次变化都必须在面板里有事件（执行或标为可疑）。
    if len(f):
        fx = f.select(
            "ts_code",
            "trade_date",
            "cum_adjfactor",
            *([pl.col("exchange")] if "exchange" in f.columns else []),
        )
        keys = ["ts_code"] + (["exchange"] if "exchange" in f.columns else [])
        fx = (
            fx.filter(pl.col("cum_adjfactor") > 0)
            .unique([*keys, "trade_date"], keep="none")
            .sort([*keys, "trade_date"])
            .with_columns(
                r=(pl.col("cum_adjfactor") / pl.col("cum_adjfactor").shift(1)).over(
                    keys
                )
            )
            .filter(pl.col("r").is_not_null() & ((pl.col("r") - 1).abs() > RATIO_NOISE))
        )
        raw = defaultdict(list)
        for code, day, r in fx.select("ts_code", "trade_date", "r").iter_rows():
            raw[(code, day)].append(r)
        for t, j in np.argwhere(src > 0):
            got = ratio[t, j]
            options = raw.get((base[j], dates[t]), [])
            if not any(abs(got / r - 1) <= 1e-9 for r in options):
                problems.append(
                    dict(
                        field="action_ratio",
                        problem="事件的因子比不在源的任何序列里",
                        ts_code=symbols[j],
                        trade_date=dates[t],
                    )
                )
        if market == "hk":
            col_of = {x: j for j, x in enumerate(symbols)}
            for (code, day), _ in raw.items():
                i, j = row_of.get(day), col_of.get(code)
                if (
                    i is not None
                    and j is not None
                    and src[i, j] == 0
                    and np.isfinite(ratio[i, j])
                    and np.isfinite(ratio[:i, j]).any()
                ):
                    problems.append(
                        dict(
                            field="action_source",
                            problem="源因子变了而面板没有事件",
                            ts_code=code,
                            trade_date=day,
                        )
                    )

    # ---- 上市与退市。
    listed = load("is_listed")
    delist = load("delist")
    if market == "hk":
        info = {
            r["ts_code"]: r
            for r in securities.unique("ts_code", keep="first").iter_rows(named=True)
        }
        want_l = np.zeros((T, N), bool)
        want_d = np.zeros((T, N), bool)
        darr = np.array(dates)
        for j, code in enumerate(symbols):
            row = info.get(code) or {}
            if not row.get("list_date"):
                continue
            on = darr >= row["list_date"]
            if row.get("delist_date"):
                on &= darr < row["delist_date"]
                after = np.flatnonzero(darr >= row["delist_date"])
                if len(after):
                    want_d[after[0], j] = True
            want_l[:, j] = on
    else:
        want_l = np.zeros((T, N), bool)
        want_d = np.zeros((T, N), bool)
        want_r = np.full((T, N), np.nan)
        # 最近 GRACE+1 个交易日（含当天）里有成交 ⇔ 上市；用累计成交次数之差写。
        counts = np.vstack([np.zeros((1, N)), np.cumsum(tradable, axis=0)])
        lag = np.maximum(np.arange(T) - GRACE, 0)
        want_l = (counts[1:] - counts[lag]) > 0

        def liquidation(j, a, b, price_b, whole):
            # 清算价 min(a 的收盘, b 的价格 × 其间执行了的倍数)，折成引擎股数；除以引擎记账价
            # （b 之前最后一个收盘，按其后执行的拆并股调整）得回收率。
            held = float(np.prod(step[a + 1 : b, j]))
            marks = np.flatnonzero(np.isfinite(close[:b, j]))
            last = marks[-1]
            mark = close[last, j] / float(np.prod(step[last + 1 : b, j]))
            return min(close[a, j], whole * price_b) / held / mark

        # 没有三大交易所因子的成交日：还在上市时连续 US_ORPHAN 个，第 US_ORPHAN 个那天退市，
        # 到下一个可交易日之前不算上市；按 min(最后可交易日收盘, 当天收盘) 清算。
        orphan_at = defaultdict(list)
        for j in range(N):
            hits = np.flatnonzero(raw_trade[:, j])
            ok = np.flatnonzero(tradable[:, j])
            streak, down = 0, None
            for i in hits:
                if has_f[i, j]:
                    if down is not None:
                        want_l[down:i, j] = False
                    streak, down = 0, None
                    continue
                streak += 1
                before = ok[ok < i]
                if (
                    streak == US_ORPHAN
                    and down is None
                    and len(before)
                    and i - before[-1] <= GRACE
                ):
                    down = int(i)
                    orphan_at[j].append(down)
                    want_d[i, j] = True
                    a = before[-1]
                    whole = float(np.prod(step[a + 1 : i + 1, j]))
                    want_r[i, j] = liquidation(j, a, i, close[i, j], whole)
            if down is not None:
                want_l[down:, j] = False
        # 拆段：同一代码的前一列在后一列第一个行情日起不再上市，并在那天退市；清算价
        # min(前一列最后成交日收盘, 后一列首日收盘 × 缺口里执行了的倍数)，相对引擎记账价的倍数。
        cut_at = {}
        for js in columns.values():
            for old, new in zip(js[:-1], js[1:], strict=True):
                cut_at[old] = (new, starts[new])
        for j in range(N):
            trades = np.flatnonzero(tradable[:, j])
            gone = bool(len(trades)) and any(x > trades[-1] for x in orphan_at[j])
            if j in cut_at:
                new, b = cut_at[j]
                want_l[b:, j] = False
                if len(trades) and not gone:
                    want_d[b, j] = True
                    a = trades[-1]
                    whole = float(np.prod(step[a + 1 : b, j])) * float(step[b, new])
                    want_r[b, j] = liquidation(j, a, b, close[b, new], whole)
                continue
            if len(trades) and trades[-1] + GRACE + 1 < T and not gone:
                want_d[trades[-1] + GRACE + 1, j] = True
        # 相邻两个可交易日之间价格跳出 [1/3, 3]、又没有执行了的事件解释的：引擎会把它当成收益。
        # 交易明显放大的（上跳看成交股数、下跳看成交额，达到前 10 个可交易日中位数的 5 倍）是交易印证的
        # 真实行情（另列计数）；
        # 其余单列一项检查，必须按代码和日期逐条登记；kind 分「相邻」（中间没有交易日）与「缺口」。
        # 同一代码拆成几列时，基准按整个代码的成交序列取（与构建一致），跳变只看同一列内部。
        # 另外：上跳超过 3 倍而成交股数没有放大的（疑似漏记的合股）构建应已拆列，还在列内就报错。
        amount = load("amount")
        dollar = np.where(np.isfinite(amount) & (amount > 0), amount, close * volume)
        for js in columns.values():
            seq = []
            for k, j in enumerate(js):
                stop = starts[js[k + 1]] if k + 1 < len(js) else T
                seq += [
                    (int(t), j)
                    for t in np.flatnonzero(tradable[:, j])
                    if starts[j] <= t < stop
                ]
            for k in range(1, len(seq)):
                (a, ja), (b, j) = seq[k - 1], seq[k]
                if ja != j or executed[a + 1 : b + 1, j].any():
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    h = close[b, j] / close[a, j]
                if 1 / 3 <= h <= 3:
                    continue
                prev = [(t, c) for t, c in seq[max(0, k - JUMP_BASE_DAYS) : k]]
                base_amount = float(np.median([dollar[t, c] for t, c in prev]))
                base_volume = float(np.median([volume[t, c] for t, c in prev]))
                record = dict(
                    ts_code=symbols[j],
                    trade_date=dates[b],
                    kind="相邻" if b == a + 1 else "缺口",
                )
                if h > QUIET_JUMP and volume[b, j] <= QUIET_VOLUME * base_volume:
                    problems.append(
                        dict(
                            record,
                            field="split",
                            problem="成交股数没有放大的上跳（疑似漏记的合股）没有拆列",
                        )
                    )
                if (
                    volume[b, j] >= JUMP_SHARES_CONFIRM * base_volume
                    if h > 1
                    else dollar[b, j] >= JUMP_AMOUNT_CONFIRM * base_amount
                ):
                    confirmed.append(record)
                else:
                    jumps.append(record)
        recovery = load("delist_recovery")
        flag(
            "delist_recovery",
            "与拆段或无因子退市的清算价不符",
            ~(
                (np.isnan(recovery) & np.isnan(want_r))
                | (np.abs(recovery - want_r) <= 1e-9 * np.abs(want_r))
            ),
        )
        overlap = np.zeros((T, N), bool)
        for js in columns.values():
            if len(js) > 1:
                total = listed[:, js].sum(axis=1)
                overlap[:, js[0]] = total > 1
        flag("is_listed", "同一代码的两列同时上市", overlap)
    flag("is_listed", "与上市规则不符", listed != want_l)
    flag("delist", "与退市规则不符", delist != want_d)

    # ---- 港股通（港股）：名单、可知、持股（错位一天）、只能卖出。
    if market == "hk":
        parts = {}
        for name in ("hk_connect_sh", "hk_connect_sz"):
            parts[name] = {
                e["partition"]: e
                for e in _complete(store, name).values()
                if e.get("rows")
            }
        known_days = set(parts["hk_connect_sh"]) & set(parts["hk_connect_sz"])
        members = defaultdict(set)
        for name in parts:
            frame = _read(store, parts[name].values())
            for code, day in (
                frame.select("ts_code", "trade_date").iter_rows() if len(frame) else []
            ):
                members[day].add(code)
        hold_frame = _read(store, _complete(store, "hk_connect_hold").values())
        held_raw = defaultdict(set)
        if len(hold_frame):
            for code, day, vol in hold_frame.select(
                "ts_code", "trade_date", "vol"
            ).iter_rows():
                if vol is not None and float(vol) > 0:
                    held_raw[day].add(code)
        k = np.array([x in known_days for x in dates])[:, None] & np.ones((1, N), bool)
        e = np.array([[c in members[x] for c in symbols] for x in dates]) & k
        h = np.zeros((T, N), bool)
        for i in range(1, T):
            h[i] = [c in held_raw[dates[i - 1]] for c in symbols]
        flag("connect_known", "与名单分区不符", load("connect_known") != k)
        flag("connect_eligible", "与当天名单不符", load("connect_eligible") != e)
        flag("connect_held", "与上一交易日南向持股不符", load("connect_held") != h)
        flag(
            "connect_sell_only",
            "与「可知、上一日持有、不在名单」不符",
            load("connect_sell_only") != (k & h & ~e),
        )
        period = np.array([HK_ROUNDED[0] <= x <= HK_ROUNDED[1] for x in dates])[:, None]
        with np.errstate(invalid="ignore"):
            flag(
                "price_rounded",
                "与「四舍五入期、有行情、收盘价 < 0.5」不符",
                load("price_rounded") != (period & quote & (close < 0.5)),
            )
    return problems, checked, (jumps, confirmed)


def _markdown(report):
    lines = [
        f"# 港美股校验（{report['market']}，{report['universe']}）",
        "",
        f"总结论：**{report['verdict']}**；生成于 {report['generated_at']}；耗时 {report['elapsed_seconds']:.1f} 秒",
        "",
        "| 检查 | 结论 | 检查数 | 失败 | 已解释 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in report["checks"].items():
        failed = c.get("count", c["failed"]) if c["verdict"] == "info" else c["failed"]
        lines.append(
            f"| {name} | {c['verdict']} | {c['checked']} | {failed} | {c['explained']} | {c.get('note', '')} |"
        )
    lines += ["", "## 样例", ""]
    for name, c in report["checks"].items():
        if c["examples"]:
            lines.append(
                f"- {name}: {json.dumps(c['examples'][:5], ensure_ascii=False)}"
            )
    k = report["known_exceptions"]
    lines += [
        "",
        f"已知例外：{k['entries']} 条，本次用到 {k['used']} 条，过期 {k['stale_count']} 条，原因未查明 {k['cause_unknown']} 条。",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["EXCEPTION_KEYS", "load_exceptions", "verify_overseas"]
