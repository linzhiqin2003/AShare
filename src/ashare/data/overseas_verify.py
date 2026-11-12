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
    "factor_vs_previous_close": {"ts_code", "trade_date", "exchange"},
    "quotes_vs_listing": {"ts_code", "side"},
    "hist_vs_day": {"dataset", "ts_code", "trade_date"},
    "calendar": {"cal_date"},
    "rows_on_closed_days": {"dataset", "ts_code", "trade_date", "date_from", "date_to"},
}
EXAMPLES = 20
FACTOR_TOL = 5e-3
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


def _read(store, entries):
    frames = [pl.read_parquet(store.root / e["file"]) for e in entries if e.get("rows")]
    if not frames:
        return pl.DataFrame()
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
    daily = _read(store, daily_parts.values())
    factor = _read(store, factor_parts.values())

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

    # 7. 因子表 close_price 与日线收盘价一致（同一源、同一天）。
    problems = []
    joined = pl.DataFrame()
    if len(daily) and len(factor):
        d1 = daily.unique(["ts_code", "trade_date"], keep="none").select(
            "ts_code", "trade_date", "close", "pre_close"
        )
        if "exchange" not in factor.columns:
            factor = factor.with_columns(pl.lit(None, pl.String).alias("exchange"))
        f1 = (
            _home_series(factor)
            .unique(["ts_code", "trade_date"], keep="none")
            .select("ts_code", "trade_date", "cum_adjfactor", "close_price", "exchange")
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

    # 8. 因子事件与价格：港股 F[t]/F[t-1] 应等于 close[t-1]/pre_close[t]（源的前收已除权）；
    #    美股前收不除权，检查拆并股日价格的相应跳变（close[t]/close[t-1]×r 落在 [1/3, 3]）。
    problems = []
    events = 0
    if len(joined):
        series = joined.sort("ts_code", "trade_date")
        codes = series["ts_code"].to_list()
        dates = series["trade_date"].to_list()
        closes = series["close"].to_numpy()
        pres = series["pre_close"].to_numpy()
        facs = series["cum_adjfactor"].to_numpy()
        venues = series["exchange"].to_list()
        previous_day = {days[k]: days[k - 1] for k in range(1, len(days))}
        for i in range(1, len(codes)):
            if codes[i] != codes[i - 1]:
                continue
            # 只核对相邻交易日：中间缺日线（停牌、无成交）时除权日落在缺口里，无从对照。
            if previous_day.get(dates[i]) != dates[i - 1]:
                continue
            a, b = facs[i - 1], facs[i]
            if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
                continue
            r = b / a
            if abs(r - 1) <= RATIO_NOISE:
                continue
            events += 1
            if market == "hk":
                expected = closes[i - 1] / pres[i] if pres[i] else math.nan
                # 前收按价位取整：允许 0.5% 或一个最小价位的差。
                implied = closes[i - 1] / r
                within = math.isfinite(expected) and (
                    abs(expected / r - 1) <= FACTOR_TOL
                    or abs(implied - pres[i]) <= hk_tick(pres[i]) + 1e-9
                )
                if not within:
                    problems.append(
                        dict(ts_code=codes[i], trade_date=dates[i], exchange=venues[i])
                    )
            else:
                move = closes[i] / closes[i - 1] * r if closes[i - 1] else math.nan
                if not (math.isfinite(move) and 1 / 3 <= move <= 3):
                    problems.append(
                        dict(ts_code=codes[i], trade_date=dates[i], exchange=venues[i])
                    )
    checks.add(
        "factor_vs_previous_close",
        events,
        problems,
        note="相邻两个交易日都有日线时因子变化的次数；中间缺日线的不检查",
    )

    # 9. 大幅跳价但因子未变（可能缺拆并股、代码复用或源错误），只计信息。
    jumps = []
    if len(joined):
        s = joined.sort("ts_code", "trade_date").with_columns(
            prev_close=pl.col("close").shift(1).over("ts_code"),
            prev_factor=pl.col("cum_adjfactor").shift(1).over("ts_code"),
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
    securities = _read(store, _complete(store, names["securities"]).values())
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

    # 13. 面板（不允许例外）：抽查与原始库逐值一致、因子比一致、字段齐全。
    panel_problems, panel_checked = _audit_panel(store, universe, daily, factor, names)
    checks.add("panel", panel_checked, panel_problems, exceptions=False)

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


def _home_series(factor):
    """一只证券有几个交易所的因子序列时只留一条：非场外优先、天数多的优先、再按字母序。"""
    if "exchange" not in factor.columns:
        return factor
    tally = defaultdict(lambda: defaultdict(set))
    for code, venue, day in factor.select(
        "ts_code", "exchange", "trade_date"
    ).iter_rows():
        tally[code][venue or ""].add(day)
    home = {
        code: min(venues, key=lambda v: (v == "OTC", -len(venues[v]), v))
        for code, venues in tally.items()
    }
    return factor.filter(
        pl.col("exchange").fill_null("")
        == pl.col("ts_code").replace_strict(home, default="", return_dtype=pl.String)
    )


def _audit_panel(store, universe, daily, factor, names):
    current = store.root / f"panel_{universe}" / "current.json"
    if not current.exists():
        return [dict(problem="面板未构建")], 0
    path = current.parent / json.loads(current.read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
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
        return problems, len(meta["fields"])
    source_daily = daily
    source_factor = factor
    if universe == "sample":
        source_daily = _read(store, _complete(store, names["daily_hist"]).values())
        source_factor = _read(store, _complete(store, names["factor_hist"]).values())
    lo, hi = dates[0], dates[-1]
    d = (
        source_daily.filter(pl.col("trade_date").is_between(pl.lit(lo), pl.lit(hi)))
        .filter(pl.col("ts_code").is_in(symbols))
        .unique(["ts_code", "trade_date"], keep="none")
    )
    if meta["market"] == "hk":
        # 港股代码复用：证券表的「代码!.HK」是已退市的旧公司，日线却记在「代码.HK」下；
        # 新公司上市日前、旧公司退市日前的行属于旧公司。
        listing = _read(store, _complete(store, names["securities"]).values())
        owners = {}
        rows = {r["ts_code"]: r for r in listing.iter_rows(named=True)}
        for code, row in rows.items():
            if "!" in code and code.replace("!", "") in rows:
                new = rows[code.replace("!", "")]
                if new["list_date"] and row["delist_date"]:
                    owners[code.replace("!", "")] = (
                        min(new["list_date"], row["delist_date"]),
                        code,
                    )
        if owners:
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
            d = d.with_columns(
                ts_code=pl.when(cut.is_not_null() & (pl.col("trade_date") < cut))
                .then(old)
                .otherwise(pl.col("ts_code"))
            ).filter(pl.col("ts_code").is_in(symbols))
        # 港股收盘价：日线与因子表 close_price 相差不超过半分时以后者为准（源日线四舍五入期）。
        precise = source_factor.select("ts_code", "trade_date", "close_price").unique(
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
    # 抽 2000 行逐值核对收盘价、成交量、has_quote。
    rng = np.random.default_rng(20261113)
    take = d[rng.choice(len(d), size=min(2000, len(d)), replace=False)] if len(d) else d
    row_of = {x: i for i, x in enumerate(dates)}
    col_of = {x: j for j, x in enumerate(symbols)}
    close = np.load(path / "close.npy", mmap_mode="r")
    volume = np.load(path / "volume.npy", mmap_mode="r")
    quote = np.load(path / "has_quote.npy", mmap_mode="r")
    for row in take.iter_rows(named=True):
        i, j = row_of[row["trade_date"]], col_of[row["ts_code"]]
        # 收盘价非正（源对无成交日的写法）面板记为缺失。
        want_close = (
            row["close"] if row["close"] is not None and row["close"] > 0 else math.nan
        )
        if not (
            close[i, j] == want_close
            or (math.isnan(close[i, j]) and math.isnan(want_close))
        ):
            problems.append(
                dict(
                    ts_code=row["ts_code"], trade_date=row["trade_date"], field="close"
                )
            )
        if row["vol"] is not None and volume[i, j] != row["vol"]:
            problems.append(
                dict(
                    ts_code=row["ts_code"], trade_date=row["trade_date"], field="volume"
                )
            )
        if bool(quote[i, j]) != math.isfinite(want_close):
            problems.append(
                dict(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    field="has_quote",
                )
            )
    # 因子：面板相邻两个有因子日的比值 = 原始前复权因子的比值。
    adj = np.load(path / "adj_factor.npy", mmap_mode="r")
    f = (
        _home_series(source_factor)
        .filter(pl.col("trade_date").is_in(dates))
        .filter(pl.col("ts_code").is_in(symbols))
        .unique(["ts_code", "trade_date"], keep="none")
        .sort("ts_code", "trade_date")
    )
    picked = sorted(set(f["ts_code"].unique().to_list()))
    picked = (
        [
            picked[k]
            for k in rng.choice(len(picked), size=min(300, len(picked)), replace=False)
        ]
        if picked
        else []
    )
    checked = len(take)
    for code, group in f.filter(pl.col("ts_code").is_in(picked)).group_by("ts_code"):
        code = code[0]
        values = group["cum_adjfactor"].to_list()
        when = group["trade_date"].to_list()
        j = col_of[code]
        for k in range(1, len(values)):
            if not values[k - 1] or not values[k]:
                continue
            want = values[k] / values[k - 1]
            got = adj[row_of[when[k]], j] / adj[row_of[when[k - 1]], j]
            checked += 1
            if not abs(got / want - 1) <= 1e-9:
                problems.append(
                    dict(ts_code=code, trade_date=when[k], field="adj_factor")
                )
                break
    source = np.load(path / "action_source.npy", mmap_mode="r")
    if int(np.max(source)) > 3:
        problems.append(dict(field="action_source", problem="枚举越界"))
    listed = np.load(path / "is_listed.npy", mmap_mode="r")
    if meta["market"] == "hk":
        outside = int((np.asarray(quote) & ~np.asarray(listed)).sum())
        if outside != meta["diagnostics"].get("quotes_outside_listing"):
            problems.append(
                dict(field="is_listed", problem="上市期外行情计数与元数据不符")
            )
    return problems, checked


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
