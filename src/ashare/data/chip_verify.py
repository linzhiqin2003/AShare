"""筹码原始值检查与复权基准证据报告。基准未证实，不提供面板换算函数。

his_low/his_high 不是上市以来的价格极值，二者比较只作诊断，不判定复权基准。
分区原始文件内的抓取时间决定候选因子，不能用清单中最后一次无变化刷新的时间。
容差 0.05 元仅统计诊断差异；发布所需证据见 DESIGN 6.1.5。
"""

from bisect import bisect_left, bisect_right
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .verify_tables import Audit

QUANTILES = ("cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct")


def fetched_day(value):
    """抓取时间按上海日期解释；无时区/损坏时间不能可靠定位基准。"""
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is not None:
            return stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    except (ValueError, TypeError):
        pass
    return None


def _calendar_index(calendar):
    """只接受有明确开闭市状态的 SSE 日历；冲突或缺失日不能作为完整性证据。"""
    states = {}
    if calendar is not None and {"cal_date", "is_open"} <= set(calendar.columns):
        if "exchange" in calendar.columns:
            calendar = calendar.filter(pl.col("exchange") == "SSE")
        for day, opened in calendar.select("cal_date", "is_open").iter_rows():
            try:
                if datetime.strptime(day, "%Y%m%d").strftime("%Y%m%d") != day:
                    continue
            except (TypeError, ValueError):
                continue
            states.setdefault(day, set()).add(str(opened))
    valid = {d: s == {"1"} for d, s in states.items() if s in ({"0"}, {"1"})}
    return sorted(valid), valid


def _full_suspensions(suspensions):
    """沿用股票面板的全日停牌定义；临时停牌和复牌记录不能解释缺行情。"""
    required = {"ts_code", "trade_date", "suspend_type", "suspend_timing"}
    if suspensions is None or not required <= set(suspensions.columns):
        return set()
    return {
        (r["ts_code"], r["trade_date"])
        for r in suspensions.iter_rows(named=True)
        if r["suspend_type"] == "S" and r["suspend_timing"] in (None, "", "全天")
    }


def _history_evidence(code, day, listing, past, calendar_days, calendar, suspended):
    """上市区间内逐交易日要有行情或显式全日停牌，日历本身也要逐自然日连续。"""
    start = listing.get("list_date")
    end = listing.get("delist_date")
    try:
        first, last = (datetime.strptime(d, "%Y%m%d") for d in (start, day))
        if first > last or (end and day >= end):
            return "不在已知上市区间", []
    except (TypeError, ValueError):
        return "缺有效上市日期", []
    days = calendar_days[
        bisect_left(calendar_days, start) : bisect_right(calendar_days, day)
    ]
    if len(days) != (last - first).days + 1:
        return "交易日历缺日或开闭市状态不明", []
    present = set(past["trade_date"])
    missing = sorted(
        d
        for d in days
        if calendar[d] and d not in present and (code, d) not in suspended
    )
    if missing:
        return "交易日缺行情且无全日停牌证据", missing
    return "", []


def basis_rows(
    chips,
    daily,
    factors,
    basic,
    *,
    calendar=None,
    suspensions=None,
    price_tolerance=0.05,
):
    """逐只逐日比较边界与候选极值，仅作诊断；历史缺失不能计为容差内。

    返回逐条诊断，保留预期低高、因子和完整性。只使用 trade_date <= 被核对交易日的极值，
    因子基准取 <= _fetched_at 上海日期的最近一条；跨日的文件逐行分别解释。
    """
    listing = {r["ts_code"]: r for r in basic.iter_rows(named=True)}
    calendar_days, calendar_state = _calendar_index(calendar)
    suspended = _full_suspensions(suspensions)
    histories = {
        k[0]: v.sort("trade_date")
        for k, v in daily.partition_by("ts_code", as_dict=True).items()
    }
    adjustments = {
        k[0]: v.sort("trade_date")
        for k, v in factors.partition_by("ts_code", as_dict=True).items()
    }
    result = []
    for (code,), group in chips.partition_by("ts_code", as_dict=True).items():
        history, adjustment = histories.get(code), adjustments.get(code)
        for row in group.iter_rows(named=True):
            day, fetched = row["trade_date"], fetched_day(row.get("_fetched_at"))
            low = high = basis = np.nan
            complete = False
            reason, missing = "缺行情、复权因子或有效抓取时间", []
            if (
                history is not None
                and adjustment is not None
                and fetched
                and fetched >= day
            ):
                before = adjustment.filter(pl.col("trade_date") <= fetched)
                past = history.filter(pl.col("trade_date") <= day)
                start = listing.get(code, {}).get("list_date")
                if start:
                    past = past.filter(pl.col("trade_date") >= start)
                if len(before) and len(past):
                    reason, missing = _history_evidence(
                        code,
                        day,
                        listing.get(code, {}),
                        past,
                        calendar_days,
                        calendar_state,
                        suspended,
                    )
                    basis = float(before["adj_factor"][-1] or np.nan)
                    joined = past.join(
                        adjustment,
                        on=["ts_code", "trade_date"],
                        how="left",
                        validate="1:1",
                    )
                    a = joined["adj_factor"].cast(pl.Float64).to_numpy()
                    lows = joined["low"].cast(pl.Float64).to_numpy() * a
                    highs = joined["high"].cast(pl.Float64).to_numpy() * a
                    valid = (
                        np.isfinite(a)
                        & (a > 0)
                        & np.isfinite(lows)
                        & np.isfinite(highs)
                    )
                    if not valid.all() and not reason:
                        reason = "行情缺有效价格或对应复权因子"
                    if (not np.isfinite(basis) or basis <= 0) and not reason:
                        reason = "缺有效抓取基准复权因子"
                    complete = not reason
                    if valid.any() and np.isfinite(basis) and basis > 0:
                        low = float(lows[valid].min() / basis)
                        high = float(highs[valid].max() / basis)
            observed_low = float(
                row.get("his_low") if row.get("his_low") is not None else np.nan
            )
            observed_high = float(
                row.get("his_high") if row.get("his_high") is not None else np.nan
            )
            within_tolerance = bool(
                complete
                and abs(observed_low - low) <= price_tolerance + 1e-10
                and abs(observed_high - high) <= price_tolerance + 1e-10
            )
            result.append(
                dict(
                    ts_code=code,
                    trade_date=day,
                    fetched_day=fetched,
                    observed_low=observed_low,
                    observed_high=observed_high,
                    expected_low=low,
                    expected_high=high,
                    basis_factor=basis,
                    history_complete=complete,
                    history_note="逐日核实完整"
                    if complete
                    else f"不足以核实：{reason}",
                    missing_trade_days=len(missing),
                    missing_trade_day_examples=missing[:10],
                    within_tolerance=within_tolerance,
                )
            )
    return pl.DataFrame(result)


def check_chip_values(table):
    """原始胜率转为比例检查范围；成本正值、分位单调与覆盖起点独立检查。空值另报覆盖。"""
    audit = Audit()
    keys = ["ts_code", "trade_date"]
    winner = table.filter(pl.col("winner_rate").is_not_null()).with_columns(
        (pl.col("winner_rate").cast(pl.Float64) / 100).alias("winner")
    )
    audit.frame(
        "chip_winner_range",
        winner,
        ~pl.col("winner").is_finite() | ~pl.col("winner").is_between(0.0, 1.0),
        columns=keys,
    )
    costs = [*QUANTILES, "weight_avg"]
    audit.frame(
        "chip_cost_positive",
        table,
        pl.any_horizontal(
            [
                (
                    pl.col(c).is_not_null()
                    & (~pl.col(c).cast(pl.Float64).is_finite() | (pl.col(c) <= 0))
                )
                for c in costs
            ]
        ),
        columns=keys,
    )
    audit.frame(
        "chip_quantile_order",
        table,
        pl.any_horizontal(
            [
                pl.col(a) > pl.col(b)
                for a, b in zip(QUANTILES, QUANTILES[1:], strict=False)
            ]
        ),
        columns=keys,
    )
    audit.frame(
        "chip_coverage_start", table, pl.col("trade_date") < "20180102", columns=keys
    )
    for column in (*costs, "winner_rate"):
        audit.frame(
            f"chip_present_{column}",
            table,
            pl.col(column).is_null(),
            columns=keys,
            note="原始字段非空覆盖；不前填、不插值",
        )
    return audit


def audit_chip_basis(store, *, sample_size=200):
    """发布证据尚缺，基准固定为未证实；极值诊断不能影响此结论或正式失败数。"""
    return dict(
        status="unverified",
        status_label="未证实",
        checked=0,
        failed=0,
        pass_rate=None,
        missing_evidence=[
            dict(
                id="cyq_chips_distribution",
                description="缺少少量 cyq_chips 分布，用于重算分位成本、均价和胜率并核对价格尺度",
            ),
            dict(
                id="corporate_action_method",
                description="缺少特殊公司行为的成本尺度口径，整段常数倍偏差尚未解释",
            ),
            dict(
                id="cross_ex_date_snapshots",
                description="缺少跨实际除权的同一证券日两份抓取快照，尚未证实换算比值不变性",
            ),
        ],
        note="his_low/his_high 历史极值判据已作废；诊断相符也不能证实复权基准，继续不发布 chip_* 字段",
        extrema_diagnostic=_audit_chip_extrema(store, sample_size=sample_size),
    )


def _extrema_report(*, data_status, checked=0, **details):
    return (
        dict(
            status="diagnostic",
            status_label="诊断（不判定）",
            data_status=data_status,
            checked=checked,
            within_tolerance=0,
            outside_tolerance=0,
            examples=[],
        )
        | details
    )


def _audit_chip_extrema(store, *, sample_size):
    """有界极值诊断：每年首三个已有分区、最多 200 只，不产生通过/失败判定。

    正式库从 2007 年起，优先选这以后上市的股票以避免把截断历史当作上市以来。
    历史完整性仍逐日核实；不足时单列说明，不能混入容差内外的数量。
    """
    entries = [
        e
        for e in store.entries("cyq_perf").values()
        if e["status"] == "complete" and e["rows"]
    ]
    if not entries:
        return _extrema_report(data_status="not_downloaded")
    parts = sorted(e["partition"] for e in entries)
    selected = [
        d
        for year in sorted({d[:4] for d in parts})
        for d in [p for p in parts if p.startswith(year)][:3]
    ]
    chips = pl.concat(
        [store.read("cyq_perf", d, d) for d in selected], how="diagonal_relaxed"
    )

    def insufficient(note):
        return _extrema_report(
            data_status="insufficient_history",
            note=f"不足以核实：{note}",
        )

    try:
        basic = store.read("stock_basic").unique("ts_code", keep="last")
    except FileNotFoundError:
        return insufficient("缺上市区间")
    first_daily = min(
        (
            e["partition"]
            for e in store.entries("daily").values()
            if e["status"] == "complete"
        ),
        default=None,
    )
    if first_daily is None:
        return insufficient("缺日线")
    codes = (
        basic.filter(
            (pl.col("list_date") >= first_daily) & (pl.col("list_date") <= selected[0])
        )
        .sort("ts_code")["ts_code"]
        .head(sample_size)
        .to_list()
    )
    chips = chips.filter(pl.col("ts_code").is_in(codes))
    if not len(chips):
        return _extrema_report(data_status="insufficient_sample")
    daily = (
        store.scan("daily")
        .filter(pl.col("ts_code").is_in(codes))
        .select("ts_code", "trade_date", "low", "high")
        .collect()
    )
    try:
        factors = (
            store.scan("adj_factor")
            .filter(pl.col("ts_code").is_in(codes))
            .select("ts_code", "trade_date", "adj_factor")
            .collect()
        )
    except FileNotFoundError:
        return insufficient("缺复权因子")
    evidence = {}
    for name in ("trade_cal", "suspend_d"):
        try:
            evidence[name] = store.read(name)
        except FileNotFoundError:
            evidence[name] = None
    rows = basis_rows(
        chips,
        daily,
        factors,
        basic,
        calendar=evidence["trade_cal"],
        suspensions=evidence["suspend_d"],
    )
    # JSON 不输出 NaN/Infinity，缺因子、缺历史保持显式空值。
    floats = pl.col(pl.Float64)
    examples = (
        rows.head(20)
        .with_columns(pl.when(floats.is_finite()).then(floats).otherwise(None))
        .to_dicts()
    )
    return _extrema_report(
        data_status="insufficient_history"
        if not rows["history_complete"].all()
        else "available",
        checked=len(rows),
        within_tolerance=int(rows["within_tolerance"].sum()),
        outside_tolerance=int(
            (rows["history_complete"] & ~rows["within_tolerance"]).sum()
        ),
        examples=examples,
        stocks=rows["ts_code"].n_unique(),
        dates=selected,
        price_tolerance=0.05,
        history_incomplete=int((~rows["history_complete"]).sum()),
        note="不足以核实的行单列；按交易日历、上市区间和全日停牌证据逐日核对；极值差异不判定复权基准",
    )
