"""年度分块的向量化不变量；跨年仅保留每只证券最后一条行情。"""

import copy

import polars as pl

from .catalog import DATASETS
from .panel_events import dividend_rows


class Audit:
    def __init__(self):
        self.checks = {}
        self.carry = None

    def add(self, name, total, examples, *, failed=None, note=""):
        failures = len(examples) if failed is None else failed
        self.checks[name] = dict(
            checked=int(total),
            failed=int(failures),
            pass_rate=(total - failures) / total if total else None,
            examples=examples[:20],
            note=note,
        )

    def frame(self, name, table, bad, *, note="", columns=None):
        failed = table.filter(bad.fill_null(False))
        self.add(
            name,
            len(table),
            (failed.select(columns) if columns else failed).head(20).to_dicts(),
            failed=len(failed),
            note=note,
        )

    def merge(self, other):
        for name, check in other.checks.items():
            if name not in self.checks:
                self.checks[name] = copy.deepcopy(check)
                continue
            old = self.checks[name]
            old["checked"] += check["checked"]
            old["failed"] += check["failed"]
            old["examples"] = (old["examples"] + check["examples"])[:20]
            for key in (
                "unequal_days",
                "positive_reference_checked",
                "positive_reference_failed",
            ):
                if key in check:
                    old[key] = old.get(key, 0) + check[key]
            for key in ("by_year_board_st", "day_examples"):
                if key in check:
                    old[key] = old.get(key, []) + check[key]
                    if key == "day_examples":
                        old[key] = old[key][:20]
            if "failure_categories" in check:
                groups = {}
                for row in (
                    old.get("failure_categories", []) + check["failure_categories"]
                ):
                    groups[row["category"]] = (
                        groups.get(row["category"], 0) + row["len"]
                    )
                old["failure_categories"] = [
                    {"category": key, "len": value}
                    for key, value in sorted(groups.items())
                ]
            old["pass_rate"] = (
                (old["checked"] - old["failed"]) / old["checked"]
                if old["checked"] and name != "unlimited_sessions"
                else None
            )


def audit_tables(tables, dates, *, prior=None, unique_names=None):
    audit = Audit()
    daily = tables["daily"]
    securities = tables["stock_basic"].unique(subset="ts_code", keep="last")
    keys = ["ts_code", "trade_date"]
    missing = sorted(set(dates) - set(daily["trade_date"]))
    audit.add("daily_coverage", len(dates), missing)
    total = failed = 0
    examples = []
    for name, table in tables.items():
        if name not in DATASETS or (
            unique_names is not None and name not in unique_names
        ):
            continue
        columns = [k for k in DATASETS[name].keys if k in table.columns]
        if not columns:
            continue
        bad = table.filter(pl.struct(columns).is_duplicated())
        total += len(table)
        failed += len(bad)
        examples += [
            {"dataset": name, **row}
            for row in bad.select(columns).head(max(0, 20 - len(examples))).to_dicts()
        ]
    audit.add("unique_keys", total, examples, failed=failed)
    audit.frame(
        "ohlc_high",
        daily,
        pl.col("high") < pl.max_horizontal("open", "close"),
        columns=keys + ["open", "high", "close"],
    )
    audit.frame(
        "ohlc_low",
        daily,
        pl.col("low") > pl.min_horizontal("open", "close"),
        columns=keys + ["open", "low", "close"],
    )
    listed = daily.join(
        securities.select("ts_code", "list_date", "delist_date"),
        on="ts_code",
        how="left",
    )
    audit.frame(
        "no_post_delist_rows",
        listed,
        pl.col("trade_date") >= pl.col("delist_date"),
        columns=keys + ["delist_date"],
    )
    factors = (
        tables["adj_factor"]
        .select(*keys, "adj_factor")
        .unique(subset=keys, keep="last")
    )
    history = daily.join(factors, on=keys, how="left")
    if prior is not None:
        history = pl.concat([prior, history], how="diagonal_relaxed")
    history = history.sort("ts_code", "trade_date")
    audit.carry = history.group_by("ts_code", maintain_order=True).last()
    joined = history.with_columns(
        pl.col("close").shift(1).over("ts_code").alias("_prior_close"),
        pl.col("adj_factor").shift(1).over("ts_code").alias("_prior_factor"),
    ).filter(pl.col("trade_date").is_in(list(dates)))
    valid = joined.filter(
        pl.col("_prior_factor").is_not_null() & (pl.col("adj_factor") > 0)
    )
    valid = valid.with_columns(
        (pl.col("_prior_close") * pl.col("_prior_factor") / pl.col("adj_factor")).alias(
            "expected"
        )
    )
    audit.frame(
        "adjustment_continuity",
        valid,
        (pl.col("pre_close") - pl.col("expected")).abs() > 0.011,
        columns=keys + ["expected", "pre_close", "_prior_factor", "adj_factor"],
        note="上个成交日收盘×旧因子/新因子；容差 0.011 元",
    )
    events, _ = dividend_rows(tables["dividend"])
    actions = (
        events.group_by("ts_code", "ex_date")
        .agg(
            pl.col("cash_div_tax").fill_null(0).sum().alias("_cash"),
            (1 + pl.col("stk_div").fill_null(0).sum()).alias("_split"),
        )
        .rename({"ex_date": "trade_date"})
    )
    ex = joined.join(actions, on=keys, how="inner").filter(
        pl.col("_prior_close").is_not_null()
    )
    ex = ex.with_columns(
        ((pl.col("_prior_close") - pl.col("_cash")) / pl.col("_split")).alias(
            "expected"
        )
    )
    audit.frame(
        "ex_dividend_reference",
        ex,
        (pl.col("expected") - pl.col("pre_close")).abs() > 0.011,
        columns=keys + ["expected", "pre_close", "_cash", "_split"],
        note="去除跨报告期相同经济条款后的税前派现与送转；差异化分红及配股仍列为不一致",
    )
    changed = valid.filter(
        (pl.col("adj_factor") - pl.col("_prior_factor")).abs() > 1e-10
    )
    absent = changed.join(
        actions.filter((pl.col("_cash") != 0) | (pl.col("_split") != 1)),
        on=keys,
        how="anti",
    )
    audit.add(
        "factor_change_has_event",
        len(changed),
        absent.select(*keys, "_prior_factor", "adj_factor").head(20).to_dicts(),
        failed=len(absent),
        note="源复权因子变化但没有可用实施事件；可能是配股、合并或源事件缺口，未猜补",
    )
    susp = (
        tables["suspend_d"]
        .filter(
            (pl.col("suspend_type") == "S")
            & (
                pl.col("suspend_timing").is_null()
                | pl.col("suspend_timing").is_in(["", "全天"])
            )
        )
        .select(keys)
        .unique()
    )
    traded = susp.join(daily, on=keys, how="inner").filter(pl.col("vol") > 0)
    audit.add(
        "suspension_no_trade",
        len(susp),
        traded.select(*keys, "vol").head(20).to_dicts(),
        failed=len(traded),
    )
    cells = (
        securities.select("ts_code", "list_date", "delist_date")
        .join(pl.DataFrame({"trade_date": dates}), how="cross")
        .filter(
            (pl.col("trade_date") >= pl.col("list_date"))
            & (
                pl.col("delist_date").is_null()
                | (pl.col("trade_date") < pl.col("delist_date"))
            )
        )
    )
    gaps = cells.join(daily.select(keys).unique(), on=keys, how="anti").join(
        susp, on=keys, how="anti"
    )
    audit.add(
        "missing_quote_has_suspension",
        len(cells),
        gaps.select(keys).head(20).to_dicts(),
        failed=len(gaps),
        note="上市且无行情、无全天停牌记录；has_quote=False，不能自动认定为停牌",
    )
    basic = (
        tables["daily_basic"]
        .filter(pl.col("total_mv").is_not_null() & pl.col("total_share").is_not_null())
        .with_columns((pl.col("close") * pl.col("total_share")).alias("expected"))
    )
    audit.frame(
        "market_cap",
        basic,
        (pl.col("total_mv") - pl.col("expected")).abs()
        > (0.02 + 1e-4 * pl.col("expected").abs()),
        columns=keys + ["expected", "total_mv"],
        note="原始单位均为万；相对容差 1e-4，绝对容差 0.02 万元",
    )
    for name in ("adj_factor", "daily_basic", "stk_limit"):
        other = tables[name].select(keys).unique()
        missing_keys = daily.select(keys).unique().join(other, on=keys, how="anti")
        daily_counts = daily.group_by("trade_date").len().rename({"len": "daily"})
        counts = other.group_by("trade_date").len().rename({"len": name})
        unequal = daily_counts.join(counts, on="trade_date", how="left").filter(
            pl.col("daily") != pl.col(name)
        )
        audit.add(
            f"daily_vs_{name}",
            len(daily),
            missing_keys.head(20).to_dicts(),
            failed=len(missing_keys),
            note="有行情但源表缺该证券；面板保留 NaN，不代填",
        )
        audit.checks[f"daily_vs_{name}"]["unequal_days"] = len(unequal)
        audit.checks[f"daily_vs_{name}"]["day_examples"] = (
            unequal.sort("trade_date").head(20).to_dicts()
        )
    return audit
