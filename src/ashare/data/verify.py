"""全历史校验按年处理，报告只保留计数和有界样例。"""

import json
import time
from collections import Counter
from datetime import UTC, datetime

import polars as pl

from .aliases import CanonicalStore
from .catalog import DATASETS, INDUSTRY_CAVEAT
from .crosscheck import fingerprint
from .panel_events import dividend_rows
from .panel_quality import securities_table
from .rules import SOURCES
from .store import atomic_json
from .sync import plan
from .verify_limits import audit_limits
from .verify_panel import audit_panel
from .verify_tables import Audit, audit_tables

__all__ = ["Audit", "audit_limits", "audit_tables", "verify"]


def coverage_report(store, dates):
    coverage = {}
    for name, spec in DATASETS.items():
        entries = store.entries(name)
        complete = [e for e in entries.values() if e["status"] == "complete"]
        counts = Counter()
        values = []
        if spec.partition == "day":
            for e in complete:
                counts[e["partition"][:4]] += e["rows"]
                if e["rows"]:
                    values.append(e["partition"])
        elif complete:
            frame = store.scan(name)
            if spec.date in frame.collect_schema():
                values = (
                    frame.select(spec.date)
                    .collect()[spec.date]
                    .drop_nulls()
                    .cast(pl.String)
                    .to_list()
                )
                counts.update(v[:4] for v in values)
        try:
            expected = [
                part for part, _ in plan(spec, dates[0], dates[-1], dates, store)
            ]
        except FileNotFoundError:
            expected = ["依赖未采集"]
        missing = [
            part
            for part in expected
            if entries.get(f"{name}/{part}", {}).get("status") != "complete"
        ]
        empty = [
            e["partition"]
            for e in complete
            if e["rows"] == 0
            and not spec.empty_allowed
            and (not spec.coverage_start or e["partition"] >= spec.coverage_start)
        ]
        coverage[name] = dict(
            rows=sum(e["rows"] for e in complete),
            start=min(values) if values else None,
            end=max(values) if values else None,
            first_nonempty_partition=min(
                (e["partition"] for e in complete if e["rows"]), default=None
            ),
            yearly_rows=dict(sorted(counts.items())),
            missing_partitions=missing,
            suspicious_empty_partitions=empty,
            present=bool(complete),
            configured_page_size=spec.limit,
            max_partition_rows=max((e["rows"] for e in complete), default=0),
            repeated_round_row_counts=[
                {"rows": count, "partitions": frequency}
                for count, frequency in sorted(
                    Counter(e["rows"] for e in complete).items()
                )
                if count >= 100 and count % 100 == 0 and frequency >= 5
            ],
            exact_page_multiple_partitions=[
                e["partition"]
                for e in complete
                if e["rows"] and e["rows"] % spec.limit == 0
            ],
        )
    return coverage


def st_name_audit(store, dates):
    """逐年按曾用名区间交叉验证 ST 源，无法解释的缺口不推断为摘帽。"""
    if not store.entries("namechange"):
        return {}
    names = (
        store.read("namechange")
        .filter(pl.col("name").str.contains("ST|退"))
        .select("ts_code", "name", "start_date", "end_date")
        .unique()
    )
    securities = securities_table(store).select("ts_code", "list_date", "delist_date")
    result = {}
    for year in sorted({d[:4] for d in dates}):
        days = [d for d in dates if d.startswith(year)]
        expected = (
            names.join(pl.DataFrame({"trade_date": days}), how="cross")
            .filter(
                (pl.col("start_date") <= pl.col("trade_date"))
                & (
                    pl.col("end_date").is_null()
                    | (pl.col("end_date") >= pl.col("trade_date"))
                )
            )
            .select("ts_code", "trade_date")
            .unique()
        )
        expected = (
            expected.join(securities, on="ts_code", how="inner")
            .filter(
                (pl.col("trade_date") >= pl.col("list_date"))
                & (
                    pl.col("delist_date").is_null()
                    | (pl.col("trade_date") < pl.col("delist_date"))
                )
            )
            .select("ts_code", "trade_date")
        )
        actual = (
            store.read("stock_st", days[0], days[-1])
            .select("ts_code", "trade_date")
            .unique()
        )
        absent = expected.join(actual, on=["ts_code", "trade_date"], how="anti")
        extra = actual.join(expected, on=["ts_code", "trade_date"], how="anti")
        result[year] = dict(
            source_rows=len(actual),
            name_implied_rows=len(expected),
            missing_from_st=len(absent),
            extra_in_st=len(extra),
            examples=absent.sort("trade_date", "ts_code").head(10).to_dicts(),
        )
    return result


def verify(store, *, output=None):
    started = time.perf_counter()
    raw_store = store.snapshot()
    store = CanonicalStore(raw_store)
    manifest = store.manifest()
    entries = store.entries("daily").values()
    parts = [e["partition"] for e in entries if e["status"] == "complete"]
    calendar = store.read("trade_cal")
    dates = sorted(
        set(
            calendar.filter(
                (pl.col("is_open") == 1)
                & pl.col("cal_date").is_between(pl.lit(min(parts)), pl.lit(max(parts)))
            )["cal_date"]
        )
    )
    static = {"stock_basic": securities_table(store), "trade_cal": calendar}
    if store.entries("namechange"):
        static["namechange"] = store.read("namechange")
    coverage = coverage_report(raw_store, dates)
    audit, yearly, carry = Audit(), {}, None
    core = (
        "daily",
        "adj_factor",
        "daily_basic",
        "stk_limit",
        "stock_st",
        "suspend_d",
        "dividend",
    )
    for year in sorted({d[:4] for d in dates}):
        block = [d for d in dates if d.startswith(year)]
        tables = {**static, **{n: store.read(n, block[0], block[-1]) for n in core}}
        part = audit_tables(tables, block, prior=carry, unique_names=core)
        carry = part.carry
        audit_limits(tables, block, part)
        panel_check = audit_panel(store, block)
        if panel_check.get("status") == "complete":
            part.add(
                "factor_change_has_panel_event",
                panel_check["checked"],
                panel_check["examples"],
                failed=panel_check["failed"],
                note="已发布面板：相邻交易日因子变化超过0.1%噪声阈值，但股数乘数/现金均无事件",
            )
            check = panel_check["reconciliation"]
            part.add(
                "action_reconciliation",
                check["checked"],
                check["examples"],
                failed=check["failed"],
                note="事件隐含比/相邻交易日因子比误差不超过构建容差；有事件必须有因子变化",
            )
        yearly[year] = part.checks
        audit.merge(part)
        del tables
    # 核心表以外的主键单独投影，避免读入数百个财务列。
    duplicates, duplicate_count, total = [], 0, 0
    for name, spec in DATASETS.items():
        if name in core or not store.entries(name):
            continue
        frame = store.scan(name)
        keys = [c for c in spec.keys if c in frame.collect_schema()]
        table = frame.select(keys).collect()
        bad = table.filter(pl.struct(keys).is_duplicated())
        total += len(table)
        duplicate_count += len(bad)
        duplicates += [
            dict(dataset=name, **r)
            for r in bad.head(max(0, 20 - len(duplicates))).to_dicts()
        ]
    extra = Audit()
    extra.add("unique_keys", total, duplicates, failed=duplicate_count)
    audit.merge(extra)
    consistency = [key for key in manifest if not store.complete(key)]
    audit.add("manifest_files", len(manifest), consistency)
    for name, item in coverage.items():
        if DATASETS[name].partition == "day" and not DATASETS[name].empty_allowed:
            bad = item["suspicious_empty_partitions"] + item["missing_partitions"]
            audit.add(f"partition_coverage_{name}", len(dates), bad)
    raw_dividend = store.read("dividend")
    dividend, event_stats = dividend_rows(
        raw_dividend,
        sorted(set(calendar.filter(pl.col("is_open") == 1)["cal_date"])),
        static["stock_basic"]["ts_code"].to_list(),
    )
    event_stats["missing_ann_date"] = raw_dividend.filter(
        (pl.col("div_proc") == "实施") & pl.col("ann_date").is_null()
    ).height
    event_stats["announcement_note"] = (
        "missing_ann_date 为预案/一般公告日；missing_announcement 为实施公告日 imp_ann_date，二者不能混淆。"
    )
    del dividend, raw_dividend
    report = dict(
        version=store.version,
        panel_version=json.loads((store.root / "panel/current.json").read_text())[
            "version"
        ]
        if (store.root / "panel/current.json").exists()
        else None,
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        coverage=coverage,
        checks=audit.checks,
        by_year=yearly,
        dividend_statistics=event_stats,
        st_name_crosscheck=st_name_audit(store, dates),
        rule_sources=SOURCES,
        caveats=[
            "历史修订不能还原首次发布版本",
            "异常不是自动修复指令",
            INDUSTRY_CAVEAT,
        ],
    )
    cross = store.root / "reports/baostock.json"
    report["baostock"] = (
        json.loads(cross.read_text())
        if cross.exists()
        else {"status": "not_run", "command": "python -m ashare.data verify --baostock"}
    )
    item = report["baostock"]
    if item.get("status") == "complete" and item.get(
        "input_fingerprint"
    ) != fingerprint(
        store, item.get("sample_size", 300), item.get("requested_status_days", 2)
    ):
        item.update(
            status="stale", note="原始输入已改变；以前抽样结果不能代表本次全历史"
        )
    gap_path = store.root / "reports/source-gap-analysis.json"
    if gap_path.exists():
        gap_report = json.loads(gap_path.read_text())
        gap_report["scope"] = (
            "第二轮原始代码口径历史解释；本轮换码规范化后的计数见checks，不沿用旧计数"
        )
        report["cross_dataset_explanations"] = gap_report
    report["elapsed_seconds"] = time.perf_counter() - started
    output = output or store.root / "reports/verify"
    atomic_json(output.with_suffix(".json"), report)
    lines = ["# 数据校验", "", "|规则|检查数|不一致|通过率|", "|---|---:|---:|---:|"]
    for name, check in audit.checks.items():
        rate = (
            f"{check['pass_rate']:.6%}"
            if check["pass_rate"] is not None
            else "未检查/不适用"
        )
        lines.append(f"|{name}|{check['checked']}|{check['failed']}|{rate}|")
    for title, value in [
        ("不变量及样例", audit.checks),
        ("覆盖率", coverage),
        ("逐年校验", yearly),
        ("分红事件", event_stats),
        ("ST 与曾用名", report["st_name_crosscheck"]),
        ("BaoStock", report["baostock"]),
    ]:
        lines += [
            "",
            f"## {title}",
            "```json",
            json.dumps(value, ensure_ascii=False, indent=2),
            "```",
        ]
    lines += ["", "## 规则来源", *SOURCES]
    output.with_suffix(".md").write_text("\n".join(lines))
    return report
