"""面板输入的完整性检查与确定性证券主表。"""

import hashlib
from pathlib import Path

import polars as pl

from .catalog import DATASETS, FIELDS, INDEX_STARTS, INDEXES
from .sync import months


def code_version():
    digest = hashlib.sha256()
    # 只绑定实际参与构建的代码；修改校验报告不会使矩阵失效。
    modules = (
        "aliases",
        "panel",
        "panel_events",
        "panel_actions",
        "panel_cache",
        "panel_increment",
        "panel_io",
        "panel_quality",
        "catalog",
        "store",
        "manifest",
        "sync",
    )
    for path in sorted(Path(__file__).parent / f"{name}.py" for name in modules):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def securities_table(store):
    table = store.read("stock_basic")
    order = [
        c
        for c in (
            "ts_code",
            "_fetched_at",
            "list_date",
            "delist_date",
            "list_status",
            "name",
        )
        if c in table.columns
    ]
    return (
        table.sort(order, nulls_last=False)
        .unique(subset="ts_code", keep="last", maintain_order=True)
        .sort("ts_code")
    )


def required(store, dates, *, strict=False):
    manifest = store.manifest()
    issues = []
    for name in {f.dataset for f in FIELDS.values()} | {"trade_cal"}:
        spec = DATASETS[name]
        entries = store.entries(name)
        if not entries:
            if name == "index_weight" and not strict:
                issues.append(dict(kind="missing_index_snapshot", index="all"))
                continue
            raise FileNotFoundError(
                f"缺少 {name}；请运行 python -m ashare.data sync --datasets {name}"
            )
        if spec.partition != "day" and any(
            e["status"] != "complete" for e in entries.values()
        ):
            raise ValueError(f"{name} 有未完成分区；请重跑 sync --datasets {name}")
        if spec.partition == "day":
            for day in dates:
                entry = manifest.get(f"{name}/{day}", {})
                if spec.coverage_start and day < spec.coverage_start:
                    continue
                if entry.get("status") != "complete":
                    raise FileNotFoundError(
                        f"缺少 {name}/{day}；请运行 python -m ashare.data sync --datasets {name} --start {dates[0]} --end {dates[-1]}"
                    )
                if not spec.empty_allowed and entry.get("rows", 0) == 0:
                    raise ValueError(
                        f"{name}/{day} 空分区可疑；请 sync --datasets {name} --force"
                    )
            ordered = sorted(
                (
                    e
                    for e in entries.values()
                    if e.get("rows") and dates[0] <= e["partition"] <= dates[-1]
                ),
                key=lambda e: e["partition"],
            )
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if (
                    previous["rows"] >= spec.drop_check_min_rows
                    and current["rows"] < previous["rows"] * spec.drop_check_ratio
                    and not spec.empty_allowed
                ):
                    issues.append(
                        dict(
                            dataset=name,
                            partition=current["partition"],
                            previous_rows=previous["rows"],
                            rows=current["rows"],
                            kind="row_collapse",
                        )
                    )
    monthly = store.entries("index_weight")
    planned = all(
        "_" in e["partition"] and len(e["partition"].rsplit("_", 1)[-1]) == 6
        for e in monthly.values()
    )
    weights = None if planned else store.read("index_weight")
    for index, code in INDEXES.items():
        snaps = (
            []
            if planned
            else weights.filter(pl.col("index_code") == code)["trade_date"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        # 当前月快照尚未发生不算缺失；上一自然月及以前必须存在快照。
        from datetime import date, timedelta

        prior = (
            date(int(dates[0][:4]), int(dates[0][4:6]), 1) - timedelta(days=1)
        ).strftime("%Y%m01")
        for begin, finish in months(prior, str(dates[-1])):
            if finish >= dates[-1] or finish < INDEX_STARTS[index]:
                continue
            entry = monthly.get(f"index_weight/{code}_{begin[:6]}", {})
            present = (
                (entry.get("status") == "complete" and entry.get("rows", 0) > 0)
                if planned
                else any(begin <= d <= finish for d in snaps)
            )
            if not present:
                message = f"指数 {index} 缺少 {begin[:6]} 快照；沿用最近已知快照，请 python -m ashare.data sync --datasets index_weight --start {begin} --end {finish} --force"
                if strict:
                    raise ValueError(message)
                issues.append(
                    dict(
                        kind="missing_index_snapshot",
                        index=index,
                        month=begin[:6],
                        message=message,
                    )
                )
    return issues
