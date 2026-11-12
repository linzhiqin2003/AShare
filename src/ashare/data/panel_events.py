"""稀疏事件向每日状态的转换；所有区间边界集中在这里。"""

import numpy as np
import polars as pl

from .catalog import INDEXES, SCALES


def listing(dates, symbols, securities, calendar):
    by_code = {r["ts_code"]: r for r in securities.iter_rows(named=True)}
    listed = np.zeros((len(dates), len(symbols)), dtype=bool)
    age = np.full(listed.shape, np.nan)
    delist = np.zeros_like(listed)
    day_numbers = np.searchsorted(calendar, dates)
    for j, symbol in enumerate(symbols):
        row = by_code[symbol]
        begin = row["list_date"]
        finish = row.get("delist_date")
        mask = dates >= begin
        if finish:
            mask &= dates < finish
            day_index = np.searchsorted(calendar, finish)
            if day_index < len(calendar):
                event_day = calendar[day_index]
                i = np.searchsorted(dates, event_day)
                if i < len(dates) and dates[i] == event_day:
                    delist[i, j] = True
        listed[:, j] = mask
        age[mask, j] = day_numbers[mask] - np.searchsorted(calendar, begin) + 1
    return listed, age, delist


def daily_mask(table, dates, symbols, *, suspension=False):
    output = np.zeros((len(dates), len(symbols)), dtype=bool)
    lookup = {symbol: j for j, symbol in enumerate(symbols)}
    for row in table.iter_rows(named=True):
        day, symbol = row.get("trade_date"), row.get("ts_code")
        if day is None or symbol not in lookup:
            continue
        i = np.searchsorted(dates, day)
        if i >= len(dates) or dates[i] != day:
            continue
        if suspension and (
            row.get("suspend_type") != "S"
            or row.get("suspend_timing") not in (None, "", "全天")
        ):
            continue
        output[i, lookup[symbol]] = True
    return output


def industry(table, dates, symbols):
    output = np.full((len(dates), len(symbols)), np.nan)
    lookup = {symbol: j for j, symbol in enumerate(symbols)}
    # 同一一级行业下的三级分类变更允许重叠；一级冲突显式失败。
    for row in table.unique(
        subset=["ts_code", "l1_code", "in_date", "out_date"]
    ).iter_rows(named=True):
        if row["ts_code"] not in lookup or not row["in_date"]:
            continue
        left = np.searchsorted(dates, row["in_date"], side="left")
        right = (
            np.searchsorted(dates, row["out_date"], side="right")
            if row["out_date"]
            else len(dates)
        )
        j = lookup[row["ts_code"]]
        value = float(row["l1_code"].split(".")[0])
        previous = output[left:right, j]
        if np.any(np.isfinite(previous) & (previous != value)):
            raise ValueError(
                f"行业区间冲突：{row['ts_code']} {row['in_date']}；请运行 verify"
            )
        output[left:right, j] = value
    return output


def weights(table, dates, symbols, index, *, membership=False):
    output = np.full((len(dates), len(symbols)), np.nan)
    lookup = {symbol: j for j, symbol in enumerate(symbols)}
    table = table.filter(pl.col("index_code") == INDEXES[index])
    snapshots = sorted(set(table["trade_date"].drop_nulls()))
    for k, day in enumerate(snapshots):
        left = np.searchsorted(dates, day, side="right")
        right = (
            np.searchsorted(dates, snapshots[k + 1], side="right")
            if k + 1 < len(snapshots)
            else len(dates)
        )
        if left >= right:
            continue
        output[left:right, :] = 0
        for row in table.filter(pl.col("trade_date") == day).iter_rows(named=True):
            if row["con_code"] in lookup:
                output[left:right, lookup[row["con_code"]]] = (
                    1.0 if membership else row["weight"] * SCALES["index_weight"]
                )
    return output


def dividend_rows(table, dates=None, symbols=None):
    """只合并经济条款完全相同的记录，保留原始表并返回每条丢弃路径的计数。"""
    table = table.filter(pl.col("div_proc") == "实施")
    stats = {"implementation_rows": len(table)}
    stats["same_day_groups"] = (
        table.group_by("ts_code", "ex_date").len().filter(pl.col("len") > 1).height
    )
    stats["same_day_rows"] = table.filter(
        pl.struct("ts_code", "ex_date").is_duplicated()
    ).height
    stats["missing_announcement"] = table.filter(
        pl.col("imp_ann_date").is_null()
    ).height
    known = (
        pl.coalesce("imp_ann_date", "ann_date")
        if "ann_date" in table.columns
        else pl.col("imp_ann_date")
    )
    keyed = table.with_columns(
        known.alias("_known"),
        (pl.col("stk_div").fill_null(0) * 1e6)
        .round(0)
        .cast(pl.Int64)
        .alias("_stock_key"),
        (pl.col("cash_div_tax").fill_null(0) * 1e6)
        .round(0)
        .cast(pl.Int64)
        .alias("_cash_key"),
    )
    subset = ["ts_code", "ex_date", "_stock_key", "_cash_key"]
    keyed = keyed.with_columns(pl.len().over(subset).alias("_records"))
    distinct = keyed.sort("_known", nulls_last=True).unique(
        subset=subset, maintain_order=True
    )
    stats["duplicate_rows_removed"] = len(table) - len(distinct)
    stats["missing_announcement_unusable"] = distinct.filter(
        pl.col("_known").is_null()
    ).height
    stats["late_announcement"] = distinct.filter(
        pl.col("_known") > pl.col("ex_date")
    ).height
    stats["unknown_symbol"] = (
        distinct.filter(~pl.col("ts_code").is_in(list(symbols))).height
        if symbols is not None
        else 0
    )
    stats["non_trading_ex_date"] = (
        distinct.filter(~pl.col("ex_date").is_in(list(dates))).height
        if dates is not None
        else 0
    )
    valid = distinct.filter(
        pl.col("ex_date").is_not_null()
        & pl.col("_known").is_not_null()
        & (pl.col("_known") <= pl.col("ex_date"))
    )
    return valid, stats


def dividends(table, dates, symbols, *, diagnostics=None):
    split = np.ones((len(dates), len(symbols)))
    cash = np.zeros_like(split)
    lookup = {symbol: j for j, symbol in enumerate(symbols)}
    table, stats = dividend_rows(table, dates, symbols)
    if diagnostics is not None:
        diagnostics.update(stats)
    for row in table.iter_rows(named=True):
        day = row.get("ex_date")
        if row["ts_code"] not in lookup:
            continue
        i = np.searchsorted(dates, day)
        if i < len(dates) and dates[i] == day:
            j = lookup[row["ts_code"]]
            split[i, j] += row.get("stk_div") or 0
            cash[i, j] += row.get("cash_div_tax") or 0
    return split, cash


def stitched_industry(store, dates, symbols):
    """旧版缺口用新版历史回填，并逐格标注重述来源；原始区间不改。"""
    current = industry(store.read("index_member_all"), dates, symbols)
    output = current.copy()
    sources = np.where(np.isfinite(current), 2, 0).astype(np.uint8)
    cut = int(np.searchsorted(dates, "20210802"))
    if store.entries("index_member"):
        old = store.read("index_member").rename(
            {"con_code": "ts_code", "index_code": "l1_code"}
        )
        previous = industry(old, dates[:cut], symbols)
        present = np.isfinite(previous)
        output[:cut] = np.where(present, previous, current[:cut])
        sources[:cut] = np.where(present, 1, np.where(np.isfinite(current[:cut]), 3, 0))
        # SW2014 发布前的旧版历史也不是当时分类，严格筛选不得误用。
        sources[: int(np.searchsorted(dates, "20140221"))] = np.where(
            np.isfinite(output[: int(np.searchsorted(dates, "20140221"))]), 3, 0
        )
    else:
        sources[:cut] = np.where(np.isfinite(current[:cut]), 3, 0)
    caveat = "0缺失，1 SW2014当时口径，2 SW2021当时口径，3历史回填/重述；2014-02-21前均为重述；缺口用SW2021回填带后见之明，严格研究仅用来源1/2；纳入剔除两端均含"
    return output, sources, caveat
