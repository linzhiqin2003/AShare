"""D3 的小量、可缓存真实核对；原始快照只写 worktree/tmp，不进入原始库。

复跑：python bench/d3_real_check.py --samples tmp/d3 --download
已有缓存：python bench/d3_real_check.py --samples tmp/d3
ETF 核对可加 --fund-root <只读数据根>；筹码逐日完整性需 --history-root 中的日历/停牌证据，
未提供则标为不足以核实。固定 2026-11-16 的研究快照日期，不自动扩到全量。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np
import polars as pl

from ashare.data.chip_verify import basis_rows
from ashare.data.sources import Tushare
from ashare.data.store import Store

ROOT = Path(__file__).resolve().parents[1]
DAYS = (
    "20180102",
    "20180103",
    "20180104",
    "20210104",
    "20210105",
    "20210106",
    "20240102",
    "20240103",
    "20240104",
    "20261110",
    "20261111",
    "20261112",
)


def cached(folder, api, params):
    return folder / (api + "_" + "_".join(str(v) for v in params.values()) + ".json")


def unpack(payload):
    data = payload["data"]
    frame = pl.DataFrame(
        data["items"], schema=data["fields"], orient="row", infer_schema_length=None
    )
    return frame.with_columns(pl.lit(payload["fetched_at"]).alias("_fetched_at"))


def eligible_codes(basic):
    """Polars 的日期字符串必须作为字面量，避免被当成列名。"""
    return basic.filter(
        pl.col("list_date").is_between(pl.lit("20070104"), pl.lit("20171231"))
    )["ts_code"].to_list()


async def collect(folder, download, history_root=None):
    source = Tushare(rate=250, concurrency=4, budget=700) if download else None

    async def fetch(api, **params):
        path = cached(folder, api, params)
        if path.exists():
            return unpack(json.loads(path.read_text()))
        if source is None:
            raise FileNotFoundError(f"缺少样本 {path.name}；用 --download 补抓")
        # 显式分页，单页参数进缓存键；文件不含鉴权、完整输入仅落 tmp。
        result = await source.fetch(api, params, timeout=180)
        frame = result.table
        payload = dict(
            data=dict(fields=frame.columns, items=frame.rows()),
            fetched_at=result.fetched_at,
        )
        path.write_text(json.dumps(payload, ensure_ascii=False))
        return unpack(payload)

    try:
        for day in ("20240102", "20261112"):
            path = folder / f"etf_nav_{day}.parquet"
            if download and not path.exists():
                result = await source.fetch(
                    "etf_share_size",
                    dict(trade_date=day),
                    fields="trade_date,ts_code,total_share,total_size,nav,close",
                    page_size=5000,
                    timeout=180,
                )
                result.table.with_columns(
                    pl.lit(result.fetched_at).alias("_fetched_at")
                ).write_parquet(path)
        statuses = {}
        for status in ("L", "D", "P"):
            statuses[status] = len(await fetch("etf_basic", list_status=status))
        early = await fetch(
            "etf_share_size", start_date="19900101", end_date="20051231", limit=5000
        )
        middle = await fetch(
            "etf_share_size", start_date="20060101", end_date="20081231", limit=5000
        )
        first = await fetch("etf_share_size", trade_date="20090105", limit=5000)
        following = await fetch("etf_share_size", trade_date="20090106", limit=5000)
        basic = await fetch("stock_basic", list_status="L", limit=6000)
        recent = set()
        for day in DAYS[-3:]:
            frame = await fetch("dividend", ex_date=day, limit=2000)
            recent.update(frame.filter(pl.col("div_proc") == "实施")["ts_code"])
        eligible = eligible_codes(basic)
        codes = sorted(set(random.Random(31415).sample(sorted(eligible), 210)) | recent)
        chips = pl.concat(
            [await fetch("cyq_perf", trade_date=d) for d in DAYS],
            how="diagonal_relaxed",
        ).filter(pl.col("ts_code").is_in(codes))
        semaphore = asyncio.Semaphore(4)

        async def history(api, code):
            async with semaphore:
                offset, pages = 0, []
                while True:
                    frame = await fetch(
                        api,
                        ts_code=code,
                        start_date="19900101",
                        end_date="20261116",
                        limit=6000,
                        offset=offset,
                    )
                    pages.append(frame)
                    if len(frame) < 6000:
                        return pl.concat(pages, how="diagonal_relaxed")
                    offset += len(frame)
                    if offset > 20000:
                        raise ValueError("超出单股预算，疑似分页失效")

        daily = pl.concat(
            await asyncio.gather(*(history("daily", c) for c in codes)),
            how="diagonal_relaxed",
        )
        factors = pl.concat(
            await asyncio.gather(*(history("adj_factor", c) for c in codes)),
            how="diagonal_relaxed",
        )
        evidence = {}
        if history_root is not None:
            reference = Store(history_root, readonly=True)
            for name in ("trade_cal", "suspend_d"):
                try:
                    evidence[name] = reference.read(name)
                except FileNotFoundError:
                    evidence[name] = None
        compared = basis_rows(
            chips,
            daily.drop("_fetched_at"),
            factors.select("ts_code", "trade_date", "adj_factor"),
            basic,
            calendar=evidence.get("trade_cal"),
            suspensions=evidence.get("suspend_d"),
        )
        compared.write_parquet(folder / "basis_comparison.parquet")
        summary = dict(
            selected_stocks=len(codes),
            recent_ex_stocks=len(recent),
            rows=len(compared),
            history_incomplete=int((~compared["history_complete"]).sum()),
            by_year={},
            etf_catalog_probe=dict(
                statuses=statuses,
                rows_before_2006=len(early),
                rows_2006_through_2008=len(middle),
                rows_20090105=len(first),
                rows_20090106=len(following),
            ),
        )
        for year in ("2018", "2021", "2024", "2026"):
            block = compared.filter(pl.col("trade_date").str.starts_with(year))
            lo, hi = (
                np.abs(
                    block[f"observed_{x}"].to_numpy()
                    - block[f"expected_{x}"].to_numpy()
                )
                for x in ("low", "high")
            )
            basis_gap = np.abs(
                (block["observed_low"] / block["expected_low"])
                / (block["observed_high"] / block["expected_high"])
                - 1
            ).to_numpy()
            summary["by_year"][year] = dict(
                stocks=block["ts_code"].n_unique(),
                rows=len(block),
                within_tolerance=int(block["within_tolerance"].sum()),
                both_within_010=int(
                    ((lo <= 0.1000000001) & (hi <= 0.1000000001)).sum()
                ),
                same_basis_within_1pct=int((basis_gap <= 0.01).sum()),
                median_abs_low_error=float(np.median(lo)),
                median_abs_high_error=float(np.median(hi)),
            )
        summary["within_tolerance"] = int(compared["within_tolerance"].sum())
        summary["status"] = "diagnostic"
        summary["status_label"] = "诊断（不判定）"
        summary["data_status"] = (
            "insufficient_history" if summary["history_incomplete"] else "available"
        )
        summary["note"] = "历史极值判据已作废；极值相符与否不能判定复权基准"
        return summary
    finally:
        if source is not None:
            await source.http.aclose()


def check_etf_samples(folder, fund_root):
    """只读已有库，不重新下载它；净值列快照由真实探针单独保存。"""
    store = Store(fund_root, readonly=True)
    result = {}
    for day in ("20240102", "20261112"):
        frame = pl.read_parquet(folder / f"etf_nav_{day}.parquet")
        shares = store.read("fund_share", day, day).select(
            "ts_code", "trade_date", "fd_share"
        )
        daily = store.read("fund_daily", day, day).select(
            "ts_code", "trade_date", pl.col("close").alias("fund_close")
        )
        joined = frame.join(shares, on=["ts_code", "trade_date"]).join(
            daily, on=["ts_code", "trade_date"], how="left"
        )
        shares = joined.filter(pl.col("fd_share") > 0)
        size = joined.filter(
            (pl.col("fund_close") > 0)
            & (pl.col("total_share") > 0)
            & (pl.col("total_size") > 0)
        )
        ratio = size["total_size"] / size["total_share"] / size["fund_close"]
        nav = frame.filter(
            (pl.col("nav") > 0)
            & (pl.col("total_share") > 0)
            & (pl.col("total_size") > 0)
        )
        nr = nav["total_size"] / nav["total_share"] / nav["nav"]
        result[day] = dict(
            rows=len(frame),
            matched_existing_shares=len(shares),
            share_exact=int((shares["total_share"] == shares["fd_share"]).sum()),
            size_compared=len(size),
            size_ratio_median=ratio.median(),
            size_ratio_min=ratio.min(),
            size_ratio_max=ratio.max(),
            nav_compared=len(nav),
            nav_ratio_median=nr.median(),
            nav_within_1ppm=int(((nr - 1).abs() < 1e-6).sum()),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "tmp/d3")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fund-root", type=Path)
    parser.add_argument(
        "--history-root", type=Path, help="只读日历与停牌证据库；不联网补抓"
    )
    args = parser.parse_args()
    folder = args.samples.resolve()
    if not folder.is_relative_to(ROOT / "tmp"):
        raise ValueError("真实样本只能写在本 worktree 的 tmp/ 下")
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(chips=asyncio.run(collect(folder, args.download, args.history_root)))
    if args.fund_root is not None:
        summary["etf"] = check_etf_samples(folder, args.fund_root)
    output = ROOT / "bench/results/d3_real_check.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
