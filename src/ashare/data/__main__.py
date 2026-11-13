"""数据层命令行；默认只采集最近两个月。"""

import argparse
import asyncio
import json
import logging
from datetime import date, timedelta

from .catalog import DATASETS, FUTURES_DATASETS
from .maintenance import collect_garbage, disk_usage
from .panel import build
from .sources import Tushare
from .store import Store
from .sync import sync
from .verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--market", default="cn")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("sync")
    collect.add_argument(
        "--start", default=(date.today() - timedelta(days=61)).isoformat()
    )
    collect.add_argument(
        "--end", default=(date.today() - timedelta(days=1)).isoformat()
    )
    collect.add_argument("--datasets", nargs="+", choices=DATASETS)
    collect.add_argument("--force", action="store_true")
    collect.add_argument("--no-refresh", action="store_true")
    collect.add_argument("--workers", type=int, default=24)
    collect.add_argument("--rate", type=float, default=300)
    collect.add_argument(
        "--budget",
        type=int,
        default=3000,
        help="本次运行的请求预算，含重试与翻页",
    )
    futures = commands.add_parser("sync-futures", help="境内期货原始库（全部交易所）")
    futures.add_argument("--start", default="1995-04-17")
    futures.add_argument(
        "--end", default=(date.today() - timedelta(days=1)).isoformat()
    )
    futures.add_argument("--datasets", nargs="+", choices=FUTURES_DATASETS)
    futures.add_argument("--force", action="store_true")
    futures.add_argument("--no-refresh", action="store_true")
    futures.add_argument("--workers", type=int, default=12)
    futures.add_argument("--rate", type=float, default=400)
    futures.add_argument("--budget", type=int, default=60000)
    commands.add_parser("build-futures", help="由期货原始库构建交易日×合约面板")
    commands.add_parser("verify-futures", help="期货原始库与面板的独立校验")
    # 港股、美股（第 2 批）：配合 --market hk / --market us，数据根 <root>/hk、<root>/us。
    overseas = commands.add_parser(
        "sync-overseas", help="港股或美股原始库（--market hk|us）"
    )
    overseas.add_argument(
        "--start", default=(date.today() - timedelta(days=3 * 365)).isoformat()
    )
    overseas.add_argument(
        "--end", default=(date.today() - timedelta(days=1)).isoformat()
    )
    overseas.add_argument("--datasets", nargs="+")
    overseas.add_argument("--force", action="store_true")
    overseas.add_argument("--no-refresh", action="store_true")
    overseas.add_argument("--workers", type=int, default=8)
    overseas.add_argument("--rate", type=float, default=200)
    overseas.add_argument("--budget", type=int, default=20000)
    sample = commands.add_parser(
        "select-sample-overseas", help="按当前市值分层抽 2010 年起的历史样本"
    )
    sample.add_argument("--asof", required=True)
    sample.add_argument("--rate", type=float, default=60)
    overseas_build = commands.add_parser(
        "build-overseas", help="港股或美股「交易日 × 证券」面板"
    )
    overseas_build.add_argument("--universe", choices=("all", "sample"), default="all")
    overseas_build.add_argument("--start")
    overseas_build.add_argument("--end")
    overseas_check = commands.add_parser(
        "verify-overseas", help="港股或美股原始库与面板的独立校验"
    )
    overseas_check.add_argument("--universe", choices=("all", "sample"), default="all")
    panel = commands.add_parser("build-panel")
    panel.add_argument("--start")
    panel.add_argument("--end")
    panel.add_argument("--incremental", action="store_true")
    panel.add_argument("--keep", type=int, default=1)
    panel.add_argument("--strict", action="store_true", help="缺指数月份时拒绝构建")
    panel.add_argument("--action-tolerance", type=float, default=0.005)
    panel.add_argument(
        "--action-k-min",
        type=float,
        default=0.85,
        help="差异化分红摊薄比例 k 的下限，进入版本哈希",
    )
    panel.add_argument(
        "--allow-unreconciled",
        action="store_true",
        help="显式允许公司行为对账失败，默认拒绝",
    )
    check = commands.add_parser("verify")
    check.add_argument("--baostock", action="store_true")
    commands.add_parser("status")
    gc = commands.add_parser("gc")
    gc.add_argument("--keep", type=int, default=1)
    confirm = commands.add_parser("confirm-partition")
    confirm.add_argument("key")
    confirm.add_argument("--reason", required=True)
    export = commands.add_parser("export-manifest")
    export.add_argument("path", help="导出兼容旧清单，覆盖前请自行保留迁移前文件")
    args = parser.parse_args()
    # 港股、美股只走 *-overseas 子命令（另可用 status、confirm-partition）：A 股和期货的子命令
    # 不认识港美股数据集，在 <root>/hk 上跑会把 A 股数据写进港股根。
    if args.market != "cn" and args.command not in (
        "sync-overseas",
        "select-sample-overseas",
        "build-overseas",
        "verify-overseas",
        "status",
        "confirm-partition",
    ):
        parser.error(
            f"{args.command} 只用于 A 股（--market cn）；港股、美股请用 sync-overseas、"
            "select-sample-overseas、build-overseas、verify-overseas"
        )
    store = Store(args.root, args.market)
    logging.basicConfig(level=logging.INFO)
    if args.command == "sync":

        async def run():
            async with Tushare(
                rate=args.rate,
                concurrency=args.workers,
                state_path=store.root / "account.json",
                budget=args.budget,
            ) as source:
                result = await sync(
                    source,
                    store,
                    args.start,
                    args.end,
                    args.datasets,
                    force=args.force,
                    refresh=not args.no_refresh,
                    workers=args.workers,
                )
                return dict(
                    partitions=len(result),
                    rows=sum(e["rows"] for e in result),
                    calls=source.calls,
                )

        result = asyncio.run(run())
    elif args.command == "sync-futures":
        from .futures_sync import sync_futures

        async def run_futures():
            async with Tushare(
                rate=args.rate,
                concurrency=args.workers,
                state_path=store.root / "account.json",
                budget=args.budget,
            ) as source:
                outcome = await sync_futures(
                    source,
                    store,
                    args.start,
                    args.end,
                    args.datasets,
                    force=args.force,
                    refresh=not args.no_refresh,
                    workers=args.workers,
                )
                return dict(stats=outcome["stats"], calls=source.calls)

        result = asyncio.run(run_futures())
    elif args.command == "build-futures":
        from .futures_panel import build_futures

        meta = build_futures(store)
        result = {k: meta[k] for k in ("version", "shape", "elapsed_seconds")}
    elif args.command == "verify-futures":
        from .futures_verify import verify_futures

        report = verify_futures(store)
        result = {
            k: {p: v[p] for p in ("verdict", "checked", "failed", "explained")}
            for k, v in report["checks"].items()
        }
        result["verdict"] = report["verdict"]
        result["cause_unknown"] = report["known_exceptions"]["cause_unknown"]
        result["report"] = str(store.root / "reports/futures_verify.md")
        if report["verdict"] == "fail":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    elif args.command in (
        "sync-overseas",
        "select-sample-overseas",
        "build-overseas",
        "verify-overseas",
    ):
        from .overseas_cli import run as run_overseas

        result = run_overseas(args, store)
    elif args.command == "build-panel":
        meta = build(
            store,
            args.start,
            args.end,
            incremental=args.incremental,
            keep=args.keep,
            strict=args.strict,
            action_tolerance=args.action_tolerance,
            action_k_min=args.action_k_min,
            allow_unreconciled=args.allow_unreconciled,
        )
        result = {
            k: meta[k] for k in ("version", "shape", "elapsed_seconds", "reused_rows")
        }
    elif args.command == "verify":
        if args.baostock:
            from .crosscheck import crosscheck

            asyncio.run(crosscheck(store))
        report = verify(store)
        result = {
            k: {p: v[p] for p in ("checked", "failed", "pass_rate")}
            for k, v in report["checks"].items()
        }
        result["report"] = str(store.root / "reports/verify.md")
    elif args.command == "gc":
        result = collect_garbage(store, args.keep)
    elif args.command == "confirm-partition":
        result = store.confirm(args.key, args.reason)
    elif args.command == "export-manifest":
        from pathlib import Path

        store.export_manifest(Path(args.path))
        result = {"exported": args.path}
    else:
        result = {
            name: dict(
                partitions=len(entries),
                rows=sum(e.get("rows", 0) for e in entries.values()),
                failed=sum(e["status"] != "complete" for e in entries.values()),
            )
            for name in DATASETS
            if (entries := store.entries(name))
        }
        # 期货单独一节，股票部分与没有期货时逐位相同。
        futures_status = {
            name: dict(
                partitions=len(entries),
                rows=sum(e.get("rows", 0) for e in entries.values()),
                failed=sum(e["status"] != "complete" for e in entries.values()),
            )
            for name in FUTURES_DATASETS
            if (entries := store.entries(name))
        }
        if futures_status:
            result["futures"] = futures_status
        # 港股、美股的数据集只出现在 <root>/hk、<root>/us 的清单里，A 股清单里永远没有。
        from .catalog import OVERSEAS_DATASETS

        overseas_status = {
            name: dict(
                partitions=len(entries),
                rows=sum(e.get("rows", 0) for e in entries.values()),
                failed=sum(e["status"] != "complete" for e in entries.values()),
            )
            for name in OVERSEAS_DATASETS
            if (entries := store.entries(name))
        }
        if overseas_status:
            result["overseas"] = overseas_status
    if args.command in ("status", "gc"):
        result["disk_usage"] = disk_usage(store.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
