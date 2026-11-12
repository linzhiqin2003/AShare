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
            k: {p: v[p] for p in ("verdict", "checked", "failed")}
            for k, v in report["checks"].items()
        }
        result["report"] = str(store.root / "reports/futures_verify.md")
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
    if args.command in ("status", "gc"):
        result["disk_usage"] = disk_usage(store.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
