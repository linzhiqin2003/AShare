"""港股、美股子命令的实现（``python -m ashare.data --market hk|us ...``）。"""

import asyncio
import json

from .sources import Tushare


def run(args, store):
    if store.market not in ("hk", "us"):
        raise SystemExit("港美股子命令要加 --market hk 或 --market us")
    if args.command == "sync-overseas":
        from .overseas_sync import sync_overseas

        def progress(name, done, total):
            if done % 100 == 0 or done == total:
                print(
                    json.dumps(dict(dataset=name, done=done, total=total)), flush=True
                )

        async def go():
            async with Tushare(
                rate=args.rate,
                concurrency=args.workers,
                state_path=store.root / "account.json",
                budget=args.budget,
            ) as source:
                outcome = await sync_overseas(
                    source,
                    store,
                    args.start,
                    args.end,
                    args.datasets,
                    force=args.force,
                    refresh=not args.no_refresh,
                    workers=args.workers,
                    progress=progress,
                )
                return dict(
                    stats=outcome["stats"],
                    snapshot=outcome["snapshot"],
                    calls=source.calls,
                )

        return asyncio.run(go())
    if args.command == "select-sample-overseas":
        from .overseas_sync import select_sample

        async def pick():
            async with Tushare(
                rate=args.rate, state_path=store.root / "account.json", budget=50
            ) as source:
                return await select_sample(source, store, args.asof)

        value = asyncio.run(pick())
        return {k: value[k] for k in ("market", "asof", "method")} | dict(
            symbols=len(value["symbols"])
        )
    if args.command == "build-overseas":
        from .overseas_panel import build_overseas

        meta = build_overseas(store, args.universe, args.start, args.end)
        return {k: meta[k] for k in ("version", "shape", "elapsed_seconds")}
    if args.command == "verify-overseas":
        from .overseas_verify import verify_overseas

        report = verify_overseas(store, universe=args.universe)
        result = {
            k: {p: v[p] for p in ("verdict", "checked", "failed", "explained")}
            for k, v in report["checks"].items()
        }
        result["verdict"] = report["verdict"]
        result["report"] = str(store.root / "reports/overseas_verify.md")
        if report["verdict"] == "fail":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        return result
    raise AssertionError(args.command)
