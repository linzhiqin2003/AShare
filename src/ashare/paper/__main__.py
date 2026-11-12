"""python -m ashare.paper：启动、状态、按日回放与十分钟冒烟。"""

import argparse
import json
import logging
from datetime import time, timedelta
from pathlib import Path

from paperbroker import Broker, replay
from paperbroker.journal import canonical, durable_write

from .runner import LiveRunner, load_targets, report
from .schedule import Schedule
from .source import LiveSource, ReplaySource, SystemClock, TushareClient
from .storage import ParquetStore, disk_usage


def read_state(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    from .profiles import from_native

    cash, config = from_native(manifest["config"])
    with Broker(
        manifest["symbols"],
        cash,
        config,
        journal=directory,
        quote_store=ParquetStore(directory / "quotes"),
        read_only=True,
    ) as broker:
        return report(broker, broker.history) | {"disk_bytes": disk_usage(directory)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A 股模拟盘（P1/P2）")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--directory", type=Path, required=True)
        p.add_argument(
            "--symbols", default="", help="逗号分隔，或由 --full-market 拉取全市场"
        )
        p.add_argument("--targets", type=Path)
        p.add_argument("--full-market", action="store_true")
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="在内存中采集和撮合，不创建或修改账户、日志、录制及日报",
        )
        p.add_argument(
            "--duration", type=float, default=600 if name == "smoke" else None
        )
        p.add_argument("--cash", type=float, default=1_000_000)
        p.add_argument(
            "--settlement-deadline", type=time.fromisoformat, default=time(15, 3)
        )
        p.add_argument("--snapshot-retention", type=int, default=3)
        p.add_argument("--quote-retention-days", type=int, default=20)
        p.add_argument(
            "--lock",
            type=Path,
            default=Path(__file__).resolve().parents[3] / ".cache/paper-tushare.lock",
        )
    p = sub.add_parser("status")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("gc", help="保留事件日志，清理旧快照及已受快照保护的旧行情")
    p.add_argument("directory", type=Path)
    p.add_argument("--snapshot-retention", type=int, default=3)
    p.add_argument("--quote-retention-days", type=int, default=20)
    p = sub.add_parser("replay")
    p.add_argument("directory", type=Path)
    p.add_argument("--day", required=True)
    p.add_argument(
        "--script",
        type=Path,
        required=True,
        help="含 symbols、initial_cash、events 的 JSON",
    )
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("import-raw", help="只读原始 gzip 响应并生成回放 Parquet")
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--day", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "import-raw":
        from .importer import import_raw

        print(
            json.dumps(
                import_raw(args.directory, args.output, args.day), ensure_ascii=False
            )
        )
        return
    if args.command == "gc":
        from .profiles import from_native

        manifest = json.loads((args.directory / "manifest.json").read_bytes())
        cash, config = from_native(manifest["config"])
        with Broker(
            manifest["symbols"],
            cash,
            config,
            journal=args.directory,
            quote_store=ParquetStore(args.directory / "quotes"),
            snapshot_retention=args.snapshot_retention,
            quote_retention_days=args.quote_retention_days,
        ) as broker:
            result = broker.journal.gc()
        print(json.dumps(result | {"disk_bytes": disk_usage(args.directory)}))
        return
    if args.command == "status":
        print(json.dumps(read_state(args.directory), ensure_ascii=False))
        return
    if args.command == "replay":
        script = json.loads(args.script.read_text())
        from .profiles import configuration

        broker = Broker(
            script["symbols"],
            script.get("initial_cash", 1_000_000),
            configuration(script["symbols"]),
        )
        source = ReplaySource(args.directory, args.day)

        def recordings():
            while (batch := source.poll()) is not None:
                yield batch.received_at, batch.quotes

        events = replay(broker, recordings(), script["events"])
        durable_write(args.output, canonical(report(broker, events)))
        print(json.dumps(broker.account()))
        return
    clock = SystemClock()
    symbols = [s for s in args.symbols.split(",") if s]
    if not symbols and not args.full_market:
        parser.error("需要 --symbols 或 --full-market")
    with TushareClient(args.lock, clock=clock) as client:
        calendar = client.call(
            "trade_cal",
            dict(
                exchange="SSE",
                start_date=(clock.now() - timedelta(days=10)).strftime("%Y%m%d"),
                end_date=(clock.now() + timedelta(days=40)).strftime("%Y%m%d"),
            ),
        )
        schedule = Schedule(calendar)
        if args.command == "smoke" and schedule.phase(clock.now()) != "continuous":
            parser.error("冒烟脚本需要在连续竞价时段运行")
        if args.full_market:
            records = []
            for pattern in ("6*.SH", "3*.SZ,0*.SZ", "9*.BJ,8*.BJ,4*.BJ"):
                records.extend(client.call("rt_k", {"ts_code": pattern}, "ts_code"))
            symbols = sorted({r["ts_code"] for r in records})
        from .profiles import configuration, from_native, instrument

        config = configuration(symbols)
        if not args.dry_run and (args.directory / "manifest.json").exists():
            args.cash, config = from_native(
                json.loads((args.directory / "manifest.json").read_bytes())["config"]
            )
            config.instruments.update(
                {s: instrument(s) for s in symbols if s not in config.instruments}
            )
        store = None if args.dry_run else ParquetStore(args.directory / "quotes")
        with Broker(
            symbols,
            args.cash,
            config,
            journal=None if args.dry_run else args.directory,
            quote_store=store,
            snapshot_retention=args.snapshot_retention,
            quote_retention_days=args.quote_retention_days,
        ) as broker:
            source = LiveSource(
                client, list(broker.symbols), full_market=args.full_market
            )
            targets = load_targets(args.targets) if args.targets else None
            if args.command == "smoke" and targets is None:
                targets = dict(
                    version=1, at="09:30:00", weights={s: 0.02 for s in symbols[:3]}
                )
            runner = LiveRunner(
                broker,
                source,
                schedule,
                clock,
                store,
                targets,
                None if args.dry_run else args.directory / "reports",
                settlement_deadline=args.settlement_deadline,
            )
            runner.run(args.duration)
            if broker.journal:
                broker.checkpoint()
            result = report(broker, runner.events)
            if not args.dry_run:
                durable_write(args.directory / "latest.json", canonical(result))
            print(
                json.dumps(
                    dict(account=result["account"], fills=result["fills"]),
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
