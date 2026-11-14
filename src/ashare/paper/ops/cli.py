"""命令行：``python -m ashare.paper daily|account|install-schedule …``，返回退出码（见 ``Exit``）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context import Context, Exit, StageResult, configure_logging


def _common(p):
    p.add_argument("--data-dir", help="数据根，缺省 $ASHARE_DATA_DIR 或仓库下 data/")
    p.add_argument(
        "--paper-dir", help="账户根，缺省 $ASHARE_PAPER_DIR 或 <数据根>/paper"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m ashare.paper", description=__doc__)
    sub = root.add_subparsers(dest="group", required=True)

    daily = sub.add_parser("daily", help="每日流水线的各个阶段")
    stages = daily.add_subparsers(dest="stage", required=True)
    p = stages.add_parser("after-close", help="增量同步、重建面板、数据校验")
    _common(p)
    p.add_argument(
        "--accept-verify", action="store_true", help="人工核对后接受校验计数"
    )
    p.add_argument("--force", action="store_true", help="当天已完成也重做")
    p = stages.add_parser("signal", help="为启用的账户生成下一开市日的目标")
    _common(p)
    p.add_argument("--account", action="append", help="只处理这些账户（可重复）")
    p = stages.add_parser("nightly", help="after-close，成功后接着 signal")
    _common(p)
    p.add_argument("--accept-verify", action="store_true")
    p = stages.add_parser("run", help="开盘前启动运行器，--until 后退出")
    _common(p)
    p.add_argument("--account", action="append")
    p.add_argument("--until", default="17:00", help="退出时刻（北京时间），默认 17:00")
    p.add_argument(
        "--dry-run", action="store_true", help="在账本副本上演练，不写账户目录"
    )
    p.add_argument("--poll-seconds", type=float, default=6.0)
    p = stages.add_parser("status", help="各阶段最近一次的结果")
    _common(p)

    account = sub.add_parser("account", help="模拟账户与策略绑定")
    actions = account.add_subparsers(dest="action", required=True)
    p = actions.add_parser("create", help="从表达式 + 设置，或从回测结果建账户")
    _common(p)
    p.add_argument("name")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--expression")
    source.add_argument("--expression-file", type=Path)
    source.add_argument("--backtest", help="回测结果 id（读回测页结果库）")
    source.add_argument(
        "--backtest-file", type=Path, help="符合「回测记录」约定的 JSON"
    )
    p.add_argument("--settings", default="{}", help="回测页设置的 JSON（白名单见文档）")
    p.add_argument("--capital", type=float, help="初始资金（万元），缺省用设置里的")
    p.add_argument("--title", help="展示名，缺省同账户名或回测名")
    p.add_argument("--disabled", action="store_true", help="建好后先停用")
    for name in ("enable", "disable", "show"):
        p = actions.add_parser(name)
        _common(p)
        p.add_argument("name")
    p = actions.add_parser("list")
    _common(p)

    p = sub.add_parser("install-schedule", help="渲染 launchd / systemd 调度文件")
    _common(p)
    kind = p.add_mutually_exclusive_group(required=True)
    kind.add_argument("--launchd", action="store_true")
    kind.add_argument("--systemd", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", default=True, help="默认：只打印"
    )
    mode.add_argument("--apply", action="store_true", help="写文件（不加载任务）")
    p.add_argument(
        "--target",
        type=Path,
        help="写到哪个目录；launchd 缺省为暂存目录，systemd 缺省为用户单元目录",
    )
    p.add_argument("--python", help="解释器路径，缺省为当前解释器")
    p.add_argument(
        "--env-file", type=Path, help="systemd 的 EnvironmentFile（放 TUSHARE_TOKEN）"
    )
    return root


def _print(result: StageResult) -> int:
    print(
        json.dumps(
            dict(
                stage=result.stage,
                status=result.code.name.lower(),
                message=result.message,
                day=result.day,
                details=result.details,
            ),
            ensure_ascii=False,
        )
    )
    return int(result.code)


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    ctx = Context.create(
        getattr(args, "data_dir", None), getattr(args, "paper_dir", None)
    )
    if args.group == "daily":
        configure_logging(ctx)
        return _daily(ctx, args)
    if args.group == "account":
        return _account(ctx, args)
    from .install import install

    install(
        "launchd" if args.launchd else "systemd",
        apply=args.apply,
        target=args.target,
        python=args.python,
        data_dir=ctx.data,
        paper_dir=ctx.paper,
        env_file=args.env_file,
    )
    return 0


def _daily(ctx: Context, args) -> int:
    if args.stage == "status":
        from .nightly import status

        print(json.dumps(status(ctx), ensure_ascii=False, indent=2))
        return 0
    if args.stage == "after-close":
        from .after_close import after_close

        return _print(
            after_close(ctx, accept_verify=args.accept_verify, force=args.force)
        )
    if args.stage == "signal":
        from .signal import signal

        return _print(signal(ctx, args.account))
    if args.stage == "nightly":
        from .nightly import nightly

        codes = [_print(r) for r in nightly(ctx, accept_verify=args.accept_verify)]
        return codes[-1]
    from .run import parse_until, run

    try:
        until = parse_until(args.until)
    except ValueError as error:
        print(
            json.dumps(
                dict(stage="run", status="usage", message=str(error)),
                ensure_ascii=False,
            )
        )
        return int(Exit.USAGE)
    return _print(
        run(
            ctx,
            args.account,
            until=until,
            dry_run=args.dry_run,
            poll_seconds=args.poll_seconds,
        )
    )


def _account(ctx: Context, args) -> int:
    from . import accounts

    try:
        if args.action == "create":
            settings = json.loads(args.settings)
            if args.backtest or args.backtest_file:
                record = (
                    json.loads(args.backtest_file.read_text())
                    if args.backtest_file
                    else accounts.BacktestRecords().get(args.backtest)
                )
                if record is None:
                    raise accounts.AccountError("没有找到这个回测", 404)
                if settings:
                    raise accounts.AccountError("从回测建账户时设置沿用回测，不能另给")
                strategy = accounts.from_backtest(
                    ctx,
                    args.name,
                    record,
                    capital=args.capital,
                    display_name=args.title,
                    enabled=not args.disabled,
                )
            else:
                source = (
                    args.expression_file.read_text()
                    if args.expression_file
                    else args.expression
                )
                strategy = accounts.create(
                    ctx,
                    args.name,
                    source=source,
                    settings=settings,
                    display_name=args.title,
                    capital=args.capital,
                    enabled=not args.disabled,
                )
            print(json.dumps(strategy.to_dict(), ensure_ascii=False, indent=2))
        elif args.action in ("enable", "disable"):
            strategy = accounts.set_enabled(ctx, args.name, args.action == "enable")
            print(
                json.dumps(
                    dict(name=args.name, enabled=strategy.enabled), ensure_ascii=False
                )
            )
        elif args.action == "show":
            directory = accounts.account_dir(ctx, args.name)
            if not (directory / "strategy.json").is_file():
                raise accounts.AccountError("这个账户没有策略定义", 404)
            print(
                json.dumps(
                    dict(
                        account=args.name,
                        strategy=json.loads((directory / "strategy.json").read_bytes()),
                        plan=accounts.describe(directory)["plan"],
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            items = [
                dict(account=name, **accounts.describe(directory))
                for name, directory in accounts.bound(ctx)
            ]
            print(json.dumps(items, ensure_ascii=False, indent=2))
    except (accounts.AccountError, json.JSONDecodeError) as error:
        print(json.dumps(dict(message=str(error)), ensure_ascii=False), file=sys.stderr)
        return int(Exit.USAGE)
    return 0
