"""python -m ashare.data.spec check|seal|info|export。"""

import argparse
import json

from . import ValidationError, build, check, open, seal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "seal", "info"):
        commands.add_parser(command).add_argument("package")
    export = commands.add_parser("export")
    export.add_argument("family", choices=["cn", "crypto", "crypto_minute"])
    export.add_argument("package")
    export.add_argument("--root")
    export.add_argument("--start")
    export.add_argument("--end")
    export.add_argument("--symbols", nargs="+")
    export.add_argument("--market", choices=["spot", "usdm"])
    export.add_argument("--kind", choices=["klines", "mark", "index"], default="klines")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = check(args.package)
        elif args.command == "seal":
            result = seal(args.package)
        elif args.command == "info":
            result = open(args.package).info()
        else:
            from .export_cn import CNAdapter
            from .export_crypto import CryptoAdapter, MinuteAdapter

            kwargs = dict(
                root=args.root, start=args.start, end=args.end, symbols=args.symbols
            )
            if args.family == "crypto_minute":
                kwargs.update(market=args.market, kind=args.kind)
            adapter = {
                "cn": CNAdapter,
                "crypto": CryptoAdapter,
                "crypto_minute": MinuteAdapter,
            }[args.family](**kwargs)
            result = build(adapter, args.package)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result.get("exit_code", 0)
    except ValidationError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2))
        return 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(error=str(exc), exit_code=2), ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
