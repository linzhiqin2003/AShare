"""python -m ashare.data.crypto --root <root> sync|build|verify。"""

import argparse
import json
from pathlib import Path

from ..store import atomic_json
from .build import build
from .sync import sync
from .verify import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("sync")
    download.add_argument("--refresh", action="store_true")
    download.add_argument(
        "--skip-announcements",
        action="store_true",
        help="显式离线/已人工导入公告时使用；结果会记录跳过",
    )
    make = sub.add_parser("build")
    make.add_argument("--allow-missing", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("--version")
    args = parser.parse_args(argv)
    if args.command == "sync":
        result = sync(
            args.root, refresh=args.refresh, announcements=not args.skip_announcements
        )
    elif args.command == "build":
        ds = build(args.root, allow_missing=args.allow_missing)
        result = ds.meta
    else:
        result = verify(args.root, args.version)
        atomic_json(args.root / "verification" / f"{result['version']}.json", result)
        # 完整逐格诊断落文件，命令行只显示摘要，避免海量输出。
        result = {**result, "diagnostics": len(result["diagnostics"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("verdict") == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
