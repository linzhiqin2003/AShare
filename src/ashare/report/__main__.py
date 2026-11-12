"""命令行：在终端里看存下来的报告数据，或把它写成 HTML 页面。

    python -m ashare.report show report.json                       # 默认摘要
    python -m ashare.report show report.json --section yearly --section warnings
    python -m ashare.report show report.json --section all --format markdown
    python -m ashare.report show report.json --from 2020-01-01 --to 2023-12-31
    python -m ashare.report show report.json --format json --lang en
    python -m ashare.report html report.json -o report.html

报告数据由 `Tearsheet.save_json(path)` 写出。
"""

from __future__ import annotations

import argparse
import sys

from . import ReportError, Tearsheet
from .sections import SECTIONS
from .text import FORMATS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ashare.report", description="查看回测报告数据"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show", help="在终端里打印报告")
    sh.add_argument("path", help="Tearsheet.save_json 写出的 JSON")
    sh.add_argument(
        "--section",
        "-s",
        action="append",
        help=f"小节，可重复或用逗号分隔：{', '.join(SECTIONS)}, all（默认 summary）",
    )
    sh.add_argument("--format", "-f", default="text", choices=FORMATS)
    sh.add_argument("--lang", default="zh", choices=("zh", "en"))
    sh.add_argument("--from", dest="start", help="区间起点 YYYY-MM-DD")
    sh.add_argument("--to", dest="end", help="区间终点 YYYY-MM-DD")
    ht = sub.add_parser("html", help="写成 HTML 页面")
    ht.add_argument("path")
    ht.add_argument("-o", "--output", required=True)
    a = p.parse_args(argv)
    try:
        ts = Tearsheet.load(a.path)
        if a.cmd == "html":
            out = ts.save(a.output)
            print(out)
            return 0
        secs = None
        if a.section:
            secs = [x for s in a.section for x in s.split(",") if x]
        sys.stdout.write(ts.text(secs, a.format, lang=a.lang, start=a.start, end=a.end))
        return 0
    except (ReportError, ValueError, OSError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
