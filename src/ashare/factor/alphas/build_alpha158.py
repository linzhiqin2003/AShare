"""由 ``docs/factors/sources/alpha158.jsonl`` 生成 ``alpha158_formulas.py``（确定性，不手写）。

    python -m ashare.factor.alphas.build_alpha158 [--check]

``--check`` 只比较不写入：生成结果与仓库里的文件不同时退出码为 1。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/factors/sources/alpha158.jsonl"
TARGET = Path(__file__).with_name("alpha158_formulas.py")


def render(source=SOURCE):
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line]
    if [r["n"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("alpha158.jsonl 的编号必须从 1 连续")
    lines = [
        '"""Alpha158 原文表达式（由 build_alpha158 从 docs/factors/sources/alpha158.jsonl 生成，勿手改）。"""',
        "",
        "FORMULAS = (",
    ]
    for r in rows:
        lines.append(f"    ({r['n']}, {r['name']!r}, {r['formula']!r}),")
    lines.append(")")
    text = "\n".join(lines) + "\n"
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(TARGET), "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return formatted.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        same = TARGET.exists() and TARGET.read_text() == text
        print("一致" if same else "不一致：请重新生成")
        return 0 if same else 1
    TARGET.write_text(text)
    print(f"写入 {TARGET.name}：{text.count(chr(10))} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
