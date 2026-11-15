"""生成平台 launchd 定义；路径统一解析，生成本身不安装或启动服务。"""

import argparse
import os
import shlex
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from ashare.paths import data_root, dataset_dir

REPO = Path(__file__).resolve().parents[2]


def locations():
    platform = dataset_dir("platform")
    results = (
        platform if os.environ.get("ASHARE_PLATFORM_DIR") else platform / "results"
    )
    return data_root(), results, platform / "logs"


def render(*, port, subnet, python=None):
    root, results, logs = locations()
    template = REPO / "deploy/platform/launchd/com.ashare.platform.plist.in"
    text = template.read_text()
    # 只固化用户明确设置的兼容变量；默认布局每次启动重新读 LOCATION。
    platform_env = (
        "<key>ASHARE_PLATFORM_DIR</key><string>" + escape(str(results)) + "</string>"
        if os.environ.get("ASHARE_PLATFORM_DIR")
        else ""
    )
    text = text.replace("{{PLATFORM_ENV}}", platform_env)
    values = dict(
        PYTHON=shlex.quote(python or sys.executable),
        REPO=str(REPO),
        PORT=str(port),
        SUBNET=subnet,
        DATA_DIR=str(root),
        LOG_DIR=str(logs),
    )
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", escape(value))
    if "{{" in text:
        raise ValueError("模板仍含占位符")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--subnet", default="10.77.0.0/24")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        text = render(port=args.port, subnet=args.subnet)
        locations()[2].mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(locations()[2])


if __name__ == "__main__":
    main()
