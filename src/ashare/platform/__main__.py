"""python -m ashare.platform：只监听本机回环地址。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AShare 本地研究平台")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口须在 1–65535 之间")
    if not (Path(__file__).parent / "static/index.html").exists():
        print("前端尚未构建，请在项目根目录运行 make web 后重试。")
        return
    try:
        import uvicorn

        from .app import create_app
    except ImportError:
        print("请先运行 uv sync --extra platform 安装平台可选依赖。")
        return
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
