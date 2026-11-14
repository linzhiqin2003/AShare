"""python -m ashare.platform：默认只监听本机回环地址。

`--host <网卡地址>` 额外监听一块网卡（例如 VPN 地址），此时必须用环境变量
`ASHARE_PLATFORM_TOKEN` 设访问口令：非本机的访问先登录，本机访问不受影响。
"""

import argparse
import ipaddress
import os
import socket
from pathlib import Path

from .access import MIN_TOKEN, Access, is_loopback

TOKEN_ENV = "ASHARE_PLATFORM_TOKEN"


def resolve_access(host, allow_hosts, token):
    """返回 (要监听的地址列表, Access)；参数不合法时抛 ValueError。"""
    binds = ["127.0.0.1"]
    if host and not is_loopback(host):
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("--host 须是一个具体的 IPv4 网卡地址") from None
        if address.version != 4 or address.is_unspecified:
            raise ValueError("--host 须是一个具体的 IPv4 网卡地址，不支持 0.0.0.0")
        binds.append(str(address))
    remote = len(binds) > 1
    if remote and not token:
        raise ValueError(
            f"监听其他网卡时必须设置访问口令：export {TOKEN_ENV}=<至少 {MIN_TOKEN} 个字符>"
        )
    hosts = set(binds[1:]) | {h.strip() for h in allow_hosts if h.strip()}
    return binds, Access(hosts=frozenset(hosts), token=token or None)


def main():
    parser = argparse.ArgumentParser(description="AShare 本地研究平台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host", help="除本机外再监听的网卡地址（如 VPN 地址），需设访问口令"
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="额外允许的访问主机名（经域名访问时用），可重复",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口须在 1–65535 之间")
    try:
        binds, access = resolve_access(
            args.host, args.allow_host, os.environ.get(TOKEN_ENV)
        )
    except ValueError as error:
        parser.error(str(error))
    if not (Path(__file__).parent / "static/index.html").exists():
        print("前端尚未构建，请在项目根目录运行 make web 后重试。")
        return
    try:
        import uvicorn

        from .app import create_app
    except ImportError:
        print("请先运行 uv sync --extra platform 安装平台可选依赖。")
        return
    sockets = []
    for address in binds:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, args.port))
        sockets.append(sock)
    for address in binds:
        print(f"AShare 研究平台：http://{address}:{args.port}")
    config = uvicorn.Config(create_app(access=access), access_log=False)
    uvicorn.Server(config).run(sockets=sockets)


if __name__ == "__main__":
    main()
