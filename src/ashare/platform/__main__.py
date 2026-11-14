"""python -m ashare.platform：默认只监听本机回环地址。

`--host <网卡地址>` 额外监听一块网卡（例如 VPN 地址），此时必须用环境变量
`ASHARE_PLATFORM_TOKEN` 设访问口令：非本机的访问先登录，本机访问不受影响。

`--host-subnet <网段>`（常驻服务用）：启动时找本机在该网段上的地址并监听；之后每隔几秒
复查，地址出现、消失或变化时以退出码 75 退出，由 launchd / systemd 重新拉起后按新地址监听。
这样开机时 VPN 还没连上、或 VPN 重连换了地址，都不用人去重启。
"""

import argparse
import ipaddress
import os
import socket
import sys
import threading
import time
from pathlib import Path

from .access import MIN_TOKEN, Access, is_loopback

TOKEN_ENV = "ASHARE_PLATFORM_TOKEN"
RESTART_CODE = 75  # EX_TEMPFAIL：网卡地址变了，交给服务管理器重启
WATCH_SECONDS = 5.0


def subnet_address(subnet, interfaces=None):
    """本机在 ``subnet`` 网段上的 IPv4 地址（按地址排序取第一个），没有时返回 None。

    ``interfaces`` 形如 ``{网卡名: [地址字符串, ...]}``，不给时用 psutil 读本机网卡。
    """
    network = ipaddress.ip_network(subnet, strict=False)
    if network.version != 4:
        raise ValueError("--host-subnet 须是 IPv4 网段，例如 10.77.0.0/24")
    if interfaces is None:
        import psutil

        interfaces = {
            name: [a.address for a in addrs if a.family == socket.AF_INET]
            for name, addrs in psutil.net_if_addrs().items()
        }
    found = sorted(
        (
            ipaddress.ip_address(a)
            for addrs in interfaces.values()
            for a in addrs
            if a and ipaddress.ip_address(a) in network
        ),  # fmt: skip
    )
    return str(found[0]) if found else None


def watch_subnet(subnet, bound, *, read=subnet_address, exit=os._exit,
                 interval=WATCH_SECONDS, sleep=time.sleep):  # fmt: skip
    """循环复查网段地址；与启动时监听的 ``bound`` 不同就退出进程（读网卡出错时跳过这一轮）。"""
    while True:
        sleep(interval)
        try:
            current = read(subnet)
        except Exception as error:  # noqa: BLE001  读网卡偶发失败不该拖垮服务
            print(f"读取网卡地址失败，跳过本轮：{type(error).__name__}", flush=True)
            continue
        if current != bound:
            print(f"网段 {subnet} 上的地址由 {bound} 变为 {current}，退出等待重启",
                  flush=True)  # fmt: skip
            sys.stdout.flush()
            exit(RESTART_CODE)
            return


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
        "--host-subnet",
        help="常驻服务用：监听本机在该网段（如 10.77.0.0/24）上的地址，地址变化时退出等待重启",
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
    if args.host and args.host_subnet:
        parser.error("--host 与 --host-subnet 只能用一个")
    host = args.host
    if args.host_subnet:
        try:
            host = subnet_address(args.host_subnet)
        except ValueError as error:
            parser.error(str(error))
        if host is None:
            print(f"网段 {args.host_subnet} 上暂无本机地址，先只监听本机", flush=True)
        if not os.environ.get(TOKEN_ENV):
            parser.error(
                f"--host-subnet 需要访问口令：export {TOKEN_ENV}=<至少 {MIN_TOKEN} 个字符>"
            )
    try:
        binds, access = resolve_access(host, args.allow_host, os.environ.get(TOKEN_ENV))
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
        print(f"AShare 研究平台：http://{address}:{args.port}", flush=True)
    if args.host_subnet:
        threading.Thread(target=watch_subnet, args=(args.host_subnet, host),
                         name="subnet-watch", daemon=True).start()  # fmt: skip
    config = uvicorn.Config(create_app(access=access), access_log=False)
    uvicorn.Server(config).run(sockets=sockets)


if __name__ == "__main__":
    main()
