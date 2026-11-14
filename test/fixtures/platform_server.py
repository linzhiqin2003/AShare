"""在子进程里真实启动研究平台（随机空闲端口），用完干净关掉并确认没有遗留进程。

    with PlatformServer(datadir, tmp_path / "platform") as server:
        server.get("/api/meta")

前端已构建（`make web`）时走正式入口 `python -m ashare.platform --port N`；没构建时
`__main__` 只打印一行提示就退出，这时用同样的 `create_app()` + uvicorn 启动，只差静态页。
所有等待都是带上限的轮询，看明确状态；子进程在 `close()` 里回收，失败时整组强杀。
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

STATIC_INDEX = (
    Path(__file__).resolve().parents[2] / "src/ashare/platform/static/index.html"
)
LAUNCHER = (
    "import sys, uvicorn\n"
    "from ashare.platform.app import create_app\n"
    "uvicorn.run(create_app(), host='127.0.0.1', port=int(sys.argv[1]), access_log=False)\n"
)
TERMINAL = ("complete", "failed", "cancelled", "interrupted")


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def process_table():
    """{pid: (ppid, command)}；用 ps，不依赖 psutil。"""
    output = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout
    table = {}
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
    return table


def descendants(pid):
    table = process_table()
    children = {}
    for child, (parent, _) in table.items():
        children.setdefault(parent, []).append(child)
    found, stack = {}, [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found[child] = table[child][1]
                stack.append(child)
    return found


def alive(pid, command=None):
    """pid 还在且（给了命令行时）命令行没变；防 pid 复用误判。"""
    table = process_table()
    if pid not in table:
        return False
    return command is None or table[pid][1] == command


class PlatformServer:
    def __init__(
        self, data_root, platform_root, *, env=None, startup_timeout=60, command=None
    ):
        self.data_root, self.platform_root = Path(data_root), Path(platform_root)
        self.platform_root.mkdir(parents=True, exist_ok=True)
        self.extra_env = dict(env or {})
        self.startup_timeout = startup_timeout
        self.process = None
        self.port = None
        self.log = self.platform_root.parent / f"{self.platform_root.name}-server.log"
        self.official_entry = STATIC_INDEX.exists()
        self.seen_children = {}
        # 只给助手自测用：port → 命令行，替代真实平台。
        self.command = command

    # 生命周期 -------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        env = dict(os.environ)
        env.update(
            ASHARE_DATA_DIR=str(self.data_root),
            ASHARE_PLATFORM_DIR=str(self.platform_root),
            # 共享机器：限制工作进程与算子线程，避免抢光核数。
            ASHARE_PLATFORM_WORKERS="2",
            ASHARE_PLATFORM_VARIANTS="2",
            ASHARE_PLATFORM_OPS_THREADS="2",
            PYTHONUNBUFFERED="1",
        )
        env.pop("TUSHARE_TOKEN", None)
        env.update(self.extra_env)
        last = None
        for _ in range(3):
            self.port = free_port()
            if self.command is not None:
                command = self.command(self.port)
            elif self.official_entry:
                command = [
                    sys.executable,
                    "-m",
                    "ashare.platform",
                    "--port",
                    str(self.port),
                ]
            else:
                command = [sys.executable, "-c", LAUNCHER, str(self.port)]
            with self.log.open("ab") as log:
                self.process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            try:
                self._wait_ready()
                return
            except RuntimeError as error:  # 端口被抢等启动失败：回收后换端口重试
                last = error
                self._kill()
        raise last

    def _wait_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"平台进程提前退出 code={self.process.returncode}\n{self.log_text()}"
                )
            try:
                response = httpx.get(self.url("/api/meta"), timeout=2)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"平台 {self.startup_timeout}s 内未就绪\n{self.log_text()}")

    def snapshot_children(self):
        """记下当前所有子孙进程（工作进程、资源跟踪器等），关闭后逐个确认已退出。"""
        if self.process is not None and self.process.poll() is None:
            self.seen_children.update(descendants(self.process.pid))
        return dict(self.seen_children)

    def close(self, timeout=30):
        """SIGINT 让 uvicorn 走正常关闭（lifespan → Manager.close），超时才强杀。

        返回 dict(returncode=…, leftovers={pid: command})；leftovers 非空说明有进程没跟着退出
        （此时已被强杀，调用方应判失败）。
        """
        if self.process is None:
            return dict(returncode=None, leftovers={})
        self.snapshot_children()
        graceful = True
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                graceful = False
                self._kill()
        deadline = time.monotonic() + 10
        leftovers = dict(self.seen_children)
        while leftovers and time.monotonic() < deadline:
            leftovers = {
                pid: command
                for pid, command in leftovers.items()
                if alive(pid, command)
            }
            if leftovers:
                time.sleep(0.2)
        for pid in leftovers:
            for kill in (os.killpg, os.kill):
                try:
                    kill(pid, signal.SIGKILL)
                    break
                except (ProcessLookupError, PermissionError):
                    continue
        result = dict(
            returncode=self.process.returncode, graceful=graceful, leftovers=leftovers
        )
        self.process = None
        return result

    def _kill(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # HTTP ------------------------------------------------------------------
    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def origin(self):
        return {"Origin": f"http://127.0.0.1:{self.port}"}

    def get(self, path, **kwargs):
        return httpx.get(self.url(path), timeout=kwargs.pop("timeout", 30), **kwargs)

    def post(self, path, json_body=None, **kwargs):
        return httpx.post(
            self.url(path),
            json=json_body,
            headers=self.origin | kwargs.pop("headers", {}),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )

    def delete(self, path):
        return httpx.delete(self.url(path), headers=self.origin, timeout=30)

    def wait_job(self, job, deadline=120):
        """带上限地轮询任务状态，直到进入终态。"""
        stop = time.monotonic() + deadline
        while time.monotonic() < stop:
            result = self.get(f"/api/jobs/{job}").json()
            if result["status"] != "running":
                return result
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"平台进程中途退出\n{self.log_text()}")
            time.sleep(0.1)
        raise TimeoutError(f"任务 {job} 在 {deadline}s 内未结束\n{self.log_text()}")

    def events(self, job, after=0):
        """读 SSE 到终态事件为止（服务端在终态后主动结束流）。"""
        events = []
        with httpx.stream(
            "GET", self.url(f"/api/jobs/{job}/events?after={after}"), timeout=60
        ) as response:
            response.raise_for_status()
            current = {}
            for line in response.iter_lines():
                if line.startswith("id: "):
                    current["id"] = int(line[4:])
                elif line.startswith("data: "):
                    current["data"] = json.loads(line[6:])
                elif not line and current:
                    events.append(current)
                    current = {}
        return events

    def log_text(self, limit=4000):
        try:
            return self.log.read_text(errors="replace")[-limit:]
        except OSError:
            return ""
