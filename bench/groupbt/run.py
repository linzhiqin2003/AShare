"""为复跑命令设置 worktree 内构建缓存和进程组超时。"""

import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    if len(sys.argv) < 3 or Path.cwd() != root:
        raise SystemExit(
            "在 worktree 根目录运行：python3 bench/groupbt/run.py 超时秒数 命令 ..."
        )
    env = os.environ.copy()
    temporary = root / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(
        CARGO_HOME=str(root / ".cache/cargo-home"),
        UV_CACHE_DIR=str(root / ".cache/uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / ".cache/uv-python"),
        MPLCONFIGDIR=str(root / ".cache/mpl"),
        GIT_OPTIONAL_LOCKS="0",
        TMPDIR=str(temporary),
    )
    process = subprocess.Popen(sys.argv[2:], env=env, start_new_session=True)
    running = root / ".cache/groupbt/processes"
    running.mkdir(parents=True, exist_ok=True)
    marker = running / f"{process.pid}.pid"
    marker.write_text(str(process.pid))

    def stop():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def interrupted(signum, _frame):
        stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        return process.wait(timeout=float(sys.argv[1]))
    except subprocess.TimeoutExpired:
        stop()
        return 124
    finally:
        stop()
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
