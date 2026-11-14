"""回测任务：独立 spawn 工作进程计算，HTTP 只读已存结果。

- 结果目录 `<平台目录>/backtests/`：`index.sqlite3` 记任务状态，`<id>.<part>.json` 是页面文档，
  `<id>.<part>.npz` 是调仓记录与成交（按页读取）。先写临时文件再改名，读不到半个文件。
- 任务身份见 `settings.identity`；已完成的同一身份直接复用，不重跑。
- 一个常驻工作进程（独立进程组），排队逐个执行；取消正在跑的任务即整组结束进程，
  下一次运行再起新进程。空闲超时回收；HTTP 进程消失时工作进程自行退出。
- 结果目录按总大小封顶（`ASHARE_PLATFORM_BACKTEST_MAX_BYTES`，默认 2 GiB）：超出时按最近打开时间
  从旧到新删整份结果，状态改为 `evicted`，再提交同一身份即重跑。正在跑、排队中和刚完成
  （`ASHARE_PLATFORM_BACKTEST_KEEP_SECONDS`，默认 600 秒内）的不删。
"""

import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np

LOG = logging.getLogger("ashare.platform")
PROGRESS = {
    "queued": 0.0,
    "loading": 0.05,
    "signal": 0.25,
    "is": 0.45,
    "validation": 0.8,
    "saving": 0.95,
}
PARTS = ("is", "validation")
SUFFIXES = ("json", "npz")
MAX_BYTES = 2 << 30
KEEP_SECONDS = 600.0
SCHEMA = """
CREATE TABLE IF NOT EXISTS backtests (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    settings TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL NOT NULL,
    finished REAL,
    elapsed REAL,
    data_version TEXT,
    code_version TEXT,
    opened REAL
)
"""


def write_part(folder, key, part, output):
    """页面文档 JSON + 调仓记录、成交 NPZ；临时文件写完再改名。"""
    document = dict(output["document"])
    path = folder / f"{key}.{part}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )
    arrays = {f"trade_{k}": v for k, v in output["trades"].items()}
    arrays["dates"] = np.asarray(output["dates"]).astype("U10")
    arrays["rebalances"] = np.frombuffer(
        json.dumps(output["rebalances"], ensure_ascii=False, allow_nan=False).encode(),
        dtype=np.uint8,
    )
    npz = folder / f"{key}.{part}.npz"
    tmp_npz = folder / f"{key}.{part}.tmp.npz"
    np.savez(tmp_npz, **arrays)
    os.replace(tmp_npz, npz)
    os.replace(tmp, path)


def worker_main(incoming, outgoing, data_root, result_root, parent):
    """独立进程组；父进程消失（被杀、崩溃）时自行退出，不留孤儿。"""
    os.setsid()
    from ..runner import PanelCache, configure_logging

    configure_logging(result_root)
    import panelops

    from ..config import ops_threads
    from ..language import parse
    from .engine import run_backtest

    panelops.set_num_threads(ops_threads())
    folder = Path(result_root) / "backtests"
    panel = None
    while True:
        if os.getppid() != parent:
            return
        try:
            message = incoming.get(timeout=1)
        except queue.Empty:
            continue
        if message is None:
            return
        key, source, settings, benchmark_name = message
        started = time.perf_counter()
        try:
            outgoing.put((key, "progress", "loading"))
            if panel is None:
                panel = PanelCache(data_root)
            expression = parse(source)[0].expression
            outputs = run_backtest(
                panel,
                expression,
                settings,
                benchmark_name,
                lambda phase, k=key: outgoing.put((k, "progress", phase)),
            )
            outgoing.put((key, "progress", "saving"))
            for part, output in outputs.items():
                write_part(folder, key, part, output)
            elapsed = time.perf_counter() - started
            LOG.info("回测完成 id=%s elapsed=%.3f", key, elapsed)
            outgoing.put(
                (key, "complete", {"elapsed": elapsed, "version": panel.version})
            )
        except Exception:
            LOG.exception("回测失败 id=%s", key)
            outgoing.put((key, "failed", {"elapsed": time.perf_counter() - started}))


class Backtests:
    def __init__(
        self,
        root,
        data_root=None,
        *,
        idle_seconds=None,
        max_bytes=None,
        keep_seconds=None,
    ):
        self.root = Path(root)
        self.folder = self.root / "backtests"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data_root = data_root
        self.idle_seconds = (
            float(os.environ.get("ASHARE_PLATFORM_IDLE_SECONDS", "60"))
            if idle_seconds is None
            else idle_seconds
        )
        self.max_bytes = int(
            os.environ.get("ASHARE_PLATFORM_BACKTEST_MAX_BYTES", MAX_BYTES)
            if max_bytes is None
            else max_bytes
        )
        self.keep_seconds = float(
            os.environ.get("ASHARE_PLATFORM_BACKTEST_KEEP_SECONDS", KEEP_SECONDS)
            if keep_seconds is None
            else keep_seconds
        )
        self.lock = threading.RLock()
        self.pending = []  # (id, source, settings, benchmark_name)
        self.current = None
        self.started = None
        self.phase = {}
        self.process = self.input = self.output = None
        self.idle_at = time.monotonic()
        self.context = mp.get_context("spawn")
        with self.connect() as db:
            db.execute(SCHEMA)
            columns = {row[1] for row in db.execute("PRAGMA table_info(backtests)")}
            if "opened" not in columns:  # 第一版结果库没有这一列
                db.execute("ALTER TABLE backtests ADD COLUMN opened REAL")
            # 上次服务退出时没跑完的任务不会再有结果。
            db.execute(
                "UPDATE backtests SET status='failed' WHERE status IN ('queued','running')"
            )
        self.prune()
        self.stopping = threading.Event()
        self.monitor = threading.Thread(
            target=self._monitor, daemon=True, name="platform-backtests"
        )
        self.monitor.start()

    @contextmanager
    def connect(self):
        """一次事务一个连接：块内正常结束即提交，出错回滚，最后关闭。"""
        with closing(sqlite3.connect(self.folder / "index.sqlite3", timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db

    # ------------------------------------------------------------ 提交与取消
    def submit(self, key, name, source, settings, benchmark_name, data_version, code):
        """已完成或在跑的同一身份直接返回；失败、取消过的重新排队。"""
        with self.lock:
            row = self.row(key)
            if row is not None and (
                row["status"] in ("queued", "running")
                # 结果被清理或文件丢失时重算。
                or (row["status"] == "complete" and self._files(key))
            ):
                return row["status"]
            with self.connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO backtests "
                    "(id,name,source,settings,status,created,data_version,code_version) "
                    "VALUES (?,?,?,?,'queued',?,?,?)",
                    (
                        key,
                        name,
                        source,
                        json.dumps(settings, sort_keys=True),
                        time.time(),
                        str(data_version),
                        code,
                    ),
                )
            self.pending.append((key, source, settings, benchmark_name))
            self.phase[key] = "queued"
            self._schedule()
            return "queued"

    def cancel(self, key):
        return bool(self.cancel_many([key]))

    def cancel_many(self, keys):
        """取消一批：先撤下排队的，再结束正在跑的那个。

        顺序要紧：先结束进程的话，调度会把批内下一组送进新起的进程，再被取消一次。
        返回实际取消的身份（已结束的不算）。
        """
        wanted = set(keys)
        cancelled = []
        with self.lock:
            for item in [item for item in self.pending if item[0] in wanted]:
                self.pending.remove(item)
                self._finish(item[0], "cancelled")
                cancelled.append(item[0])
            if self.current in wanted:
                key = self.current
                self._stop_process()
                self._finish(key, "cancelled")
                cancelled.append(key)
            if cancelled:
                self._schedule()
        return cancelled

    def _paths(self, key):
        return [
            self.folder / f"{key}.{part}.{suffix}"
            for part in PARTS
            for suffix in SUFFIXES
        ]

    def _files(self, key):
        return all(path.exists() for path in self._paths(key))

    # ------------------------------------------------------------ 容量
    def touch(self, key):
        """记一次打开；清理按这个时间从旧到新删。"""
        with self.connect() as db:
            db.execute("UPDATE backtests SET opened=? WHERE id=?", (time.time(), key))

    def usage(self):
        """每份结果占用的字节数（只数结果文件，不含索引库）。"""
        sizes = {}
        for path in self.folder.iterdir():
            parts = path.name.split(".")
            if len(parts) == 3 and parts[1] in PARTS and parts[2] in SUFFIXES:
                try:
                    sizes[parts[0]] = sizes.get(parts[0], 0) + path.stat().st_size
                except FileNotFoundError:
                    pass
        return sizes

    def prune(self):
        """总大小超过上限时，从最久没打开的结果开始删，直到回到上限以内。"""
        with self.lock:
            sizes = self.usage()
            total = sum(sizes.values())
            if total <= self.max_bytes:
                return []
            busy = {self.current, *(item[0] for item in self.pending)}
            now = time.time()
            with self.connect() as db:
                rows = db.execute(
                    "SELECT id, status, finished, created, opened FROM backtests"
                ).fetchall()
            known = {row["id"]: row for row in rows}

            def last_used(key):
                row = known.get(key)
                if row is None:
                    return 0.0  # 索引里没有的孤儿文件最先删
                return row["opened"] or row["finished"] or row["created"] or 0.0

            removed = []
            for key in sorted(sizes, key=lambda k: (last_used(k), k)):
                if total <= self.max_bytes:
                    break
                row = known.get(key)
                if key in busy or (
                    row is not None and row["status"] in ("queued", "running")
                ):
                    continue
                if (
                    row is not None
                    and row["finished"] is not None
                    and now - row["finished"] < self.keep_seconds
                ):
                    continue
                # 先改状态再删文件：读到 evicted 就不会去打开半删的文件。
                if row is not None:
                    with self.connect() as db:
                        db.execute(
                            "UPDATE backtests SET status='evicted' WHERE id=?", (key,)
                        )
                for path in self._paths(key):
                    path.unlink(missing_ok=True)
                total -= sizes[key]
                removed.append(key)
            if removed:
                LOG.info(
                    "回测结果清理 removed=%d total=%d limit=%d",
                    len(removed),
                    total,
                    self.max_bytes,
                )
            return removed

    def _finish(self, key, status, elapsed=None):
        with self.connect() as db:
            db.execute(
                "UPDATE backtests SET status=?, finished=?, elapsed=? WHERE id=?",
                (status, time.time(), elapsed, key),
            )
        self.phase.pop(key, None)
        if self.current == key:
            self.current, self.started = None, None
            self.idle_at = time.monotonic()

    # ------------------------------------------------------------ 进程
    def _schedule(self):
        if self.current is not None or not self.pending:
            return
        if self.process is None:
            self.input, self.output = self.context.Queue(), self.context.Queue()
            self.process = self.context.Process(
                target=worker_main,
                args=(
                    self.input,
                    self.output,
                    self.data_root,
                    str(self.root),
                    os.getpid(),
                ),
            )
            self.process.start()
        item = self.pending.pop(0)
        self.current, self.started = item[0], time.monotonic()
        with self.connect() as db:
            db.execute("UPDATE backtests SET status='running' WHERE id=?", (item[0],))
        self.input.put(item)

    def _stop_process(self):
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.join(timeout=2)
        for q in (self.input, self.output):
            q.cancel_join_thread()
            q.close()
        self.process = self.input = self.output = None

    def _monitor(self):
        while not self.stopping.wait(0.05):
            with self.lock:
                process = self.process
                if process is None:
                    continue
                try:
                    while True:
                        key, kind, payload = self.output.get_nowait()
                        if key != self.current:
                            continue
                        if kind == "progress":
                            self.phase[key] = payload
                        elif kind == "complete":
                            with self.connect() as db:
                                db.execute(
                                    "UPDATE backtests SET data_version=? WHERE id=?",
                                    (str(payload["version"]), key),
                                )
                            self._finish(key, "complete", payload["elapsed"])
                            self.prune()
                        elif kind == "failed":
                            self._finish(key, "failed", payload["elapsed"])
                except queue.Empty:
                    pass
                except Exception:
                    LOG.exception("回测结果处理失败")
                    self._stop_process()
                    if self.current:
                        self._finish(self.current, "failed")
                if self.process is not None and not process.is_alive():
                    self._stop_process()
                    if self.current:
                        self._finish(self.current, "failed")
                if (
                    self.process is not None
                    and self.current is None
                    and not self.pending
                    and time.monotonic() - self.idle_at >= self.idle_seconds
                ):
                    self._stop_process()
                self._schedule()

    def close(self):
        self.stopping.set()
        self.monitor.join(timeout=5)
        with self.lock:
            self._stop_process()

    # ------------------------------------------------------------ 读取
    def row(self, key):
        with self.connect() as db:
            return db.execute("SELECT * FROM backtests WHERE id=?", (key,)).fetchone()

    def status(self, key):
        row = self.row(key)
        if row is None:
            return None
        with self.lock:
            phase = self.phase.get(key)
            position = next(
                (i for i, item in enumerate(self.pending) if item[0] == key), None
            )
        status = row["status"]
        if status == "complete" and not self._files(key):
            status = "evicted"  # 文件在服务之外被删：同样提示重跑
        return dict(
            id=key,
            name=row["name"],
            status=status,
            progress=PROGRESS.get(phase) if status == "running" else None,
            queue_position=position,
            elapsed=row["elapsed"],
            settings=json.loads(row["settings"]),
        )

    @property
    def running(self):
        with self.lock:
            return (self.current is not None) + len(self.pending)

    def document(self, key, part):
        path = self.folder / f"{key}.{part}.json"
        return path.read_text() if path.exists() else None

    def arrays(self, key, part):
        path = self.folder / f"{key}.{part}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
