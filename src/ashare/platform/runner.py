"""每 tab 一个 spawn 常驻进程；调度线程只落盘与转发事件。"""

import logging
import logging.handlers
import multiprocessing as mp
import os
import queue
import resource
import shutil
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ashare.factor import groupbt
from ashare.ledger import code_version

from .analysis import evaluate_signal, run_variant
from .config import (
    expand_settings,
    job_concurrency,
    max_workers,
    ops_threads,
    parallel_budget,
)
from .language import parse

LOG = logging.getLogger("ashare.platform")


def configure_logging(root):
    directory = Path(root) / "logs"
    directory.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("ashare.platform")
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            directory / "platform.log", maxBytes=4 << 20, backupCount=3
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


class PanelCache:
    """只读 mmap 按字段缓存，整个 tab 的数据版本固定；没有调用写数据接口。"""

    def __init__(self, root):
        import ashare.data

        self.data = ashare.data.open(root)
        meta = self.data.panel([])
        available = np.asarray(meta.dates)
        allowed = available[available < groupbt.VALIDATION[1]]
        if not len(allowed):
            raise ValueError("研究窗口没有数据")
        self.end = str(allowed[-1])
        self.dates, self.symbols = allowed, meta.symbols
        self.version = meta.version
        self.arrays = {}
        self.diagnostics = {}

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def ensure(self, fields):
        wanted = set(fields) - self.arrays.keys()
        raw = wanted - {"industry_l1", "benchmark_open", "benchmark_close"}
        if "industry_l1" in wanted:
            raw.add("industry")
        if raw:
            panel = self.data.panel(sorted(raw), end=self.end)
            self.arrays.update({name: panel[name] for name in raw})
        if "industry_l1" in wanted:
            a = self.arrays["industry"].astype(np.int64)
            self.arrays["industry_l1"] = np.where(a > 0, a, -1)
        benchmark = wanted & {"benchmark_open", "benchmark_close"}
        if benchmark:
            import polars as pl

            frame = self.data.table(
                "index_daily", end=self.end, symbols=["000300.SH"]
            ).collect()
            for name in benchmark:
                prices = frame.select(
                    "date", pl.col(name.removeprefix("benchmark_"))
                ).rows()
                lookup = dict(prices)
                self.arrays[name] = np.array(
                    [lookup.get(day, np.nan) for day in self.dates]
                )
        for value in self.arrays.values():
            value.flags.writeable = False

    def __getitem__(self, name):
        return self.arrays[name]


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def prepare_signals(runs, panel, folder, progress):
    """同摘要只求值一次，配置变体及两个窗口共享只读信号文件。"""
    folder.mkdir(parents=True, exist_ok=True)
    panel.diagnostics.clear()
    expressions = [parse(run["source"])[0].expression for run in runs]
    panel.ensure(set(groupbt.FIELDS).union(*(e.fields for e in expressions)))
    prepared = {}
    for index, expression in enumerate(expressions):
        progress(index, "evaluate")
        if expression.digest not in prepared:
            alpha = evaluate_signal(expression, panel)
            path = folder / f"{expression.digest}.npy"
            np.save(path, alpha, allow_pickle=False)
            prepared[expression.digest] = str(path)
            del alpha
        runs[index] = dict(
            runs[index], signal=prepared[expression.digest], ordinal=index
        )
        progress(index, "prepared")
    return runs


def calculate_run(run, panel, job, outgoing):
    started = time.perf_counter()
    alpha = np.load(run["signal"], mmap_mode="r", allow_pickle=False)
    timings = {}
    summary, arrays = run_variant(
        None,
        panel,
        run["settings"],
        lambda phase: outgoing.put(
            (job, "progress", {"ordinal": run["ordinal"], "phase": phase})
        ),
        signal=alpha,
        timings=timings,
        diagnostics=panel.diagnostics.setdefault(
            (run["signal"], run["settings"]["universe"], run["settings"]["trade_at"]),
            {},
        ),
    )
    return dict(
        run=run["id"],
        summary=summary,
        arrays=arrays,
        elapsed=time.perf_counter() - started,
        data_version=panel.version,
        timings=timings,
    )


_CHILD = None


def init_child(data_root, fields, outgoing):
    global _CHILD
    import panelops

    panelops.set_num_threads(1)
    panel = PanelCache(data_root)
    panel.ensure(fields)
    _CHILD = (panel, outgoing)


def child_run(run, job):
    panel, outgoing = _CHILD
    result = calculate_run(run, panel, job, outgoing)
    outgoing.put((job, "memory", {"pid": os.getpid(), "rss": rss_bytes()}))
    return result


def worker_main(incoming, outgoing, data_root, result_root, concurrency, mode):
    # 独立进程组保证取消时协调进程和所有子进程一起回收。
    os.setsid()
    configure_logging(result_root)
    import panelops

    panelops.set_num_threads(ops_threads())
    panel = None
    while True:
        message = incoming.get()
        if message is None:
            break
        job, runs, children, parallel = message
        started = time.perf_counter()
        try:
            if panel is None:
                panel = PanelCache(data_root)
            before = time.perf_counter()
            runs = prepare_signals(
                runs,
                panel,
                Path(result_root) / "work" / job,
                lambda index, phase, current=job: outgoing.put(
                    (current, "progress", {"ordinal": index, "phase": phase})
                ),
            )
            evaluate_seconds = time.perf_counter() - before
            if mode == "processes" and children >= 2:
                with ProcessPoolExecutor(
                    max_workers=children,
                    mp_context=mp.get_context("spawn"),
                    initializer=init_child,
                    initargs=(data_root, set(panel.arrays), outgoing),
                ) as executor:
                    futures = [executor.submit(child_run, run, job) for run in runs]
                    for future in as_completed(futures):
                        outgoing.put((job, "result", future.result()))
            else:
                with ThreadPoolExecutor(
                    max_workers=parallel if mode == "threads" else 1
                ) as executor:
                    futures = [
                        executor.submit(calculate_run, run, panel, job, outgoing)
                        for run in runs
                    ]
                    for future in as_completed(futures):
                        outgoing.put((job, "result", future.result()))
            elapsed = time.perf_counter() - started
            LOG.info(
                "任务完成 mode=%s variants=%d elapsed=%.3f evaluation=%.3f peak_rss=%d fields=%d children=%d",
                mode,
                len(runs),
                elapsed,
                evaluate_seconds,
                rss_bytes(),
                len(panel.arrays),
                children,
            )
            outgoing.put(
                (
                    job,
                    "complete",
                    {
                        "elapsed": elapsed,
                        "evaluation_seconds": evaluate_seconds,
                        "rss": rss_bytes(),
                    },
                )
            )
        except Exception:
            LOG.exception("任务失败 job=%s", job)
            outgoing.put(
                (
                    job,
                    "failed",
                    {
                        "message": "运行失败，请稍后重试",
                        "elapsed": time.perf_counter() - started,
                    },
                )
            )


class Manager:
    def __init__(
        self,
        store,
        *,
        limit=None,
        data_root=None,
        concurrency=None,
        mode=None,
        idle_seconds=None,
    ):
        self.store, self.data_root = store, data_root
        self.limit = max_workers() if limit is None else limit
        self.concurrency = (
            max(1, min(8, concurrency)) if concurrency else parallel_budget()
        )
        self.mode = mode or os.environ.get("ASHARE_PLATFORM_EXECUTOR", "threads")
        if self.mode not in ("threads", "processes"):
            raise ValueError("运行模式须为 threads 或 processes")
        self.idle_seconds = (
            float(os.environ.get("ASHARE_PLATFORM_IDLE_SECONDS", "60"))
            if idle_seconds is None
            else idle_seconds
        )
        self.context = mp.get_context("spawn")
        self.tabs = {}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.monitor = threading.Thread(
            target=self._monitor, daemon=True, name="platform-results"
        )
        self.store.recover()
        self.monitor.start()

    def create_tab(self, tab=None):
        with self.lock:
            key = tab or uuid.uuid4().hex
            if key not in self.tabs:
                self.tabs[key] = {
                    "job": None,
                    "last_job": None,
                    "process": None,
                    "input": None,
                    "output": None,
                    "pending": None,
                    "slots": 0,
                    "idle_at": time.monotonic(),
                }
            return key

    def submit(self, tab, source, settings, name):
        variants = parse(source)
        configs = expand_settings(settings, len(variants))
        pairs = [(v, c) for c in configs for v in variants]
        with self.lock:
            if tab not in self.tabs:
                raise ValueError("研究 tab 已关闭，请重新打开")
            state = self.tabs[tab]
            if state["job"]:
                raise ValueError("当前 tab 正在运行，请完成或取消后重试")
            job, runs = self.store.begin(
                tab, name, pairs, code_version(Path(__file__).parent)
            )
            state.update(job=job, last_job=job, started=time.monotonic(), pending=runs)
            self.store.event(job, {"kind": "queued"})
            self._schedule()
            return job

    def _schedule(self):
        pending = sorted(
            (s for s in self.tabs.values() if s["pending"]), key=lambda s: s["started"]
        )
        for state in pending:
            active = sum(
                s["slots"]
                for s in self.tabs.values()
                if s is not state and s["job"] and not s["pending"]
            )
            available = self.limit - active
            if available <= 0:
                continue
            # 有等待任务时提前回收其他 tab 的空闲缓存；进程总数不突破预算。
            for other in self.tabs.values():
                if other is not state and not other["job"] and other["process"]:
                    self._stop_process(other)
            children = (
                min(self.concurrency, available - 1) if self.mode == "processes" else 0
            )
            if children < 2:
                children = 0
            if state["process"] is None:
                incoming, outgoing = self.context.Queue(), self.context.Queue()
                process = self.context.Process(
                    target=worker_main,
                    args=(
                        incoming,
                        outgoing,
                        self.data_root,
                        str(self.store.root),
                        self.concurrency,
                        self.mode,
                    ),
                )
                process.start()
                state.update(process=process, input=incoming, output=outgoing)
            state["slots"] = 1 + children
            others = sum(
                1
                for s in self.tabs.values()
                if s is not state and s["job"] and not s["pending"]
            )
            parallel = job_concurrency(self.concurrency, len(state["pending"]), others)
            LOG.info(
                "调度 job=%s variants=%d parallel=%d budget=%d other_jobs=%d ops_threads=%d mode=%s",
                state["job"],
                len(state["pending"]),
                parallel,
                self.concurrency,
                others,
                ops_threads(),
                self.mode,
            )
            state["input"].put((state["job"], state["pending"], children, parallel))
            state["pending"] = None

    def _stop_process(self, state):
        process = state["process"]
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
                process.join(timeout=2)
            for key in ("input", "output"):
                state[key].cancel_join_thread()
                state[key].close()
            state.update(process=None, input=None, output=None, slots=0)

    def _finish(self, state, status, elapsed=None):
        job = state["job"]
        self.store.finish(job, status, elapsed)
        self.store.event(
            job,
            {
                "kind": "memory",
                "pid": os.getpid(),
                "rss": rss_bytes(),
                "role": "server",
            },
        )
        shutil.rmtree(self.store.root / "work" / job, ignore_errors=True)
        state.update(
            job=None,
            pending=None,
            slots=1 if state["process"] else 0,
            idle_at=time.monotonic(),
        )

    def cancel(self, job):
        with self.lock:
            for state in self.tabs.values():
                if state["job"] == job:
                    self._stop_process(state)
                    self._finish(
                        state, "cancelled", time.monotonic() - state["started"]
                    )
                    return True
        return False

    def close_tab(self, tab):
        with self.lock:
            state = self.tabs.get(tab)
            if state is None:
                return
            if state["job"]:
                self.cancel(state["job"])
            self._stop_process(state)
            del self.tabs[tab]

    def _monitor(self):
        while not self.stopping.wait(0.05):
            with self.lock:
                for state in list(self.tabs.values()):
                    process = state["process"]
                    if process is None:
                        continue
                    try:
                        while True:
                            job, kind, payload = state["output"].get_nowait()
                            if state["job"] != job:
                                continue
                            if kind == "result":
                                self.store.save(**payload)
                            elif kind in ("complete", "failed"):
                                self.store.event(job, {"kind": "timing", **payload})
                                self._finish(
                                    state, kind, time.monotonic() - state["started"]
                                )
                            else:
                                self.store.event(job, {"kind": kind, **payload})
                    except queue.Empty:
                        pass
                    except Exception:
                        LOG.exception("结果落盘失败")
                        self._stop_process(state)
                        if state["job"]:
                            self._finish(state, "failed")
                    if state["process"] is not None and not process.is_alive():
                        self._stop_process(state)
                        if state["job"]:
                            self._finish(state, "failed")
                    if (
                        state["process"]
                        and not state["job"]
                        and time.monotonic() - state["idle_at"] >= self.idle_seconds
                    ):
                        self._stop_process(state)
                self._schedule()

    @property
    def running(self):
        with self.lock:
            return sum(bool(s["job"]) for s in self.tabs.values())

    def close(self):
        self.stopping.set()
        self.monitor.join(timeout=5)
        for tab in list(self.tabs):
            self.close_tab(tab)
