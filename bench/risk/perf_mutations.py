"""性能轮（P-004…P-008）新逻辑的源码变异检查。

先跑不改动的对照（必须全绿），再逐项改坏源码、只跑对应测试（必须失败），
最后恢复源码并核对哈希。改 Rust 的变异会先重建扩展、确认导入的就是新构建。

    python bench/risk/perf_mutations.py [--only P-004]
"""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 改动前的源码先备份到这里；进程被杀没来得及还原时，下次启动先从备份还原。
BACKUP = ROOT / ".cache/perf_mutations"

CASES = {
    "P-004": [
        (
            "不核对快照键",
            [
                (
                    "src/ashare/risk/cache.py",
                    'raise ValueError("快照键不符")',
                    "pass",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
        (
            "读失败不当未命中",
            [
                (
                    "src/ashare/risk/cache.py",
                    "_UNREADABLE = (\n    OSError,\n    ValueError,\n    KeyError,\n    EOFError,\n    zipfile.BadZipFile,\n    zlib.error,\n    NotImplementedError,\n    RuntimeError,\n)",
                    "_UNREADABLE = (KeyError,)",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
        (
            "后台写失败被吞掉",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                self.error = error\n",
                    "                pass\n",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
        (
            "旧格式不判失效",
            [
                (
                    "src/ashare/risk/cache.py",
                    'if content.get("format") == FORMAT:',
                    "if True:",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
        (
            "命令行为不存在的目录建目录",
            [
                (
                    "src/ashare/risk/cache.py",
                    "    if not args.directory.is_dir():",
                    "    if False:",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
        (
            "证券表坏了不重写",
            [
                (
                    "src/ashare/risk/cache.py",
                    "        if symbols is not None:\n            self._symbol_table(symbols)",
                    "        pass",
                )
            ],
            "test/risk/test_cache_format.py",
        ),
    ],
    "P-006": [
        (
            "最老残块不延伸到覆盖滞后对",
            [
                (
                    "src/ashare/risk/specific.py",
                    "edge = (start + lags) - (start + lags) % BLOCK + BLOCK - 1",
                    "edge = start - start % BLOCK + BLOCK - 1",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "去掉条件数闸",
            [
                (
                    "src/ashare/risk/specific.py",
                    "| (variance * TAU < magnitude)",
                    "| (variance * 0 < 0)",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "去掉均值闸",
            [
                (
                    "src/ashare/risk/specific.py",
                    "| (mean * mean > MEAN_RATIO**2 * variance)",
                    "| (mean * 0 > 1)",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "去掉阈值闸",
            [
                (
                    "src/ashare/risk/specific.py",
                    "bad |= np.abs(v - floor) <= THRESHOLD_GUARD * floor",
                    "pass",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "重取移位时不清空整块",
            [
                (
                    "src/ashare/risk/specific.py",
                    "        self.blocks.clear()\n        self.sums.clear()\n",
                    "",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "有效天数出界行不减",
            [
                (
                    "src/ashare/risk/specific.py",
                    "                self.count -= np.isfinite(leaving)",
                    "                pass",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "Rust 滞后对忽略下限",
            [
                (
                    "crates/portopt-core/src/lib.rs",
                    "if i < lag || i - lag < floor {",
                    "if i < lag {",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
        (
            "Rust Bartlett 权重按全部滞后而不是实际滞后",
            [
                (
                    "crates/portopt-core/src/lib.rs",
                    "* (1. - lag as f64 / (used + 1) as f64)",
                    "* (1. - lag as f64 / (lags + 1) as f64)",
                )
            ],
            "test/risk/test_specific_incremental.py",
        ),
    ],
    "P-005": [
        (
            "窗口末端差一行",
            [
                (
                    "src/ashare/risk/window.py",
                    "end = last % self.capacity + self.capacity + 1",
                    "end = last % self.capacity + self.capacity",
                )
            ],
            "test/risk/test_residual_window.py",
        ),
        (
            "只写一份",
            [
                (
                    "src/ashare/risk/window.py",
                    "        self.rows[slot + self.capacity] = residual\n",
                    "",
                )
            ],
            "test/risk/test_residual_window.py",
        ),
        (
            "出界行取错",
            [
                (
                    "src/ashare/risk/window.py",
                    "return self.rows[day % self.capacity]",
                    "return self.rows[(day + 1) % self.capacity]",
                )
            ],
            "test/risk/test_residual_window.py",
        ),
        (
            "从头算模式窗口少取一天",
            [
                (
                    "src/ashare/risk/series.py",
                    "rs = residual_window.window(max(0, day - c.history + 1), day)",
                    "rs = residual_window.window(max(0, day - c.history + 2), day)",
                )
            ],
            "test/risk/test_residual_window.py",
        ),
    ],
    "P-007": [
        (
            "描述子不预先顺序算好，线程共用有状态的递推",
            [
                (
                    "src/ashare/risk/series.py",
                    "descriptors = _Precomputed({day: rolling.at(day) for day in days})",
                    "descriptors = rolling",
                )
            ],
            "test/risk/test_parallel_exposures.py",
        ),
        (
            "结果与日期错位",
            [
                (
                    "src/ashare/risk/series.py",
                    "prepared.update(zip(days, exposure_batch(days), strict=True))",
                    "prepared.update(zip(days, exposure_batch(days)[::-1], strict=True))",
                )
            ],
            "test/risk/test_parallel_exposures.py",
        ),
        (
            "1 个线程也走线程池",
            [
                (
                    "src/ashare/risk/series.py",
                    "if threads == 1 or len(days) < 2:",
                    "if len(days) < 2:",
                )
            ],
            "test/risk/test_parallel_exposures.py",
        ),
        (
            "线程数不校验",
            [
                (
                    "src/ashare/risk/series.py",
                    "    if (\n        not isinstance(threads, int)\n        or isinstance(threads, bool)\n        or not 1 <= threads <= MAX_THREADS\n    ):",
                    "    if False:",
                )
            ],
            "test/risk/test_parallel_exposures.py",
        ),
        (
            "忽略环境变量",
            [
                (
                    "src/ashare/risk/series.py",
                    'value = os.environ.get("ASHARE_RISK_THREADS")',
                    "value = None",
                )
            ],
            "test/risk/test_parallel_exposures.py",
        ),
    ],
    "P-008": [
        (
            "索引不核对逐日输入就沿用",
            [
                (
                    "src/ashare/risk/series.py",
                    "reuse = limit if same.all() else int(np.argmin(same))",
                    "reuse = limit",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
        (
            "索引不核对校验和",
            [
                (
                    "src/ashare/risk/cache.py",
                    "            or hashlib.sha256(rows.tobytes() + prefixes.tobytes()).hexdigest()\n            != checksum\n",
                    "",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
        (
            "仍从第 0 天逐个读快照",
            [
                (
                    "src/ashare/risk/series.py",
                    "        first = min(start, window)",
                    "        first = 0",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
        (
            "预读失败不提前起点",
            [
                (
                    "src/ashare/risk/series.py",
                    "                miss = day\n                break\n",
                    "                continue\n",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
        (
            "交出之后才发现损坏时不重新计划",
            [
                (
                    "src/ashare/risk/series.py",
                    "if cached is None and day < miss:",
                    "if False:",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
        (
            "较短的运行截掉较长的索引",
            [
                (
                    "src/ashare/risk/series.py",
                    "or len(index[0]) < end) and end:",
                    "or len(index[0]) != end) and end:",
                )
            ],
            "test/risk/test_prefix_index.py",
        ),
    ],
    "复核修补": [
        (
            "写队列不设上限",
            [
                (
                    "src/ashare/risk/cache.py",
                    "queue.Queue(maxsize=_QUEUE_LIMIT)",
                    "queue.Queue(maxsize=0)",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "读前不等待自己的写入",
            [
                (
                    "src/ashare/risk/cache.py",
                    "        if path in self._pending:\n            self.flush()",
                    "        if False:\n            self.flush()",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "出错后仍继续执行排队的写",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                if self.error is None:\n                    job()",
                    "                job()",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "阈值闸带宽缩到 1e-15",
            [
                (
                    "src/ashare/risk/specific.py",
                    "THRESHOLD_GUARD = 1e-9",
                    "THRESHOLD_GUARD = 1e-15",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "线程池不关闭",
            [
                (
                    "src/ashare/risk/series.py",
                    "        if pool is not None and self.pid == os.getpid():\n            pool.shutdown()",
                    "        if False:\n            pool.shutdown()",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "逐日摘要分段差一天",
            [
                (
                    "src/ashare/risk/series.py",
                    "range(i, min(i + step, len(days)))",
                    "range(i, min(i + step - 1, len(days)))",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "部分命中仍从窗口起点交出（P0-1 原样）",
            [
                (
                    "src/ashare/risk/series.py",
                    "        first = min(start, window)",
                    "        first = window",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "只在主循环起点 > 0 时重新计划",
            [
                (
                    "src/ashare/risk/series.py",
                    "if cached is None and day < miss:",
                    "if cached is None and first > 0 and day < miss:",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "fork 后沿用继承的写线程",
            [
                (
                    "src/ashare/risk/cache.py",
                    "if writer is not None and not writer.owned():",
                    "if False:",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "fork 后沿用继承的线程池",
            [
                (
                    "src/ashare/risk/series.py",
                    "if self.pool is None or self.pid != os.getpid():",
                    "if self.pool is None:",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "写失败后 load 照常读",
            [
                (
                    "src/ashare/risk/cache.py",
                    '            raise ValueError(f"fields: 需要 {STATE_FIELDS} 的非空子集")\n        self._check()',
                    '            raise ValueError(f"fields: 需要 {STATE_FIELDS} 的非空子集")',
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "生成器被回收时不报 stderr",
            [
                (
                    "src/ashare/risk/series.py",
                    '                        report_failure(cache, error, "生成器关闭时")',
                    "                        pass",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "解释器退出时不报 stderr",
            [
                (
                    "src/ashare/risk/cache.py",
                    '            report_failure(cache, error, "解释器退出时")',
                    "            pass",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "每个实例登记一次 atexit",
            [
                (
                    "src/ashare/risk/cache.py",
                    "            _LIVE.add(self)",
                    "            _LIVE.add(self)\n            atexit.register(lambda: None)",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "证券表不核对 SHA-256",
            [
                (
                    "src/ashare/risk/cache.py",
                    "            if stored is None or stored[1] != symbols_sha:",
                    "            if stored is None:",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "证券表坏了删掉完好的快照",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                return None\n            symbols = stored[0]",
                    "                self._discard(day, path)\n                return None\n            symbols = stored[0]",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "fields 不核对字段名",
            [
                (
                    "src/ashare/risk/cache.py",
                    "            or not set(fields) <= set(STATE_FIELDS)\n",
                    "",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "zipfile 的其他异常不当未命中",
            [
                (
                    "src/ashare/risk/cache.py",
                    "    zlib.error,\n    NotImplementedError,\n    RuntimeError,\n",
                    "",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "默认线程数不设上限",
            [
                (
                    "src/ashare/risk/series.py",
                    "    return max(1, min(count or os.cpu_count() or 1, MAX_DEFAULT_THREADS))",
                    "    return max(1, count or os.cpu_count() or 1)",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "环境变量接受非 ASCII 数字",
            [
                (
                    "src/ashare/risk/series.py",
                    "not (value.isascii() and value.isdigit())",
                    "not value.isdigit()",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "残留临时文件不清理",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                        path.unlink()",
                    "                        pass",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
        (
            "specific_incremental 不校验类型",
            [
                (
                    "src/ashare/risk/fundamental.py",
                    "        if not isinstance(self.specific_incremental, bool) or isinstance(",
                    "        if False and isinstance(",
                )
            ],
            "test/risk/test_review_fixes.py",
        ),
    ],
}


def native():
    return next((ROOT / "packages/portopt/python/portopt").glob("_core*.so"))


def rebuild():
    before = hashlib.sha256(native().read_bytes()).hexdigest()
    subprocess.run(
        ["make", "dev"], cwd=ROOT, check=True, capture_output=True, timeout=1800
    )
    return before, hashlib.sha256(native().read_bytes()).hexdigest()


def pytest(target):
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            target,
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.returncode, proc.stdout.strip().splitlines()[-1:]


def _restore_backups():
    for saved in sorted(BACKUP.rglob("*")) if BACKUP.is_dir() else []:
        if saved.is_file():
            target = ROOT / saved.relative_to(BACKUP)
            target.write_bytes(saved.read_bytes())
            saved.unlink()
            print(json.dumps({"restored_from_backup": str(target.relative_to(ROOT))}))


def _backup(file):
    saved = BACKUP / file.relative_to(ROOT)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(file.read_bytes())


def main():
    _restore_backups()
    # SIGTERM 也走 finally 还原源码。
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    args = parser.parse_args()
    report = []
    for item, cases in CASES.items():
        if args.only and item != args.only:
            continue
        targets = sorted({target for *_, target in cases})
        for target in targets:
            code, tail = pytest(target)
            report.append(
                {"item": item, "case": "对照（不改动）", "code": code, "tail": tail}
            )
            assert code == 0, tail
        for label, edits, target in cases:
            originals = {}
            native_edit = False
            try:
                for path, old, new in edits:
                    file = ROOT / path
                    if file not in originals:
                        _backup(file)
                    text = originals.setdefault(file, file.read_text())
                    current = file.read_text()
                    assert current.count(old) == 1, (label, old)
                    file.write_text(current.replace(old, new))
                native_edit = any(
                    p.startswith(("crates/", "packages/")) for p, *_ in edits
                )
                if native_edit:
                    before, after = rebuild()
                    assert before != after, "扩展没有重建"
                code, tail = pytest(target)
            finally:
                for file, text in originals.items():
                    file.write_text(text)
                    (BACKUP / file.relative_to(ROOT)).unlink(missing_ok=True)
                if native_edit:
                    rebuild()
            caught = code != 0
            report.append({"item": item, "case": label, "caught": caught, "tail": tail})
            print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    digest = hashlib.sha256(
        b"".join(
            p.read_bytes() for p in sorted((ROOT / "src/ashare/risk").glob("*.py"))
        )
    ).hexdigest()[:16]
    print(json.dumps({"restored_source_sha": digest}))
    missed = [r for r in report if r.get("caught") is False]
    if missed:
        raise SystemExit(f"未抓到的变异：{[r['case'] for r in missed]}")


if __name__ == "__main__":
    main()
