"""性能轮（P-004…P-008）新逻辑的源码变异检查。

先跑不改动的对照（必须全绿），再逐项改坏源码、只跑对应测试（必须失败），
最后恢复源码并核对哈希。改 Rust 的变异会先重建扩展、确认导入的就是新构建。

    python bench/risk/perf_mutations.py [--only P-004]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
                    "_UNREADABLE = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)",
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
                    "                self.count -= np.isfinite(self.rows[slot])",
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
                    "if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:",
                    "if False:",
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


def main():
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
