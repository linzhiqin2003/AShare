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
}


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
            try:
                for path, old, new in edits:
                    file = ROOT / path
                    text = originals.setdefault(file, file.read_text())
                    current = file.read_text()
                    assert current.count(old) == 1, (label, old)
                    file.write_text(current.replace(old, new))
                code, tail = pytest(target)
            finally:
                for file, text in originals.items():
                    file.write_text(text)
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
