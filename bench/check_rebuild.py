"""实际修改内核一行并 make dev，检查 uv 的跨目录 Rust 缓存键。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
CACHE = ROOT / ".cache"


def marker():
    return subprocess.check_output(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "from panelbt import _core; print(_core.BUILD_MARKER)",
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def build(name):
    with (CACHE / name).open("w") as log:
        subprocess.run(
            ["make", "dev"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def main():
    source = CORE.read_text()
    before = marker()
    assert before == "panelbt-core-v2"
    try:
        CORE.write_text(
            source.replace('"panelbt-core-v2"', '"panelbt-core-cache-check"')
        )
        build("cache-changed-build.log")
        changed = marker()
        assert changed == "panelbt-core-cache-check", changed
    finally:
        CORE.write_text(source)
        build("cache-restored-build.log")
    restored = marker()
    assert restored == before
    result = {
        "before": before,
        "changed": changed,
        "restored": restored,
        "command": "make dev（未用 --reinstall、--refresh 或清缓存）",
    }
    (CACHE / "cache-verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
