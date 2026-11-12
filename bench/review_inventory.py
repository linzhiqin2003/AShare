"""Inventory reviewable new files and root-file changes against this rework's start."""

import difflib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT, text=True
    ).split("\0")
    excluded = (
        ".worktrees/",
        "data/",
        "packages/paperbroker/",
        "crates/paperbroker-core/",
        "src/ashare/",
    )
    records = []
    reasons = {
        "test/fixtures/legacy_sha256.npz": "20000 portfolios × 2 versions × 32 SHA-256 bytes; original and corrected regression digests, no full arrays.",
        "bench/results/legacy-differences.json.gz": "Per-seed, per-field, per-day differences required by review; compressed losslessly instead of a multi-megabyte Markdown table.",
        "bench/results/compare.json": "Complete comparison matrix with both timing scopes, all raw repetitions, correctness checks, memory and version metadata; reproducible performance baseline.",
    }
    for relative in sorted(x for x in paths if x and not x.startswith(excluded)):
        size = (ROOT / relative).stat().st_size
        records.append(
            dict(
                path=relative,
                bytes=size,
                reason=reasons.get(relative, "") if size > 200000 else "",
            )
        )
    changes = {}
    for name in [
        ".gitignore",
        "Cargo.toml",
        "Cargo.lock",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ]:
        prior = ROOT / ".cache/rework/start" / name
        if prior.exists():
            delta = list(
                difflib.unified_diff(
                    prior.read_text().splitlines(),
                    (ROOT / name).read_text().splitlines(),
                    fromfile="start/" + name,
                    tofile=name,
                    lineterm="",
                )
            )
            if delta:
                changes[name] = delta
    result = dict(
        files=records,
        total_bytes=sum(x["bytes"] for x in records),
        root_changes=changes,
    )
    (ROOT / "bench/results/review-inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 本轮新增文件清单",
        "",
        "|文件|字节|超过 200 KB 的理由|",
        "|---|---:|---|",
    ]
    lines += [f"|{x['path']}|{x['bytes']}|{x['reason']}|" for x in records]
    lines += ["", "## 根目录共享文件：相对于返工开始", ""]
    for name, delta in changes.items():
        lines += [f"### {name}", "", "```diff", *delta, "```", ""]
    (ROOT / "docs/step1b/INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(len(records), "new files;", result["total_bytes"], "bytes")


if __name__ == "__main__":
    main()
