"""D1 定向变异；只改临时源码副本，检查导入位置后跑基线和审查反例。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TEST = "test/paths/test_paths.py"
REVIEW = "test/paths/test_review.py"


def run(args, env):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def verify_import(env, expected, module="ashare.paths"):
    result = run(
        [
            "-c",
            f"import importlib; print(importlib.import_module({module!r}).__file__)",
        ],
        env,
    )
    if result.returncode or Path(result.stdout.strip()).resolve() != expected.resolve():
        raise RuntimeError("测试导入位置不符，停止变异")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"变异锚点数量不等于 1：{old!r}")
    return source.replace(old, new, 1)


def mutants(source):
    yield (
        "missing_target_guard",
        replace_once(source, "if redirected and not folder.is_dir():", "if False:"),
    )
    start = source.index("    if path is not None:\n", source.index("def _location("))
    end = source.index(
        "    folder = data_root(root, cwd_default=cwd_default) / name", start
    )
    block = source[start:end]
    split = block.index("    legacy = ")
    yield (
        "legacy_over_explicit",
        source[:start] + block[split:] + block[:split] + source[end:],
    )
    yield "skip_sha256", replace_once(source, "_sha256(child)", '"unchecked"')
    early = replace_once(
        source, "        _verify_copy(source, destination, before)\n", ""
    )
    yield (
        "publish_before_verify",
        replace_once(
            early,
            "        _write_pointer(source, value)\n",
            "        _write_pointer(source, value)\n        _verify_copy(source, destination, before)\n",
        ),
    )
    yield "allow_hot", replace_once(source, "if spec.hot and not force:", "if False:")


def review_mutants(source):
    flag = replace_once(
        source, "    owned = False\n", "    owned = False\n    committed = False\n"
    )
    flag = replace_once(
        flag,
        "        _write_pointer(source, value)\n",
        "        _write_pointer(source, value)\n        committed = True\n",
    )
    flag = replace_once(
        flag,
        "if owned and not os.path.lexists(source / POINTER):",
        "if owned and not committed:",
    )
    yield (
        "publication_memory_flag",
        "ashare/paths.py",
        flag,
        "test_move_interrupt_publication",
    )
    for name, relative, old, new, test in (
        (
            "datastatus_reconstruct_root",
            "ashare/platform/datastatus.py",
            "read_manifest(self._root_option)",
            "read_manifest(self.root.parent)",
            "test_datastatus_renamed_dataset",
        ),
        (
            "cb_reconstruct_root",
            "ashare/data/cb_panel.py",
            "else store.data_root, dates",
            "else store.root.parent, dates",
            "test_cb_panel_renamed_dataset",
        ),
        (
            "cwd_uses_checkout_root",
            "ashare/paths.py",
            'or (Path("data") if cwd_default else DEFAULT_DATA_ROOT)',
            "or DEFAULT_DATA_ROOT",
            "test_cwd_defaults_outside_checkout",
        ),
        (
            "bench_loses_main_checkout",
            "ashare/paths.py",
            'return data_root(Path(common).parent / "data")',
            "return root",
            "test_bench_main_checkout_fallback",
        ),
    ):
        original = (ROOT / "src" / relative).read_text()
        yield name, relative, replace_once(original, old, new), test


def main():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ASHARE_")
        and k not in ("CRYPTO_DATA_ROOT", "TUSHARE_TOKEN")
    }
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        POLARS_MAX_THREADS="2",
        RAYON_NUM_THREADS="2",
    )
    expected = ROOT / "src/ashare/paths.py"
    verify_import(env, expected)
    baseline = run(["-m", "pytest", "test/paths", "-q"], env)
    if baseline.returncode:
        raise RuntimeError("未变异基线失败，停止；先运行 test/paths")
    print("worktree import verified; baseline passed", flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="ashare-d1-mutations-") as temporary:
        overlay = Path(temporary) / "src"
        shutil.copytree(
            ROOT / "src/ashare",
            overlay / "ashare",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        original = expected.read_text()
        env["PYTHONPATH"] = str(overlay)
        cases = [
            (name, "ashare/paths.py", changed, TEST)
            for name, changed in mutants(original)
        ] + [
            (name, relative, changed, f"{REVIEW}::{test}")
            for name, relative, changed, test in review_mutants(original)
        ]
        for name, relative, changed, test in cases:
            file = overlay / relative
            saved = file.read_text()
            file.write_text(changed)
            module = relative.removesuffix(".py").replace("/", ".")
            verify_import(env, overlay / "ashare/paths.py")
            verify_import(env, file, module)
            started = perf_counter()
            result = run(["-m", "pytest", test, "-q"], env)
            file.write_text(saved)
            failed = re.findall(r"^FAILED (\S+)", result.stdout, re.MULTILINE)
            killed = (
                result.returncode == 1
                and bool(failed)
                and "ERROR collecting" not in result.stdout
            )
            row = dict(
                mutation=name,
                file=f"src/{relative}",
                tests=test,
                killed=killed,
                returncode=result.returncode,
                failed_tests=failed,
                seconds=round(perf_counter() - started, 3),
                import_verified=True,
            )
            results.append(row)
            print(f"{name}: {'killed' if killed else 'SURVIVED / ERROR'}", flush=True)
    output = dict(
        baseline_passed=True,
        baseline_summary=baseline.stdout.strip().splitlines()[-1],
        worktree_import_verified=True,
        killed=sum(r["killed"] for r in results),
        total=len(results),
        results=results,
    )
    target = ROOT / "bench/results/mutations_data_paths.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return int(output["killed"] != output["total"])


if __name__ == "__main__":
    raise SystemExit(main())
