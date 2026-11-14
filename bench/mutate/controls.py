"""Control experiments: prove the tools test the mutated code, not the installed original.

Python (per shard, on the work directory a ``full`` run left behind):
  1. import path: the target module resolves inside ``.cache/mutate/py/<shard>/mutants/src``
     and carries mutmut's trampolines;
  2. mutant 0: the shard's tests pass on the trampolined copy with no mutant active;
  3. sure-kill: a ``raise`` appended to the copy makes the tests fail with that message.

Rust (one scratch copy of the tree, driven through the same cargo shim as cargo-mutants):
  1. mutant 0: unmutated copy builds, the proof step passes, pytest passes;
  2. sure-kill: an obvious kernel bug (weight-mode turnover halved) is rebuilt, proven loaded (the
     embedded source digest changes), and the tests fail;
  3. stale-extension guard: dropping the original extension into the mutated copy makes the
     proof step fail with the infra sentinel instead of counting as a kill.

Writes ``.cache/testing/mutation_controls.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from bench.mutate import python as py
from bench.mutate.cargo_shim import INFRA_SENTINEL
from bench.mutate.common import (
    PYTHON,
    ROOT,
    TESTING,
    WORK,
    child_env,
    git,
    load_shards,
    log,
    now,
)

PY_TARGETS = {
    "ashare-platform": ("ashare.platform.config", "ashare/platform/config.py"),
    "ashare-factor-groupbt": (
        "ashare.factor.groupbt.api",
        "ashare/factor/groupbt/api.py",
    ),
}
# the weight-mode kernel behind groupbt
RUST_FILE = "crates/panelbt-core/src/weights.rs"
RUST_TESTS = [
    "test/factor/test_groupbt_weights.py",
    "test/factor/test_groupbt_bitwise.py",
]


def run(cmd, cwd, env, timeout=1200):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout + proc.stderr


def python_controls():
    shards = load_shards()
    out = {}
    for name, (module, inner) in PY_TARGETS.items():
        shard = shards[name]
        work = py.workdir(shard, "full")
        copy = work / "mutants/src" / inner
        if not copy.exists() or "__mutmut_" not in copy.read_text():
            out[name] = {"skipped": "run `make mutate MODULE=src/" + inner + "` first"}
            continue
        env = child_env(PYTHONPATH=py.pythonpath(work, shard))
        env.pop("MUTANT_UNDER_TEST", None)
        cwd = work / "mutants"
        code, text = run(
            [str(PYTHON), "-c", f"import {module} as m; print(m.__file__)"],
            cwd,
            env,
            120,
        )
        loaded = text.strip().splitlines()[-1] if text.strip() else ""
        import_ok = code == 0 and loaded.startswith(str(work / "mutants/src"))
        tests = [str(t) for t in shard.test_args()]
        pytest = [
            str(PYTHON),
            "-m",
            "pytest",
            *tests,
            "-x",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ]
        t0 = time.monotonic()
        clean_code, clean_text = run(pytest, cwd, env)
        clean_seconds = time.monotonic() - t0
        original = copy.read_text()
        marker = "mutate control: sure kill"
        try:
            copy.write_text(original + f"\nraise RuntimeError({marker!r})\n")
            kill_code, kill_text = run(pytest, cwd, env)
        finally:
            copy.write_text(original)
        out[name] = {
            "module": module,
            "imported_from": os.path.relpath(loaded, ROOT) if loaded else None,
            "import_from_mutated_copy": import_ok,
            "mutant0_exit": clean_code,
            "mutant0_passed": clean_code == 0,
            "mutant0_seconds": round(clean_seconds, 1),
            "sure_kill_exit": kill_code,
            "sure_kill_caught": kill_code != 0 and marker in kill_text,
        }
        log(f"python control {name}: {out[name]}")
    return out


def scratch_copy(dest):
    if dest.exists():
        shutil.rmtree(dest)
    files = git("ls-files", "-co", "--exclude-standard", "-z").split("\0")
    for f in files:
        if not f or not (ROOT / f).is_file():
            continue
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, target)


def shim(args, cwd, env):
    cmd = [str(PYTHON), str(ROOT / "bench/mutate/cargo_shim.py"), *args]
    return run(cmd, cwd, env, timeout=1800)


def rust_controls():
    dest = WORK / "control-rs"
    scratch_copy(dest)
    real = shutil.which("cargo")
    env = child_env(
        MUTATE_REAL_CARGO=real,
        MUTATE_PYTHON=str(PYTHON),
        MUTATE_PYTEST=json.dumps(RUST_TESTS),
        PYO3_PYTHON=str(PYTHON),
        CARGO_PROFILE_RELEASE_LTO="off",
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS="16",
        CARGO_PROFILE_RELEASE_INCREMENTAL="true",
        CARGO_BUILD_JOBS="2",
        MUTATE_SKIP_CARGO_TEST="1",
    )
    manifest = ["--manifest-path", str(dest / "Cargo.toml"), "--package=panelbt-core"]
    result = {}

    def phase(label):
        b_code, b_text = shim(["test", "--no-run", *manifest], dest, env)
        t_code, t_text = shim(["test", *manifest], dest, env)
        digest = None
        for line in t_text.splitlines():
            if "loaded from scratch:" in line:
                digest = json.loads(line.split("loaded from scratch:", 1)[1])[
                    "panelbt_source_sha256"
                ]
        (dest / f"{label}.log").write_text(b_text + "\n----\n" + t_text)
        return {
            "build_exit": b_code,
            "test_exit": t_code,
            "panelbt_digest": digest,
            "infra_error": INFRA_SENTINEL in t_text,
            "log": os.path.relpath(dest / f"{label}.log", ROOT),
        }

    result["mutant0"] = phase("mutant0")
    result["mutant0"]["passed"] = result["mutant0"]["test_exit"] == 0
    source = dest / RUST_FILE
    original = source.read_text()
    old = "sum(&scratch.select)"
    assert old in original, (
        "control mutation site moved; update bench/mutate/controls.py"
    )
    source.write_text(original.replace(old, "0.5 * sum(&scratch.select)", 1))
    result["sure_kill"] = phase("sure_kill")
    result["sure_kill"]["caught"] = (
        result["sure_kill"]["test_exit"] != 0 and not result["sure_kill"]["infra_error"]
    )
    result["sure_kill"]["digest_changed"] = (
        result["sure_kill"]["panelbt_digest"] is not None
        and result["sure_kill"]["panelbt_digest"] != result["mutant0"]["panelbt_digest"]
    )
    # Stale guard: keep the mutated source, put the unmutated extension back, test only.
    stale = next((ROOT / "packages/panelbt/python/panelbt").glob("_core*.so"))
    target_dir = dest / "packages/panelbt/python/panelbt"
    for old_so in target_dir.glob("_core*.so"):
        old_so.unlink()
    shutil.copy2(stale, target_dir / stale.name)
    t_code, t_text = shim(["--prove", str(dest)], dest, env)
    result["stale_extension"] = {
        "test_exit": t_code,
        "infra_error_reported": INFRA_SENTINEL in t_text,
    }
    log(f"rust control: {result}")
    return result


def main(language=None):
    TESTING.mkdir(parents=True, exist_ok=True)
    path = TESTING / "mutation_controls.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if language in (None, "python"):
        data["python"] = python_controls()
    if language in (None, "rust"):
        data["rust"] = rust_controls()
    data["generated"] = now()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    log(f"wrote {path.relative_to(ROOT)}")
