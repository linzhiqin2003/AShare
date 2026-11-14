"""Stand-in for `cargo` that cargo-mutants calls (via the CARGO environment variable).

cargo-mutants can only run `cargo test`, but the Rust kernels here are mostly guarded by the
Python tests. This shim keeps every cargo call as is and adds two steps:

* build phase (`cargo test --no-run`): also build the four PyO3 extensions from the mutated
  scratch tree and copy them next to their Python packages inside that tree;
* test phase (`cargo test`): after the Rust tests pass, prove the scratch extensions are the ones
  being imported (test/build_guard.py checks the embedded source digest), then run the shard's
  pytest selection against the scratch tree.

A build failure makes the mutant unviable, a failing test catches it. If the proof step fails,
the shim prints INFRA_SENTINEL; bench/mutate/rust.py reclassifies such mutants as errors
instead of counting them as caught.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

INFRA_SENTINEL = "MUTATE-SHIM INFRA-ERROR"

# (cargo package, cdylib stem, Python package, file name of the installed module)
EXTENSIONS = (
    ("panelbt-python", "panelbt_ext", "panelbt", "_core.abi3.so"),
    ("panelops-python", "panelops_ext", "panelops", None),
    ("paperbroker-python", "paperbroker_native", "paperbroker", None),
    ("portopt-python", "portopt_ext", "portopt", None),
)


def say(*parts):
    print("[mutate-shim]", *parts, flush=True)


def real_cargo():
    return os.environ["MUTATE_REAL_CARGO"]


def scratch_root(args):
    if "--manifest-path" in args:
        return Path(args[args.index("--manifest-path") + 1]).resolve().parent
    for a in args:
        if a.startswith("--manifest-path="):
            return Path(a.split("=", 1)[1]).resolve().parent
    return Path.cwd().resolve()


def dylib_name(stem):
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    if sys.platform.startswith("win"):
        return f"{stem}.dll"
    return f"lib{stem}.so"


def build_extensions(root):
    link = []
    if sys.platform == "darwin":
        # What maturin passes for extension modules: leave Python symbols to the loader.
        link = ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for package, stem, module, name in EXTENSIONS:
        cmd = [
            real_cargo(),
            "rustc",
            "--release",
            "--lib",
            "-p",
            package,
            "--manifest-path",
            str(root / "Cargo.toml"),
            *link,
        ]
        code = subprocess.call(cmd)
        if code:
            say("extension build failed:", package)
            return code
        built = target / "release" / dylib_name(stem)
        dest_dir = root / "packages" / module / "python" / module
        for old in dest_dir.glob("_core*.so"):
            old.unlink()
        dest = dest_dir / (name or "_core" + suffix)
        shutil.copy2(built, dest)
    return 0


def python_env(root):
    env = dict(os.environ)
    paths = [str(root / "src")]
    paths += [str(p) for p in sorted((root / "packages").glob("*/python"))]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop("CARGO", None)
    return env


PROOF = """
import hashlib, json, sys
from pathlib import Path
root = Path.cwd().resolve()
sys.path.insert(0, str(root))
from test.build_guard import assert_current_build
digest = assert_current_build(root)
import panelbt, panelops, paperbroker, portopt, ashare
from panelbt import _core as a
from panelops import _core as b
from paperbroker import _core as c
from portopt import _core as d
mods = dict(panelbt=a, panelops=b, paperbroker=c, portopt=d, ashare=ashare)
out = {}
for k, m in mods.items():
    p = Path(m.__file__).resolve()
    assert p.is_relative_to(root), (k, str(p))
    out[k] = str(p.relative_to(root))
out["panelbt_source_sha256"] = digest
print(json.dumps(out))
"""


def prove_and_test(root):
    env = python_env(root)
    py = os.environ["MUTATE_PYTHON"]
    proof = subprocess.run(
        [py, "-B", "-c", PROOF], cwd=root, env=env, capture_output=True, text=True
    )
    if proof.returncode:
        print(proof.stdout + proof.stderr)
        say(INFRA_SENTINEL, "scratch extensions are not the ones imported")
        return 97
    say("loaded from scratch:", proof.stdout.strip())
    tests = json.loads(os.environ.get("MUTATE_PYTEST", "[]"))
    if not tests:
        return 0
    cmd = [
        py,
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
    if "-n" not in tests:
        cmd += ["-p", "no:xdist"]
    code = subprocess.call(cmd, cwd=root, env=env)
    say("pytest exit", code)
    # 0: survived; anything else (failure, crash, collection error) is a catch:
    # the proof step above already ruled out a stale or missing extension.
    return 0 if code == 0 else 1


def main(args):
    if args[:1] == ["--prove"]:  # used by bench/mutate/controls.py
        return prove_and_test(Path(args[1]).resolve())
    testing = args[:1] == ["test"]
    if (
        testing
        and "--no-run" not in args
        and os.environ.get("MUTATE_SKIP_CARGO_TEST") == "1"
    ):
        code = 0  # controls only: isolate the Python side of the test phase
    else:
        code = subprocess.call([real_cargo(), *args])
    if not testing or code:
        return code
    root = scratch_root(args)
    if "--no-run" in args:
        return build_extensions(root)
    return prove_and_test(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
