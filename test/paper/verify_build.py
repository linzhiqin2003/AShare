"""Build panelbt then paperbroker into a clean shared target and inspect installed wheels."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / ".cache/round3/collision"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    source = directory / "source"
    source.mkdir()
    for name in ("Cargo.toml", "Cargo.lock"):
        shutil.copy2(ROOT / name, source / name)
    for name in (
        "crates/panelbt-core",
        "crates/paperbroker-core",
        "packages/panelbt",
        "packages/paperbroker",
    ):
        shutil.copytree(
            ROOT / name,
            source / name,
            ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
        )
    source_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "crates/paperbroker-core/src/lib.rs",
            "packages/paperbroker/src/lib.rs",
            "packages/paperbroker/Cargo.toml",
        )
    }
    env = os.environ | dict(
        CARGO_HOME=str(ROOT / ".cache/cargo-home"),
        CARGO_TARGET_DIR=str(directory / "target"),
        PYO3_PYTHON=str(ROOT / ".venv/bin/python"),
        TMPDIR=str(ROOT / ".cache/round3/tmp"),
    )

    def run(args, name):
        process = subprocess.run(
            args, cwd=ROOT, env=env, text=True, capture_output=True
        )
        (directory / (name + ".log")).write_text(process.stdout + process.stderr)
        assert process.returncode == 0, name + ": " + process.stderr[-2000:]
        return process.stdout

    for name in ("panelbt", "paperbroker"):
        run(
            [
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "build",
                "--release",
                "--manifest-path",
                str(source / f"packages/{name}/Cargo.toml"),
                "--out",
                str(directory / "wheels"),
            ],
            name,
        )
    artifacts = {
        name: hashlib.sha256(
            (directory / "target/release" / name).read_bytes()
        ).hexdigest()
        for name in ("lib_core.dylib", "libpaperbroker_native.dylib")
    }
    # Exercise Cargo's cache after both artifacts exist.
    run(
        [
            "uv",
            "run",
            "--no-sync",
            "maturin",
            "build",
            "--release",
            "--manifest-path",
            str(source / "packages/paperbroker/Cargo.toml"),
            "--out",
            str(directory / "wheels"),
        ],
        "paperbroker-cached",
    )
    run(
        [
            "uv",
            "venv",
            str(directory / "venv"),
            "--python",
            str(ROOT / ".venv/bin/python"),
        ],
        "venv",
    )
    python = directory / "venv/bin/python"
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            *map(str, sorted((directory / "wheels").glob("*.whl"))),
        ],
        "install",
    )
    probe = """import json,hashlib
from pathlib import Path
import panelbt._core as panel
import paperbroker._core as paper
from paperbroker import Broker
assert hasattr(panel,"run") and hasattr(panel,"BUILD_MARKER") and not hasattr(panel,"BrokerCore")
assert hasattr(paper,"BrokerCore") and not hasattr(paper,"run")
assert Broker(["synthetic"]).account()["cash"] == 10000000000
print(json.dumps({"panelbt": {"file":panel.__file__, "sha256":hashlib.sha256(Path(panel.__file__).read_bytes()).hexdigest()},
"paperbroker": {"file":paper.__file__, "sha256":hashlib.sha256(Path(paper.__file__).read_bytes()).hexdigest()}, "imports_correct":True}))"""
    checked = json.loads(run([str(python), "-c", probe], "imports"))
    for value in checked.values():
        if isinstance(value, dict):
            assert Path(value["file"]).resolve().is_relative_to(directory / "venv")
            value["file"] = str(Path(value["file"]).relative_to(ROOT))
    result = dict(
        order=["panelbt", "paperbroker", "paperbroker_cached"],
        artifacts=artifacts,
        source_hashes=source_hashes,
        **checked,
    )
    (ROOT / ".cache/round3/build-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
