# ruff: noqa: E402
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)

import subprocess
import sys


def test_fingerprint_tracks_python_in_workspace_and_sdist(tmp_path):
    script = (
        (ROOT / "packages/panelops/build.rs")
        .read_text()
        .split('r#"', 1)[1]
        .split('"#;', 1)[0]
    )

    def layout(name, relocated):
        base = tmp_path / name
        package = base / "packages/panelops"
        files = {
            package
            / "Cargo.toml": '[dependencies]\npanelops-core = { path = "../../crates/panelops-core" }\n',
            package / "build.rs": "// fingerprint build script\n",
            package / "src/lib.rs": "// binding\n",
            base / "crates/panelops-core/src/lib.rs": "// core\n",
            (base if relocated else package) / "python/panelops/api.py": "VALUE = 1\n",
            (base if relocated else package)
            / "python/panelops/__init__.py": "from .api import VALUE\n",
        }
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        return package, (base if relocated else package) / "python/panelops/api.py"

    def identity(package):
        run = subprocess.run(
            [sys.executable, "-c", script],
            cwd=package,
            capture_output=True,
            text=True,
            check=True,
        )
        return next(
            line.split("=", 2)[-1]
            for line in run.stdout.splitlines()
            if line.startswith("cargo:rustc-env=PANELOPS_BUILD_ID=")
        )

    workspace, original = layout("workspace", False)
    sdist, relocated = layout("sdist", True)
    assert identity(workspace) == identity(sdist)
    previous = identity(sdist)
    relocated.write_text("VALUE = 2\n")
    assert identity(sdist) != previous
    original.write_text("VALUE = 2\n")
    assert identity(workspace) == identity(sdist)
    previous = identity(sdist)
    (sdist / "../../crates/panelops-core/src/lib.rs").write_text("// new core\n")
    assert identity(sdist) != previous
