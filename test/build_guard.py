"""Reject stale panelbt extensions in tests and maintenance tools."""

import hashlib
from pathlib import Path


def assert_current_build(root=None):
    import panelbt
    from panelbt import _core

    root = Path(root or Path(__file__).resolve().parents[1])
    assert Path(panelbt.__file__).resolve().is_relative_to(root)
    digest = hashlib.sha256()
    for relative in (
        "packages/panelbt",
        "crates/panelbt-core",
        "crates/tradecore",
        "crates/panelops-core",
    ):
        directory = root / relative
        files = list((directory / "src").rglob("*.rs"))
        if (directory / "build.rs").exists():
            files.append(directory / "build.rs")
        for path in sorted(files):
            digest.update(path.read_bytes())
    assert getattr(_core, "SOURCE_SHA256", None) == digest.hexdigest(), (
        "panelbt extension is stale: run make dev before testing or benchmarking"
    )
    return digest.hexdigest()
