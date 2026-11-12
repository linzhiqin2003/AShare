import hashlib
from pathlib import Path

import panelbt
from panelbt import _core

ROOT = Path(__file__).resolve().parents[1]
assert Path(panelbt.__file__).resolve().is_relative_to(ROOT)
h = hashlib.sha256()
for relative in ("packages/panelbt", "crates/panelbt-core", "crates/tradecore"):
    root = ROOT / relative
    files = list((root / "src").rglob("*.rs"))
    if (root / "build.rs").exists():
        files.append(root / "build.rs")
    for path in sorted(files):
        h.update(path.read_bytes())
assert getattr(_core, "SOURCE_SHA256", None) == h.hexdigest(), (
    "panelbt extension is stale: run make dev before testing"
)
