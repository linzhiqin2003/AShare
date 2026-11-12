"""Remove superseded editable native extensions after switching to abi3."""

from pathlib import Path

repo = Path(__file__).resolve().parents[1]
root = repo / "packages/panelbt/python/panelbt"
if (root / "_core.abi3.so").exists():
    for path in root.glob("_core.cpython-*.so"):
        path.unlink()

# The binding's Rust crate is now panelbt_ext; remove stale artifacts of the
# old _core crate, including Cargo's deps directory, without cleaning other crates.
for profile in ("debug", "release"):
    directory = repo / "target" / profile
    for pattern in ("lib_core.*", "lib_core-*"):
        for path in directory.rglob(pattern):
            if path.is_file():
                path.unlink()
