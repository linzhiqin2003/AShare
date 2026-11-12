"""Maturin PEP 517 hooks with portable compiler source-path remapping.

Only the build environment changes. The wheel has no dependency on this module.
Caller-provided Rust flags are preserved; local roots become virtual build paths.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shlex
import zipfile
from contextlib import contextmanager
from pathlib import Path

import maturin


@contextmanager
def _portable_paths():
    here = Path(__file__).resolve().parent
    root = here
    for candidate in (here, *here.parents):
        manifest = candidate / "Cargo.toml"
        if manifest.is_file() and "[workspace]" in manifest.read_text():
            root = candidate
            break
    cargo = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).resolve()
    prior = os.environ.get("CARGO_ENCODED_RUSTFLAGS")
    flags = (
        prior.split("\x1f")
        if prior is not None
        else shlex.split(os.environ.get("RUSTFLAGS", ""))
    )
    mappings = (
        (Path.home(), "/build-home"),
        (root, "/panelbt-source"),
        (cargo, "/cargo-source"),
    )
    flags.extend(
        f"--remap-path-prefix={source}={target}" for source, target in mappings
    )
    os.environ["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("CARGO_ENCODED_RUSTFLAGS", None)
        else:
            os.environ["CARGO_ENCODED_RUSTFLAGS"] = prior


def _sanitize_wheel(path):
    """Normalize generated SBOM path identities and regenerate wheel RECORD."""
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    record = next(
        info.filename
        for info, _ in entries
        if info.filename.endswith(".dist-info/RECORD")
    )
    normalized = []
    for info, data in entries:
        if ".dist-info/sboms/" in info.filename and info.filename.endswith(".json"):
            text = data.decode()
            # SBOM cargo package IDs contain local file URLs independently of
            # rustc's source-map flags. Keep dependency identity, normalize roots.
            import re

            text = re.sub(
                r"path\+file://[^\"#]+/(crates|packages)/([^\"#]+)",
                r"path+file:///panelbt-source/\1/\2",
                text,
            )
            text = text.replace(str(Path.home()), "/build-home")
            data = text.encode()
        if info.filename != record:
            normalized.append((info, data))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for info, data in normalized:
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((info.filename, "sha256=" + digest, len(data)))
    writer.writerow((record, "", ""))
    temporary = path.with_suffix(".portable.whl")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info, data in normalized:
            archive.writestr(info, data)
        archive.writestr(record, buffer.getvalue().encode())
    temporary.replace(path)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _portable_paths():
        name = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
        _sanitize_wheel(Path(wheel_directory) / name)
        return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _portable_paths():
        return maturin.build_editable(
            wheel_directory, config_settings, metadata_directory
        )


def build_sdist(sdist_directory, config_settings=None):
    with _portable_paths():
        return maturin.build_sdist(sdist_directory, config_settings)


def __getattr__(name):
    return getattr(maturin, name)
