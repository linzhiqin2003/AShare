"""Verify a built wheel has portable bytes and correct RECORD digests."""

import base64
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
wheel = next((root / "dist").glob("panelbt-*-abi3-*.whl"))
with zipfile.ZipFile(wheel) as archive:
    paths = archive.namelist()
    forbidden = (str(Path.home()).encode(), str(root).encode())
    leaks = [
        name
        for name in paths
        if any(value in archive.read(name) for value in forbidden)
    ]
    assert not leaks, leaks
    record = next(name for name in paths if name.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
    verified = 0
    for name, digest, size in rows:
        if name == record:
            assert digest == size == ""
            continue
        payload = archive.read(name)
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode()
        )
        assert digest == "sha256=" + actual and int(size) == len(payload), name
        verified += 1
    assert set(paths) == {row[0] for row in rows}
sdist = next((root / "dist").glob("panelbt-*.tar.gz"))
with tarfile.open(sdist) as archive:
    members = archive.getmembers()
    sdist_leaks = []
    for member in members:
        assert not member.name.startswith("/") and ".." not in Path(member.name).parts
        if member.isfile():
            payload = archive.extractfile(member).read()
            if any(value in payload for value in forbidden):
                sdist_leaks.append(member.name)
    assert not sdist_leaks, sdist_leaks
    assert any("crates/tradecore/src/lib.rs" in member.name for member in members)
    assert any("crates/panelbt-core/src/lib.rs" in member.name for member in members)
report = {
    "wheel": wheel.name,
    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "local_path_leaks": leaks,
    "sdist": sdist.name,
    "sdist_local_path_leaks": sdist_leaks,
    "verified_record_entries": verified,
}
(root / "docs/step1b/wheel-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
