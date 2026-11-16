"""D2 合规基线；测试签名器独立于产品 seal，避免成对实现互相掩盖错误。"""

import hashlib
import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from ashare.data.spec import check
from docs.examples.generate_datapackage import SyntheticAdapter


def read_manifest(root):
    return json.loads((root / "datapackage.json").read_text())


def resign(root, manifest=None):
    manifest = manifest or read_manifest(root)
    for r in manifest["resources"]:
        files, payload = [], bytearray()
        for path in sorted((root / r["name"]).glob("*.parquet")):
            data = path.read_bytes()
            payload.extend(data)
            files.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    bytes=len(data),
                    hash="sha256:" + hashlib.sha256(data).hexdigest(),
                    rows=pq.read_metadata(path).num_rows,
                )
            )
        r.update(
            path=[f["path"] for f in files],
            files=files,
            bytes=len(payload),
            rows=sum(f["rows"] for f in files),
            hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
    write_manifest(root, manifest)
    return manifest


def write_manifest(root, manifest, *, version=True):
    if version:
        data = {k: v for k, v in manifest.items() if k not in {"version", "created"}}
        manifest["version"] = hashlib.sha256(
            json.dumps(
                data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
    manifest.setdefault("created", datetime.now(UTC).isoformat())
    (root / "datapackage.json").write_text(json.dumps(manifest, ensure_ascii=False))


def item(report, code):
    return next(x for x in report["checks"] if x["code"] == code)


def assert_good(root):
    report = check(root)
    assert report["exit_code"] == 0, [
        x for x in report["checks"] if x["status"] == "fail"
    ]


def assert_bad(root, code, exit_code=1):
    report = check(root)
    assert report["exit_code"] == exit_code, report
    assert item(report, code)["status"] == "fail", report
    assert 0 < len(item(report, code)["examples"]) <= 20
    return report


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    adapter = SyntheticAdapter()
    for table, partition in adapter.partitions():
        folder = root / table
        folder.mkdir(parents=True, exist_ok=True)
        adapter.read(table, partition).write_parquet(
            folder / f"{partition}.parquet", compression="zstd"
        )
    resign(root, adapter.describe())
    assert_good(root)
    return root
