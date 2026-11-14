"""独立清单与内容寻址原始库；旧版本从不覆盖。"""

import hashlib
import json
from pathlib import Path

from ..manifest import Manifest
from ..store import atomic_json


def encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class RawStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = Manifest(self.root)

    def close(self):
        self.manifest.db.close()

    def put(self, key, payload, **meta):
        sha = digest(payload)
        path = self.root / "objects" / sha
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or digest(path.read_bytes()) != sha:
            tmp = path.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(path)
        previous = self.manifest.read().get(key, {})
        revisions = previous.get("revisions", [])
        entry = dict(sha256=sha, bytes=len(payload), **meta)
        if entry not in revisions:
            revisions = [*revisions, entry]
        return self.manifest.put(
            key, dict(**entry, status="complete", revisions=revisions)
        )

    def get(self, key, *, sha=None):
        entry = self.manifest.read()[key]
        sha = sha or entry["sha256"]
        payload = (self.root / "objects" / sha).read_bytes()
        if digest(payload) != sha:
            raise ValueError(f"本地原始数据损坏: {key}")
        return payload

    def complete(self, key):
        entry = self.manifest.read().get(key, {})
        if entry.get("status") != "complete":
            return False
        try:
            self.get(key)
            return True
        except (OSError, ValueError):
            return False

    def freeze(self):
        entries = {
            k: {n: v for n, v in e.items() if n != "revisions"}
            for k, e in self.manifest.read().items()
            if e.get("status") == "complete"
        }
        version = digest(encoded(entries))
        atomic_json(self.root / "raw_versions" / f"{version}.json", entries)
        return version
