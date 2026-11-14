"""双时间事件账本。修订新增记录，不回写旧版的替代指针。"""

import json
import math
from pathlib import Path

from ..store import atomic_json, lock
from .storage import digest, encoded
from .timeutil import millis

KINDS = {"delist", "settlement", "delivery", "suspend", "resume", "migration"}


def normalize_event(event):
    e = dict(event)
    for name in (
        "event_id",
        "instrument",
        "kind",
        "source_url",
        "original_text",
        "known_at",
        "effective_at",
    ):
        if name not in e or e[name] in (None, ""):
            raise ValueError(f"事件缺字段 {name}")
    if e["kind"] not in KINDS:
        raise ValueError("未知事件类型")
    e["known_at"], e["effective_at"] = millis(e["known_at"]), millis(e["effective_at"])
    e.setdefault("status", "active")
    if e["status"] not in ("active", "cancelled"):
        raise ValueError("未知事件状态")
    e.setdefault("version_unknown", False)
    e.setdefault("manual", False)
    e.setdefault("supersedes", None)
    e.setdefault("settlement_price", None)
    if e["settlement_price"] is not None:
        if (
            not math.isfinite(float(e["settlement_price"]))
            or float(e["settlement_price"]) <= 0
        ):
            raise ValueError("结算价必须为正")
    if e["kind"] == "migration":
        if (
            not e.get("replacement")
            or not math.isfinite(e.get("ratio", 0))
            or e["ratio"] <= 0
        ):
            raise ValueError("迁移事件需要替代币及正转换比例")
    supplied = e.pop("version_id", None)
    version = digest(encoded(e))
    if supplied is not None and supplied != version:
        raise ValueError("事件内容与 version_id 不符")
    return dict(e, version_id=version)


def validate_events(events):
    seen, times = {}, set()
    result = sorted(
        (normalize_event(e) for e in events),
        key=lambda e: (e["known_at"], e["version_id"]),
    )
    for e in result:
        key = (e["event_id"], e["known_at"])
        if key in times:
            raise ValueError("同一事件同一 known_at 不能有两个版本")
        times.add(key)
        prior = e["supersedes"]
        if prior:
            if (
                prior not in seen
                or seen[prior]["event_id"] != e["event_id"]
                or seen[prior]["known_at"] >= e["known_at"]
            ):
                raise ValueError("替代关系必须指向同事件的较早版本")
        elif any(p["event_id"] == e["event_id"] for p in seen.values()):
            raise ValueError("后续版本必须声明 supersedes")
        seen[e["version_id"]] = e
    return result


def asof(events, at, *, instrument=None):
    latest = {}
    for e in events:
        if e["known_at"] <= millis(at) and (
            instrument is None or e["instrument"] == instrument
        ):
            old = latest.get(e["event_id"])
            if old is None or e["known_at"] > old["known_at"]:
                latest[e["event_id"]] = e
    return sorted(
        (e for e in latest.values() if e["status"] == "active"),
        key=lambda e: (e["effective_at"], e["event_id"]),
    )


class EventLedger:
    def __init__(self, root):
        self.root = Path(root) / "events"

    def read(self):
        return validate_events(
            [json.loads(p.read_text()) for p in sorted(self.root.glob("*.json"))]
        )

    def append(self, event):
        e = normalize_event(event)
        self.root.mkdir(parents=True, exist_ok=True)
        with lock(self.root / ".lock"):
            old = self.read()
            if e["version_id"] not in {x["version_id"] for x in old}:
                validate_events([*old, e])
                atomic_json(self.root / f"{e['version_id']}.json", e)
        return e
