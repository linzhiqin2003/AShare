"""清单读取、显式默认值和第 7 节的规范 JSON。校验器不导入本模块。"""

import copy
import hashlib
import json
from pathlib import Path

from .constants import FIELDS


def canonical(manifest):
    return json.dumps(
        {k: v for k, v in manifest.items() if k not in {"version", "created"}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def version(manifest):
    return hashlib.sha256(canonical(manifest)).hexdigest()


def load(root):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"重复 JSON 键：{k}")
            result[k] = v
        return result

    return json.loads(
        (Path(root) / "datapackage.json").read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
    )


def normalized(manifest):
    """给展示/读取补词典默认值；版本始终按原始声明的规范 JSON 计算。"""
    result = copy.deepcopy(manifest)
    for resource in result["resources"]:
        if resource["kind"] == "bars":
            for field in resource["schema"]["fields"]:
                if field["name"] in FIELDS:
                    field.setdefault("knownAt", FIELDS[field["name"]][2])
                    if FIELDS[field["name"]][1] in {"1", "ratio", "count"}:
                        field.setdefault("unit", FIELDS[field["name"]][1])
    return result
