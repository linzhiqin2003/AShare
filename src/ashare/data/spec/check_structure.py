"""独立的包结构校验；不依赖 manifest、seal、build 或 reader。"""

import hashlib
import json
import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .constants import KINDS, PARTITION, RESOURCE_NAME, SPEC_VERSION, SYMBOLOGY


def descriptor(root, report):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"重复键 {k}")
            result[k] = v
        return result

    def invalid(value):
        raise ValueError(f"非法 JSON 常量 {value}")

    try:
        manifest = json.loads(
            (root / "datapackage.json").read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid,
        )
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("resources"), list)
            or not manifest["resources"]
        ):
            raise ValueError("清单必须含非空 resources 数组")
        names = set()
        for resource in manifest["resources"]:
            if not isinstance(resource, dict):
                raise ValueError("resource 必须是对象")
            name = resource.get("name")
            if (
                not isinstance(name, str)
                or not re.fullmatch(RESOURCE_NAME, name)
                or name in names
            ):
                raise ValueError("表名非法或重复")
            names.add(name)
            if resource.get("kind") not in KINDS:
                raise ValueError("kind 非法")
            schema = resource.get("schema", {})
            fields = schema.get("fields", [])
            if (
                not fields
                or not isinstance(fields, list)
                or any(
                    not isinstance(f, dict) or not isinstance(f.get("name"), str)
                    for f in fields
                )
            ):
                raise ValueError("fields 必须含字段描述")
            field_names = [f["name"] for f in fields]
            if len(field_names) != len(set(field_names)):
                raise ValueError("重复字段名")
            key = schema.get("primaryKey")
            if (
                not isinstance(key, list)
                or not key
                or any(not isinstance(k, str) for k in key)
                or len(key) != len(set(key))
                or set(key) - set(field_names)
            ):
                report.fail("primary_key", value=f"{name}: 非法主键声明")
            paths, files = resource.get("path"), resource.get("files")
            if (
                not isinstance(paths, list)
                or not paths
                or not isinstance(files, list)
                or len(paths) != len(files)
            ):
                raise ValueError("每张表必须有 path 列表和逐文件 files")
            if len(set(paths)) != len(paths):
                raise ValueError("同一文件重复登记")
            for path, entry in zip(paths, files, strict=True):
                if not isinstance(path, str) or not re.fullmatch(
                    re.escape(name) + "/" + PARTITION + r"\.parquet", path
                ):
                    report.fail("paths", value=path)
                    report.fatal = True
                    continue
                if (
                    (root / name).is_symlink()
                    or (root / path).is_symlink()
                    or not (root / path).is_file()
                ):
                    report.fail(
                        "paths", file=path, value="文件缺失、符号链接或非普通文件"
                    )
                    report.fatal = True
                if (
                    not isinstance(entry, dict)
                    or entry.get("path") != path
                    or any(
                        type(entry.get(k)) is not int or entry[k] < 0
                        for k in ("bytes", "rows")
                    )
                    or not isinstance(entry.get("hash"), str)
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", entry["hash"])
                ):
                    raise ValueError("逐文件 path/bytes/hash/rows 非法")
        if (
            manifest.get("specVersion") != SPEC_VERSION
            or manifest.get("symbology") not in SYMBOLOGY
        ):
            report.fail("manifest", value="specVersion 或 symbology 不支持")
        if not isinstance(manifest.get("name"), str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", manifest["name"]
        ):
            report.fail("manifest", value="name 非法")
        for key in ("title", "source"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                report.fail("manifest", value=f"缺少 {key}")
        licenses = manifest.get("licenses")
        if (
            not isinstance(licenses, list)
            or not licenses
            or any(
                not isinstance(x, dict)
                or not any(isinstance(x.get(k), str) and x[k] for k in ("name", "path"))
                for x in licenses
            )
        ):
            report.fail("manifest", value="licenses 非法")
        try:
            created = datetime.fromisoformat(manifest["created"].replace("Z", "+00:00"))
            if created.tzinfo is None:
                raise ValueError("created 无时区")
        except (KeyError, ValueError, TypeError, AttributeError):
            report.fail("manifest", value="created 必须是带时区的 RFC3339 时间")
        try:
            calendar = manifest["calendar"]
            ZoneInfo(calendar["tz"])
            if not isinstance(calendar["name"], str) or not calendar["name"]:
                raise ValueError("日历名为空")
            if calendar["name"] != "24x7":
                opening, closing = (
                    time.fromisoformat(calendar[k])
                    for k in ("sessionOpen", "sessionClose")
                )
                if opening >= closing:
                    raise ValueError("不支持跨午夜日盘；夜盘归属交易日")
        except (KeyError, ValueError, TypeError):
            report.fail("calendar_schema", value="日历、IANA 时区或日盘时段非法")
        body = {k: v for k, v in manifest.items() if k not in ("version", "created")}
        encoded = json.dumps(
            body,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(encoded).hexdigest()
        if manifest.get("version") != expected:  # mutation: version
            report.fail("version", value="清单版本不匹配")
        registered = {"datapackage.json"} | {
            p for r in manifest["resources"] for p in r["path"]
        }
        for path in root.rglob("*"):
            if path.is_symlink() or (
                not path.is_dir()
                and path.relative_to(root).as_posix() not in registered
            ):
                report.fail(
                    "registered_files",
                    file=path.relative_to(root).as_posix(),
                    value="未登记文件或符号链接",
                )
        return manifest
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report.fail("manifest", value=str(exc))
        report.fatal = True
        return None
