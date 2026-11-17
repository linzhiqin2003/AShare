"""按分区与列裁剪的读取入口；每个被读文件默认核对 SHA-256。"""

import hashlib
import os
import re
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ashare.paths import dataset_dir

from .constants import ASSET_UNITS, FIELDS
from .dependencies import validate_time_dependencies
from .manifest import load, normalized, version
from .scan import verified_scan
from .timing import declarations

_VERIFIED = {}
_CACHE_LOCK = threading.Lock()
# 文件系统的时间戳按时钟节拍取（Linux ext4 毫秒级，FAT 两秒）：校验后同一节拍内原地改写、
# 再改回 mtime，身份五元组完全不变。所以只缓存改动早于校验开始 2 秒以上的文件，较新的
# 每次都重算哈希（git 的 racily clean 规则）。
_SETTLED_NS = 2 * 10**9
_clock = time.time_ns


def _identity(stat):
    return (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino, stat.st_ctime_ns)


def _hash_file(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verify(path, entry, stream=None):
    if stream is None:
        with path.open("rb") as opened:
            return _verify(path, entry, opened)
    stat = os.fstat(stream.fileno())
    key = (str(path.resolve()), *_identity(stat))
    if stat.st_size != entry["bytes"]:
        raise ValueError(f"文件大小不匹配：{path.name}")
    with _CACHE_LOCK:
        digest = _VERIFIED.get(key)
    payload = None
    if digest is None:
        started = _clock()
        stream.seek(0)
        payload = stream.read()
        digest = _hash_file(payload)
        if _identity(os.fstat(stream.fileno())) != key[1:]:
            raise ValueError("文件在校验中改变")
        settled = max(stat.st_mtime_ns, stat.st_ctime_ns) < started - _SETTLED_NS
        if digest == entry["hash"] and settled:
            with _CACHE_LOCK:
                if len(_VERIFIED) >= 4096:
                    _VERIFIED.pop(next(iter(_VERIFIED)))
                _VERIFIED[key] = digest
    if digest != entry["hash"]:
        raise ValueError(f"文件哈希不匹配：{path.name}")
    return payload


def boundary(value, daily):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raise TypeError("时间边界不得使用整数 epoch")
    text = str(value)
    if daily:
        return (
            date.fromisoformat(text[:10])
            if not re.fullmatch(r"\d{8}", text)
            else datetime.strptime(text, "%Y%m%d").date()
        )
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(text.replace("Z", "+00:00"))
    )
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _schema(resource):
    mapping = {
        "date": pl.Date,
        "datetime": pl.Datetime("us", "UTC"),
        "number": pl.Float64,
        "integer": pl.Int64,
        "boolean": pl.Boolean,
        "string": pl.String,
    }
    return {
        f["name"]: pl.UInt32 if f["name"] == "quality" else mapping[f["type"]]
        for f in resource["schema"]["fields"]
    }


class Package:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = load(self.root)
        if version(self.manifest) != self.manifest.get("version"):
            raise ValueError("清单版本不匹配")
        self.version = self.manifest["version"]
        self.resources = {r["name"]: r for r in self.manifest["resources"]}
        for resource in self.resources.values():
            for entry in resource["files"]:
                path = self.root / entry["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(self.root.resolve())
                    or not re.fullmatch(
                        re.escape(resource["name"])
                        + r"/(?:\d{4}(?:-\d{2})?|all)\.parquet",
                        entry["path"],
                    )
                ):
                    raise ValueError("不安全的文件路径")

    def _scan(
        self,
        table,
        start=None,
        end=None,
        symbols=None,
        fields=None,
        as_of=None,
        *,
        allow_unknown=False,
    ):
        resource = self.resources[table]
        validate_time_dependencies(declarations(resource))
        schema = _schema(resource)
        time_col = (
            "ts"
            if "ts" in schema
            else "date"
            if "date" in schema
            else "ex_date"
            if "ex_date" in schema
            else None
        )
        daily = time_col != "ts"
        first, last = boundary(start, daily), boundary(end, daily)
        if first is not None and last is not None and first > last:
            raise ValueError("start 必须不晚于 end")
        instant = boundary(as_of, False) if as_of is not None else None
        if instant is not None and resource["kind"] == "bars":
            cutoff = (
                instant.astimezone(ZoneInfo(self.manifest["calendar"]["tz"])).date()
                if daily
                else instant + timedelta(microseconds=1)
            )
            last = min(last, cutoff) if last is not None else cutoff
        if time_col is None and (start is not None or end is not None):
            raise ValueError("静态表不接受日期范围")
        selected = (
            list(schema)
            if fields is None
            else list(
                dict.fromkeys(
                    [
                        *resource["schema"]["primaryKey"],
                        *([fields] if isinstance(fields, str) else fields),
                    ]
                )
            )
        )
        if set(selected) - schema.keys():
            raise KeyError(f"未知字段：{sorted(set(selected) - schema.keys())}")
        entries = []
        for entry in resource["files"]:
            path = self.root / entry["path"]
            label = path.stem
            if time_col and label != "all":
                lo = datetime.strptime(
                    label, "%Y" if len(label) == 4 else "%Y-%m"
                ).replace(tzinfo=UTC)
                hi = (
                    lo.replace(year=lo.year + 1, month=1)
                    if len(label) == 4 or lo.month == 12
                    else lo.replace(month=lo.month + 1)
                )
                lo, hi = (lo.date(), hi.date()) if daily else (lo, hi)
                if (first is not None and first >= hi) or (
                    last is not None and (last < lo if daily else last <= lo)
                ):
                    continue
            entries.append((path, entry))
        frame = verified_scan(entries, schema, _verify, _identity)
        if first is not None:
            frame = frame.filter(pl.col(time_col) >= first)
        if last is not None:
            frame = frame.filter(
                pl.col(time_col) <= last if daily else pl.col(time_col) < last
            )
        if symbols is not None:
            if "symbol" not in schema:
                raise ValueError("该表没有 symbol")
            frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
        if as_of is not None:
            from .timing import as_of_view

            frame = as_of_view(
                frame,
                resource,
                self.manifest,
                selected,
                instant,
                allow_unknown=allow_unknown,
            )
        return frame.select(selected)

    def read(
        self,
        table="bars",
        start=None,
        end=None,
        symbols=None,
        fields=None,
        *,
        as_of=None,
        allow_unknown=False,
    ):
        """返回长表；未传 as_of 是原始行，不是时点视图。"""
        from .dependencies import readonly_metadata

        result = self._scan(
            table, start, end, symbols, fields, as_of, allow_unknown=allow_unknown
        ).collect(engine="streaming")
        result.field_metadata = readonly_metadata(
            declarations(self.resources[table]), result.columns
        )
        return result

    def panel(
        self,
        fields,
        start=None,
        end=None,
        symbols=None,
        table="bars",
        *,
        allow_unknown=False,
    ):
        from .panel import make_panel

        return make_panel(
            self, fields, start, end, symbols, table, allow_unknown=allow_unknown
        )

    def info(self):
        result = {
            k: self.manifest.get(k)
            for k in (
                "name",
                "title",
                "source",
                "version",
                "created",
                "symbology",
                "calendar",
                "gaps",
            )
        }
        instruments = next(
            (r["name"] for r in self.resources.values() if r["kind"] == "instruments"),
            None,
        )
        instrument_frame = (
            self.read(instruments, fields=["currency", "asset_type"])
            if instruments
            else None
        )
        result["resources"] = []
        for resource in normalized(self.manifest)["resources"]:
            time_col = next(
                (k for k in ("ts", "date", "ex_date") if k in _schema(resource)), None
            )
            bounds = None
            if time_col:
                row = (
                    self._scan(resource["name"], fields=[time_col])
                    .select(
                        pl.col(time_col).min().alias("start"),
                        pl.col(time_col).max().alias("end"),
                    )
                    .collect()
                    .row(0)
                )
                bounds = [None if x is None else x.isoformat() for x in row]
            units = []
            if instrument_frame is not None and "symbol" in _schema(resource):
                used = (
                    self._scan(resource["name"], fields=["symbol"])
                    .select("symbol")
                    .unique()
                    .collect()
                )
                units = [
                    (currency, ASSET_UNITS.get(asset))
                    for currency, asset in instrument_frame.join(
                        used, on="symbol", how="semi"
                    )
                    .select("currency", "asset_type")
                    .unique()
                    .iter_rows()
                ]
            fields = resource["schema"]["fields"]
            for field in fields:
                if field["name"] in FIELDS and "unit" not in field:
                    category = FIELDS[field["name"]][1]
                    inferred = {
                        c
                        if category == "currency"
                        else q
                        if category == "quantity"
                        else f"{c}/{q}"
                        if category == "price"
                        else category
                        for c, q in units
                    }
                    field["unit"] = (
                        next(iter(inferred)) if len(inferred) == 1 else sorted(inferred)
                    )
            result["resources"].append(
                dict(
                    name=resource["name"],
                    kind=resource["kind"],
                    range=bounds,
                    rows=resource["rows"],
                    bytes=resource["bytes"],
                    fields=fields,
                )
            )
        return result


def open(name_or_path):
    if (
        isinstance(name_or_path, Path)
        or "/" in str(name_or_path)
        or str(name_or_path).startswith(".")
    ):
        root = Path(name_or_path).expanduser()
    else:
        root = dataset_dir("ext") / str(name_or_path)
    return Package(root)
