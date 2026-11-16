"""独立校验 Table Schema、Arrow 类型和可知时点。"""

import re

import pyarrow as pa

from .constants import (
    ACTION_FIELDS,
    ASSET_UNITS,
    CURRENCIES,
    FIELDS,
    FREQUENCIES,
    INSTRUMENT_FIELDS,
    INTRADAY,
    KNOWN_AT,
    NAME,
    PRICE_UNITS,
    QUANTITIES,
    STRUCTURAL,
    TYPES,
    UNITS,
)


def time_dependencies(resource, report):
    """独立遍历声明；不使用读取侧的依赖检查或规范化函数。"""
    fields = {f["name"]: f for f in resource["schema"]["fields"]}
    referenced = {"available_at"} if "available_at" in fields else set()
    for field in fields.values():
        declaration = field.get("knownAt")
        if isinstance(declaration, str) and declaration.startswith("column:"):
            referenced.add(declaration.removeprefix("column:"))
    for target in sorted(referenced):
        declaration = fields.get(target, {}).get("knownAt")
        if declaration not in ("start", "end"):
            report.fail(
                "known_at_dependency",
                value=dict(table=resource["name"], field=target, knownAt=declaration),
            )


def declarations(manifest, resource, report):
    fields = {f["name"]: f for f in resource["schema"]["fields"]}
    time_dependencies(resource, report)
    kind, name = resource["kind"], resource["name"]
    key = resource["schema"].get("primaryKey")
    time = (
        "ex_date" if kind == "corporate_actions" else "ts" if "ts" in fields else "date"
    )
    temporal = kind in {"bars", "events", "corporate_actions"}
    if temporal:
        allowed_times = {"ex_date"} if kind == "corporate_actions" else {"date", "ts"}
        if len(allowed_times & fields.keys()) != 1 or (
            kind == "corporate_actions" and {"date", "ts"} & fields.keys()
        ):
            report.fail("time_column", value=name)
        if "symbol" not in fields:
            report.fail("symbols", value=f"{name}: 缺少 symbol")
    if kind == "bars":
        if key != ["symbol", time]:
            report.fail("primary_key", value=f"{name}: 行情主键必须是 symbol+时间")
        if (
            not isinstance(resource.get("frequency"), str)
            or resource.get("frequency") not in FREQUENCIES
        ):
            report.fail("frequency", value=name)
        needs_ts = (
            str(resource.get("frequency")) in INTRADAY
            or manifest.get("calendar", {}).get("name") == "24x7"
        )
        if needs_ts != (time == "ts"):
            report.fail("time_column", value=f"{name}: 频率/市场与时间列不符")
    if kind == "calendar" and (set(fields) != {"date"} or key != ["date"]):
        report.fail("calendar_schema", value=name)
    if kind == "instruments" and (
        not {"symbol", "asset_type", "currency"} <= fields.keys() or key != ["symbol"]
    ):
        report.fail("instruments_schema", value=name)
    if kind == "corporate_actions" and not ACTION_FIELDS.keys() <= fields.keys():
        report.fail("actions_schema", value=name)
    if kind in {"events", "corporate_actions"} and "available_at" not in fields:
        report.fail("available_at", value=f"{name}: 缺少 available_at")
    for field, desc in fields.items():
        declared_type = desc.get("type")
        if not isinstance(declared_type, str) or declared_type not in TYPES:
            report.fail("column_types", value=f"{name}.{field}: {declared_type}")
        standard = dict(STRUCTURAL)
        if kind == "instruments":
            standard.update(INSTRUMENT_FIELDS)
        if kind == "corporate_actions":
            standard.update(ACTION_FIELDS)
        expected_type = standard.get(field)
        if kind == "bars" and field in FIELDS:
            expected_type = FIELDS[field][0]
            if declared_type != expected_type:
                report.fail("dictionary", value=f"{name}.{field}: 改写词典类型")
        elif field not in standard:
            if not re.fullmatch(NAME, field) or any(
                not isinstance(desc.get(k), str) or not desc[k].strip()
                for k in ("unit", "knownAt", "description")
            ):
                report.fail("extension", value=f"{name}.{field}")
        if expected_type and declared_type != expected_type:
            report.fail("column_types", value=f"{name}.{field}: 应是 {expected_type}")
        unit = desc.get("unit")
        if "unit" in desc and (not isinstance(unit, str) or unit not in UNITS):
            report.fail("units", value=f"{name}.{field}: {unit}")
        unit_type = (
            "number"
            if isinstance(unit, str) and unit in PRICE_UNITS | CURRENCIES | QUANTITIES
            else "integer"
            if unit == "count"
            else None
        )
        if unit_type is not None and declared_type != unit_type:
            report.fail(
                "column_types", value=f"{name}.{field}: 单位 {unit} 要求 {unit_type}"
            )
        known = desc.get(
            "knownAt", FIELDS[field][2] if kind == "bars" and field in FIELDS else None
        )
        if ("knownAt" in desc or known is not None) and (
            not isinstance(known, str) or not re.fullmatch(KNOWN_AT, known)
        ):
            report.fail("known_at", value=f"{name}.{field}: {known}")
        elif known == "unknown":
            report.fail(
                "known_at_unknown",
                value=dict(table=name, field=field, knownAt=known),
                warning=True,
            )
        elif known and known.startswith("column:"):
            target = fields.get(known[7:], {})
            if target.get("type") != "datetime":
                report.fail(
                    "known_at", value=f"{name}.{field}: 可知时点引用的列必须是 datetime"
                )
        if (
            kind in {"events", "corporate_actions"}
            and field != "available_at"
            and known != "column:available_at"
        ):
            report.fail(
                "known_at",
                value=f"{name}.{field}: 非行情时间序列必须 column:available_at",
            )
        if "allowNegative" in desc and (
            type(desc["allowNegative"]) is not bool
            or not (
                isinstance(unit, str)
                and unit in PRICE_UNITS
                or (kind == "bars" and field in FIELDS and FIELDS[field][1] == "price")
            )
        ):
            report.fail("dictionary", value=f"{name}.{field}: allowNegative 只用于价格")
    return time


def arrow_schema(resource, arrow, report, file):
    fields = {f["name"]: f for f in resource["schema"]["fields"]}
    if arrow.names != list(fields):
        report.fail(
            "columns", file, value=dict(declared=list(fields), actual=arrow.names)
        )
    basic = {
        "date": pa.date32(),
        "number": pa.float64(),
        "integer": pa.int64(),
        "boolean": pa.bool_(),
    }
    for field in arrow:
        dtype = field.type
        if field.name not in fields:
            continue
        expected = fields[field.name].get("type")
        if expected == "datetime":
            ok = (
                pa.types.is_timestamp(dtype)
                and dtype.unit in {"ms", "us"}
                and dtype.tz == "UTC"
            )  # mutation: timezone
        elif expected == "string":
            ok = (
                pa.types.is_string(dtype)
                or pa.types.is_large_string(dtype)
                or pa.types.is_string_view(dtype)
            )
        else:
            ok = dtype == (
                pa.uint32()
                if field.name == "quality" and expected == "integer"
                else basic.get(expected)
            )
        if not ok:
            report.fail(
                "column_types", file, value=f"{field.name}: {dtype} != {expected}"
            )
        if field.name in {"date", "ex_date", "ts", "available_at"}:
            valid = (
                (dtype == pa.date32())
                if field.name in {"date", "ex_date"}
                else (
                    pa.types.is_timestamp(dtype)
                    and dtype.unit in {"ms", "us"}
                    and dtype.tz == "UTC"
                )
            )
            if not valid:  # mutation: epoch
                report.fail("time_type", file, value=f"{field.name}: {dtype}")


def dictionary_units(resource, instruments, report):
    """按资产和报价货币独立推导词典单位，不接受更换含义的显式 unit。"""
    if resource["kind"] != "bars":
        return
    for field in resource["schema"]["fields"]:
        name = field["name"]
        if name not in FIELDS or not isinstance(field.get("unit"), str):
            continue
        category = FIELDS[name][1]
        expected = {category} if category in {"1", "ratio", "count"} else set()
        if (
            category not in {"1", "ratio", "count"}
            and instruments is not None
            and {"currency", "asset_type"} <= set(instruments.columns)
        ):
            for currency, asset in (
                instruments.select("currency", "asset_type").unique().iter_rows()
            ):
                quantity = ASSET_UNITS.get(asset)
                expected.add(
                    currency
                    if category == "currency"
                    else quantity
                    if category == "quantity"
                    else f"{currency}/{quantity}"
                )
        if expected and expected != {field["unit"]}:
            report.fail(
                "dictionary",
                value=f"{name}: 单位与品种不符，期待 {sorted(str(x) for x in expected)}",
            )
