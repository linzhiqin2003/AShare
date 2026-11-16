"""读取侧拒绝不支持的时间依赖；普通未知值的元数据不做引用链传播。"""

from types import MappingProxyType


def validate_time_dependencies(fields):
    """整张表的声明约束，不能通过投影、原始读取或 allow_unknown 绕过。"""
    targets = {
        known[7:]
        for desc in fields.values()
        if isinstance(known := desc.get("knownAt"), str) and known.startswith("column:")
    }
    targets |= {"available_at"} & fields.keys()
    for name in sorted(targets):
        desc = fields.get(name, {})
        if desc.get("type") != "datetime" or desc.get("knownAt") not in (
            "start",
            "end",
        ):
            raise ValueError(
                f"known_at_dependency: 时间依赖 {name} 必须是随行可知的 datetime，"
                f"knownAt 必须为 start 或 end，实际为 {desc.get('knownAt')!r}"
            )


def unknown_sources(fields, selected):
    return sorted(name for name in selected if fields[name].get("knownAt") == "unknown")


def require_known(fields, selected, allow_unknown):
    unknown = unknown_sources(fields, selected)
    if unknown and not allow_unknown:
        raise ValueError(
            f"字段 {unknown} 的 knownAt=unknown；显式 allow_unknown=True 才可使用"
        )


def marked_metadata(desc, fields, selected):
    result = dict(desc)
    unknown = unknown_sources(fields, selected)
    if unknown:
        result["knownAt"] = "unknown"
        result["unknownDependencies"] = tuple(unknown)
    return result


def readonly_metadata(fields, selected):
    return MappingProxyType(
        {
            name: MappingProxyType(marked_metadata(fields[name], fields, [name]))
            for name in selected
        }
    )
