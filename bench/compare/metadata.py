"""Deduplicate descriptions without changing their per-record meaning."""


def compact(report):
    descriptions = report.setdefault("descriptions", {})
    diagnostics = report.setdefault("case_diagnostics", {})
    for row in report["records"]:
        for name in ("timing_boundary", "rss_scope", "semantics"):
            if name not in row:
                continue
            values = descriptions.setdefault(name, [])
            value = row[name]
            if isinstance(value, str):
                if value not in values:
                    values.append(value)
                row[name] = values.index(value)
            else:
                assert 0 <= value < len(values)
        if "panelbt_diagnostics" in row:
            key = "-".join(str(row[x]) for x in ("task", "T", "N", "interval", "K"))
            value = row.pop("panelbt_diagnostics")
            assert diagnostics.setdefault(key, value) == value
    return report
