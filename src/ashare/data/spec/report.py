"""校验报告：稳定错误代码、逐项结果、最多 20 个定位样例。"""

from datetime import date, datetime

# 按 DATA_SPEC 第 8 节的检查顺序排列。
CHECKS = {
    "manifest": "2,11",
    "paths": "2",
    "registered_files": "8.1",
    "file_bytes": "7",
    "file_hash": "7",
    "file_rows": "7",
    "resource_integrity": "7",
    "version": "7",
    "columns": "2",
    "column_types": "2,5.3",
    "time_type": "3.1",
    "time_column": "3.1",
    "frequency": "3.2",
    "units": "5.2",
    "known_at": "3.3",
    "known_at_dependency": "3.3",
    "known_at_unknown": "3.3",
    "dictionary": "5.1",
    "extension": "5.4",
    "calendar_schema": "3.4",
    "instruments_schema": "4",
    "actions_schema": "5.5",
    "compression": "2",
    "symbols": "4",
    "primary_key": "6",
    "sorted": "6",
    "nan": "6",
    "finite": "6",
    "missing": "6",
    "prices": "6",
    "nonnegative": "6",
    "partition_range": "2",
    "calendar_membership": "8.4",
    "lifecycle": "4",
    "available_at": "3.3",
    "cross_partition": "6,8.4",
    "gaps": "6,8.4",
    "quality": "6",
    "actions_consistency": "8.5",
    "price_jumps": "8.5",
}


def safe(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return str(value)
    return value


class Report:
    def __init__(self):
        self.items = {
            code: dict(code=code, section=section, status="pass", count=0, examples=[])
            for code, section in CHECKS.items()
        }
        self.fatal = False

    def fail(
        self,
        code,
        file="datapackage.json",
        row=None,
        value=None,
        *,
        count=1,
        warning=False,
    ):
        item = self.items[code]
        item["status"] = "warning" if warning else "fail"
        item["count"] += count
        if len(item["examples"]) < 20:
            item["examples"].append(dict(file=file, row=row, value=safe(value)))

    def frame(self, code, frame, mask, file, *, warning=False):
        bad = frame.filter(mask.fill_null(False))
        if bad.height:
            for index, row in enumerate(bad.head(20).iter_rows(named=True)):
                self.fail(
                    code,
                    file,
                    row.pop("_row", None),
                    row,
                    count=bad.height if index == 0 else 0,
                    warning=warning,
                )

    def result(self):
        failed = any(x["status"] == "fail" for x in self.items.values())
        code = 2 if self.fatal else int(failed)
        return dict(ok=code == 0, exit_code=code, checks=list(self.items.values()))
