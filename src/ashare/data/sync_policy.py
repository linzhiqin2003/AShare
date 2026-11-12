"""分区定稿与刷新策略；刷新失败不损坏已经完成的修订。"""

import calendar
from datetime import UTC, date, datetime, timedelta


def month_final(entry, *, now=None, grace_days=3):
    if entry.get("rows", 0) <= 0:
        return False
    params = entry.get("params", {})
    end = params.get("end_date", "")
    if len(end) != 8:
        return False
    year, month = int(end[:4]), int(end[4:6])
    finish = date(year, month, calendar.monthrange(year, month)[1])
    if end != finish.strftime("%Y%m%d"):
        return False
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"]).date()
    except (KeyError, ValueError):
        return False
    return fetched >= finish + timedelta(days=grace_days)


def should_refresh(spec, partition, entry, days, end, *, now=None):
    now = now or datetime.now(UTC).date()
    kind = spec.partition
    if kind == "legacy_industry":
        return False  # 冻结的旧分类仅显式强制刷新。
    if kind in ("static", "status", "classification", "industry"):
        return True
    if kind in ("index_month", "index_month_daily"):
        return not month_final(entry)
    if kind == "report":
        # 保守保留最近 190 天的披露/修订窗口，最多覆盖三个季度；不重抓更早历史。
        cutoff = now - timedelta(days=190)
        return partition >= cutoff.strftime("%Y%m%d")
    if kind == "year":
        return partition == end[:4]
    recent = set(days[-spec.refresh :]) if spec.refresh > 0 else set()
    return partition in recent
