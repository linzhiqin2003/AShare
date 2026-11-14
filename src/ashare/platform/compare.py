"""因子对比：只读结果库里已存的结果，不重跑回测。

验证期（2025）数值只在调用方显式要求时返回；默认响应里不出现任何验证期数字，
图表也只到样本内末，避免看着验证期调参。
"""

import json
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from .analysis import packed

LOG = logging.getLogger("ashare.platform.compare")
MAX_ITEMS = 8
METRICS = (
    "sharpe",
    "return",
    "turnover",
    "maxdrawdown",
    "margin",
    "fitness",
    "t",
    "ic",
    "coverage",
    "holding_days",
)
YEAR_METRICS = ("period", "sharpe", "return", "maxdrawdown")

router = APIRouter()


def _metrics(stats):
    """只传页面画得到的字段；不传持仓、分季明细等大块内容。"""
    if not stats:
        return None
    return {
        **{key: stats.get(key) for key in METRICS},
        "years": [
            {key: year.get(key) for key in YEAR_METRICS}
            for year in stats.get("years") or []
        ],
        "exposure": stats.get("exposure") or {},
        "subs": {
            name: {"sharpe": sub.get("sharpe")}
            for name, sub in (stats.get("subs") or {}).items()
        },
    }


def _parse_ids(ids):
    values = [value for value in ids.split(",") if value]
    if not values or len(values) > MAX_ITEMS or len(set(values)) != len(values):
        raise ValueError(f"对比需要 1–{MAX_ITEMS} 个不同因子")
    return values


def compare(store, ids, *, validation=False):
    """读取若干条目的最新结果，按共同日期轴对齐多空日收益。"""
    with store.connect() as db:
        rows = db.execute(
            "SELECT e.id,e.name,e.source,e.settings,e.status,e.latest,r.summary FROM entries e"
            " LEFT JOIN runs r ON r.id=e.latest AND r.status='complete'"
            " WHERE e.id IN (" + ",".join("?" for _ in ids) + ")",
            ids,
        ).fetchall()
    found = {row["id"]: row for row in rows}
    missing = [value for value in ids if value not in found]
    if missing:
        raise HTTPException(404, "没有找到这个因子")
    parts = ("is", "validation") if validation else ("is",)
    items, pnl = [], {}
    for entry in ids:
        row = found[entry]
        summary = json.loads(row["summary"]) if row["summary"] else None
        item = {
            "id": row["id"],
            "name": row["name"],
            "source": row["source"],
            "settings": json.loads(row["settings"]),
            "status": row["status"],
            "latest": row["latest"] if summary else None,
            "is": _metrics(summary and summary.get("is")),
            "correlation": (
                {
                    "name": summary["correlations"][0]["name"],
                    "value": summary["correlations"][0]["value"],
                }
                if summary and summary.get("correlations")
                else None
            ),
        }
        if validation:  # 关时连键都不出现
            item["validation"] = _metrics(summary and summary.get("validation"))
        items.append(item)
        if not summary:
            continue
        # latest 来自数据库，不接受用户提供的路径。
        with np.load(
            store.root / "series" / f"{row['latest']}.npz", allow_pickle=False
        ) as file:
            series = {}
            for part in parts:
                if f"{part}.dates" not in file:
                    continue
                for date, value in zip(
                    file[f"{part}.dates"].tolist(),
                    file[f"{part}.pnl.long_short"].tolist(),
                    strict=True,
                ):
                    series[str(date)] = value
            pnl[entry] = series
    dates = sorted({date for series in pnl.values() for date in series})
    LOG.debug("对比 %d 个因子 validation=%s 日期 %d", len(ids), validation, len(dates))
    return {
        "items": items,
        "dates": dates,
        "series": {
            entry: packed(
                np.array([series.get(date, np.nan) for date in dates], dtype=float)
            )
            for entry, series in pnl.items()
        },
        "correlation": store.correlation_matrix(ids)["matrix"],
    }


@router.get("/api/compare")
def compare_route(
    request: Request,
    ids: str = Query(..., max_length=800),
    validation: bool = False,
):
    return compare(request.app.state.store, _parse_ids(ids), validation=validation)
