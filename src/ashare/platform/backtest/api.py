"""回测页接口：`create_app` 里一行 `include_router` 注册。

只开放两个窗口（样本内、验证期），参数里没有窗口、起止日期；TEST 没有任何路由可达。
页面文档按窗口分开返回，验证期只在页面打开验证结果开关时才请求。

表达式里的候选列表（`ts_std(returns, [20, 60])`）展开成多组，每组一个独立任务，身份与单独
提交那一组相同（已算过的直接复用）；组数上限与研究台一致（`language.MAX_VARIANTS`）。
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ashare.factor.windows import IS, VALIDATION
from ashare.ledger import code_version

from ..benchmarks import Benchmarks
from ..language import MAX_VARIANTS, parse, suggested_name
from . import settings as config
from .jobs import Backtests

Part = Literal["is", "validation"]
Key = Field(pattern=r"^[0-9a-f]{24}$")


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=16000)
    name: str = Field("", max_length=80)
    settings: dict = Field(default_factory=dict)


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[Annotated[str, Key]] = Field(min_length=1, max_length=MAX_VARIANTS)


def _iso(day):
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def router(root, data_root=None, *, idle_seconds=None, default_benchmark=None):
    """default_benchmark：返回设置页默认基准代码的函数（没有时用数据目录的默认指数）。"""
    jobs = Backtests(root, data_root, idle_seconds=idle_seconds)
    benchmarks = Benchmarks(data_root)
    api = APIRouter(prefix="/api/backtests", on_shutdown=[jobs.close])
    api.jobs = jobs  # 测试和关闭时使用
    version = {"at": 0.0, "value": None}

    def data_version():
        if time.monotonic() - version["at"] > 60:
            import ashare.data

            version["value"] = str(ashare.data.open(data_root).panel([]).version)
            version["at"] = time.monotonic()
        return version["value"]

    @lru_cache(maxsize=1)
    def code():
        # 提交号与工作区是否有改动；一次服务只取一次，不在每个请求里跑 git。
        return json.dumps(code_version(Path(__file__).parent.parent), sort_keys=True)

    @lru_cache(maxsize=8)
    def arrays(key, part, stamp):
        # stamp 是文件修改时间：清理后重跑的同一身份不会读到旧缓存。
        raw = jobs.arrays(key, part)
        if raw is None:
            return None
        rebalances = json.loads(bytes(raw.pop("rebalances")).decode())
        return raw, rebalances

    def complete(key):
        state = jobs.status(key)
        if state is None:
            raise HTTPException(404, "没有找到这次回测")
        if state["status"] == "evicted":
            raise HTTPException(410, "结果已清理，请重新运行")
        if state["status"] != "complete":
            raise HTTPException(409, "回测尚未完成")
        jobs.touch(key)
        return state

    def loaded(key, part):
        path = jobs.folder / f"{key}.{part}.npz"
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            raise HTTPException(410, "结果已清理，请重新运行") from None
        result = arrays(key, part, stamp)
        if result is None:
            raise HTTPException(410, "结果已清理，请重新运行")
        return result

    def preferred():
        return default_benchmark() if default_benchmark else None

    @api.get("/options")
    def options():
        selected = benchmarks.selection(preferred())
        return {
            "defaults": config.DEFAULTS
            | ({"benchmark": selected["symbol"]} if selected["symbol"] else {}),
            "choices": config.CHOICES,
            "limits": config.LIMITS,
            "windows": {
                "is": [_iso(IS[0]), _iso(IS[1])],
                "validation": [_iso(VALIDATION[0]), _iso(VALIDATION[1])],
            },
        }

    @api.get("/status")
    def status():
        return {"running": jobs.running}

    @api.post("", status_code=202)
    def submit(body: BacktestRequest):
        # 组数超过上限（与研究台同一口径）时 parse 以 422 拒绝，并指出列表位置。
        variants = parse(body.source)
        try:
            settings = config.normalize(body.settings)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        selected = benchmarks.selection(settings.get("benchmark"))
        if selected["symbol"]:
            settings["benchmark"] = selected["symbol"]
        else:
            settings.pop("benchmark", None)
        name = body.name.strip() or suggested_name(body.source)
        version = data_version()
        groups = {}  # 身份 → (组名, 这一组的表达式)；同一身份的两组只跑一次
        for variant in variants:
            key = config.identity(variant.expression.digest, settings, version)
            # 单个表达式照旧存原文（含注释）；多组存展开后的那一组，工作进程逐个解析。
            text = body.source if len(variants) == 1 else variant.source
            groups.setdefault(key, (" · ".join(variant.parameters), text))
        # 同一批连续入队：工作进程依次计算，面板字段与证券上下文只读一次。
        for key, (_, text) in groups.items():
            jobs.submit(
                key,
                name,
                text,
                settings,
                selected["name"] if selected["symbol"] else None,
                version,
                code(),
            )
        items = [
            jobs.status(key) | {"label": label} for key, (label, _) in groups.items()
        ]
        first = {k: v for k, v in items[0].items() if k != "label"}
        return first | ({"name": name} if len(items) > 1 else {}) | {"variants": items}

    @api.get("/states")
    def states(ids: str = Query(max_length=25 * MAX_VARIANTS)):
        """一次问一批任务的状态；页面轮询整批进度用。"""
        keys = ids.split(",")
        if not 1 <= len(keys) <= MAX_VARIANTS or not all(
            len(k) == 24 and all(c in "0123456789abcdef" for c in k) for k in keys
        ):
            raise HTTPException(422, "请求参数不正确")
        found = (jobs.status(key) for key in keys)
        return {"items": [state for state in found if state is not None]}

    @api.post("/cancel")
    def cancel_many(body: CancelRequest):
        """取消整批：先撤下排队的，再结束正在跑的，不为批内下一组重起进程。"""
        return {"cancelled": jobs.cancel_many(body.ids)}

    @api.get("/{key}")
    def detail(key: str = PathParam(pattern=r"^[0-9a-f]{24}$")):
        result = jobs.status(key)
        if result is None:
            raise HTTPException(404, "没有找到这次回测")
        return result

    @api.get("/{key}/result")
    def result(
        key: str = PathParam(pattern=r"^[0-9a-f]{24}$"),
        part: Part = "is",
    ):
        complete(key)
        document = jobs.document(key, part)
        if document is None:
            raise HTTPException(410, "结果已清理，请重新运行")
        # 已存结果不可变：直接返回落盘文本，不再解析和重编码。
        return Response(document, media_type="application/json")

    @api.get("/{key}/rebalances")
    def rebalances(
        key: str = PathParam(pattern=r"^[0-9a-f]{24}$"),
        part: Part = "is",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        complete(key)
        rows = loaded(key, part)[1][::-1]
        return {"items": rows[offset : offset + limit], "total": len(rows)}

    @api.get("/{key}/trades")
    def trades(
        key: str = PathParam(pattern=r"^[0-9a-f]{24}$"),
        part: Part = "is",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        complete(key)
        a = loaded(key, part)[0]
        total = len(a["trade_day"])
        # 最新的在前；同一天按引擎成交顺序。
        order = np.arange(total)[::-1][offset : offset + limit]
        names = names_lookup()
        items = []
        for i in order:
            symbol = str(a["trade_symbol"][i])
            qty = float(a["trade_qty"][i])
            price = float(a["trade_price"][i])
            items.append(
                dict(
                    date=str(a["dates"][a["trade_day"][i]]),
                    symbol=symbol,
                    name=names.get(symbol, symbol),
                    side="buy" if qty > 0 else "sell",
                    qty=abs(qty),
                    price=price,
                    amount=abs(qty) * price,
                    fees=float(
                        a["trade_commission"][i]
                        + a["trade_tax"][i]
                        + a["trade_other"][i]
                    ),
                    slippage=float(a["trade_slippage"][i]),
                )
            )
        return {"items": items, "total": total}

    @api.post("/{key}/cancel")
    def cancel(key: str = PathParam(pattern=r"^[0-9a-f]{24}$")):
        return {"cancelled": jobs.cancel(key)}

    @lru_cache(maxsize=1)
    def names_lookup():
        import ashare.data

        data = ashare.data.open(data_root)
        frame = data.securities().select("symbol", "name").collect()
        return dict(frame.rows())

    return api
