"""回测页接口：`create_app` 里一行 `include_router` 注册。

只开放两个窗口（样本内、验证期），参数里没有窗口、起止日期；TEST 没有任何路由可达。
页面文档按窗口分开返回，验证期只在页面打开验证结果开关时才请求。
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ashare.factor.windows import IS, VALIDATION
from ashare.ledger import code_version

from ..benchmarks import Benchmarks
from ..language import parse, suggested_name
from . import settings as config
from .jobs import Backtests

Part = Literal["is", "validation"]


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=16000)
    name: str = Field("", max_length=80)
    settings: dict = Field(default_factory=dict)


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
        variants = parse(body.source)
        if len(variants) != 1:
            raise HTTPException(422, "回测一次只跑一个表达式，请去掉候选列表")
        try:
            settings = config.normalize(body.settings)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        selected = benchmarks.selection(settings.get("benchmark"))
        if selected["symbol"]:
            settings["benchmark"] = selected["symbol"]
        else:
            settings.pop("benchmark", None)
        key = config.identity(variants[0].expression.digest, settings, data_version())
        jobs.submit(
            key,
            body.name.strip() or suggested_name(body.source),
            body.source,
            settings,
            selected["name"] if selected["symbol"] else None,
            data_version(),
            code(),
        )
        return jobs.status(key)

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
