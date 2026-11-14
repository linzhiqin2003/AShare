"""设置页的「默认回测设定」：新建研究 tab 和「恢复默认」时套用，存在平台目录。

放在服务端而不是浏览器：默认值本来就由 ``/api/settings`` 下发，校验规则（``CHOICES``）
也只在服务端有一份；存服务端后研究台不用改一行代码就能套用。主题等纯外观项是每个浏览器
自己的偏好，留在前端 localStorage。
"""

import json
import logging
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from .config import DEFAULTS, expand_settings

LOG = logging.getLogger("ashare.platform")
# 可设默认值的项目。中性化不在内：新建 tab 固定以「行业、市值 + 行业」两组扫描开始。
KEYS = (
    "universe",
    "trade_at",
    "trade_limitation",
    "n_groups",
    "decay",
    "rebalance",
    "weights_portfolio",
    "by",
    "sub_universes",
)


def clean(values):
    """只保留合法的标量设定；不合法的项目报错（不静默丢弃用户的输入）。"""
    if not isinstance(values, dict):
        raise ValueError("设置格式不正确")
    unknown = set(values) - set(KEYS) - {"benchmark"}
    if unknown:
        raise ValueError("设置含不支持的项目")
    output = {}
    for key in KEYS:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, list):
            raise ValueError("默认设定只能取单个值")
        expand_settings({key: value})
        output[key] = value
    benchmark = values.get("benchmark")
    if benchmark is not None:
        if not isinstance(benchmark, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,32}", benchmark
        ):
            raise ValueError("基准代码不正确")
        output["benchmark"] = benchmark
    return output


class Preferences:
    def __init__(self, root):
        self.path = Path(root) / "preferences.json"
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            try:
                stored = json.loads(self.path.read_text())
            except FileNotFoundError:
                return {}
            except (OSError, ValueError):
                LOG.warning("默认设定文件无法读取，使用内置默认")
                return {}
        try:
            return clean(stored.get("backtest", {}))
        except (ValueError, AttributeError):
            LOG.warning("默认设定文件含不受支持的值，使用内置默认")
            return {}

    def save(self, values):
        values = clean(values)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps({"backtest": values}, ensure_ascii=False, sort_keys=True)
            )
            os.replace(temporary, self.path)
        LOG.info("默认设定已保存 keys=%s", ",".join(sorted(values)))
        return values

    def defaults(self):
        """研究台用的默认值：内置默认叠加用户默认（基准另行解析）。"""
        stored = self.load()
        return DEFAULTS | {k: v for k, v in stored.items() if k != "benchmark"}

    def benchmark(self):
        return self.load().get("benchmark")


class BacktestDefaults(BaseModel):
    model_config = ConfigDict(extra="allow")


def routes(preferences, benchmarks):
    router = APIRouter(prefix="/api/preferences")

    def payload():
        selected = benchmarks.selection(preferences.benchmark())
        return dict(
            backtest=preferences.defaults()
            | ({"benchmark": selected["symbol"]} if selected["symbol"] else {}),
            builtin=DEFAULTS,
            keys=list(KEYS),
        )

    @router.get("")
    def read():
        return payload()

    @router.put("")
    def write(body: BacktestDefaults):
        preferences.save(body.model_dump())
        return payload()

    @router.delete("")
    def reset():
        preferences.save({})
        return payload()

    return router
