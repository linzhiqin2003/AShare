"""回测页的设置白名单、默认值与结果身份。

接口只收这里列出的键；窗口、起止日期、TEST 一律不收（窗口由 `ashare.factor.windows` 固定）。
"""

import hashlib
import json
import re

DEFAULTS = dict(
    universe="csi300",
    rebalance="month",
    top_n=30,
    trade_at="open",
    capital=1000,  # 万元
    commission=0.00025,
    commission_min=5.0,
    slippage_bps=5.0,
    dividend_tax=0.2,
    carry_blocked=True,
    lots="board",
)
CHOICES = dict(
    universe=("csi300", "csi500", "csi1000", "TOP800", "TOP1800", "ALL"),
    rebalance=("day", "week", "month", "quarter", "year"),
    trade_at=("open", "close"),
    dividend_tax=(0.2, 0.1, 0.0),
    carry_blocked=(True, False),
    lots=("board", "none"),
)
LIMITS = dict(
    top_n=(1, 500),
    capital=(10, 1_000_000),
    commission=(0.0, 0.003),
    commission_min=(0.0, 100.0),
    slippage_bps=(0.0, 100.0),
)
_SYMBOL = re.compile(r"[A-Za-z0-9._-]{1,32}")


def normalize(settings):
    """校验并补全设置；不认识的键（包括 window/start/end）直接拒绝。"""
    if not isinstance(settings, dict):
        raise ValueError("设置格式不正确")
    unknown = set(settings) - set(DEFAULTS) - {"benchmark"}
    if unknown:
        raise ValueError("设置含不支持的项目")
    out = {}
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if key in CHOICES:
            # bool 不能冒充数字，0 与 0.0 视为同一个税率。
            matched = [
                c
                for c in CHOICES[key]
                if type(value) is type(c)
                or (
                    isinstance(c, float)
                    and type(value) is int
                    and not isinstance(value, bool)
                )
            ]
            if not any(value == c for c in matched):
                raise ValueError(f"设置 {key} 的值不受支持")
            value = next(c for c in matched if value == c)
        else:
            lo, hi = LIMITS[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"设置 {key} 须为数字")
            if key == "top_n" and type(value) is not int:
                raise ValueError("持仓只数须为整数")
            if not lo <= value <= hi:
                raise ValueError(f"设置 {key} 超出范围")
            value = int(value) if key == "top_n" else float(value)
        out[key] = value
    benchmark = settings.get("benchmark")
    if benchmark is not None:
        if not isinstance(benchmark, str) or not _SYMBOL.fullmatch(benchmark):
            raise ValueError("基准代码不正确")
        out["benchmark"] = benchmark
    return out


def identity(digest, settings, data_version):
    """表达式摘要 + 规范化设置 + 数据版本：同一份数据上的同一配置只算一次。"""
    text = json.dumps(
        {"expression": digest, "settings": settings, "data": str(data_version)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()[:24]
