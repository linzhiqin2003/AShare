"""策略定义文件与「信号 → 目标持仓」。

账户目录里的 ``strategy.json``（版本 1）::

    {
      "version": 1,
      "name": "低波动 30",                 # 展示名，可含中文和空格
      "source": "rank(-ts_std(ret, 20))",   # 平台表达式，只允许一个候选
      "settings": {...},                    # 回测页设置，经 backtest.settings.normalize 规范化
      "initial_cash": 10000000.0,           # 元；缺省为 settings.capital × 1 万
      "enabled": true,
      "created_at": "2026-11-14T10:00:00+08:00",
      "origin": {"kind": "expression"}      # 或 {"kind": "backtest", "id": "…", "data_version": "…"}
    }

选股与回测页逐位同一套函数：信号用 ``ashare.factor.Evaluator`` 在整张面板上求值（与回测页
``evaluate_signal`` 相同的因果算子，只是不在验证期末截断——模拟盘本来就是向前跑的），股票池
用回测页的 ``universe_mask``，``ashare.portfolio.top_equal`` 取前 N 只等权，再经
``ashare.portfolio.to_targets`` 规整（回测页 ``build_targets`` 的同一条路径）。

模拟盘实际用到的设置是 ``universe``、``rebalance``、``top_n``、``trade_at``、``capital``；费用、整手、
红利税、受限顺延由模拟盘内核自己的规则决定（DESIGN 8.3/8.4），这里只原样记录，便于日后对账。信号日收盘后得到的目标在下一个开市日成交（回测的
``execution_targets(lag=1)``；最后一个信号日在面板里没有下一行，用交易日历的下一开市日）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

from ashare.platform.backtest import settings as contract

NAME = re.compile(r"\w[\w.-]{0,63}")
VERSION = 1
UNTIL = "14:56:50"
# 回测的「开盘成交」对应参加开盘集合竞价（运行器在 9:15–9:25 用昨收参考价下单，9:25 撮合，
# 没成交的部分连续竞价里继续对齐）；「收盘成交」对应 14:50 起在连续竞价末段对齐。运行器不在
# 收盘集合竞价里下单。
AT = {"open": "09:15:00", "close": "14:50:00"}


@dataclass(frozen=True)
class Strategy:
    name: str
    source: str
    settings: dict
    initial_cash: float
    enabled: bool
    created_at: str
    origin: dict

    def to_dict(self) -> dict:
        return dict(
            version=VERSION,
            name=self.name,
            source=self.source,
            settings=self.settings,
            initial_cash=self.initial_cash,
            enabled=self.enabled,
            created_at=self.created_at,
            origin=self.origin,
        )

    @classmethod
    def from_dict(cls, raw: dict) -> Strategy:
        if not isinstance(raw, dict) or raw.get("version") != VERSION:
            raise ValueError("策略定义版本不受支持")
        known = {
            "version",
            "name",
            "source",
            "settings",
            "initial_cash",
            "enabled",
            "created_at",
            "origin",
        }
        if set(raw) - known:
            raise ValueError("策略定义含不认识的项目")
        cash = raw["initial_cash"]
        if isinstance(cash, bool) or not isinstance(cash, (int, float)) or cash <= 0:
            raise ValueError("初始资金须为正数")
        if not isinstance(raw.get("enabled"), bool):
            raise ValueError("启用标记须为布尔值")
        return cls(
            name=str(raw["name"]),
            source=str(raw["source"]),
            settings=contract.normalize(raw["settings"]),
            initial_cash=float(cash),
            enabled=raw["enabled"],
            created_at=str(raw["created_at"]),
            origin=dict(raw["origin"]),
        )

    def expression(self):
        variants = parse_source(self.source)
        return variants[0].expression

    @property
    def execution(self) -> dict:
        return dict(
            at=AT[self.settings["trade_at"]],
            until=UNTIL,
            rebalance_seconds=30,
            protection_ticks=2,
        )


def parse_source(source: str):
    """平台表达式语言（与研究台、回测页同一个解释器）；只接受一个候选。"""
    from ashare.platform.language import parse

    if not isinstance(source, str) or not source.strip() or len(source) > 16000:
        raise ValueError("表达式不能为空或过长")
    variants = parse(source)
    if len(variants) != 1:
        raise ValueError("模拟账户只跑一个表达式，请去掉候选列表")
    return variants


def load(directory: Path) -> Strategy:
    return Strategy.from_dict(json.loads((directory / "strategy.json").read_bytes()))


class LivePanel:
    """整张最新面板（不截断）的只读字段缓存；字段处理与平台 ``PanelCache`` 相同。"""

    def __init__(self, root):
        import ashare.data

        self.data = ashare.data.open(root)
        meta = self.data.panel([])
        self.dates = np.asarray(meta.dates)
        self.symbols = meta.symbols
        self.version = str(meta.version)
        self.arrays: dict[str, np.ndarray] = {}

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def ensure(self, fields) -> None:
        wanted = set(fields) - self.arrays.keys()
        raw = wanted - {"industry_l1", "benchmark_open", "benchmark_close"}
        if "industry_l1" in wanted:
            raw.add("industry")
        if raw:
            panel = self.data.panel(sorted(raw))
            self.arrays.update({name: panel[name] for name in raw})
        if "industry_l1" in wanted:
            a = self.arrays["industry"].astype(np.int64)
            self.arrays["industry_l1"] = np.where(a > 0, a, -1)
        benchmark = wanted & {"benchmark_open", "benchmark_close"}
        if benchmark:
            import polars as pl

            frame = self.data.table("index_daily", symbols=["000300.SH"]).collect()
            for name in benchmark:
                lookup = dict(
                    frame.select("date", pl.col(name.removeprefix("benchmark_"))).rows()
                )
                self.arrays[name] = np.array(
                    [lookup.get(d, np.nan) for d in self.dates]
                )
        for value in self.arrays.values():
            value.flags.writeable = False

    def __getitem__(self, name):
        return self.arrays[name]


class _Row:
    """面板的一行（保持二维），给只按行计算的股票池函数用。"""

    def __init__(self, panel, row):
        self.panel, self.row = panel, row
        self.dates = np.asarray(panel.dates)[row : row + 1]
        self.symbols = panel.symbols

    def __getitem__(self, name):
        return np.asarray(self.panel[name])[self.row : self.row + 1]


def pool_fields(universe: str) -> set[str]:
    from ashare.factor.groupbt.api import UNIVERSES
    from ashare.platform.backtest.engine import INDEX_POOLS

    if universe in INDEX_POOLS:
        return {"is_listed", INDEX_POOLS[universe]}
    return {"is_listed", *UNIVERSES[universe]}


def pool_row(panel, universe: str, row: int) -> np.ndarray:
    """回测页 ``universe_mask`` 在第 ``row`` 行的结果（它只按行计算，传一行即可）。"""
    from ashare.platform.backtest.engine import universe_mask

    return np.asarray(universe_mask(_Row(panel, row), universe))[0]


def signal_alpha(panel, expression) -> np.ndarray:
    from ashare.factor import Evaluator

    panel.ensure(set(expression.fields))
    return Evaluator().evaluate(expression, panel)


def target_weights(alpha_row, pool, top_n: int, symbols) -> dict[str, float]:
    """前 N 只等权 → ``to_targets`` 规整后的权重（证券 → 权重，按面板列序）。"""
    from ashare.portfolio import to_targets, top_equal

    w = top_equal(np.asarray(alpha_row), top_n, eligible=pool)
    columns = np.flatnonzero(w).astype(np.int64)
    sparse = to_targets(np.array([0], dtype=np.int64), w[columns][None, :], columns)
    cols = np.asarray(sparse.cols)
    weights = np.asarray(sparse.weights, dtype=np.float64)
    return {str(symbols[int(c)]): float(x) for c, x in zip(cols, weights, strict=True)}


def runner_safe(weights: dict[str, float]) -> tuple[dict[str, float], int]:
    """运行器按十进制（``Decimal(str(w))``）检查权重和不超过 1。1/N 的最短十进制表示偶尔
    合计多出末位（如 N=11），这时把最大的权重向下挪一个浮点末位，直到合计不超过 1。返回挪动次数。"""
    out = dict(weights)
    moved = 0
    while sum((Decimal(str(v)) for v in out.values()), Decimal(0)) > 1:
        key = max(out, key=lambda k: (out[k], k))
        out[key] = float(np.nextafter(out[key], 0.0))
        moved += 1
        if moved > 64:
            raise ValueError("目标权重无法规整到合计不超过 1")
    return out, moved
