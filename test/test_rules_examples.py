"""冒烟测试：真实数据例子 bench/rules_examples.py 的代码路径。

测试不许读真实数据（`ashare.data` 在 pytest 下拒绝真实数据根目录），所以这里给脚本
一个形状相同的小型假数据句柄：同样经过 `ashare.backtest.market_from_panel` 和
`index_daily` 长表。`ashare.backtest` 或数据层接口变了，这里会先失败。
真实数据上的数字用 `python bench/rules_examples.py` 跑。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "rules_examples", ROOT / "bench/rules_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Panel:
    def __init__(self, arrays, dates, symbols):
        self._arrays, self.dates, self.symbols = arrays, dates, symbols
        self.shape = (len(dates), len(symbols))

    def __getitem__(self, name):
        return self._arrays[name]


class FakeData:
    """The parts of `ashare.data.Data` the examples use, on random prices."""

    def __init__(self, t=320, n=40, seed=0):
        rng = np.random.default_rng(seed)
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0))
        open_ = close * np.exp(rng.normal(0, 0.01, (t, n)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.03, (t, n)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.03, (t, n)))
        ones = np.ones((t, n), bool)
        days = np.datetime64("2020-01-01", "D") + np.arange(t)
        self.dates = np.array([str(d).replace("-", "") for d in days])
        self.symbols = np.array([f"{600000 + j:06d}.SH" for j in range(n)])
        self.arrays = {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full((t, n), 1e8),
            "split": np.full((t, n), np.nan),
            "cash_dividend": np.full((t, n), np.nan),
            "delist": ~ones,
            "can_buy_open": ones,
            "can_sell_open": ones,
            "adj_factor": np.ones((t, n)),
            "member_csi300": ones,
        }
        o, c = open_.mean(axis=1), close.mean(axis=1)
        self.index = pl.DataFrame(
            {
                "symbol": "000300.SH",
                "date": self.dates,
                "open": o,
                "high": np.maximum(o, c) * 1.01,
                "low": np.minimum(o, c) * 0.99,
                "close": c,
            }
        )

    def panel(self, names, start=None, end=None, symbols=None):
        return Panel({k: self.arrays[k] for k in names}, self.dates, self.symbols)

    def securities(self):
        return pl.DataFrame(
            {"symbol": list(self.symbols), "board": ["主板"] * len(self.symbols)}
        )

    def table(self, name, start=None, end=None, symbols=None):
        assert name == "index_daily"
        return self.index.lazy().filter(pl.col("symbol").is_in(symbols))


def test_rules_examples_code_path():
    module = load()
    module.REPEAT = 1
    data = FakeData()
    timing = module.index_timing(data)
    stocks = module.stock_selection(data)
    for out, name in (
        (timing, "no_rules"),
        (timing, "trailing_8pct"),
        (stocks, "no_rules"),
        (stocks, "stop_10pct"),
    ):
        assert math.isfinite(out[name]["annual_return"]), name
        assert out[name]["trades"] > 0, name
