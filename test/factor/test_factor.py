# ruff: noqa: E402
from pathlib import Path

import panelops

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import json
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from ashare.factor import (
    Evaluator,
    Registry,
    adjusted_close,
    call,
    default_registry,
    field,
)


@dataclass
class FakeSource:
    arrays: dict
    version: str = "v1"

    @property
    def dates(self):
        return np.arange(len(next(iter(self.arrays.values()))))

    @property
    def symbols(self):
        return ["A", "B", "C"][: next(iter(self.arrays.values())).shape[1]]

    def __getitem__(self, name):
        return self.arrays[name]


def test_hash_and_lookback():
    expression = call("ts_mean", adjusted_close(), window=20, min_periods=5)
    reversed_params = call("ts_mean", adjusted_close(), min_periods=5, window=20)
    assert expression.digest == reversed_params.digest
    assert hash(expression) == hash(reversed_params)
    assert (
        expression.digest
        != call("ts_mean", adjusted_close(), window=21, min_periods=5).digest
    )
    assert (
        expression.digest
        != call("ts_mean", adjusted_close(), window=20, min_periods=6).digest
    )
    assert expression.fields == {"close", "adj_factor"}
    assert expression.lookback == 19
    assert call("ts_delay", expression, periods=5).lookback == 24
    assert call("ts_ema", expression, window=20).lookback is None
    assert "window=20" in str(expression)
    code = (
        "import ashare, pathlib; assert pathlib.Path(ashare.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
        "import json; from ashare.factor import call,adjusted_close; "
        'x=call("ts_mean",adjusted_close(),window=20,min_periods=5); print(json.dumps([x.digest,hash(x)]))'
    )
    result = json.loads(
        subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True)
    )
    assert result == [expression.digest, hash(expression)]
    with pytest.raises(ValueError):
        call("not_an_operator", expression)
    with pytest.raises(TypeError):
        call("ts_mean", expression, window=[])


def test_registry():
    registry = Registry()
    factor = registry.register("x", field("close"), "说明")
    assert factor.fields == {"close"}
    with pytest.raises(ValueError, match="重名"):
        registry.register("x", field("amount"))
    assert len(default_registry()) == 14


def test_cache_and_common_subexpressions(tmp_path, monkeypatch):
    x = np.arange(18, dtype=float).reshape(6, 3) + 1
    source = FakeSource({"close": x, "adj_factor": np.ones_like(x)})
    common = call("ts_mean", adjusted_close(), window=2)
    expr = common + common
    engine = Evaluator(tmp_path)
    first = engine.evaluate(expr, source)
    assert engine.last_counts[common.digest] == 1
    second = engine.evaluate(expr, source)
    assert engine.cache_hit and isinstance(second, np.memmap)
    assert not second.flags.writeable
    np.testing.assert_array_equal(first, second)
    for change in ("version", "universe", "mask", "params", "panelops"):
        kwargs = {}
        expression = expr
        if change == "version":
            source.version = "v2"
        elif change == "universe":
            kwargs["universe_id"] = "other"
        elif change == "mask":
            kwargs["mask"] = np.array([[False, True, True]] * 6)
        elif change == "params":
            expression = call("ts_mean", adjusted_close(), window=3)
        else:
            monkeypatch.setattr(panelops, "build_id", "different")
        engine.evaluate(expression, source, **kwargs)
        assert not engine.cache_hit
    assert not list(tmp_path.glob("*.tmp"))


def test_each_builtin_hand_calculated():
    t = np.arange(270)[:, None]
    close = np.ascontiguousarray(20 + t * np.array([[0.1, 0.2, -0.01]]))
    adj = np.ones_like(close)
    adj[130:] = 2
    fields = {
        "close": close,
        "adj_factor": adj,
        "total_market_cap": close * 1e7,
        "volume": np.ones_like(close) * 100,
        "amount": close * 100,
        "turnover": np.ones_like(close) * 0.02,
        "pe_ttm": close,
        "pb": close / 10,
        "ps_ttm": close / 5,
        "dividend_yield_ttm": np.ones_like(close) * 0.03,
    }
    price = close * adj
    daily = np.full_like(price, np.nan)
    for i in range(1, len(price)):
        for j in range(3):
            daily[i, j] = price[i, j] / price[i - 1, j] - 1
    engine, source = Evaluator(), FakeSource(fields)
    for factor in default_registry():
        expected = np.full_like(price, np.nan)
        name = factor.name
        for i in range(len(price)):
            for j in range(3):
                if name == "log_market_cap":
                    value = np.log(fields["total_market_cap"][i, j])
                elif name.startswith("reversal_"):
                    w = int(name.split("_")[-1])
                    value = -(price[i, j] / price[i - w, j] - 1) if i >= w else np.nan
                elif name == "momentum_12_1":
                    value = (
                        price[i - 21, j] / price[i - 252, j] - 1 if i >= 252 else np.nan
                    )
                elif name.startswith("volatility_"):
                    w = int(name.split("_")[-1])
                    value = (
                        np.nanstd(daily[max(0, i - w + 1) : i + 1, j], ddof=1)
                        if i >= int(np.ceil(0.8 * w))
                        else np.nan
                    )
                elif name == "turnover_20":
                    value = 0.02 if i >= 15 else np.nan
                elif name == "amihud_20":
                    value = (
                        np.nanmean(
                            abs(daily[max(0, i - 19) : i + 1, j])
                            / fields["amount"][max(0, i - 19) : i + 1, j]
                        )
                        if i >= 16
                        else np.nan
                    )
                elif name in ("ep", "bp", "sp"):
                    value = (
                        1
                        / fields[{"ep": "pe_ttm", "bp": "pb", "sp": "ps_ttm"}[name]][
                            i, j
                        ]
                    )
                elif name == "dividend_yield":
                    value = 0.03
                elif name == "max_return_20":
                    value = (
                        np.nanmax(daily[max(0, i - 19) : i + 1, j])
                        if i >= 16
                        else np.nan
                    )
                elif name == "amount_20":
                    value = (
                        np.mean(fields["amount"][max(0, i - 19) : i + 1, j])
                        if i >= 15
                        else np.nan
                    )
                expected[i, j] = value
        actual = engine.evaluate(factor.expression, source)
        np.testing.assert_allclose(
            actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True, err_msg=name
        )


def test_missing_and_no_shift():
    x = np.array([[1.0, 2.0], [0.0, -2.0]])
    source = FakeSource({"pe_ttm": x})
    result = Evaluator().evaluate(default_registry()["ep"].expression, source)
    np.testing.assert_equal(result, [[1.0, 0.5], [np.nan, -0.5]])
    assert source.arrays["pe_ttm"][1, 0] == 0


def test_expression_equality_obeys_hash_contract():
    a = call("ts_mean", field("close"), window=1)
    with pytest.raises(TypeError):
        call("ts_mean", field("close"), window=1.0)
    b = field("close") * 1
    c = field("close") * 1.0
    assert b != c
    assert len({a, b, c}) == 3


def test_group_and_neutralization_expression_nodes():
    from ashare.factor import neutralize

    x = np.array([[1.0, 3.0, 8.0], [1.0, 3.0, 8.0]])
    groups = np.array([[0, 0, 1], [-1, 0, 0]], dtype=np.int64)
    weights = np.array([[1.0, 3.0, 1.0], [1.0, 3.0, 1.0]])
    source = FakeSource(
        {"score": x, "sector": groups, "beta": np.ones_like(x), "weight": weights}
    )
    expr = neutralize(
        field("score"),
        field("beta"),
        groups=field("sector", dtype="int64"),
        weights=field("weight"),
    )
    assert expr.fields == {"score", "sector", "beta", "weight"}
    actual = Evaluator().evaluate(expr, source)
    np.testing.assert_allclose(
        actual,
        [[-1.5, 0.5, 0], [np.nan, -1.25, 3.75]],
        atol=2e-10,
        rtol=2e-10,
        equal_nan=True,
    )
    demean = call("cs_group_demean", field("score"), field("sector", dtype="int64"))
    np.testing.assert_allclose(
        Evaluator().evaluate(demean, source),
        [[-1, 1, 0], [np.nan, -2.5, 2.5]],
        equal_nan=True,
    )
    mask = np.array([[True, True, False], [True, True, True]])
    masked = Evaluator().evaluate(expr, source, mask=mask)
    assert np.isnan(masked[0, 2])
    assert field("sector", dtype="int64").digest != field("sector").digest
