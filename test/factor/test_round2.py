# ruff: noqa: E402
from pathlib import Path

import panelops

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import math
import os

import numpy as np
import pytest

import ashare.factor as af


class Source:
    def __init__(self, **fields):
        self.fields = fields
        self.dates = list(range(len(next(iter(fields.values())))))
        first = next(iter(fields.values()))
        self.symbols = list(range(first.shape[1] if first.ndim == 2 else 3))
        self.version = "v1"
        self.reads = 0

    def __getitem__(self, name):
        self.reads += 1
        return self.fields[name]


def test_cache_corruption_axes_build_id_and_readonly(tmp_path, monkeypatch):
    source = Source(x=np.arange(24.0, dtype=float).reshape(8, 3))
    expression = af.call("ts_mean", af.field("x"), window=3)
    ev = af.Evaluator(tmp_path)
    expected = ev.evaluate(expression, source)
    assert not expected.flags.writeable and not ev.cache_hit
    assert not ev.evaluate(expression, source).flags.writeable and ev.cache_hit
    path = next(tmp_path.glob("*.npy"))
    payload = path.read_bytes()
    for damaged in (
        payload[: len(payload) // 2],
        payload[:-8] + bytes(8),
        b"invalid header",
    ):
        path.write_bytes(damaged)
        got = ev.evaluate(expression, source)
        assert not ev.cache_hit and not got.flags.writeable
        np.testing.assert_array_equal(got, expected)
    for axis in ("dates", "symbols"):
        original = getattr(source, axis)
        setattr(source, axis, [f"changed-{a}" for a in original])
        ev.evaluate(expression, source)
        assert not ev.cache_hit
        setattr(source, axis, original)
    ev.evaluate(expression, source)
    assert ev.cache_hit
    monkeypatch.setattr(panelops, "build_id", "changed-implementation")
    ev.evaluate(expression, source)
    assert not ev.cache_hit
    assert source.fields["x"].flags.writeable


def test_stale_temporary_cleanup_preserves_live_writer(tmp_path):
    (tmp_path / "legacy.tmp").write_text("unfinished")
    (tmp_path / "key.999999999.dead.tmp").write_text("unfinished")
    live = tmp_path / f"key.{os.getpid()}.alive.tmp"
    live.write_text("active writer")
    af.Evaluator(tmp_path)
    assert live.exists()
    assert sorted(p.name for p in tmp_path.glob("*.tmp")) == [live.name]


def test_parameters_defaults_nested_lookback_and_long_chain():
    x = af.field("x")
    assert af.call("ts_mean", x) == af.call("ts_mean", x, window=20, min_periods=20)
    assert af.call("ts_ema", x) == af.call(
        "ts_ema", x, window=20, min_periods=20, alpha=2 / 21
    )
    for params in (
        {"window": 20.0},
        {"window": True},
        {"min_periods": 1.5},
        {"q": 0.2},
        {"layout": 0},
        {"ddof": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            af.call("ts_mean", x, **params)
    nested = af.call(
        "ts_mean",
        af.call("ts_std", af.call("ts_delta", x, periods=3), window=20),
        window=5,
    )
    assert nested.lookback == 26
    paired = af.call(
        "ts_corr",
        af.call("ts_delay", x, periods=3),
        af.call("ts_mean", x, window=10),
        window=5,
    )
    assert paired.lookback == 13
    assert af.call("ts_ffill", x, limit=7).lookback == 7
    assert af.call("ts_ffill", x).lookback is None
    long = x
    for _ in range(800):
        long = long + 1
    source = Source(x=np.ones((3, 2)))
    ev = af.Evaluator()
    np.testing.assert_equal(ev.evaluate(long, source), np.full((3, 2), 801.0))
    assert ev.last_peak_arrays <= 3
    assert source.reads == 1
    common = af.call("ts_mean", x, window=2)
    ev.evaluate((common + 1) * (common - 1), source)
    assert ev.last_counts[common.digest] == 1


def _scalar_reference(op, args):
    """Independent scalar truth table; no calls into the evaluator."""
    finite = [math.isfinite(a) for a in args]
    if op == "is_nan":
        return float(not finite[0])
    if op == "fill_nan":
        return args[0] if finite[0] else args[1]
    if op == "where":
        return (args[1] if args[0] != 0 else args[2]) if finite[0] else math.nan
    if not all(finite):
        return math.nan
    a = args[0]
    b = args[1] if len(args) > 1 else None
    if op in ("gt", "ge", "lt", "le", "eq", "ne"):
        return float(
            {
                "gt": a > b,
                "ge": a >= b,
                "lt": a < b,
                "le": a <= b,
                "eq": a == b,
                "ne": a != b,
            }[op]
        )
    if op == "and_":
        return float(a != 0 and b != 0)
    if op == "or_":
        return float(a != 0 or b != 0)
    if op == "not_":
        return float(a == 0)
    if op == "maximum":
        return max(a, b)
    if op == "minimum":
        return min(a, b)
    if op == "sign":
        return float(a > 0) - float(a < 0)
    if op == "clip":
        return min(2.0, max(-1.0, a))
    try:
        z = {
            "sqrt": lambda: math.sqrt(a),
            "exp": lambda: math.exp(a),
            "power": lambda: math.pow(a, b),
            "signed_power": lambda: (float(a > 0) - float(a < 0)) * math.pow(abs(a), b),
        }[op]()
        return z if math.isfinite(z) else math.nan
    except (ValueError, OverflowError):
        return math.nan


@pytest.mark.parametrize(
    "op",
    "gt ge lt le eq ne and_ or_ not_ where maximum minimum sign power signed_power sqrt exp clip is_nan fill_nan".split(),
)
def test_elementwise_scalar_spec(op):
    x = np.array([[-2.0, 0.0, 1.0, np.nan], [2.0, np.nan, 1000.0, -0.5]])
    y = np.array([[2.0, 0.0, 0.5, 2.0], [np.nan, 3.0, 2.0, -1.0]])
    z = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]])
    arity = af.expression.ELEMENTWISE[op]
    raw = [x, y, z][:arity]
    args = [af.field(name) for name in ("x", "y", "z")[:arity]]
    expression = af.clip(args[0], -1.0, 2.0) if op == "clip" else getattr(af, op)(*args)
    source = Source(x=x, y=y, z=z)
    expected = np.full_like(x, np.nan)
    for ij in np.ndindex(x.shape):
        expected[ij] = _scalar_reference(op, [a[ij] for a in raw])
    actual = af.Evaluator().evaluate(expression, source)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True)
    mask = np.array([[True, False, True, True], [False, True, True, True]])
    actual = af.Evaluator().evaluate(expression, source, mask=mask)
    expected[~mask] = np.nan
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True)


def test_scalar_broadcast_market_field_and_data_dtypes():
    source = Source(
        x=np.ones((5, 3)),
        market=np.arange(5.0, dtype=float),
        turnover=np.full((5, 3), 0.03, dtype=np.float32),
    )
    x = af.field("market", broadcast=True)
    ev = af.Evaluator()
    np.testing.assert_equal(
        ev.evaluate(x + 2, source), np.broadcast_to(np.arange(5.0)[:, None] + 2, (5, 3))
    )
    np.testing.assert_allclose(
        ev.evaluate(af.field("turnover"), source), 0.03, rtol=1e-7
    )
    source.fields["market"] = np.arange(5.0, dtype=float)[:, None]
    np.testing.assert_equal(
        ev.evaluate(x, source), np.broadcast_to(np.arange(5.0)[:, None], (5, 3))
    )
    assert af.call("ts_mean", x, window=3).fields == {"market"}
    assert af.call("ts_mean", x, window=3).lookback == 2
    mask = np.ones((5, 3), bool)
    mask[2, 1] = False
    values = ev.evaluate(af.fill_nan(x / 0, 1), source, mask=mask)
    assert np.isnan(values[2, 1]) and np.all(values[mask] == 1)


def test_authoritative_fields_minimum_counts_and_registry():
    allowed = set(
        "open high low close previous_close vwap volume amount adj_factor turnover total_shares float_shares free_shares total_market_cap float_market_cap pe pe_ttm pb ps ps_ttm dividend_yield dividend_yield_ttm upper_limit lower_limit is_listed is_st is_suspended has_quote listing_age industry cash_dividend split delist".split()
    )
    registry = af.default_registry()
    assert set().union(*(factor.fields for factor in registry)) <= allowed
    assert (
        "total_mv" not in af.FIELDS
        and "turnover_rate" not in af.FIELDS
        and "dv_ttm" not in af.FIELDS
    )
    assert af.FIELDS["turnover"][0] == "比例"
    assert af.FIELDS["dividend_yield_ttm"][0] == "比例"
    assert len(registry.names()) == 14
    assert {f.name for f in registry.using_field("turnover")} == {"turnover_20"}
    for name in (
        "volatility_20",
        "volatility_250",
        "turnover_20",
        "amihud_20",
        "max_return_20",
        "amount_20",
    ):
        p = dict(registry[name].expression.params)
        assert p["min_periods"] == math.ceil(0.8 * p["window"])
    assert (
        dict(
            af.default_registry(min_periods={"volatility_250": 120})[
                "volatility_250"
            ].expression.params
        )["min_periods"]
        == 120
    )
    assert "亏损股为缺失" in registry["ep"].description
    assert "缺失" in registry["dividend_yield"].description


def test_bare_field_respects_mask():
    from ashare.factor import Evaluator, field

    source = Source(close=np.arange(9.0, dtype=float).reshape(3, 3) + 1)
    mask = np.ones((len(source.dates), len(source.symbols)), dtype=bool)
    mask[0, 0] = False
    out = Evaluator().evaluate(field("close"), source, mask=mask)
    assert np.isnan(out[0, 0])
    assert np.isfinite(source["close"][0, 0])


def test_clip_parameter_order_is_canonical():
    x = af.field("close")
    a = af.call("clip", x, lower=-1.0, upper=2.0)
    b = af.call("clip", x, upper=2.0, lower=-1.0)
    assert a == b
    assert a.digest == b.digest
    assert a.params == b.params
    assert a == af.clip(x, -1.0, 2.0)
