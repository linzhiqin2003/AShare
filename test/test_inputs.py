from pathlib import Path

import numpy as np
import panelbt as bt
import pytest

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def test_targets_dense_roundtrip_and_empty_rows():
    dense = np.array([[np.nan, np.nan], [0, 0], [np.nan, 0.5], [0.4, 0]])
    t = bt.Targets.from_dense(dense)
    np.testing.assert_array_equal(t.days, [1, 2, 3])
    np.testing.assert_array_equal(t.indptr, [0, 0, 1, 2])
    np.testing.assert_array_equal(t.cols, [1, 0])
    expected = dense.copy()
    expected[2, 0] = 0
    np.testing.assert_array_equal(t.to_dense(4, 2), expected)
    assert t.days.dtype == np.int64 and t.weights.dtype == np.float64
    empty = bt.Targets([], [0], [], [])
    assert np.isnan(empty.to_dense(2, 3)).all()
    t = bt.Targets([0], [0, 2], [1, 0], [0.3, 0.2])
    np.testing.assert_array_equal(t.cols, [0, 1])
    np.testing.assert_array_equal(t.weights, [0.2, 0.3])


@pytest.mark.parametrize(
    "args,match",
    [
        (([1, 0], [0, 0, 0], [], []), "days"),
        (([0, 0], [0, 0, 0], [], []), "days"),
        (([-1], [0, 0], [], []), "days"),
        (([0.1], [0, 0], [], []), "days"),
        (([0], [1, 1], [], []), "indptr"),
        (([0], [0, 2], [0], [0.5]), "indptr"),
        (([0], [0, 2], [0, 0], [0.1, 0.2]), "cols"),
        (([0], [0, 1], [-1], [0.5]), "cols"),
        (([0], [0, 1], [0], [-0.5]), "weights"),
        (([0], [0, 1], [0], [np.nan]), "weights"),
        (([0], [0, 1], [0], [np.inf]), "weights"),
        (([0], [0, 2], [0, 1], [0.5, 0.500000002]), "weights"),
    ],
)
def test_invalid_targets(args, match):
    with pytest.raises(ValueError, match=match):
        bt.Targets(*args)


def test_dense_invalid_and_tolerance():
    for w in ([[np.nan, -0.1]], [[np.inf, 0]], [[0.8, 0.3]]):
        with pytest.raises(ValueError, match="weights"):
            bt.Targets.from_dense(w)
    bt.Targets.from_dense([[0.5, 0.5 + 5e-10]])


def market():
    return bt.Market([[10]], [[10]], [[True]], [[True]])


def test_context_bounds_and_threads():
    for target, match in [
        (bt.Targets([1], [0, 0], [], []), "days"),
        (bt.Targets([0], [0, 1], [1], [1]), "cols"),
    ]:
        with pytest.raises(ValueError, match=match):
            bt.run(market(), target)
        with pytest.raises(ValueError, match=match):
            target.to_dense(1, 1)
    for threads in (-1, 1.5, True):
        with pytest.raises(ValueError, match="threads"):
            bt.run_batch(market(), [], threads=threads)


@pytest.mark.parametrize(
    "name,value",
    [
        ("volume", [[np.inf]]),
        ("split", [[0]]),
        ("cash_div", [[-1]]),
        ("close", [[0]]),
        ("exec_price", [[-1]]),
        ("can_buy", [[1, 1]]),
    ],
)
def test_bad_panels(name, value):
    values = dict(exec_price=[[10]], close=[[10]], can_buy=[[1]], can_sell=[[1]])
    values[name] = value
    with pytest.raises(ValueError, match=name):
        bt.Market(**values)


def test_conversion_contiguity_and_readonly_borrow():
    p = np.arange(1, 13, dtype=np.float32).reshape(3, 4).T
    m = bt.Market(p, p, np.ones_like(p), np.ones_like(p))
    assert m.exec_price.flags.c_contiguous and m.exec_price.dtype == np.float64
    assert m.can_buy.dtype == np.bool_
    p = np.ones((3, 4)) * 10
    m = bt.Market(p, p, np.ones_like(p, bool), np.ones_like(p, bool))
    assert np.shares_memory(m.exec_price, p)
    assert not m.exec_price.flags.writeable and p.flags.writeable


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_cash", 0),
        ("min_qty", -1),
        ("qty_step", [1, 1]),
        ("cash_buffer", 1.1),
        ("delist_recovery", np.nan),
        ("max_participation", -1),
    ],
)
def test_bad_config(key, value):
    with pytest.raises(ValueError, match=key):
        bt.run(market(), bt.Targets.from_dense([[1]]), config=bt.Config(**{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("commission", -1),
        ("min_commission", np.inf),
        ("tax_buy", [0, 1]),
        ("tax_sell", np.nan),
        ("slippage", 1),
    ],
)
def test_bad_costs(key, value):
    with pytest.raises(ValueError, match=key):
        bt.run(market(), bt.Targets.from_dense([[1]]), costs=bt.Costs(**{key: value}))


@pytest.mark.parametrize("key,value", [("commission", "bad"), ("tax_buy", ["bad"])])
def test_nonnumeric_costs_name_parameter(key, value):
    with pytest.raises(ValueError, match=key):
        bt.run(market(), bt.Targets.from_dense([[1]]), costs=bt.Costs(**{key: value}))


def test_nonnumeric_lots_name_parameter():
    with pytest.raises(ValueError, match="min_qty"):
        bt.run(
            market(), bt.Targets.from_dense([[1]]), config=bt.Config(min_qty=["bad"])
        )
