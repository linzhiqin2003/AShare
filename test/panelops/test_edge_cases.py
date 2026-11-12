# ruff: noqa: E402
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)

import threading

import numpy as np
import pytest
from panelops import reference


def test_signed_zero_tie_order():
    x = np.array([[-0.0, 0.0, -0.0, 0.0]])
    t = po.targets(x, k=2)
    np.testing.assert_array_equal(t.cols, [0, 1])
    np.testing.assert_array_equal(po.cs_rank(x), [[2.5] * 4])
    np.testing.assert_allclose(
        po.group_returns(x, np.array([[1.0, 2.0, 3.0, 4.0]]), q=2), [[1.5, 3.5]]
    )


def test_autocorr_pairwise_ranks_and_turnover_zero_lag():
    x = np.array(
        [[1.0, 2.0, 4.0, np.nan], [3.0, np.nan, 1.0, 7.0], [4.0, 2.0, 3.0, 1.0]]
    )
    lag = reference.ts(x, "delay")
    np.testing.assert_allclose(
        po.factor_autocorr(x),
        reference.ic(x, lag, rank=True),
        rtol=2e-10,
        atol=2e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        po.group_turnover(x, q=2, periods=0), np.zeros((3, 2))
    )
    assert np.isnan(po.group_turnover(x, periods=10)).all()


def test_evaluation_thread_determinism():
    rng = np.random.default_rng(404)
    x, y = rng.normal(size=(2, 33, 19))
    functions = [
        lambda: po.ic(x, y),
        lambda: po.rank_ic(x, y),
        lambda: po.group_returns(x, y),
        lambda: po.group_turnover(x),
        lambda: po.factor_autocorr(x),
    ]
    try:
        for fn in functions:
            po.set_num_threads(1)
            a = fn()
            po.set_num_threads(3)
            b = fn()
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        po.set_num_threads(1)
        a = po.targets(x)
        po.set_num_threads(3)
        b = po.targets(x)
        for key in ("days", "indptr", "cols", "weights"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    finally:
        po.set_num_threads(0)


def test_native_releases_gil():
    x = np.ones((1000, 200))
    started, done = threading.Event(), threading.Event()
    counter = [0]

    def work():
        started.wait()
        while not done.is_set():
            counter[0] += 1

    thread = threading.Thread(target=work)
    thread.start()
    try:
        po.set_num_threads(1)
        # Prebuild every argument before measuring progress inside the binding.
        args = (
            x,
            np.empty((0, 0)),
            np.empty((0, 0), bool),
            "skew",
            250,
            1,
            1,
            0.5,
            False,
            0.5,
            1,
            1,
            0,
        )
        started.set()
        before = counter[0]
        po._api._core.ts(*args)
        after = counter[0]
        assert after - before > 1000
    finally:
        done.set()
        thread.join()
        po.set_num_threads(0)


def test_input_validation_and_readonly():
    x = np.ones((2, 3))
    x.flags.writeable = False
    np.testing.assert_equal(po.ts_mean(x, 1), x)
    for function, kwargs in [
        (po.cs_neutralize, {"exposures": [np.ones((3, 2))]}),
        (po.cs_group_demean, {}),
        (po.targets, {"days": np.array([1, 0])}),
        (po.targets, {"days": np.array([2])}),
        (po.targets, {"k": 1, "fraction": 0.2}),
        (po.cs_winsor_quantile, {"lower": 0.9, "upper": 0.1}),
        (po.ts_ema, {"alpha": 0}),
        (po.ts_corr, {"y": np.ones((2, 2))}),
    ]:
        with pytest.raises(ValueError):
            function(x, **kwargs)
    for name in (
        "ts_mean",
        "cs_rank",
        "ic",
        "group_returns",
        "group_turnover",
        "targets",
    ):
        args = (x, x) if name in ("ic", "group_returns") else (x,)
        with pytest.raises(ValueError):
            getattr(po, name)(*args, backend="unknown")


def test_masked_delay_history_and_ema_gaps():
    x = np.array([[1.0], [9.0], [3.0], [np.nan], [5.0]])
    mask = np.array([[True], [False], [True], [True], [True]])
    assert np.isnan(po.ts_delay(x, mask=mask)[2, 0])
    expected = reference.ts(x, "ema", window=2, min_periods=1, mask=mask)
    np.testing.assert_allclose(
        po.ts_ema(x, 2, min_periods=1, mask=mask),
        expected,
        rtol=2e-10,
        atol=2e-10,
        equal_nan=True,
    )


def test_neutralization_singletons_all_missing_and_zero_columns():
    x = np.array([[1.0, 2.0, 3.0], [np.nan, 2.0, 5.0]])
    groups = np.array([[0, 1, 2], [-1, 1, 1]], dtype=np.int64)
    result = po.cs_neutralize(x, groups=groups, exposures=[np.zeros_like(x)])
    np.testing.assert_allclose(
        result, [[0, 0, 0], [np.nan, -1.5, 1.5]], atol=2e-10, rtol=2e-10, equal_nan=True
    )


def test_higher_moment_cancellation_fallback():
    # Offset, ramp and constant windows exercise the guarded direct path.
    for base in (0.0, 1e10):
        x = np.ascontiguousarray(
            base + np.arange(1100)[:, None] * np.array([[0.001, 0.002]])
        )
        x[400:700] = base + 1
        for op in ("skew", "kurt"):
            for window in (5, 60, 250):
                expected = reference.ts(x, op, window=window, min_periods=4)
                actual = getattr(po, "ts_" + op)(x, window, min_periods=4)
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True
                )


def test_numpy_backend_does_not_call_native(monkeypatch):
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [4.0, 1.0, 3.0]])

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy backend entered native code")

    for name in ("ts", "cs", "ic", "grouped", "targets"):
        monkeypatch.setattr(po._api._core, name, forbidden)
    np.testing.assert_allclose(
        po.factor_autocorr(x, backend="numpy"),
        reference.factor_autocorr(x),
        atol=2e-10,
        rtol=2e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        po.ts_delta(x, backend="numpy"), reference.ts(x, "delta"), equal_nan=True
    )
    np.testing.assert_allclose(po.cs_rank(x, backend="numpy"), reference.cs(x, "rank"))
    np.testing.assert_allclose(
        po.group_returns(x, x, backend="numpy"),
        reference.group_returns(x, x),
        equal_nan=True,
    )
    target = po.targets(x, backend="numpy")
    np.testing.assert_array_equal(target.cols, reference.targets(x)[2])


def test_native_binding_rejects_fortran_layout():
    x = np.asfortranarray(np.arange(12.0, dtype=float).reshape(4, 3))
    with pytest.raises(ValueError, match="C-contiguous"):
        po._api._core.ts(
            x,
            np.empty((0, 0)),
            np.empty((0, 0), bool),
            "mean",
            2,
            1,
            1,
            0.5,
            False,
            0.5,
            1,
            1,
            0,
        )


def test_reference_and_production_empty_target_rows():
    x = np.full((3, 4), np.nan)
    expected = reference.targets(x)
    for backend in ("rust", "numpy"):
        actual = po.targets(x, backend=backend)
        for a, b in zip(
            (actual.days, actual.indptr, actual.cols, actual.weights),
            expected,
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype
    np.testing.assert_array_equal(expected[1], [0, 0, 0, 0])


def test_invalid_weights_and_exposures_are_missing():
    x = np.array([[1.0, 3.0, 7.0, 9.0]])
    weights = np.array([[1.0, 0.0, -1.0, np.inf]])
    for backend in ("rust", "numpy"):
        grouped = po.group_returns(x, x, q=1, weights=weights, backend=backend)
        np.testing.assert_equal(grouped, [[1.0]])
        residual = po.cs_neutralize(x, weights=weights, backend=backend)
        np.testing.assert_allclose(
            residual,
            [[0.0, np.nan, np.nan, np.nan]],
            atol=2e-10,
            rtol=2e-10,
            equal_nan=True,
        )
        exposure = np.array([[1.0, np.nan, np.inf, 2.0]])
        residual = po.cs_neutralize(x, exposures=[exposure], backend=backend)
        np.testing.assert_allclose(
            residual,
            [[0.0, np.nan, np.nan, 0.0]],
            atol=2e-10,
            rtol=2e-10,
            equal_nan=True,
        )


def test_extreme_intermediate_state_recovers_after_window():
    x = np.array([[1e308], [1e308], [1.0], [2.0], [3.0], [4.0], [5.0]])
    for op in ("sum", "mean", "std", "decay_linear"):
        actual = getattr(po, "ts_" + op)(x, 3, min_periods=1)
        expected = reference.ts(x[2:].copy(), op, window=3, min_periods=3)
        np.testing.assert_allclose(
            actual[4:], expected[2:], rtol=2e-10, atol=2e-10, equal_nan=True, err_msg=op
        )
