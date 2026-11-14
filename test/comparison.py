"""Full-result comparison shared by deterministic tests and corpus validation."""

from dataclasses import astuple

import numpy as np
from panelbt.models import METRICS


def _differ(name, actual, expected):
    """字典比较失败时只列出不同的键。这个文件不是测试模块，pytest 不改写这里的 assert，
    不带说明的 assert 失败时消息是空的（2026-11-14 Linux CI 上 seed 151 就只报了空串）。"""
    keys = sorted(set(actual) | set(expected), key=str)
    return (
        name
        + ": "
        + ", ".join(
            f"{k}: {actual.get(k)!r} != {expected.get(k)!r}"
            for k in keys
            if actual.get(k) != expected.get(k)
        )
    )


def assert_results_equal(actual, expected):
    for name in (*METRICS, "positions", "contribution"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b, name
        else:
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=name)
    assert actual.diagnostics == expected.diagnostics, _differ(
        "diagnostics", actual.diagnostics, expected.diagnostics
    )
    assert actual.coverage == expected.coverage, _differ(
        "coverage", actual.coverage, expected.coverage
    )
    assert actual.bankruptcy_events == expected.bankruptcy_events, (
        "bankruptcy_events",
        actual.bankruptcy_events,
        expected.bankruptcy_events,
    )
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b, (name, a is None, b is None)
            continue
        assert len(a) == len(b), (name, len(a), len(b))
        for left, right in zip(a, b, strict=True):
            assert (left.day, left.col) == (right.day, right.col), (name, left, right)
            np.testing.assert_allclose(
                astuple(left), astuple(right), rtol=1e-9, atol=1e-9, err_msg=name
            )
