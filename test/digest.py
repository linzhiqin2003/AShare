"""Bitwise digest of a Result (used by the frozen no-rules fixture).

The fixture `fixtures/no_rules_digest.json` was produced by the engine build
before the stepping interface and declarative rules were added; matching it
proves that runs without rules are unchanged bit for bit.

Re-frozen on 2026-11-14 for one change only: the engine now writes +0.0 instead
of -0.0 for a zero external cash flow (`max` of two zeros has a platform-dependent
sign; Linux CI disagreed with macOS). All 600 cases were compared field by field
against the previous build: 124 differed, every one only in the sign of zeros in
`cash_flow`, with equal values everywhere.
"""

import hashlib
import platform
import sys
from dataclasses import astuple

import numpy as np
import pytest
from panelbt.models import METRICS

from . import generators, generators_futures

# 冻结的逐位摘要只在生成它们的平台上成立：引擎里的 powf、生成随机输入时 numpy 的 exp/log 走
# 系统数学库，macOS 与 Linux glibc 在最后一位上可能不同，求和的 SIMD 顺序也不同（2026-11-14 第一次
# 在 Linux CI 上跑，spot-0 起就不同）。别的平台上跨实现的正确性由按容差比对参考实现的测试守住。
frozen_platform = pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason="冻结的逐位摘要在 macOS arm64 上生成；系统数学库和 SIMD 求和顺序随平台不同",
)


def digest(result) -> str:
    h = hashlib.sha256()
    for name in (*METRICS, "positions", "contribution"):
        a = getattr(result, name)
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(sorted(result.diagnostics.items())).encode())
    h.update(repr(sorted(result.coverage.items())).encode())
    h.update(repr(result.bankruptcy_events).encode())
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        items = getattr(result, name) or []
        for x in items:
            h.update(np.asarray(astuple(x)[:9], dtype=float).tobytes())
    return h.hexdigest()


def corpora(count):
    from .test_extended import extended_case

    for seed in range(count):
        yield f"spot-{seed}", generators.case(seed)
        yield f"margin-{seed}", extended_case(seed)
        yield f"futures-{seed}", generators_futures.case(seed)
        yield f"futures-rich-{seed}", generators_futures.rich_case(seed)
