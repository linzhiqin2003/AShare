"""Deterministic NumPy-only operators on time-by-security panels."""

from . import api as _api

__version__ = "0.1.0"
build_id = _api._core.BUILD_ID
__all__ = [
    *("ts_" + op for op in _api.TS_OPS),
    *("cs_" + op for op in _api.CS_OPS),
    "elementwise",
    "ic",
    "rank_ic",
    "rank_ic_many",
    "factor_autocorr",
    "group_returns",
    "group_turnover",
    "targets",
    "SparseTargets",
    "set_num_threads",
    "get_num_threads",
]
globals().update({name: getattr(_api, name) for name in __all__})
