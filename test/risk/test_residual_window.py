"""P-005：残差历史的双份环形缓冲。"""

from dataclasses import replace

import numpy as np
import pytest

from ashare.risk.series import model_series
from ashare.risk.window import ResidualWindow
from test.risk.test_risk_review import config
from test.risk.test_series import source


def test_windows_are_contiguous_views_in_time_order():
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(50, 6))
    rows[rng.random(rows.shape) < 0.2] = np.nan
    ring = ResidualWindow(6, 7)
    for day in range(50):
        leaving = ring.leaving(day)
        if day < 7:
            assert leaving is None
        else:
            np.testing.assert_array_equal(leaving, rows[day - 7])
        ring.push(day, rows[day])
        for first in range(max(0, day - 6), day + 1):
            view = ring.window(first, day)
            assert view.flags.c_contiguous and np.shares_memory(view, ring.rows)
            np.testing.assert_array_equal(view, rows[first : day + 1])
    with pytest.raises(ValueError, match="覆盖"):
        ring.window(42, 45)
    with pytest.raises(ValueError, match="不在缓冲内"):
        ring.window(43, 50)
    with pytest.raises(ValueError, match="逐日连续"):
        ring.push(51, rows[0])


@pytest.mark.parametrize("history", (6, 15, 40))
def test_scratch_mode_window_equals_stacked_history(history, monkeypatch):
    """从头算模式下窗口来自环形缓冲，与旧实现「把最近 history 天残差拼成数组」逐位相同。"""
    import ashare.risk.series as series

    p = source(60, 70)
    c = replace(config(history=history), specific_incremental=False)
    pushed, checked = [], []
    push = ResidualWindow.push

    def record_push(self, day, residual):
        pushed.append(np.array(residual))
        return push(self, day, residual)

    original = series.estimate_covariances

    def record(fs, rs, *args, **kwargs):
        expected = np.asarray(pushed[-history:])
        assert rs.dtype == expected.dtype and rs.flags.c_contiguous
        np.testing.assert_array_equal(rs, expected)
        checked.append(len(rs))
        return original(fs, rs, *args, **kwargs)

    monkeypatch.setattr(ResidualWindow, "push", record_push)
    monkeypatch.setattr(series, "estimate_covariances", record)
    models = list(model_series(p, config=c))
    assert len(models) == 60 and len(pushed) == 60
    assert len(checked) > 30 and max(checked) == history
