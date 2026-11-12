"""P-007：一批日期的风格暴露并行计算，与单线程逐位相同。"""

import numpy as np
import pytest

import ashare.risk.series as series
from ashare.risk.series import default_threads, model_series
from test.risk.test_risk_review import config
from test.risk.test_series import source


def _same(a, b):
    assert a.date == b.date and a.metadata == b.metadata
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("batch_size", (1, 7, 64))
def test_threads_do_not_change_any_model(batch_size, tmp_path):
    p, c = source(70, 60), config()
    rng = np.random.default_rng(4)
    external = rng.normal(size=(70, 60, 2))
    options = dict(config=c, external=external, external_names=("x", "y"))
    single = list(model_series(p, threads=1, batch_size=batch_size, **options))
    for threads in (2, 5, 16):
        many = list(model_series(p, threads=threads, batch_size=batch_size, **options))
        for a, b in zip(many, single, strict=True):
            _same(a, b)
    # 带缓存的追加同样逐位相同。
    list(model_series(p, cache=tmp_path, end=69, threads=4, **options))
    appended = list(model_series(p, cache=tmp_path, start=69, threads=4, **options))
    _same(appended[0], single[-1])


def test_exposures_really_run_on_worker_threads(monkeypatch):
    import threading

    p, c = source(40, 50), config()
    names = set()
    original = series.exposures_at

    def record(*args, **kwargs):
        names.add(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(series, "exposures_at", record)
    list(model_series(p, config=c, threads=3, batch_size=16))
    assert any(name.startswith("risk-exposure") for name in names)
    names.clear()
    list(model_series(p, config=c, threads=1, batch_size=16))
    assert names == {threading.main_thread().name}


@pytest.mark.parametrize("value", (0, -1, True, 2.0, "4"))
def test_threads_validated(value):
    with pytest.raises(ValueError, match="threads"):
        next(model_series(source(10, 5), config=config(), threads=value))


def test_default_threads_environment(monkeypatch):
    monkeypatch.setenv("ASHARE_RISK_THREADS", "3")
    assert default_threads() == 3
    monkeypatch.setenv("ASHARE_RISK_THREADS", "0")
    with pytest.raises(ValueError, match="ASHARE_RISK_THREADS"):
        default_threads()
    monkeypatch.delenv("ASHARE_RISK_THREADS")
    assert default_threads() >= 1


def test_rolling_descriptors_advance_in_order_on_one_thread(monkeypatch):
    """滚动描述子是有状态的逐日递推：只许在调用线程上按日期递增调用，不能交给线程池。"""
    import threading

    from ashare.risk.rolling import RollingDescriptors

    calls = []
    original = RollingDescriptors.at

    def record(self, day):
        calls.append((threading.current_thread().name, day))
        return original(self, day)

    monkeypatch.setattr(RollingDescriptors, "at", record)
    list(model_series(source(50, 40), config=config(), threads=4, batch_size=9))
    assert {name for name, _ in calls} == {threading.main_thread().name}
    days = [day for _, day in calls]
    assert days == sorted(days) and len(days) == 50
