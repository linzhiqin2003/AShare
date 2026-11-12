"""注册表的元数据标记、从翻译模块构造注册表、外部因子包的发现与加载。"""

import types
from importlib import metadata

import numpy as np
import pytest

from ashare.factor import Evaluator, call, field
from ashare.factor.formula import (
    FactorSet,
    Hub,
    discover,
    load_registry,
    plugins,
    reference_frames,
    registry_from_module,
)
from ashare.factor.formula import (
    dsl as d,
)
from ashare.factor.formula.testing import synthetic_source
from ashare.factor.library import Registry


def test_register_validates_notes_and_filters_by_tag():
    registry = Registry()
    registry.register("a", field("x"), "", [("两读", "D-13")])
    registry.register("b", field("x"))
    assert registry["a"].notes == (("两读", "D-13"),) and registry["b"].notes == ()
    assert [f.name for f in registry.with_note("两读")] == ["a"]
    for bad in [
        ("两读",),
        "两读",
        None,
        [("两读", "")],
        [("两读", 3)],
        [("a", "b", "c")],
        [3],
    ]:
        with pytest.raises(ValueError):
            Registry().register("c", field("x"), "", bad)
    # 生成器只能遍历一次：标记不许被静默丢掉。
    gen = registry.register("g", field("x"), "", (n for n in [("照抄", "X-1")]))
    assert gen.notes == (("照抄", "X-1"),)


def _fake_module():
    module = types.ModuleType("fake_set")

    def mine_001():
        """argmax(high, 7)"""
        return d.ts_argmax(d.high, 7)

    def mine_002():
        """close"""
        return d.close

    module.mine_001, module.mine_002 = mine_001, mine_002
    module.helper = lambda: None
    return module


def test_registry_from_module_notes_and_argmax_reading():
    module = _fake_module()
    notes = {"mine_001": (("两读", "D-13"),), "mine_003": (("照抄", "X"),)}
    registry = registry_from_module(module, "mine", 10, notes=notes)
    assert registry.names() == ("mine_001", "mine_002")
    assert registry["mine_001"].notes == (("两读", "D-13"),)
    assert registry["mine_001"].description == "argmax(high, 7)"
    assert registry["mine_001"].expression == call(
        "ts_argmax", d.high, window=7, min_periods=7
    )
    with d.argmax_reading("position"):
        position = registry_from_module(module, "mine", 10)
    assert position["mine_001"].expression == call(
        "ts_idxmax", d.high, window=7, min_periods=7, ties="earliest"
    )
    # 构造完以后回到默认读法。
    assert d.ts_argmax(d.high, 7) == call("ts_argmax", d.high, window=7, min_periods=7)
    with pytest.raises(ValueError, match="不收"):
        registry_from_module(module, "mine", 10, excluded={2: "X-2"})
    with pytest.raises(ValueError, match="越界"):
        registry_from_module(module, "mine", 1)
    with pytest.raises(ValueError, match="不属于"):
        registry_from_module(module, "mine", 10, notes={"other_001": ()})


def test_argmax_readings_evaluate_consistently():
    """无并列时 位置 = 窗口 − 距今天数；有并列时最早的位置更靠前。"""
    source = synthetic_source(seed=7)
    e = Evaluator()
    a = e.evaluate(d.ts_argmax(d.close, 5), source)
    with d.argmax_reading("position"):
        expression = d.ts_argmax(d.close, 5)
    b = e.evaluate(expression, source)
    ok = np.isfinite(a)
    assert ok.any() and (np.isfinite(b) == ok).all()
    assert (b[ok] <= 5 - a[ok]).all()
    assert (b[ok] == 5 - a[ok]).sum() > 0.5 * ok.sum()


# ---- 外部因子包的发现 ------------------------------------------------------------


def _build(scale="1"):
    registry = Registry()
    registry.register("toy_close", d.close * float(scale), "close * scale")
    return registry


def register_toy(hub):
    hub.add(FactorSet("toy", _build, options=("scale",), description="测试"))


def register_twice(hub):
    register_toy(hub)
    register_toy(hub)


def register_broken(hub):
    raise RuntimeError("坏了")


def register_not_a_set(hub):
    hub.add("toy")


def register_bad_name(hub):
    hub.add(FactorSet("not a name", _build))


def _entry_point(name, target):
    return metadata.EntryPoint(
        name=name, value=f"{__name__}:{target}", group=plugins.ENTRY_POINT_GROUP
    )


def test_discover_with_entry_points():
    sets = discover(entry_points=[_entry_point("toy", "register_toy")])
    assert set(sets) == {"example", "toy"}
    toy = sets["toy"]
    assert toy.origin == f"ashare.factors:toy = {__name__}:register_toy"
    assert toy.registry(scale="2")["toy_close"].expression == d.close * 2.0
    with pytest.raises(ValueError, match="不认识选项"):
        toy.registry(window="3")
    with pytest.raises(plugins.FactorPluginError, match="ashare.factors:twice.*重名"):
        discover(entry_points=[_entry_point("twice", "register_twice")])
    with pytest.raises(
        plugins.FactorPluginError, match="ashare.factors:broken.*坏了"
    ) as info:
        discover(entry_points=[_entry_point("broken", "register_broken")])
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(
        plugins.FactorPluginError, match="ashare.factors:not_a_set.*FactorSet"
    ):
        discover(entry_points=[_entry_point("not_a_set", "register_not_a_set")])
    with pytest.raises(
        plugins.FactorPluginError, match="ashare.factors:bad_name.*标识符"
    ):
        discover(entry_points=[_entry_point("bad_name", "register_bad_name")])
    with pytest.raises(ImportError, match="missing"):
        discover(entry_points=[_entry_point("missing", "no_such_function")])
    with pytest.raises(TypeError):
        discover(entry_points=[_entry_point("bad", "_entry_point")])


def test_hub_rejects_bad_sets():
    hub = Hub()
    with pytest.raises(TypeError):
        hub.add(object())
    with pytest.raises(ValueError):
        hub.add(FactorSet("not a name", _build))
    hub.add(FactorSet("bad_build", lambda: 3))
    with pytest.raises(TypeError, match="Registry"):
        hub.sets["bad_build"].registry()


def test_builtin_example_set_without_external_packages():
    registry = load_registry("example")
    assert registry.names() == ("example_spread_mean", "example_amount_peak")
    assert all(f.notes == (("示例", "EX-1"),) for f in registry)
    with pytest.raises(KeyError):
        load_registry("no_such_set")


def test_reference_frames_returns_match_dsl():
    """参考实现的收益与 DSL 同一口径（上市首日为 NaN）。"""
    source = synthetic_source(seed=4)
    frames = reference_frames(source)
    got = Evaluator().evaluate(d.returns, source)
    np.testing.assert_array_equal(frames["returns"].to_numpy(), got)


def test_indneutralize_through_public_source():
    """数据层的行业编码是 uint32：经 PublicSource 求 industry_l1 和行业中性化，与 pandas 对拍。"""
    source = synthetic_source(seed=5)
    assert np.asarray(source.arrays["industry"]).dtype == np.uint32
    groups = source["industry_l1"]
    assert groups.dtype == np.int64 and (groups == -1).any()
    got = Evaluator().evaluate(d.indneutralize(d.close, "sector"), source)
    frames = reference_frames(source)
    close, industry = frames["close"], frames["industry"]
    expected = np.full(close.shape, np.nan)
    for t in range(close.shape[0]):
        row = close.iloc[t]
        ok = row.notna() & industry.iloc[t].notna()
        means = row[ok].groupby(industry.iloc[t][ok]).transform("mean")
        expected[t, np.flatnonzero(ok.to_numpy())] = (row[ok] - means).to_numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert np.isnan(got[np.isnan(expected)]).all()
