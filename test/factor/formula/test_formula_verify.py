"""核对工具的机制：比较统计与结论、REFERENCES 接口、选项透传、已知例外、命令行。"""

import json

import numpy as np
import pytest

from ashare.factor.formula import plugins
from ashare.factor.formula import verify as v
from ashare.factor.formula.testing import synthetic_source


def test_compare_statistics_and_verdicts():
    ref = np.array([[1.0, 2.0, np.nan], [3.0, np.inf, 5.0]])
    same = v.compare(ref, ref)
    assert same["mask_mismatch"] == 0 and same["max_err"] == 0
    assert same["reference_valid_cells"] == 4  # inf 按缺失
    assert v.verdict(same) == "一致"
    noisy = ref * (1 + 1e-9)
    stats = v.compare(noisy, ref)
    assert v.verdict(stats) == "仅浮点差"
    assert stats["max_err"] == pytest.approx(5e-9 / 6, rel=1e-3)
    holed = ref.copy()
    holed[0, 0] = np.nan
    stats = v.compare(holed, ref, dates=["d0", "d1"], symbols=["a", "b", "c"])
    assert stats["mask_mismatch"] == 1 and stats["reference_only"] == 1
    assert stats["first_mask_mismatch"]["symbol"] == "a"
    assert v.verdict(stats) == "不一致"
    far = ref + 1e-3
    assert v.verdict(v.compare(far, ref)) == "不一致"


# ---- REFERENCES 字典、整截面计算 ------------------------------------------------


def _rank_close(frames):
    return frames["close"].rank(axis=1, pct=True)


def _rank_close_wrong(frames):
    # (秩−1)/(个数−1)：另一种百分位口径，必须被报出来。
    return (
        frames["close"]
        .rank(axis=1, pct=False)
        .sub(1)
        .div(frames["close"].count(axis=1) - 1, axis=0)
    )


REFERENCES = {"x_rank": _rank_close, "x_rank_wrong": _rank_close_wrong}


def test_references_dict_and_cross_sections():
    from ashare.factor.formula import dsl as d

    source = synthetic_source(seed=6)
    candidates = {"x_rank": d.rank(d.close), "x_rank_wrong": d.rank(d.close)}
    rows = v.verify(candidates, __name__, source, keep=100, log=lambda _: None)
    verdicts = {r["name"]: r["verdict"] for r in rows}
    assert verdicts["x_rank"] == "一致"
    assert verdicts["x_rank_wrong"] == "不一致"


def _scaled_close(frames, scale="1"):
    return frames["close"] * float(scale)


def _many(frames, names, **options):
    return {n: _scaled_close(frames, **options).to_numpy() for n in names}


REFERENCES["x_scaled"] = _scaled_close


def test_options_reach_reference(monkeypatch):
    from ashare.factor.formula import dsl as d

    source = synthetic_source(seed=6)
    rows = v.verify(
        {"x_scaled": d.close * 2},
        __name__,
        source,
        keep=50,
        options={"scale": "2"},
        log=lambda _: None,
    )
    assert rows[0]["verdict"] == "一致"
    rows = v.verify(
        {"x_scaled": d.close * 2}, __name__, source, keep=50, log=lambda _: None
    )
    assert rows[0]["verdict"] == "不一致"


# ---- 已知例外 ------------------------------------------------------------------


def test_apply_exceptions_requires_declared_candidate_value():
    from ashare.factor import field
    from ashare.factor.formula.known_exceptions import KnownException

    rule = KnownException("X-1", field("c"), 0.0, "测试")
    cand = np.array([[0.0, 0.0, 1.0, 0.0, 2.0]])
    ref = np.array([[5.0, 0.0, 9.0, np.nan, 2.0]])
    cond = np.array([[1.0, 1.0, 1.0, 1.0, np.nan]])
    stats = v.compare(cand, ref, exceptions=[(rule, cond)])
    # 第 0 格：条件成立、不过关、候选为 0 → 例外；第 1 格本来就一致 → 不算例外；
    # 第 2 格：候选不是 0 → 照样不一致；第 3 格掩码不同、候选为 0 → 例外；第 4 格条件 NaN。
    assert stats["exception_cells"] == 2
    assert stats["known_exceptions"][0]["condition_cells"] == 4
    assert v.verdict(stats) == "不一致"
    cand[0, 2] = 9.0
    stats = v.compare(cand, ref, exceptions=[(rule, cond)])
    assert v.verdict(stats) == "一致（已知例外 2 格）"
    assert v.category(stats) == "一致（含已知例外）"
    assert v.summary([stats, v.compare(ref, ref)]) == {
        "一致（含已知例外）": 1,
        "一致": 1,
    }


def test_exception_condition_is_evaluated_with_candidates():
    """例外条件是表达式，与候选在同一份数据上求值；只救条件成立、候选为声明值的格子。"""
    from ashare.factor.formula import dsl as d
    from ashare.factor.formula.known_exceptions import KnownException

    source = synthetic_source(seed=2)
    candidate = d.where(d.gt(d.volume, 0), d.close * 0, d.close)
    rule = KnownException("X-2", d.gt(d.volume, 0), 0.0, "测试")
    rows = v.verify(
        {"x_close": candidate},
        __name__,
        source,
        keep=100,
        exceptions={"x_close": (rule,), "not_verified": (rule,)},
        log=lambda _: None,
    )
    n = rows[0]["exception_cells"]
    assert n > 0 and rows[0]["verdict"] == f"一致（已知例外 {n} 格）"
    assert rows[0]["known_exceptions"][0]["condition_cells"] == n
    plain = v.verify(
        {"x_close": candidate}, __name__, source, keep=100, log=lambda _: None
    )
    assert plain[0]["verdict"] == "不一致"


def _x_close(frames):
    return frames["close"]


REFERENCES["x_close"] = _x_close


# ---- 命令行（内置示例集） ----------------------------------------------------------


def test_cli_runs_example_set(tmp_path, monkeypatch):
    source = synthetic_source(seed=3)
    monkeypatch.setattr(v, "load_inputs", lambda **_: source)
    out = tmp_path / "r.json"
    code = v.main(["example", "--days", "100", "--workers", "0", "--out", str(out)])
    result = json.loads(out.read_text())
    assert code == 0
    assert result["summary"] == {"一致": 2}
    assert {r["name"] for r in result["factors"]} == {
        "example_range_mean",
        "example_volume_turn",
    }
    with pytest.raises(SystemExit):
        v.main(["no_such_set", "--workers", "0"])
    with pytest.raises(SystemExit):
        v.main(["example", "--factors", "nope", "--workers", "0"])
    with pytest.raises(ValueError):
        v.main(["example", "--option", "argmax=position", "--workers", "0"])
    assert "example" in plugins.discover()
