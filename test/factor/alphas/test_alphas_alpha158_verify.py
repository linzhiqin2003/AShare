"""核对工具：158 条在合成数据上与参考实现一致；故意改错的因子必须报不一致。"""

import json

import numpy as np
import pytest

from ashare.factor import call
from ashare.factor.alphas import ALPHA158_FORMULAS
from ashare.factor.formula import qlib
from ashare.factor.formula import verify as v
from ashare.factor.formula.testing import synthetic_source

MODULE = "ashare.factor.alphas.reference.alpha158"


def candidates(names=None):
    out = {}
    for _, name, formula in ALPHA158_FORMULAS:
        key = f"alpha158_{name.lower()}"
        if names is None or key in names:
            out[key] = qlib.to_expr(formula)
    return out


def test_all_alpha158_match_reference_on_synthetic_panel():
    source = synthetic_source(seed=1)
    rows = v.verify(
        candidates(), MODULE, source, keep=120, group=40, chunk=5, log=lambda _: None
    )
    assert len(rows) == 158
    bad = [
        (r["name"], r["mask_mismatch"], r["max_err"])
        for r in rows
        if r["verdict"] == "不一致"
    ]
    assert bad == []
    # 合成数据里有停牌、上市前后空行和并列价格：确实覆盖了缺失的格子。
    assert all(r["valid_cells"] < r["cells"] for r in rows)


@pytest.mark.parametrize(
    "name, wrong",
    [
        # 窗口差一天
        ("alpha158_ma5", lambda: qlib.to_expr("Mean($close, 6)/$close")),
        # 比较不填 0（NaN 传播）：只在停牌附近不同
        (
            "alpha158_cntp5",
            lambda: call(
                "ts_mean",
                call(
                    "gt",
                    qlib.FIELDS["close"],
                    call("ts_delay", qlib.FIELDS["close"], periods=1),
                ),
                window=5,
                min_periods=1,
            ),
        ),
        # IdxMax 并列取最近而不是最早
        (
            "alpha158_imax10",
            lambda: (
                call("ts_idxmax", qlib.FIELDS["high"], window=10, min_periods=1) / 10
            ),
        ),
    ],
)
def test_wrong_candidate_is_reported(name, wrong):
    source = synthetic_source(seed=2)
    rows = v.verify({name: wrong()}, MODULE, source, keep=120, log=lambda _: None)
    assert rows[0]["verdict"] == "不一致"


def test_cli_writes_json(tmp_path, monkeypatch):
    source = synthetic_source(seed=3)
    monkeypatch.setattr(v, "load_inputs", lambda **_: source)
    out = tmp_path / "r.json"
    code = v.main(
        [
            "alpha158",
            "--factors",
            "alpha158_kmid,alpha158_ma5,alpha158_imxd10",
            "--days",
            "100",
            "--workers",
            "0",
            "--out",
            str(out),
        ]
    )
    result = json.loads(out.read_text())
    assert code == 0
    assert result["summary"] and sum(result["summary"].values()) == 3
    assert {r["name"] for r in result["factors"]} == {
        "alpha158_kmid",
        "alpha158_ma5",
        "alpha158_imxd10",
    }


# ---- 已知例外 ------------------------------------------------------------------

WVMA5 = "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)"


def _zero_windows(frames):
    from ashare.factor.alphas.reference import qlib_alpha158 as ref

    return (
        ref.compute(
            frames, formula="Max(Abs($close/Ref($close, 1)-1)*$volume, 5)"
        ).to_numpy()
        == 0
    )


def _wvma_with_residue(frames):
    """模拟 pandas 的残差：窗口全为 0 的格子给出很大的数。"""
    from ashare.factor.alphas.reference import qlib_alpha158 as ref

    out = ref.compute(frames, formula=WVMA5).to_numpy().copy()
    out[_zero_windows(frames) & np.isfinite(out)] = 3.5e10
    return out


def _wvma_garbage_elsewhere(frames):
    out = _wvma_with_residue(frames)
    zero = _zero_windows(frames)
    finite = np.isfinite(out) & ~zero
    t, j = np.argwhere(finite)[-1]
    out[t, j] += 1.0  # 条件不成立的格子出错：例外不能救
    return out


REFERENCES = {}
REFERENCES.update(
    {
        "alpha158_wvma5": _wvma_with_residue,
        "wvma_garbage_elsewhere": _wvma_garbage_elsewhere,
    }
)


def test_known_exception_rescues_only_matching_cells():
    from ashare.factor.alphas import exceptions as k

    source = synthetic_source(seed=1)
    candidate = qlib.to_expr(WVMA5)
    rules = k.ALPHA158["alpha158_wvma5"]
    rows = v.verify(
        {"alpha158_wvma5": candidate, "wvma_garbage_elsewhere": candidate},
        __name__,
        source,
        keep=120,
        exceptions={"alpha158_wvma5": rules, "wvma_garbage_elsewhere": rules},
        log=lambda _: None,
    )
    by = {r["name"]: r for r in rows}
    good = by["alpha158_wvma5"]
    n = good["exception_cells"]
    assert n > 0  # 合成数据里有常数价格的股票（第 5 只）
    assert good["verdict"] in (
        f"一致（已知例外 {n} 格）",
        f"仅浮点差（已知例外 {n} 格）",
    )
    assert v.category(good).endswith("（含已知例外）")
    assert good["known_exceptions"][0]["ref"] == "Q-Q-06"
    assert len(good["known_exceptions"][0]["samples"]) == min(5, n)
    assert good["known_exceptions"][0]["samples"][0]["candidate"] == 0.0
    # 例外只救条件成立的格子：别处出错照样不一致。
    assert by["wvma_garbage_elsewhere"]["verdict"] == "不一致"
    # 不带例外时，同样的数据是不一致。
    plain = v.verify(
        {"alpha158_wvma5": candidate}, __name__, source, keep=120, log=lambda _: None
    )
    assert plain[0]["verdict"] == "不一致" and plain[0]["exception_cells"] == 0
