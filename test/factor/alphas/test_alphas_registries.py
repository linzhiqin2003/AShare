"""注册表：notes 标记、D-13 读法开关、国君不收的条目；标记表与约定文件一致。"""

import re
from pathlib import Path

import numpy as np
import pytest

from ashare.factor import Evaluator, call
from ashare.factor.alphas import (
    alpha101,
    alpha101_registry,
    alpha158_registry,
    gtja191,
    gtja191_registry,
    notes,
)
from ashare.factor.formula import dsl as d
from ashare.factor.formula import load_registry
from ashare.factor.formula.testing import synthetic_source

ROOT = Path(__file__).resolve().parents[3]


def test_alpha158_notes():
    registry = alpha158_registry()
    assert registry["alpha158_imax5"].notes == (("照抄", "D-16"),)
    assert registry["alpha158_imxd60"].notes == (("照抄", "D-16"),)
    assert registry["alpha158_roc10"].notes == (("照抄", "Q-Q-02"),)
    assert registry["alpha158_kmid"].notes == ()
    assert len(registry.with_note("照抄")) == 20


def _alpha101_001():
    """(rank(Ts_ArgMax(x, 5)) - 0.5)"""
    return d.rank(d.ts_argmax(d.close, 5)) - 0.5


def test_alpha101_registry_argmax_switch_and_notes(monkeypatch):
    monkeypatch.setattr(alpha101, "alpha101_001", _alpha101_001, raising=False)
    days = alpha101_registry()["alpha101_001"]
    position = alpha101_registry(argmax="position")["alpha101_001"]
    assert days.notes == (("两读", "D-13"),)
    assert (
        days.expression
        == d.rank(call("ts_argmax", d.close, window=5, min_periods=5)) - 0.5
    )
    assert (
        position.expression
        == d.rank(call("ts_idxmax", d.close, window=5, min_periods=5, ties="earliest"))
        - 0.5
    )
    # 两种读法求值：无并列时 位置 = 窗口 − 距今天数。
    source = synthetic_source(seed=7)
    e = Evaluator()
    a = e.evaluate(call("ts_argmax", d.close, window=5, min_periods=5), source)
    b = e.evaluate(
        call("ts_idxmax", d.close, window=5, min_periods=5, ties="earliest"), source
    )
    ok = np.isfinite(a)
    assert ok.any()
    # 无并列时 位置 = 窗口 − 距今天数；有并列时最早的位置更靠前。
    assert (b[ok] <= 5 - a[ok]).all()
    assert (b[ok] == 5 - a[ok]).sum() > 0.5 * ok.sum()
    # 注册表构造完以后，默认读法恢复。
    assert d.ts_argmax(d.close, 5) == call(
        "ts_argmax", d.close, window=5, min_periods=5
    )
    with pytest.raises(ValueError):
        alpha101_registry(argmax="index")


def test_gtja191_excluded_entries_are_rejected(monkeypatch):
    def gtja191_030():
        return d.close

    monkeypatch.setattr(gtja191, "gtja191_030", gtja191_030, raising=False)
    with pytest.raises(ValueError, match="不收"):
        gtja191_registry()


def test_gtja191_notes_attach(monkeypatch):
    def gtja191_149():
        """REGBETA(...)"""
        return d.regbeta(d.returns, d.benchmark_close, 252, min_pairs=20)

    monkeypatch.setattr(gtja191, "gtja191_149", gtja191_149, raising=False)
    factor = gtja191_registry()["gtja191_149"]
    assert ("缺失口径自定", "Q-G-149") in factor.notes


def test_notes_table_is_consistent():
    for table in (notes.ALPHA101_NOTES, notes.GTJA191_NOTES):
        for name, items in table.items():
            assert re.fullmatch(r"(alpha101|gtja191)_\d{3}", name)
            assert len(set(items)) == len(items)
            assert all(tag in notes.TAGS for tag, _ in items)
    assert not set(notes.GTJA191_EXCLUDED) & {int(n[-3:]) for n in notes.GTJA191_NOTES}
    # 每个编号都在约定文件里出现。
    text = (ROOT / "docs/factors/CONVENTIONS.md").read_text()
    refs = {
        ref
        for t in (notes._ALPHA101, notes._GTJA191, notes._ALPHA158)
        for _, ref, _ in t
    }
    assert all(ref in text for ref in refs), sorted(r for r in refs if r not in text)
    # 量纲混用：adv 与 volume 相除、相乘或比较的条目，按原文自动核对。
    import json

    rows = [
        json.loads(line)
        for line in (ROOT / "docs/factors/sources/alpha101.jsonl")
        .read_text()
        .splitlines()
    ]
    mixed = {
        r["n"]
        for r in rows
        if re.search(
            r"volume\s*/\s*adv|adv\d+\s*[<>]\s*volume|/\s*adv\d+", r["formula"]
        )
    }
    tagged = {
        int(n[-3:])
        for n, items in notes.ALPHA101_NOTES.items()
        if ("量纲混用", "D-05") in items
    }
    assert mixed == tagged
    industry = {
        r["n"]
        for r in rows
        if re.search(r"IndClass\.(industry|subindustry)", r["formula"])
    }
    tagged = {
        int(n[-3:])
        for n, items in notes.ALPHA101_NOTES.items()
        if ("行业层级降级", "D-06") in items
    }
    assert industry == tagged
    argmax = {r["n"] for r in rows if re.search(r"(?i)ts_arg(max|min)", r["formula"])}
    tagged = {
        int(n[-3:])
        for n, items in notes.ALPHA101_NOTES.items()
        if ("两读", "D-13") in items
    }
    assert argmax == tagged


def test_builtin_discovery():
    """三套公开因子是内置因子集（2026-11-14 起随仓库公开，不再经入口点）。"""
    from ashare.factor.formula import discover

    sets = discover(refresh=True)
    assert {"alpha158", "alpha101", "gtja191", "example"} <= set(sets)
    assert {sets[n].origin for n in ("alpha158", "alpha101", "gtja191")} == {"ashare"}
    assert len(load_registry("alpha158")) == 158
    assert sets["alpha158"].reference == "ashare.factor.alphas.reference.alpha158"
    assert set(sets["alpha158"].exceptions) == {
        f"alpha158_wvma{n}" for n in (5, 10, 20, 30, 60)
    }
    assert load_registry("alpha101", argmax="position").names() == (
        load_registry("alpha101").names()
    )
