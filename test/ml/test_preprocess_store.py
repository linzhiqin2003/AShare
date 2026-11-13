"""截面预处理（对逐日循环的参考实现）与特征库（内存/磁盘一致、缓存、增量追加）。"""

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402
from ashare.factor import Evaluator, call, field  # noqa: E402

from .synthetic import SynthPanel, make  # noqa: E402


def _reference(x, mask, pre, groups=None, size=None):
    """逐日、逐步的直白实现，与正式实现不共用代码。"""
    t, n = x.shape
    out = np.full((t, n), np.nan)
    for d in range(t):
        m = mask[d]
        v = np.where(m & np.isfinite(x[d]), x[d], np.nan)
        ok = np.isfinite(v)
        if pre.winsor == "quantile" and ok.sum():
            lo, hi = np.quantile(v[ok], [pre.lower, pre.upper])
            v[ok] = np.clip(v[ok], lo, hi)
        if pre.neutralize and ok.sum():
            cols = [np.ones(ok.sum())]
            if "size" in pre.neutralize:
                good = ok & np.isfinite(size[d])
                v[ok & ~good] = np.nan
                ok = good
                cols = [np.ones(ok.sum()), size[d][ok]]
            if "industry" in pre.neutralize:
                g = groups[d][ok]
                for k in np.unique(g)[1:]:
                    cols.append((g == k).astype(float))
            A = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
            v[ok] = v[ok] - A @ coef
        if pre.standardize == "zscore" and ok.sum():
            s = v[ok].std()
            v[ok] = (v[ok] - v[ok].mean()) / s if s > 0 else np.nan
        elif pre.standardize == "rank" and ok.sum():
            r = v[ok].argsort().argsort().astype(float)
            v[ok] = (r / (ok.sum() - 1) if ok.sum() > 1 else 0.5) - 0.5
        if pre.fill == "zero":
            v[m & ~np.isfinite(v)] = 0.0
        v[~m] = np.nan
        out[d] = v
    return out


@pytest.mark.parametrize(
    "pre",
    [
        ml.Preprocess(),
        ml.Preprocess(standardize="rank", fill="none"),
        ml.Preprocess(winsor=None, neutralize=("industry",)),
        ml.Preprocess(neutralize=("industry", "size"), fill="none"),
    ],
)
def test_preprocess_matches_daily_loop(pre):
    rng = np.random.default_rng(1)
    t, n = 30, 40
    x = rng.standard_normal((t, n)) ** 3
    x[rng.random((t, n)) < 0.1] = np.nan
    mask = rng.random((t, n)) < 0.85
    groups = rng.integers(0, 4, (t, n))
    cap = np.exp(rng.standard_normal((t, n)) + 20)
    panel = SynthPanel(
        {"industry": groups.astype(np.uint32), "total_market_cap": cap},
        [f"2020{i:04d}" for i in range(t)],
        [f"S{i}" for i in range(n)],
    )
    got = pre.apply(x, mask, pre.context(panel))
    ref = _reference(x, mask, pre, groups, np.log(cap))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got, ref, atol=1e-9, equal_nan=True)


def test_preprocess_uses_only_same_day():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, 30))
    mask = np.ones_like(x, bool)
    pre = ml.Preprocess()
    a = pre.apply(x, mask)
    y = x.copy()
    y[12:] = rng.standard_normal((8, 30)) * 100
    b = pre.apply(y, mask)
    np.testing.assert_array_equal(a[:12], b[:12])


def _expr_panel(t=120, n=15, seed=0):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0))
    return SynthPanel(
        {"close": close, "volume": rng.random((t, n)) + 1},
        [f"2020{i:04d}" for i in range(t)],
        [f"S{i}" for i in range(n)],
    )


def test_store_memory_disk_and_reference(tmp_path):
    panel = _expr_panel()
    mask = np.random.default_rng(1).random(panel.shape) < 0.9
    feats = {
        "mom": field("close") / call("ts_delay", field("close"), periods=5) - 1,
        "vol": call("ts_mean", field("volume"), window=10),
        "arr": np.random.default_rng(3).standard_normal(panel.shape),
    }
    pre = ml.Preprocess()
    mem = ml.build_store(feats, panel, mask, pre, memory=True)
    disk = ml.build_store(feats, panel, mask, pre, root=tmp_path)
    assert mem.n_samples == mask.sum()
    for name in feats:
        np.testing.assert_array_equal(np.asarray(disk.column(name)), mem.column(name))
        value = feats[name]
        raw = (
            value
            if isinstance(value, np.ndarray)
            else Evaluator().evaluate(value, panel)
        )
        expect = pre.apply(raw, mask)[mask].astype(np.float32)
        np.testing.assert_array_equal(mem.column(name), expect)
    # 样本按（日，列）排序，day_ptr 切片就是那一天
    d = 50
    sl = mem.samples(d, d + 1)
    np.testing.assert_array_equal(mem.cols[sl], np.flatnonzero(mask[d]))
    m = mem.matrix(sl)
    assert (
        m.dtype == np.float32 and m.flags.c_contiguous and m.shape == (mask[d].sum(), 3)
    )
    panel_back = mem.scatter(mem.column("arr").astype(np.float64))
    np.testing.assert_array_equal(np.isfinite(panel_back), mask)
    # 第二次命中缓存：不重写文件
    files = {p: p.stat().st_mtime_ns for p in disk.path.glob("*.f32")}
    again = ml.build_store(feats, panel, mask, pre, root=tmp_path)
    assert {p: p.stat().st_mtime_ns for p in again.path.glob("*.f32")} == files
    # 加一个特征只算新的那一列
    more = dict(feats, vol20=call("ts_mean", field("volume"), window=20))
    third = ml.build_store(more, panel, mask, pre, root=tmp_path)
    assert {
        p: p.stat().st_mtime_ns for p in third.path.glob("*.f32") if p in files
    } == files
    assert len(list(third.path.glob("*.f32"))) == 4


def test_store_incremental_append_equals_full(tmp_path):
    full = _expr_panel(t=150)
    mask = np.random.default_rng(1).random(full.shape) < 0.9
    feats = {
        "mom": field("close") / call("ts_delay", field("close"), periods=5) - 1,
        "ema": call("ts_ema", field("volume"), window=10),
    }
    head = SynthPanel(
        {k: full[k][:100] for k in ("close", "volume")},
        full.dates[:100],
        full.symbols,
        version="v1",
    )
    logs = []
    ml.build_store(
        feats,
        head,
        mask[:100],
        root=tmp_path,
        incremental=True,
        universe_id="u",
        log=logs.append,
    )
    grown = SynthPanel(
        {k: full[k] for k in ("close", "volume")},
        full.dates,
        full.symbols,
        version="v2",
    )
    inc = ml.build_store(
        feats,
        grown,
        mask,
        root=tmp_path,
        incremental=True,
        universe_id="u",
        log=logs.append,
    )
    assert any("mom 追加 50 天" in m for m in logs), logs
    ref = ml.build_store(feats, grown, mask, memory=True)
    for name in feats:
        np.testing.assert_array_equal(np.asarray(inc.column(name)), ref.column(name))
    # 已存日期的数据被改过：尾部核对不一致 → 整列重算
    changed = full["close"].copy()
    changed[90:, ::2] *= 1.5  # 只改一半股票：整体缩放会被截面标准化抵消
    bad = SynthPanel(
        {"close": changed, "volume": full["volume"]},
        full.dates,
        full.symbols,
        version="v3",
    )
    longer_mask = mask
    logs.clear()
    # 先把库截回 100 天的状态：用新的 universe_id 建 100 天再追加
    head2 = SynthPanel(
        {k: full[k][:100] for k in ("close", "volume")},
        full.dates[:100],
        full.symbols,
        version="v1",
    )
    ml.build_store(
        feats, head2, mask[:100], root=tmp_path, incremental=True, universe_id="u2"
    )
    out = ml.build_store(
        feats,
        bad,
        longer_mask,
        root=tmp_path,
        incremental=True,
        universe_id="u2",
        log=logs.append,
    )
    assert any("尾部核对不一致" in m for m in logs), logs
    ref = ml.build_store(feats, bad, mask, memory=True)
    for name in feats:
        np.testing.assert_array_equal(np.asarray(out.column(name)), ref.column(name))


def test_store_rejects_bad_inputs():
    panel, feats, _ = make(t=30, n=5, k=1)
    with pytest.raises(ValueError):
        ml.build_store(
            {"x": np.zeros((3, 3))}, panel, np.ones(panel.shape, bool), memory=True
        )
    with pytest.raises(ValueError):
        ml.build_store({}, panel, np.ones(panel.shape, bool), memory=True)
    with pytest.raises(ValueError):
        ml.build_store(feats, panel, np.ones((2, 2), bool), memory=True)


def test_basic_features_evaluate_and_use_only_past():
    from ashare.factor import FIELDS

    feats = ml.basic_features()
    assert len(feats) >= 40 and len(set(feats)) == len(feats)
    fields = ml.required_fields(feats, "ret5")
    assert set(fields) <= set(FIELDS) | {"can_buy_open", "can_sell_open"}
    rng = np.random.default_rng(0)
    t, n = 320, 12
    arrays = {f: np.abs(rng.standard_normal((t, n))) + 1 for f in fields}
    for b in ("is_listed", "has_quote", "can_buy_open", "can_sell_open"):
        arrays[b] = np.ones((t, n), bool)
    arrays["is_st"] = np.zeros((t, n), bool)
    arrays["listing_age"] = np.full((t, n), 500)
    panel = SynthPanel(
        arrays, [f"2020{i:04d}" for i in range(t)], [f"S{i}" for i in range(n)]
    )
    store = ml.build_store(feats, panel, np.ones((t, n), bool), memory=True)
    # 后半段数据全换掉，前半段的特征值逐位不变（表达式只看过去）
    changed = SynthPanel(
        {
            k: (
                v.copy()
                if v.dtype == bool
                else np.where(np.arange(t)[:, None] >= 200, v * 3 + 1, v)
            )
            for k, v in arrays.items()
        },
        panel.dates,
        panel.symbols,
    )
    other = ml.build_store(feats, changed, np.ones((t, n), bool), memory=True)
    cut = store.samples(0, 200)
    for name in feats:
        np.testing.assert_array_equal(store.column(name)[cut], other.column(name)[cut])
        assert np.isfinite(store.column(name)[store.samples(260, 320)]).any(), name


def test_default_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ASHARE_ML_DIR", raising=False)
    monkeypatch.setenv("ASHARE_DATA_DIR", str(tmp_path / "d"))
    assert ml.default_root() == tmp_path / "d" / "ml"
    monkeypatch.setenv("ASHARE_ML_DIR", str(tmp_path / "x"))
    assert ml.default_root() == tmp_path / "x"
    assert ml.Ledger().path == tmp_path / "x" / "ledger.jsonl"
