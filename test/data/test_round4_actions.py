"""第四轮：差异化分红来源 5、裁决边界与前视守卫、独立校验。数字均为手算合成值。"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data.panel_actions as actions
from ashare.data.panel import build
from ashare.data.panel_actions import choose_action, dilution_ratio, reconcile
from ashare.data.panel_events import dividend_rows
from ashare.data.sources.base import FetchResult
from ashare.data.verify_panel import audit_actions, audit_panel

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

TOL = 0.005


def row(stock=0.0, cash=0.0, *, copies=1):
    return dict(stk_div=stock, cash_div_tax=cash, _records=copies)


def verify_k(previous, ratio, split, cash, source=5, k_min=0.85):
    """把单格交给独立校验，返回 (是否失败, k)。"""
    bad, _, k, _ = audit_actions(
        np.array([[previous]]),
        np.array([[ratio]]),
        np.array([[split]]),
        np.array([[cash]]),
        np.array([[source]], np.uint8),
        np.ones((1, 1), bool),
        tolerance=TOL,
        k_min=k_min,
    )
    return bool(bad[0, 0]), float(k[0, 0])


# (前收, 因子比, 公告送转合计, 公告每股现金, 手算 k)；k 用分数手算，不调实现。
HAND_CASES = [
    (46.10, 1.064173308808502, 0.0, 3.0, 0.9266628256833758),  # 000651.SZ 20210823
    (10.68, 1.37805, 0.4, 0.153, 0.9006729024211685),  # 688597.SH 20250623
    (21.83, 1.36918, 0.4, 0.0, 0.92295),  # 600699.SH 20190729
]


@pytest.mark.parametrize("previous,ratio,stock,cash,k", HAND_CASES)
def test_differential_dividend_keeps_announced_values(previous, ratio, stock, cash, k):
    assert choose_action([row(stock, cash)], ratio, previous, tolerance=TOL) == (
        1 + stock,
        cash,
        5,
    )
    assert dilution_ratio(previous, ratio, 1 + stock, cash) == pytest.approx(
        k, abs=1e-12
    )
    bad, independent = verify_k(previous, ratio, 1 + stock, cash)
    assert not bad and independent == pytest.approx(k, abs=1e-12)


def test_dilution_bounds_and_order():
    # 纯送转 s=1：k = a - 1。
    assert choose_action([row(1, 0)], 1.85, 10, tolerance=TOL) == (2, 0, 5)
    assert choose_action([row(1, 0)], 1.849, 10, tolerance=TOL) == (1.849, 0, 3)
    assert choose_action([row(1, 0)], 1.85, 10, tolerance=TOL, dilution_min=0.9) == (
        1.85,
        0,
        3,
    )
    # 因子比大于公告值（k>1）不是差异化分红。
    assert choose_action([row(1, 0)], 2.02, 10, tolerance=TOL) == (2.02, 0, 3)
    # 偏差 0.3% 时普通候选先采信，不能被摊薄候选抢走。
    assert choose_action([row(1, 0)], 1.994, 10, tolerance=TOL) == (2, 0, 1)
    assert verify_k(10, 1.849, 2, 0) == (True, pytest.approx(0.849))
    assert verify_k(10, 2.02, 2, 0)[0]
    assert verify_k(10, 1.85, 2, 0, k_min=0.9)[0]
    assert not verify_k(10, 1.85, 2, 0)[0]


def test_candidate_order_and_tolerance_boundary():
    # X1：全部与仅现金都在容差内时取全部（保留送转）。
    ratio = 100 * 1.002 / 99
    assert choose_action([row(0.002, 1.0)], ratio, 100, tolerance=TOL) == (
        1.002,
        1.0,
        1,
    )
    # X2：偏差 -0.398% 采信，-0.596% 不采信（k>1，也不是差异化分红）。
    theta = 10 / 9
    assert choose_action([row(0, 1)], theta * 1.004, 10, tolerance=TOL) == (1, 1, 1)
    out = choose_action([row(0, 1)], theta * 1.006, 10, tolerance=TOL)
    assert out[1:] == (0, 3) and out[0] == pytest.approx(theta * 1.006)


def table_of(*records):
    return pl.DataFrame(
        [
            dict(
                ts_code="A",
                ex_date=ex,
                imp_ann_date=known,
                ann_date=known,
                div_proc="实施",
                stk_div=stock,
                cash_div_tax=cash,
            )
            for ex, known, stock, cash in records
        ]
    )


DAYS = np.array(
    ["20260105", "20260106", "20260107", "20260108", "20260109", "20260112", "20260113"]
)


def arrays_of(close, factor):
    n = len(close)
    return dict(
        close=np.array(close, float)[:, None],
        adj_factor=np.array(factor, float)[:, None],
        is_listed=np.ones((n, 1), bool),
        split=np.ones((n, 1)),
        cash_dividend=np.zeros((n, 1)),
        action_source=np.zeros((n, 1), np.uint8),
    )


def test_late_announcement_filtered_by_dividend_rows():
    # X11：公告日晚于除权日的记录不进入有效集合。
    valid, stats = dividend_rows(
        table_of(("20260106", "20260107", 0, 1), ("20260108", "20260105", 0, 1))
    )
    assert valid["ex_date"].to_list() == ["20260108"]
    assert stats["late_announcement"] == 1


def test_shift_requires_known_target():
    # X4：最近的因子变化日（d2）早于公告，必须挪到 d5。
    f = 1.05
    arrays = arrays_of(
        [10, 10, 10, np.nan, 10, 9, 9], [1, 1, f, f, f, f * 10 / 9, f * 10 / 9]
    )
    meta = reconcile(arrays, DAYS, ["A"], table_of((DAYS[3], DAYS[3], 0, 1)))
    assert arrays["action_source"][:, 0].tolist() == [0, 0, 3, 0, 0, 2, 0]
    assert arrays["cash_dividend"][5, 0] == 1 and meta["shifted_records"] == 1


def test_reconcile_guard_holds_without_upstream_filter(monkeypatch):
    # X5：即便上游过滤失效，记账层也不得采信公告晚于记账日的记录。
    real = dividend_rows

    def leaky(table, dates=None, symbols=None):
        valid, stats = real(
            table.with_columns(
                pl.col("ex_date").alias("imp_ann_date"),
                pl.col("ex_date").alias("ann_date"),
            ),
            dates,
            symbols,
        )
        return valid.with_columns(pl.lit("20260107").alias("_known")), stats

    monkeypatch.setattr(actions, "dividend_rows", leaky)
    arrays = arrays_of([10, 9, 9], [1, 10 / 9, 10 / 9])
    reconcile(arrays, DAYS[:3], ["A"], table_of((DAYS[1], DAYS[2], 0, 1)))
    assert arrays["action_source"][:, 0].tolist() == [0, 3, 0]
    assert arrays["cash_dividend"][1, 0] == 0


def test_differential_requires_known_record():
    # 时点：差异化分红的记录晚公告时只能按因子合成。
    a = 1.064173308808502
    late = arrays_of([46.10, 46.10 / a], [1, a])
    reconcile(late, DAYS[:2], ["A"], table_of((DAYS[1], DAYS[1], 0, 3.0)))
    assert late["action_source"][1, 0] == 5 and late["cash_dividend"][1, 0] == 3
    late = arrays_of([46.10, 46.10 / a], [1, a])
    reconcile(late, DAYS[:2], ["A"], table_of((DAYS[1], "20260107", 0, 3.0)))
    assert late["action_source"][1, 0] == 3 and late["cash_dividend"][1, 0] == 0


def test_factor_noise_threshold():
    # X7：0.3% 的无记录因子变化要合成；0.05% 视为噪声。
    arrays = arrays_of([10, 10, 10], [1, 1.003, 1.003 * 1.0005])
    empty = table_of(("20990101", "20990101", 0, 1)).head(0)
    reconcile(arrays, DAYS[:3], ["A"], empty)
    assert arrays["action_source"][:, 0].tolist() == [0, 3, 0]
    assert arrays["split"][1, 0] == pytest.approx(1.003)


def test_shift_tie_prefers_earlier_day():
    # X10：前后等距两个因子变化日时取较早日。
    arrays = arrays_of(
        [10, 10, 9, np.nan, 9, 9, 9], [1, 1, 10 / 9, 10 / 9, 1.2, 1.2, 1.2]
    )
    reconcile(arrays, DAYS, ["A"], table_of((DAYS[3], DAYS[1], 0, 1)))
    assert arrays["action_source"][:, 0].tolist() == [0, 0, 2, 0, 3, 0, 0]
    assert arrays["cash_dividend"][2, 0] == 1


def tamper(store, name, i, j, value):
    import json

    path = (
        store.root
        / "panel"
        / json.loads((store.root / "panel/current.json").read_text())["version"]
    )
    target = path / f"{name}.npy"
    target.chmod(0o644)
    array = np.load(target, mmap_mode="r+")
    array[i, j] = value
    array.flush()
    del array


def test_verify_fails_on_bad_panel(tables, make_store):
    # X24：对账项与来源 5 项各给一个应该失败的面板。
    store = make_store(tables)
    meta = build(store)
    dates = meta["dates"]
    clean = audit_panel(store, dates)["reconciliation"]
    assert clean["failed"] == 0 and clean["checked"] >= 1
    tamper(store, "cash_dividend", 3, 2, 0.3)  # 偏差约 2%
    broken = audit_panel(store, dates)["reconciliation"]
    assert broken["failed"] == 1 and broken["examples"][0]["date"] == "20260107"
    tamper(store, "action_source", 3, 2, 5)  # k≈0.833 不在 [0.85,1)
    broken = audit_panel(store, dates)["reconciliation"]
    assert broken["failed"] == 1 and broken["diluted_failed"] == 1
    from ashare.data.verify import verify

    report = verify(store)
    assert report["checks"]["action_reconciliation"]["failed"] == 1


def test_tolerance_and_k_min_enter_version(tables, make_store):
    # X21 与 k_min：改参数必须得到新代，不能拿回旧面板。
    store = make_store(tables)
    first = build(store)
    second = build(store, action_tolerance=0.004)
    third = build(store, action_k_min=0.9)
    assert len({first["version"], second["version"], third["version"]}) == 3
    assert second["action_tolerance"] == 0.004 and third["action_k_min"] == 0.9
    assert first["diagnostics"]["actions"]["source_counts"].keys() == {
        str(k) for k in range(6)
    }


def test_snapshot_age_warning_and_strict(tables, make_store, caplog):
    # X20：快照超过 40 天告警，--strict 拒绝。
    from ashare.data.catalog import INDEXES

    store = make_store({k: v for k, v in tables.items() if k != "index_weight"})
    for code in INDEXES.values():
        snapshot = tables["index_weight"].filter(
            (pl.col("index_code") == code) & (pl.col("trade_date") == "20251231")
        )
        store.write(
            "index_weight",
            f"{code}_202512",
            {},
            FetchResult(
                snapshot.with_columns(pl.lit("20251127").alias("trade_date")),
                "2026-01-08T00:00:00Z",
                1,
            ),
        )
    with caplog.at_level(logging.WARNING, logger="ashare.data.panel"):
        meta = build(store)
    assert meta["diagnostics"]["index_snapshots"]["csi300"]["age_days"] == 41
    assert "快照已 41 天" in caplog.text
    with pytest.raises(ValueError, match="快照已 41 天"):
        build(store, strict=True)


def test_build_reconciliation_rejects_out_of_range_k(monkeypatch):
    # 构建期对账：来源 5 的 k 不在 [k_min,1) 时必须拦截。
    monkeypatch.setattr(actions, "choose_action", lambda *a, **k: (2.0, 0.0, 5))
    arrays = arrays_of([10, 10], [1, 1.5])  # s=1 时 k=0.5
    with pytest.raises(ValueError, match="对账失败"):
        reconcile(arrays, DAYS[:2], ["A"], table_of((DAYS[1], DAYS[0], 1, 0)))


def test_verify_source_event_consistency():
    # 有事件却标 0、无事件却标 1，都要被独立校验判失败。
    bad, *_ = audit_actions(
        np.array([[10.0, 10.0, 10.0]]),
        np.array([[10 / 9, 1.0, 1.0]]),
        np.array([[1.0, 1.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0, 1, 4]], np.uint8),
        np.ones((1, 3), bool),
        tolerance=TOL,
        k_min=0.85,
    )
    assert bad[0].tolist() == [True, True, False]
