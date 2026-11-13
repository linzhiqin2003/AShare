"""条目身份、试验与过滤计数同源。"""

import json

import numpy as np
import pytest

from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from ashare.platform.storage import Store


def test_identity_name_and_experiment_count(tmp_path):
    store = Store(tmp_path)
    a = parse("x=rank(close)\nx", factors={})[0]
    b = parse("rank(close) # 同表达式", factors={})[0]
    store.begin("tab", "first", [(a, DEFAULTS), (a, DEFAULTS | {"decay": 5})], {})
    store.begin("tab", "renamed", [(b, DEFAULTS)], {})
    assert store.trials() == 2
    assert {r["name"] for r in store.library()["items"]} == {"renamed"}
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 3
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_filtered_counts_groups_and_page_use_same_population(tmp_path):
    store = Store(tmp_path)
    a = parse("rank(close)", factors={})[0]
    for i in range(1010):
        store.begin(
            "tab",
            f"factor_{i % 10}",
            [(a, DEFAULTS | {"decay": i, "universe": "ALL" if i % 2 else "TOP800"})],
            {},
        )
    with store.connect() as db:
        db.execute("UPDATE entries SET sharpe=json_extract(settings,'$.decay')/100.0")
    all_rows = store.library(limit=200)
    assert all_rows["total"] == 1010 and len(all_rows["items"]) == 200
    for group in ("none", "expression", "config"):
        result = store.library(
            universe="ALL", metric="sharpe", minimum=3, maximum=5, group=group, limit=20
        )
        assert result["total"] == 100
        assert sum(result["counts"].values()) == 100
        assert len(result["items"]) == 20
        if group != "none":
            all_groups = store.library(
                universe="ALL",
                metric="sharpe",
                minimum=3,
                maximum=5,
                group=group,
                limit=200,
            )
            assert sum(g["count"] for g in all_groups["groups"]) == 100
            full_counts = {g["key"]: g["count"] for g in all_groups["groups"]}
            assert {g["key"]: g["count"] for g in result["groups"]} == {
                key: full_counts[key]
                for key in {
                    r["expression_id"]
                    if group == "expression"
                    else json.dumps(r["settings"], separators=(",", ":"))
                    for r in result["items"]
                }
            }
            assert len(result["groups"]) <= len(result["items"])
        assert all(
            json.loads(json.dumps(r["settings"]))["universe"] == "ALL"
            for r in result["items"]
        )
    assert store.library(search="%' OR 1=1 --")["total"] == 0
    assert store.library(universe="TOP800", minimum=30)["total"] == 0


def test_events_resume_and_restart_recovery(tmp_path):
    store = Store(tmp_path)
    job, _ = store.begin(
        "tab", "alpha", [(parse("rank(close)", factors={})[0], DEFAULTS)], {}
    )
    first = store.events(job, 0)[0]["id"]
    store.event(job, {"kind": "progress", "phase": "is"})
    store.recover()
    events = store.events(job, first)
    assert [e["kind"] for e in events] == ["progress", "interrupted"]
    assert events[0]["id"] > first
    assert store.job(job)["status"] == "interrupted"


def test_matrix_aligns_dates_and_bounds_visible_entries(tmp_path):
    store = Store(tmp_path)
    a = parse("rank(close)", factors={})[0]
    _, runs = store.begin(
        "tab", "alpha", [(a, DEFAULTS | {"decay": i}) for i in range(3)], {}
    )
    for i, run in enumerate(runs):
        dates = np.array([20240101, 20240102, 20240103, 20240104])
        pnl = np.array([0.01, 0.03, 0.02, 0.04])
        if i == 1:
            dates, pnl = dates[1:], -pnl[1:]
        if i == 2:
            pnl[:] = 0
        np.savez_compressed(
            tmp_path / "series" / f"{run['id']}.npz",
            **{"is.dates": dates, "is.pnl.long_short": pnl},
        )
        with store.connect() as db:
            db.execute(
                "UPDATE entries SET latest=? WHERE id=?", (run["id"], run["entry"])
            )
    ids = [r["entry"] for r in runs]
    assert store.correlation_matrix(ids) == {
        "ids": ids,
        "matrix": [[1, -1, None], [-1, 1, None], [None, None, None]],
    }
    assert store.correlation_matrix([]) == {"ids": [], "matrix": []}
    assert store.correlation_matrix(["missing"])["matrix"] == [[None]]
    with pytest.raises(ValueError):
        store.correlation_matrix(ids * 2)
    with pytest.raises(ValueError):
        store.correlation_matrix([str(i) for i in range(13)])


def test_benchmark_is_not_part_of_entry_identity(tmp_path):
    store = Store(tmp_path)
    variant = parse("rank(close)")[0]
    first, runs = store.begin(
        "tab", "factor", [(variant, DEFAULTS | {"benchmark": "000001.SH"})], {}
    )
    second, other = store.begin(
        "tab", "factor", [(variant, DEFAULTS | {"benchmark": "000300.SH"})], {}
    )
    assert store.trials() == 1
    assert runs[0]["entry"] == other[0]["entry"]
    assert store.job(first)["runs"][0]["settings"]["benchmark"] == "000001.SH"
    assert store.job(second)["runs"][0]["settings"]["benchmark"] == "000300.SH"


def test_correlations_exclude_job_and_parameter_family(tmp_path):
    store = Store(tmp_path)
    dates, pnl = np.arange(10).astype(str), np.sin(np.arange(10))

    def insert(source, name):
        return store.begin("tab", name, [(parse(source)[0], DEFAULTS)], {})

    target, runs = insert("-ts_std(returns,20)", "target")
    family, related = insert("-ts_std(returns,60)", "same_family")
    independent, others = insert("rank(close)", "independent")
    same_job, siblings = insert("rank(open)", "same_job")
    for row in [*related, *others, *siblings]:
        np.savez_compressed(
            store.root / "series" / f"{row['id']}.npz",
            **{"is.dates": dates, "is.pnl.long_short": pnl},
        )
        with store.connect() as db:
            db.execute(
                "UPDATE entries SET latest=? WHERE id=?", (row["id"], row["entry"])
            )
    with store.connect() as db:
        db.execute("UPDATE runs SET job=? WHERE id=?", (target, siblings[0]["id"]))
    values = store.correlations(runs[0]["entry"], dates, pnl, target)
    assert [row["name"] for row in values] == ["independent"]
    assert values[0]["value"] == 1
