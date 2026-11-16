"""第 3.3 节审查回归：提前公告、unknown 警告与面板字段元数据。"""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from ashare.data.reader import Panel
from ashare.data.spec import build, check, open
from docs.examples.generate_datapackage import SyntheticAdapter
from test.spec.conftest import (
    assert_bad,
    assert_good,
    item,
    read_manifest,
    resign,
    write_manifest,
)
from test.spec.test_boundaries import events
from test.spec.test_contract import add_field, change


@pytest.mark.parametrize("kind", ["events", "corporate_actions"])
def test_s3_3_announcement_before_event_date(package, kind):
    # 来源：审查决定，公告早于事件/除权日合法，但不能缺失公告时刻。
    if kind == "corporate_actions":
        events(package)
    else:
        m = read_manifest(package)
        m["resources"].append(
            dict(
                name=kind,
                kind=kind,
                schema=dict(
                    fields=[
                        dict(
                            name=k,
                            type=t,
                            knownAt="end"
                            if k == "available_at"
                            else "column:available_at",
                        )
                        for k, t in [
                            ("symbol", "string"),
                            ("date", "date"),
                            ("available_at", "datetime"),
                        ]
                    ],
                    primaryKey=["symbol", "date"],
                ),
            )
        )
        (package / kind).mkdir()
        pl.DataFrame(
            dict(
                symbol=["000001.SZ"],
                date=[date(2024, 1, 3)],
                available_at=[datetime(2024, 1, 3, 7, tzinfo=UTC)],
            )
        ).write_parquet(package / kind / "all.parquet")
        resign(package, m)
        assert_good(package)
    path = kind + "/all.parquet"
    change(
        package,
        lambda f: f.with_columns(pl.col("available_at") - pl.duration(days=30)),
        path,
    )
    assert_good(package)
    change(
        package,
        lambda f: f.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("available_at")
        ),
        path,
    )
    assert_bad(package, "available_at")


@pytest.mark.parametrize("extension", [False, True])
def test_s3_3_unknown_warning_and_typo_rejected(package, extension):
    # 来源：审查决定，合法 unknown 必须报警告，拼写错误仍按 known_at 失败。
    name = "vendor_score" if extension else "close"
    if extension:
        add_field(
            package,
            dict(
                name=name,
                type="number",
                unit="1",
                knownAt="end",
                description="合成评分",
            ),
            pl.lit(0.5),
        )
        assert_good(package)
    m = read_manifest(package)
    desc = next(f for f in m["resources"][0]["schema"]["fields"] if f["name"] == name)
    desc["knownAt"] = "unknown"
    write_manifest(package, m)
    report = check(package)
    assert report["ok"] and report["exit_code"] == 0
    warning = item(report, "known_at_unknown")
    assert warning["status"] == "warning" and warning["count"] == 1
    assert warning["examples"][0]["value"] == dict(
        table="bars", field=name, knownAt="unknown"
    )
    assert item(report, "known_at")["status"] == "pass"
    desc["knownAt"] = "unkown"
    write_manifest(package, m)
    assert_bad(package, "known_at")


@pytest.mark.parametrize("unknown_first", [False, True])
def test_s3_3_unknown_warning_never_hides_invalid_known_at(package, unknown_first):
    m = read_manifest(package)
    first, second = m["resources"][0]["schema"]["fields"][2:4]
    first["knownAt"], second["knownAt"] = (
        ("unknown", "unkown") if unknown_first else ("unkown", "unknown")
    )
    write_manifest(package, m)
    report = assert_bad(package, "known_at")
    assert item(report, "known_at_unknown")["status"] == "warning"


def test_s9_unknown_visible_in_info_and_panel(package, monkeypatch):
    # 来源：审查要求，面板不能丢掉来源不能保证时点的字段标记。
    m = read_manifest(package)
    next(f for f in m["resources"][0]["schema"]["fields"] if f["name"] == "close")[
        "knownAt"
    ] = "unknown"
    write_manifest(package, m)
    dataset = open(package)
    info = dataset.info()
    close = next(f for f in info["resources"][0]["fields"] if f["name"] == "close")
    assert close["knownAt"] == "unknown"

    def no_full_info():
        raise AssertionError("panel 字段元数据不得触发 info 的全表范围扫描")

    monkeypatch.setattr(dataset, "info", no_full_info)
    panel = dataset.panel(["open", "close"], allow_unknown=True)
    assert isinstance(panel, Panel)
    assert panel.field_metadata["close"]["knownAt"] == "unknown"
    assert panel.field_metadata["open"]["knownAt"] == "start"
    with pytest.raises(TypeError):
        panel.field_metadata["close"]["knownAt"] = "start"
    assert dataset.manifest == m


def test_s9_unknown_dependency_visible_in_trading_mask(package):
    add_field(
        package,
        dict(name="limit_up", type="number", knownAt="unknown"),
        pl.col("high") + 1,
    )
    panel = open(package).panel(["can_buy"], allow_unknown=True)
    assert panel.field_metadata["can_buy"]["knownAt"] == "unknown"


def test_s9_build_accepts_unknown_warning(tmp_path):
    class Unknown(SyntheticAdapter):
        def describe(self):
            m = super().describe()
            m["resources"][0]["schema"]["fields"][2]["knownAt"] = "unknown"
            return m

    target = tmp_path / "unknown"
    assert build(Unknown(), target)["published"]
    report = check(target)
    assert report["exit_code"] == 0
    assert item(report, "known_at_unknown")["status"] == "warning"
