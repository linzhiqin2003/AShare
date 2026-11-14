import copy

import polars as pl
import pytest

from ashare.data.crypto.dataset import write_panel
from ashare.data.crypto.events import normalize_event
from ashare.data.crypto.instruments import mapping_at
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.verify import verify, verify_frame

PERP = "usdm:PERPUSDT"
QUARTER = "usdm:FUTUSDT_240628"


@pytest.mark.parametrize(
    "field,value,check",
    [
        ("funding", 999.0, "funding_cash"),
        ("settlement", 999.0, "settlement_value"),
        ("mark_open", 999.0, "mark_ohlc"),
        ("open", 999.0, "raw_value"),
        ("volume_engine", 1.0, "volume_unit"),
        ("qty_step", 99.0, "rule_qty_step"),
        ("min_qty", 99.0, "rule_min_qty"),
        ("min_notional", 99.0, "rule_min_notional"),
        ("tradable", False, "tradable"),
        ("coin_units", 1000, "unit_mapping"),
        ("expires", True, "expiry_boundary"),
        ("lifecycle", "ended", "lifecycle"),
    ],
)
def test_independent_verifier_catches_mutations(fixture, field, value, check):
    """来源：6.10 独立校验；直接变异产物，不能与生产实现共享结论。"""
    frame, _ = build_frame(**fixture)
    target = (pl.col("instrument") == PERP) & (
        pl.col("time") == fixture["start"] + 3 * 3_600_000
    )
    mutant = frame.with_columns(
        pl.when(target).then(pl.lit(value)).otherwise(pl.col(field)).alias(field)
    )
    report = verify_frame(mutant, fixture)
    assert report["verdict"] == "fail"
    assert check in {e["check"] for e in report["errors"]}


def test_delivery_and_spot_price_differences_are_diagnostics(fixture):
    spot = copy.deepcopy(fixture["profiles"][0])
    spot.update(
        instrument="spot:PERPUSDT",
        market="spot",
        contract_type="SPOT",
        margin_asset=None,
    )
    fixture["profiles"].append(spot)
    bars = copy.deepcopy(fixture["data"][PERP]["klines"])
    for r in bars:
        for field in ("open", "high", "low", "close"):
            r[field] *= 2
    fixture["data"][spot["instrument"]] = dict(klines=bars)
    frame, _ = build_frame(**fixture)
    report = verify_frame(frame, fixture, price_threshold=0.01)
    assert report["verdict"] == "pass", report["errors"]
    assert {"delivery_index_deviation", "spot_perpetual_deviation"} <= {
        d["check"] for d in report["diagnostics"]
    }


def test_missing_price_and_hour_gap_verifier(fixture):
    frame, _ = build_frame(**fixture)
    bad = copy.deepcopy(fixture)
    bad["data"][PERP]["fundingRate"].pop(1)
    assert "funding_missing" in {e["check"] for e in verify_frame(frame, bad)["errors"]}
    cut = frame.filter(pl.col("time") != fixture["start"] + 3 * 3_600_000)
    assert "hour_axis" in {e["check"] for e in verify_frame(cut, fixture)["errors"]}


def test_read_version_and_detect_corrupt_chunk(tmp_path, fixture):
    frame, notes = build_frame(**fixture)
    ds = write_panel(
        tmp_path,
        frame,
        profiles=fixture["profiles"],
        events=fixture["events"],
        data=fixture["data"],
        notes=notes,
    )
    assert verify(tmp_path, ds.version)["verdict"] == "pass"
    chunk = ds.folder / next(iter(ds.meta["chunks"]))
    chunk.write_bytes(b"broken")
    with pytest.raises(ValueError, match="分块损坏"):
        ds.read()


def test_migration_mapping_is_asof_and_preserves_value(fixture):
    p = fixture["profiles"][0]
    p.update(coin="MATIC")
    event = normalize_event(
        dict(
            event_id="migration",
            kind="migration",
            instrument=PERP,
            replacement="POL",
            ratio=2,
            known_at=fixture["start"],
            effective_at=fixture["start"] + 3_600_000,
            source_url="fixture://token-swap",
            original_text="Synthetic 1 old to 2 new tokens",
        )
    )
    assert mapping_at(p, [event], fixture["start"])["coin"] == "MATIC"
    mapped = mapping_at(p, [event], fixture["start"] + 3_600_000)
    assert mapped["coin"] == "POL" and mapped["coin_units"] == 2
    assert (100 / mapped["coin_units"]) * (10 * mapped["coin_units"]) == 100 * 10


def test_verifier_does_not_import_builder_or_parsers():
    import ast
    from pathlib import Path

    import ashare.data.crypto.verify as unused

    del unused
    base = Path(__file__).parents[2] / "src/ashare/data/crypto"
    forbidden = {
        "panel",
        "formats",
        "funding",
        "instruments",
        "events",
        "build",
        "delivery",
    }
    for file in base.glob("verify*.py"):
        tree = ast.parse(file.read_text())
        assert not any(
            isinstance(n, ast.ImportFrom) and n.module in forbidden
            for n in ast.walk(tree)
        )
