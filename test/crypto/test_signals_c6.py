"""季度信号归档全月列举、独立复算、内容版本读取的离线验收。"""

import csv
import io
import json
import zipfile

import polars as pl
import pytest

from ashare.data.crypto.archive import archive_key, archive_prefix
from ashare.data.crypto.signals import SignalDataset, sync_signals
from ashare.data.crypto.storage import digest
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify_signals import verify_signals


class SyntheticClient:
    def __init__(self):
        self.symbol = "BTCUSDT_240329"
        self.keys = [
            archive_key("usdm", "klines", self.symbol, label)
            for label in ("2023-12", "2024-01")
        ]
        self.listed = []

    def listing(self, prefix, delimiter=None):
        self.listed.append((prefix, delimiter))
        if delimiter:
            return [archive_prefix("usdm", "klines") + self.symbol + "/"]
        return list(reversed(self.keys))

    def json(self, url):
        return {"symbols": []}

    def archive(self, store, key, refresh=False):
        month = "2023-12-31T23:00:00Z" if key == self.keys[0] else "2024-01-01"
        t = millis(month)
        buf = io.StringIO()
        csv.writer(buf).writerow(
            [
                t,
                100,
                102,
                99,
                101 if key == self.keys[0] else 102,
                2,
                t + HOUR - 1,
                202,
                1,
                1,
                101,
                0,
            ]
        )
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("synthetic.csv", buf.getvalue())
        payload = data.getvalue()
        sha = digest(payload)
        store.put(
            key, payload, checksum=sha, checksum_text=sha + " " + key.rsplit("/", 1)[1]
        )
        return payload


def test_sync_earliest_month_independent_verify_and_corruption(tmp_path):
    client = SyntheticClient()
    ds = sync_signals(
        tmp_path, start="2024-01-01", end="2024-01-01T01:00:00Z", client=client
    )
    assert ds.metadata[0]["listed_at"] == millis("2023-12-31T23:00:00Z")
    assert ds.metadata[0]["archives"] == client.keys
    assert verify_signals(tmp_path)["verdict"] == "pass"
    data = ds.read()
    path = ds.folder / "hours.parquet"
    data.with_columns((pl.col("close") + 1).alias("close")).write_parquet(path)
    # 即使攻击者同步改了 meta 的分块摘要，内容版本仍保护真实解析结果。
    meta = json.loads((ds.folder / "meta.json").read_text())
    meta["frame_sha256"] = digest(path.read_bytes())
    (ds.folder / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="元数据"):
        SignalDataset(tmp_path)


def test_metadata_mutation_rejected(tmp_path):
    ds = sync_signals(
        tmp_path,
        start="2024-01-01",
        end="2024-01-01T01:00:00Z",
        client=SyntheticClient(),
    )
    p = ds.folder / "inputs.json"
    content = json.loads(p.read_text())
    content["metadata"][0]["listed_at"] -= HOUR
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="损坏"):
        ds.read()


def test_explicit_exchange_metadata_conflict(tmp_path):
    c = SyntheticClient()
    c.json = lambda _: dict(
        symbols=[
            dict(
                symbol=c.symbol,
                quoteAsset="USDT",
                marginAsset="USDT",
                underlyingType="COIN",
                onboardDate=millis("2023-01-01"),
                deliveryDate=millis("2024-03-29T09:00:00Z"),
            )
        ]
    )
    with pytest.raises(ValueError, match="deliveryDate"):
        sync_signals(tmp_path, start="2024-01-01", end="2024-01-01T01:00:00Z", client=c)


def test_post_delivery_auto_exception_retains_raw_and_never_overrides_metadata(
    tmp_path,
):
    from ashare.data.crypto.signals import contract_metadata, write_signals
    from test.crypto.test_factors_c6 import raw_row

    symbol = "BTCUSDT_240329"
    expiry = millis("2024-03-29T08:00:00Z")
    rows = [
        raw_row(expiry - HOUR, 100),
        raw_row(expiry, 101),
        raw_row(expiry + HOUR, 101, 0),
    ]
    m = contract_metadata(symbol, rows, ["synthetic.zip"])
    ds = write_signals(
        tmp_path,
        raw={symbol: rows},
        metadata=[m],
        references={},
        start=expiry - HOUR,
        end=expiry + 2 * HOUR,
    )
    assert len(ds.inputs()["raw"][symbol]) == 3
    assert ds.read().height == 1
    assert verify_signals(tmp_path)["verdict"] == "pass"
    with pytest.raises(ValueError, match="deliveryDate"):
        contract_metadata(
            symbol,
            rows,
            ["synthetic.zip"],
            dict(deliveryDate=expiry + HOUR),
        )


# 来源：审查人 DESIGN 4.18 v5.1，先在 v5 实现上复现失败。
def test_v51_delivery_hour_trade_accepted_and_post_delivery_hashed(tmp_path):
    from ashare.data.crypto.signals import contract_metadata, write_signals
    from test.crypto.test_factors_c6 import raw_row

    symbol = "BTCUSDT_240329"
    expiry = millis("2024-03-29T08:00:00Z")
    rows = [raw_row(expiry - HOUR, 100), raw_row(expiry, 101)]
    m = contract_metadata(symbol, rows, ["synthetic.zip"])
    versions = []
    for zero_tail in (False, True):
        raw = rows + ([raw_row(expiry + HOUR, 101, 0)] if zero_tail else [])
        ds = write_signals(
            tmp_path,
            raw={symbol: raw},
            metadata=[m],
            references={},
            start=expiry - HOUR,
            end=expiry + 2 * HOUR,
        )
        versions.append(ds.version)
        assert len(ds.inputs()["raw"][symbol]) == 2 + int(zero_tail)
        assert ds.read()["time"].to_list() == [expiry - HOUR]
        assert ds.meta["metadata"][0]["post_delivery_excluded_rows"] == 1 + int(
            zero_tail
        )
        exceptions = [
            e for e in ds.inputs()["exceptions"] if e["kind"] == "post_delivery"
        ]
        assert len(exceptions) == 1
        assert exceptions[0]["count"] == 1 + int(zero_tail)
        assert verify_signals(tmp_path)["verdict"] == "pass"
    assert versions[0] != versions[1]


@pytest.mark.parametrize("offset", [1, 2, -25])
def test_v51_last_trade_outside_delivery_window_rejected(offset):
    from ashare.data.crypto.signals import contract_metadata
    from test.crypto.test_factors_c6 import raw_row

    expiry = millis("2024-03-29T08:00:00Z")
    rows = [raw_row(expiry + offset * HOUR, 100)]
    if offset < 0:
        rows.append(raw_row(expiry, 100, 0))
    with pytest.raises(ValueError, match="最后成交"):
        contract_metadata("BTCUSDT_240329", rows, ["synthetic.zip"])


def test_v51_exact_24_hour_lower_boundary_is_accepted():
    from ashare.data.crypto.signals import contract_metadata
    from test.crypto.test_factors_c6 import raw_row

    expiry = millis("2024-03-29T08:00:00Z")
    rows = [raw_row(expiry - 24 * HOUR), raw_row(expiry, volume=0)]
    m = contract_metadata("BTCUSDT_240329", rows, ["synthetic.zip"])
    assert m["last_trade_at"] == expiry - 24 * HOUR
    assert m["delivery_check"] == "pass"


def test_v51_excluded_tail_does_not_change_selected_contract(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from ashare.data.crypto.signals import write_signals
    from ashare.factor.crypto import crypto_factors
    from test.crypto.test_factors_c6 import START, Synthetic, quarters, raw_row

    panel = Synthetic()
    original = quarters(panel)
    raw = {}
    metadata = []
    for m in original.metadata:
        symbol, expiry = m["symbol"], m["delivery_at"]
        raw[symbol] = [
            raw_row(r["time"], r["close"])
            for r in original.frame.filter(pl.col("symbol") == symbol).iter_rows(
                named=True
            )
            if r["time"] < expiry
        ]
        raw[symbol] += [raw_row(expiry, 110), raw_row(expiry + HOUR, 110, 0)]
        metadata.append(
            dict(
                symbol=symbol,
                archives=["synthetic.zip"],
                evidence=dict(
                    onboardDate=m["listed_at"], listed_source="synthetic listing"
                ),
            )
        )
    ds = write_signals(
        tmp_path,
        raw=raw,
        metadata=metadata,
        references={},
        start=START,
        end=panel.meta["end"],
    )
    baseline = SimpleNamespace(
        metadata=ds.metadata, version="synthetic-pre-filter", read=original.read
    )
    before = crypto_factors(panel, baseline, ["BTCUSDT"], START, panel.meta["end"])
    after = crypto_factors(panel, ds, ["BTCUSDT"], START, panel.meta["end"])
    np.testing.assert_array_equal(
        before.selected_contract[1], after.selected_contract[1]
    )
    np.testing.assert_allclose(before["basis_q"][1], after["basis_q"][1])


def test_v51_no_listing_is_excluded_instead_of_a_fabricated_delivery_check(tmp_path):
    # 来源：v5.1 生命周期复核；无归档且无上市证据仍须遵守设计的排除清单规则。
    from ashare.data.crypto.signals import write_signals

    symbol = "BTCUSDT_240329"
    ds = write_signals(
        tmp_path,
        raw={symbol: []},
        metadata=[dict(symbol=symbol, archives=[])],
        references={},
        start="2024-03-01",
        end="2024-04-01",
    )
    assert ds.meta["excluded"] == [symbol]
    assert ds.metadata[0]["delivery_check"] == "excluded_no_listing"
    assert ds.read().is_empty()
    assert verify_signals(tmp_path)["verdict"] == "pass"


def test_v51_unexpired_status_does_not_bypass_matured_lower_bound():
    from ashare.data.crypto.signals import contract_metadata
    from test.crypto.test_factors_c6 import raw_row

    expiry = millis("2024-03-29T08:00:00Z")
    rows = [raw_row(expiry - 48 * HOUR)]
    m = contract_metadata(
        "BTCUSDT_240329", rows, ["synthetic.zip"], coverage_end=expiry - 47 * HOUR
    )
    assert m["delivery_check"] == "pending_not_expired"
    with pytest.raises(ValueError, match="最后成交"):
        contract_metadata(
            "BTCUSDT_240329", rows, ["synthetic.zip"], coverage_end=expiry
        )
