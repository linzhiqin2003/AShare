import csv
import hashlib
import importlib
import io
import json
import zipfile

import httpx

from ashare.data.crypto.__main__ import main
from ashare.data.crypto.dataset import CryptoDataset
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.sync import prepare_profiles
from ashare.data.crypto.timeutil import HOUR
from ashare.data.crypto.verify import verify


def zip_rows(rows):
    csv_buf = io.StringIO()
    csv.writer(csv_buf).writerows(rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("sample.csv", csv_buf.getvalue())
    return buf.getvalue()


def test_offline_sync_build_verify_cli(tmp_path, fixture, monkeypatch, capsys):
    """全链路离线 HTTP：真实 ZIP/校验、归档目录、资金费、交割与快照。"""
    profiles = fixture["profiles"]
    for p in profiles:
        p["index_symbol"] = p["symbol"]
    config = {k: fixture[k] for k in ("start", "end", "events")}
    config.update(
        instruments=[p["instrument"] for p in profiles],
        profiles={p["instrument"]: p for p in profiles},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    symbols = [
        dict(
            symbol=p["symbol"],
            contractType=p["contract_type"],
            underlyingType="COIN",
            quoteAsset="USDT",
            marginAsset="USDT",
            baseAsset=p["base_asset"],
            onboardDate=p["listed_at"],
            filters=[dict(filterType="LOT_SIZE", stepSize="0.01", minQty="0.01")],
        )
        for p in profiles
    ]
    zips = {}
    for p in profiles:
        for dataset, rows in fixture["data"][p["instrument"]].items():
            if dataset in ("funding_prices", "deliveries"):
                continue
            if dataset == "fundingRate":
                csv_rows = [
                    ["calc_time", "funding_interval_hours", "last_funding_rate"],
                    *[[r["raw_time"], r["interval"], r["rate"]] for r in rows],
                ]
                key = f"/data/futures/um/monthly/fundingRate/{p['symbol']}/{p['symbol']}-fundingRate-2024-06.zip"
            else:
                csv_rows = [
                    [
                        r["time"],
                        r["open"],
                        r["high"],
                        r["low"],
                        r["close"],
                        r["volume"],
                        r["time"] + HOUR - 1,
                        r["quote_volume"],
                        1,
                        0,
                        r["taker_buy_quote_volume"],
                        0,
                    ]
                    for r in rows
                ]
                key = f"/data/futures/um/monthly/{dataset}/{p['symbol']}/1h/{p['symbol']}-1h-2024-06.zip"
            zips[key] = zip_rows(csv_rows)
    requests = []

    def handle(request):
        requests.append(str(request.url))
        path = request.url.path
        if path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, json=dict(symbols=symbols))
        if path == "/data.binance.vision":
            prefix = request.url.params["prefix"]
            items = "".join(
                f"<CommonPrefixes><Prefix>{prefix}{p['symbol']}/</Prefix></CommonPrefixes>"
                for p in profiles
            )
            return httpx.Response(
                200,
                text=f"<ListBucketResult><IsTruncated>false</IsTruncated>{items}</ListBucketResult>",
            )
        if path.endswith(".CHECKSUM"):
            key = path.removesuffix(".CHECKSUM")
            if key not in zips:
                return httpx.Response(404)
            return httpx.Response(
                200,
                text=hashlib.sha256(zips[key]).hexdigest()
                + "  "
                + key.rsplit("/", 1)[-1],
            )
        if path in zips:
            return httpx.Response(200, content=zips[path])
        if path == "/fapi/v1/fundingRate":
            return httpx.Response(
                200,
                json=fixture["data"]["usdm:" + request.url.params["symbol"]][
                    "funding_prices"
                ],
            )
        if path == "/futures/data/delivery-price":
            return httpx.Response(
                200, json=fixture["data"]["usdm:FUTUSDT_240628"]["deliveries"]
            )
        raise AssertionError(str(request.url))

    module = importlib.import_module("ashare.data.crypto.sync")
    monkeypatch.setattr(
        module,
        "PublicClient",
        lambda **kwargs: PublicClient(
            transport=httpx.MockTransport(handle), sleep=lambda _: None, **kwargs
        ),
    )
    assert main(["--root", str(tmp_path), "sync", "--skip-announcements"]) == 0
    assert main(["--root", str(tmp_path), "build"]) == 0
    assert main(["--root", str(tmp_path), "verify"]) == 0
    ds = CryptoDataset(tmp_path)
    assert ds.read().height == 24
    assert verify(tmp_path)["archives_checked"] == 7
    # 再同步只刷新快照与交割；历史 ZIP 和资金费断点续跑。
    before = sum(url.endswith(".zip") for url in requests)
    assert main(["--root", str(tmp_path), "sync", "--skip-announcements"]) == 0
    assert sum(url.endswith(".zip") for url in requests) == before
    assert '"verdict": "pass"' in capsys.readouterr().out


def test_sync_appends_changed_snapshot_rules_without_reselecting_unit(
    tmp_path, fixture
):
    """开发自检：冻结乘数不等于忽略以后快照里的规则变化。"""
    p = fixture["profiles"][0]
    (tmp_path / "profiles.json").write_text(json.dumps({p["instrument"]: p}))
    symbol = dict(
        symbol=p["symbol"],
        contractType="PERPETUAL",
        underlyingType="COIN",
        quoteAsset="USDT",
        marginAsset="USDT",
        baseAsset=p["base_asset"],
        filters=[
            dict(filterType="LOT_SIZE", stepSize="0.02", minQty="0.02"),
            dict(filterType="MIN_NOTIONAL", notional="10"),
        ],
    )
    store = RawStore(tmp_path)
    result = prepare_profiles(
        None,
        store,
        dict(instruments=[p["instrument"]]),
        fixture["end"] + HOUR,
        dict(usdm=dict(symbols=[symbol])),
    )
    assert result[0]["multiplier"] == 0.01
    assert len(result[0]["rules"]) == 2
    assert result[0]["rules"][0] == p["rules"][0]
    assert result[0]["rules"][1]["min_notional"] == 10
    store.close()
