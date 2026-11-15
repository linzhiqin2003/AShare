"""季度合约仅信号库：内容寻址，不要求交割价格，也不能交给交易引擎。"""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..store import atomic_json, lock
from .archive import archive_prefix, read_series
from .sources import FAPI, PublicClient
from .storage import RawStore, digest, encoded
from .timeutil import HOUR, millis

SIGNAL_DEFINITION = "quarter-signals-v2-design4.18v5.1"
BAD_PRICE = 1 | 8 | 128 | 512 | 1024 | 2048
QUARTER_BAD_PRICE = BAD_PRICE | 2 | 4


def contract_metadata(symbol, rows, archives, evidence=None, *, coverage_end=None):
    """交割日取代码；上市取证据或全部月归档首行，绝不使用查询起点。"""
    match = re.fullmatch(r"(BTCUSDT|ETHUSDT)_(\d{6})", symbol)
    if not match:
        raise ValueError("仅支持 BTC/ETH U 本位季度代码")
    delivery = int(
        datetime.strptime(match[2], "%y%m%d").replace(tzinfo=UTC, hour=8).timestamp()
        * 1000
    )
    evidence = evidence or {}
    if (
        evidence.get("deliveryDate") is not None
        and int(evidence["deliveryDate"]) != delivery
    ):
        raise ValueError(f"{symbol} deliveryDate 与代码冲突")
    post_delivery = [r for r in rows if r["close_time"] + 1 > delivery]
    last_trade = max((r["time"] for r in rows if r["volume"] > 0), default=None)
    # 尚未到期的合约不能用未来最后成交作前提；到期后必须完成窗口下界检查。
    observed_end = max(
        millis(coverage_end) if coverage_end is not None else delivery,
        max((r["close_time"] + 1 for r in rows), default=0),
    )
    matured = observed_end >= delivery
    if last_trade is not None and last_trade > delivery:
        raise ValueError(f"{symbol} 最后成交晚于交割")
    first = min((r["time"] for r in rows), default=None)
    listed = evidence.get("onboardDate")
    source = evidence.get("listed_source")
    if listed is not None:
        listed = millis(listed)
        if not source:
            raise ValueError("onboardDate/上市公告必须有来源")
        if first is not None and listed > first:
            raise ValueError(f"{symbol} 上市时刻晚于最早归档")
    elif first is not None:
        listed = first
        source = dict(kind="earliest_archive_proxy", key=min(archives))
    if listed is not None and listed >= delivery:
        raise ValueError("上市不得晚于交割")
    if (
        listed is not None
        and matured
        and (last_trade is None or last_trade < delivery - 24 * HOUR)
    ):
        raise ValueError(f"{symbol} 最后成交早于交割24小时窗口或无成交")
    return dict(
        symbol=symbol,
        pair=match[1],
        listed_at=listed,
        listed_source=source,
        delivery_at=delivery,
        delivery_source=dict(
            kind="symbol_date_08_utc",
            symbol=symbol,
            exchange=evidence.get("delivery_source"),
        ),
        archives=sorted(archives),
        evidence=evidence,
        last_trade_at=last_trade,
        delivery_check="excluded_no_listing"
        if listed is None
        else "pass"
        if matured
        else "pending_not_expired",
        post_delivery_excluded_rows=len(post_delivery),
    )


def signal_rows(metadata, raw, start, end, exceptions=()):
    """缺小时需显式例外；冻结从第二个连续同收盘小时起标记，不事后回填。"""
    if metadata["listed_at"] is None:
        return []
    lo = max(start, (metadata["listed_at"] + HOUR - 1) // HOUR * HOUR)
    hi = min(end, metadata["delivery_at"])
    by_time = {r["time"]: r for r in raw}
    if len(by_time) != len(raw):
        raise ValueError("季度小时重复")
    out = []
    for t in range(lo, hi, HOUR):
        r = by_time.get(t)
        if r is None:
            allowed = any(
                e.get("kind") != "post_delivery"
                and e["symbol"] == metadata["symbol"]
                and millis(e["start"]) <= t < millis(e["end"])
                and e.get("reason")
                for e in exceptions
            )
            if not allowed:
                raise ValueError(f"季度缺小时，需质量例外: {metadata['symbol']} {t}")
            out.append(dict(time=t, symbol=metadata["symbol"], close=None, quality=1))
            continue
        if r["close_time"] + 1 > metadata["delivery_at"]:
            continue
        q = {"normal": 0, "partial_halt": 512, "invalid_close_time": 2048}[
            r["time_quality"]
        ]
        if r["volume"] <= 0 or r["quote_volume"] <= 0 or r["count"] <= 0:
            q |= 2
        previous = by_time.get(t - HOUR)
        if previous and previous["close"] == r["close"]:
            q |= 4
        if min(r[f] for f in ("open", "high", "low", "close")) <= 0:
            raise ValueError("季度价格必须为正")
        out.append(dict(time=t, symbol=metadata["symbol"], close=r["close"], quality=q))
    return out


def write_signals(
    root, *, raw, metadata, references, start, end, exceptions=(), provenance=None
):
    """冻结原始引用、完整解析行、规范化元数据、例外与定义。仅供构建及合成夹具。"""
    root = Path(root)
    start, end = millis(start), millis(end)
    if start >= end or start % HOUR or end % HOUR:
        raise ValueError("季度信号范围必须是递增整小时")
    exceptions = [
        dict(e, start=millis(e["start"]), end=millis(e["end"])) for e in exceptions
    ]
    if any(
        e.get("kind") in ("post_delivery", "post_delivery_archive") for e in exceptions
    ):
        raise ValueError("post_delivery 由原始归档自动计数，不接受手工覆盖")
    rows = []
    normalized = []
    for m in metadata:
        item = contract_metadata(
            m["symbol"],
            raw[m["symbol"]],
            m["archives"],
            m.get("evidence"),
            coverage_end=end,
        )
        excluded = [
            r for r in raw[item["symbol"]] if r["close_time"] + 1 > item["delivery_at"]
        ]
        if excluded:
            exceptions.append(
                dict(
                    kind="post_delivery",
                    symbol=item["symbol"],
                    start=min(r["time"] for r in excluded),
                    end=max(r["time"] for r in excluded) + HOUR,
                    count=len(excluded),
                    reason="DESIGN 4.18 v5.1：收盘晚于交割的归档行保留原始记录并剔除",
                )
            )
        normalized.append(item)
        rows.extend(signal_rows(item, raw[item["symbol"]], start, end, exceptions))
    inputs = dict(
        definition=SIGNAL_DEFINITION,
        raw=raw,
        metadata=normalized,
        references=references,
        start=start,
        end=end,
        exceptions=list(exceptions),
        provenance=provenance,
    )
    frame = pl.DataFrame(
        rows,
        schema={
            "time": pl.Int64,
            "symbol": pl.String,
            "close": pl.Float64,
            "quality": pl.Int64,
        },
    ).sort(["time", "symbol"])
    buf = io.BytesIO()
    frame.write_parquet(buf)
    inputs["frame_sha256"] = digest(buf.getvalue())
    payload = encoded(inputs)
    version = digest(payload)
    folder = root / "signals" / "versions" / version
    with lock(root / "signals" / ".build.lock"):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "inputs.json").write_bytes(payload)
        (folder / "hours.parquet").write_bytes(buf.getvalue())
        meta = dict(
            version=version,
            definition=SIGNAL_DEFINITION,
            rows=len(frame),
            start=start,
            end=end,
            metadata=normalized,
            excluded=[m["symbol"] for m in normalized if m["listed_at"] is None],
            frame_sha256=digest((folder / "hours.parquet").read_bytes()),
        )
        atomic_json(folder / "meta.json", meta)
        atomic_json(root / "signals" / "latest.json", dict(version=version))
    return SignalDataset(root, version)


class SignalDataset:
    def __init__(self, root, version=None):
        self.root = Path(root)
        self.version = (
            version
            or json.loads((self.root / "signals/latest.json").read_text())["version"]
        )
        if not re.fullmatch(r"[0-9a-f]{64}", self.version):
            raise ValueError("非法信号版本")
        self.folder = self.root / "signals/versions" / self.version
        self.meta = json.loads((self.folder / "meta.json").read_text())
        self.inputs()

    def inputs(self):
        payload = (self.folder / "inputs.json").read_bytes()
        if digest(payload) != self.version:
            raise ValueError("季度信号输入/元数据损坏")
        value = json.loads(payload)
        if any(
            value[k] != self.meta[k]
            for k in ("metadata", "definition", "frame_sha256", "start", "end")
        ):
            raise ValueError("季度信号元数据与内容版本冲突")
        for ref in value["references"].values():
            data = (self.root / "signals/raw/objects" / ref["sha256"]).read_bytes()
            if digest(data) != ref["sha256"]:
                raise ValueError("季度原始归档损坏")
        return value

    @property
    def metadata(self):
        return self.inputs()["metadata"]

    def read(self, start=None, end=None):
        self.inputs()
        path = self.folder / "hours.parquet"
        if digest(path.read_bytes()) != self.meta["frame_sha256"]:
            raise ValueError("季度信号小时表损坏")
        frame = pl.read_parquet(path)
        if start is not None:
            frame = frame.filter(pl.col("time") >= millis(start))
        if end is not None:
            frame = frame.filter(pl.col("time") < millis(end))
        return frame


def sync_signals(
    root,
    *,
    start,
    end,
    symbols=("BTCUSDT", "ETHUSDT"),
    client=None,
    refresh=False,
    exceptions=(),
    evidence=None,
):
    """每期列举全部月归档并下载全生命周期；独立库不会改变严格面板指针。"""
    root = Path(root)
    own = client is None
    client = client or PublicClient(max_bytes=300_000_000)
    store = RawStore(root / "signals/raw")
    try:
        prefix = archive_prefix("usdm", "klines")
        universe = sorted(
            k[len(prefix) :].strip("/") for k in client.listing(prefix, delimiter="/")
        )
        selected = [
            s
            for s in universe
            if re.fullmatch(
                r"(?:" + "|".join(re.escape(s) for s in symbols) + r")_\d{6}", s
            )
        ]
        snapshot = client.json(FAPI + "/fapi/v1/exchangeInfo")
        current = {r["symbol"]: r for r in snapshot["symbols"]}
        raw, metadata, references = {}, [], {}
        for symbol in selected:
            keys = sorted(
                k
                for k in client.listing(archive_prefix("usdm", "klines", symbol))
                if k.endswith(".zip")
            )
            for key in keys:
                if client.archive(store, key, refresh=refresh) is None:
                    raise ValueError("目录中的季度归档缺 CHECKSUM")
            raw[symbol] = read_series(store, keys, "usdm", "klines")
            ev = dict((evidence or {}).get(symbol, {}))
            if symbol in current:
                entry = current[symbol]
                if (
                    entry.get("quoteAsset") != "USDT"
                    or entry.get("marginAsset") != "USDT"
                    or entry.get("underlyingType") != "COIN"
                ):
                    raise ValueError("季度合约市场类型冲突")
                for key in ("onboardDate", "deliveryDate"):
                    if key in ev and int(ev[key]) != int(entry[key]):
                        raise ValueError("季度元数据来源冲突")
                    ev[key] = entry[key]
                source = dict(
                    kind="exchangeInfo",
                    url=FAPI + "/fapi/v1/exchangeInfo",
                    snapshot_sha256=digest(encoded(snapshot)),
                )
                ev.update(listed_source=source, delivery_source=source)
            metadata.append(dict(symbol=symbol, archives=keys, evidence=ev))
            manifest = store.manifest.read()
            references.update(
                {
                    k: {
                        field: manifest[k][field]
                        for field in ("sha256", "checksum", "checksum_text")
                    }
                    for k in keys
                }
            )
            print(
                f"signals: {symbol}, {len(keys)} months, {len(raw[symbol])} hours",
                flush=True,
            )
        atomic_json(
            root / "signals/sync_plan.json",
            dict(
                metadata=metadata,
                references=references,
                start=millis(start),
                end=millis(end),
                exceptions=list(exceptions),
                provenance=dict(universe=universe, exchangeInfo=snapshot),
            ),
        )
        return write_signals(
            root,
            raw=raw,
            metadata=metadata,
            references=references,
            start=start,
            end=end,
            exceptions=exceptions,
            provenance=dict(universe=universe, exchangeInfo=snapshot),
        )
    finally:
        store.close()
        if own:
            client.client.close()


def build_signals(root, *, exceptions=None):
    """从冻结同步计划离线构建；重试质量处理不再联网或重取归档。"""
    root = Path(root)
    plan = json.loads((root / "signals/sync_plan.json").read_text())

    class Frozen:
        def get(self, key):
            sha = plan["references"][key]["sha256"]
            payload = (root / "signals/raw/objects" / sha).read_bytes()
            if digest(payload) != sha:
                raise ValueError("季度归档对象损坏")
            return payload

    raw = {
        m["symbol"]: read_series(Frozen(), m["archives"], "usdm", "klines")
        for m in plan["metadata"]
    }
    if exceptions is not None:
        plan["exceptions"] = exceptions
    return write_signals(root, raw=raw, **plan)
