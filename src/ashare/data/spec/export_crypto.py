"""币安小时和分钟视图：毫秒转 UTC，逐月投影，明确去掉来源补齐行。"""

import hashlib
from pathlib import Path

import polars as pl

from ashare.data.crypto import minute
from ashare.data.crypto.dataset import CryptoDataset
from ashare.data.crypto.panel import QUALITY as SOURCE_QUALITY
from ashare.data.crypto.timeutil import HOUR, millis, month_end

from .export_common import clean, field, manifest, resource, timestamp, utc_expr

HOUR_MAPPING = {k: k for k in ("open", "high", "low", "close")}
HOUR_MAPPING.update(
    volume="volume_engine",
    amount="quote_volume",
    base_volume="volume",
    taker_buy_amount="taker_buy_quote_volume",
)
HOUR_EXTENSIONS = {"base_volume": "coin", "taker_buy_amount": "USDT"}
QUALITY_FLAGS = {
    str(1 << (16 + i)): "来源质量位：" + key for i, key in enumerate(SOURCE_QUALITY)
}
MINUTE_MAPPING = {
    k: k for k in ("open", "high", "low", "close", "volume", "taker_buy_volume")
}
MINUTE_MAPPING.update(
    amount="quote_volume",
    trade_count="count",
    taker_buy_amount="taker_buy_quote_volume",
)


def _verified(path, expected):
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("来源分区 SHA-256 不匹配")


def _instrument_resource():
    return resource(
        "instruments",
        "instruments",
        [
            field(k, t)
            for k, t in [
                ("symbol", "string"),
                ("asset_type", "string"),
                ("currency", "string"),
                ("list_date", "date"),
                ("multiplier", "number"),
            ]
        ],
        ["symbol"],
    )


def _fields(mapping):
    return [
        field("symbol", "string"),
        field("ts", "datetime"),
        *[
            field(
                k,
                "integer" if k == "trade_count" else "number",
                unit=HOUR_EXTENSIONS.get(k),
                description="来源基础资产成交量"
                if k == "base_volume"
                else "主动买入成交金额"
                if k == "taker_buy_amount"
                else None,
            )
            for k in mapping
        ],
    ]


def _select(frame, column, start, end, symbol, wanted):
    return frame.filter(
        (pl.col(column) >= start)
        & (pl.col(column) < end)
        & pl.col(symbol).is_in(wanted)
    )


class CryptoAdapter:
    def __init__(self, root=None, *, start=None, end=None, symbols=None):
        self.data = CryptoDataset(root)
        self.start = millis(start) if start is not None else self.data.meta["start"]
        self.end = millis(end) if end is not None else self.data.meta["end"]
        self.symbols = list(symbols or self.data.meta["instruments"])
        if self.start >= self.end or set(self.symbols) - set(
            self.data.meta["instruments"]
        ):
            raise ValueError("范围或交易对不合法")
        self.months = [
            f.removesuffix(".parquet")
            for f in sorted(self.data.meta["chunks"])
            if millis(f[:7] + "-01") < self.end and month_end(f[:7]) > self.start
        ]
        self.gaps = []
        for month in self.months:
            source = self._source(month, ["instrument", "time", "quality", "lifecycle"])
            missing = source.filter(
                (pl.col("lifecycle") == "active")
                & ((pl.col("quality") & SOURCE_QUALITY["missing"]) != 0)
            )
            for part in missing.partition_by("instrument"):
                times = sorted(part["time"].to_list())
                if not times:
                    continue
                lo, last = times[0], times[0]
                for t in [*times[1:], None]:
                    if t is None or t != last + HOUR:
                        self.gaps.append(
                            dict(
                                table="bars",
                                symbol=part["instrument"][0],
                                start=timestamp(lo).isoformat(),
                                end=timestamp(last + HOUR).isoformat(),
                                reason="来源 missing 位：缺少行情/标记/指数，不导出其前填占位",
                            )
                        )
                        lo = t
                    last = t

    def _source(self, month, fields=None):
        path = self.data.folder / f"{month}.parquet"
        _verified(path, self.data.meta["chunks"][path.name])
        frame = _select(
            pl.scan_parquet(path),
            "time",
            self.start,
            self.end,
            "instrument",
            self.symbols,
        )
        return (frame.select(fields) if fields else frame).collect(engine="streaming")

    def describe(self):
        fields = _fields(HOUR_MAPPING) + [
            field("quality", "integer"),
            field("suspended", "boolean"),
        ]
        bars = resource("bars", "bars", fields, ["symbol", "ts"], "1h")
        bars["bounds"] = dict(
            start=timestamp(self.start).isoformat(), end=timestamp(self.end).isoformat()
        )
        result = manifest(
            "ashare-crypto-hourly",
            f"AShare币安小时面板 {self.data.version}",
            "crypto",
            [bars, _instrument_resource()],
        )
        result.update(gaps=self.gaps, qualityFlags=QUALITY_FLAGS)
        return result

    def partitions(self):
        yield "instruments", "all"
        for month in self.months:
            yield "bars", month

    def read(self, table, partition):
        if table == "instruments":
            rows = [
                dict(
                    symbol=p["instrument"],
                    asset_type="spot"
                    if p["market"] == "spot"
                    else "perpetual"
                    if p["contract_type"] == "PERPETUAL"
                    else "future",
                    currency=p["quote_asset"],
                    list_date=timestamp(p["listed_at"]).date(),
                    multiplier=float(p.get("multiplier", 1)),
                )
                for p in self.data.profiles
                if p["instrument"] in self.symbols
            ]
            return pl.DataFrame(rows).sort("symbol")
        source = self._source(partition)
        source = source.filter(
            (pl.col("lifecycle") == "active")
            & ((pl.col("quality") & SOURCE_QUALITY["missing"]) == 0)
        )
        quality = pl.col("quality").cast(pl.UInt32)
        translated = quality * 65536
        # 高位逐一保留源标记；低位给统一解释。
        suspect = (
            SOURCE_QUALITY["funding_approximation"]
            | SOURCE_QUALITY["settlement_approximation"]
            | SOURCE_QUALITY["rule_assumption"]
            | SOURCE_QUALITY["invalid_close_time"]
            | SOURCE_QUALITY["funding_interval_inferred"]
        )
        translated = translated | pl.when((quality & suspect) != 0).then(1).otherwise(
            0
        ).cast(pl.UInt32)
        translated = translated | pl.when((quality & SOURCE_QUALITY["rest"]) != 0).then(
            2
        ).otherwise(0).cast(pl.UInt32)
        translated = translated | pl.when(
            (quality & SOURCE_QUALITY["frozen_zero"]) != 0
        ).then(8).otherwise(0).cast(pl.UInt32)
        return clean(
            source.select(
                pl.col("instrument").alias("symbol"),
                utc_expr("time").alias("ts"),
                *[pl.col(raw).alias(name) for name, raw in HOUR_MAPPING.items()],
                translated.alias("quality"),
                ((quality & SOURCE_QUALITY["suspended"]) != 0).alias("suspended"),
            ).sort(["symbol", "ts"])
        )


class MinuteAdapter:
    def __init__(
        self,
        root=None,
        *,
        start=None,
        end=None,
        symbols=None,
        market=None,
        kind="klines",
    ):
        self.root = minute.data_root(root)
        self.meta = minute.metadata(root=self.root)
        candidates = {
            key: part
            for key, part in self.meta["partitions"].items()
            if part["kind"] == kind
            and (market is None or market == part["market"])
            and (
                symbols is None
                or part["symbol"] in symbols
                or f"{part['market']}:{part['symbol']}" in symbols
            )
        }
        if not candidates:
            raise ValueError("未找到所选分钟序列")
        self.start = (
            millis(start)
            if start
            else min(millis(p["month"] + "-01") for p in candidates.values())
        )
        self.end = (
            millis(end)
            if end
            else max(month_end(p["month"]) for p in candidates.values())
        )
        if self.start >= self.end:
            raise ValueError("start 必须早于 end")
        self.parts = {
            k: p
            for k, p in candidates.items()
            if millis(p["month"] + "-01") < self.end
            and month_end(p["month"]) > self.start
        }
        self.symbols = sorted(
            {f"{p['market']}:{p['symbol']}" for p in self.parts.values()}
        )
        self.months = sorted({p["month"] for p in self.parts.values()})
        self.mapping = (
            MINUTE_MAPPING
            if kind == "klines"
            else {k: k for k in ("open", "high", "low", "close")}
        )
        self.kind = kind

    def describe(self):
        bars = resource("bars", "bars", _fields(self.mapping), ["symbol", "ts"], "1m")
        bars["bounds"] = dict(
            start=timestamp(self.start).isoformat(), end=timestamp(self.end).isoformat()
        )
        result = manifest(
            "ashare-crypto-minute-" + self.kind,
            f"AShare币安分钟 {self.kind} {self.meta['version']}",
            "crypto",
            [bars, _instrument_resource()],
        )
        gaps = []
        for p in self.parts.values():
            for gap in p["quality"]["gaps"]:
                lo, hi = max(self.start, gap["start"]), min(self.end, gap["end"])
                if lo < hi:
                    gaps.append(
                        dict(
                            table="bars",
                            symbol=f"{p['market']}:{p['symbol']}",
                            start=timestamp(lo).isoformat(),
                            end=timestamp(hi).isoformat(),
                            reason="分钟来源缺口，保留原缺行",
                        )
                    )
        result["gaps"] = gaps
        return result

    def partitions(self):
        yield "instruments", "all"
        for month in self.months:
            yield "bars", month

    def read(self, table, partition):
        if table == "instruments":
            return pl.DataFrame(
                dict(
                    symbol=self.symbols,
                    asset_type=[
                        "spot" if s.startswith("spot:") else "perpetual"
                        for s in self.symbols
                    ],
                    currency=["USDT"] * len(self.symbols),
                    list_date=pl.Series([None] * len(self.symbols), dtype=pl.Date),
                    multiplier=[1.0] * len(self.symbols),
                )
            )
        frames = []
        for p in self.parts.values():
            if p["month"] != partition:
                continue
            path = self.root / p["file"]
            _verified(path, p["sha256"])
            frames.append(
                pl.scan_parquet(path)
                .filter(
                    (pl.col("open_time") >= self.start)
                    & (pl.col("open_time") < self.end)
                )
                .select(
                    pl.lit(f"{p['market']}:{p['symbol']}").alias("symbol"),
                    utc_expr("open_time").alias("ts"),
                    *[pl.col(raw).alias(name) for name, raw in self.mapping.items()],
                )
                .collect(engine="streaming")
            )
        return clean(pl.concat(frames).sort(["symbol", "ts"]))
