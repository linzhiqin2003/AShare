"""按月不可变面板、读取、数组、日线与截断安全的成交额排名。"""

import json

import numpy as np
import polars as pl

from ashare.paths import dataset_dir

from ..store import atomic_json, lock
from .coverage import coverage
from .storage import digest, encoded
from .timeutil import DAY, HOUR, millis


def write_panel(
    root,
    frame,
    *,
    profiles,
    events,
    data,
    notes,
    raw_version=None,
    exceptions=(),
    recent=None,
):
    root = dataset_dir("crypto", root)
    inputs = dict(
        profiles=profiles,
        events=events,
        data=data,
        notes=notes,
        raw_version=raw_version,
        exceptions=list(exceptions),
    )
    if recent is not None:
        inputs["recent"] = recent
    inferred = bool(recent and recent["allow_inferred_funding"])
    # JSON 不写 NaN；原始规则中缺清算费等参数保持 null，由 C3 明确处理。
    payload = encoded(inputs)
    import io

    buf = io.BytesIO()
    frame.write_ipc(buf)
    version = digest(payload + buf.getvalue())
    folder = root / "panels" / version
    with lock(root / ".panel.lock"):
        if not (folder / "meta.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "inputs.json").write_bytes(payload)
            chunks = {}
            labelled = frame.with_columns(
                pl.from_epoch("time", time_unit="ms")
                .dt.strftime("%Y-%m")
                .alias("month")
            )
            for key, group in labelled.partition_by("month", as_dict=True).items():
                name = key[0] + ".parquet"
                path = folder / name
                group.drop("month").write_parquet(path)
                chunks[name] = digest(path.read_bytes())
            meta = dict(
                version=version,
                schema_version=1,
                chunks=chunks,
                inputs_sha256=digest(payload),
                start=frame["time"].min(),
                end=frame["time"].max() + HOUR,
                instruments=sorted(frame["instrument"].unique().to_list()),
                rows=frame.height,
                raw_version=raw_version,
                notes=notes,
                coverage=coverage(data),
            )
            if recent is not None:
                meta.update(allow_inferred_funding=inferred, recent=recent)
            atomic_json(folder / "meta.json", meta)
        atomic_json(
            root / ("latest_inferred.json" if inferred else "latest.json"),
            dict(version=version),
        )
    return CryptoDataset(root, version)


class CryptoDataset:
    def __init__(self, root=None, version=None, *, allow_inferred_funding=False):
        self.root = dataset_dir("crypto", root)
        self.version = (
            version
            or json.loads(
                (
                    self.root
                    / (
                        "latest_inferred.json"
                        if allow_inferred_funding
                        else "latest.json"
                    )
                ).read_text()
            )["version"]
        )
        if len(self.version) != 64 or any(
            c not in "0123456789abcdef" for c in self.version
        ):
            raise ValueError("非法数据版本")
        self.folder = self.root / "panels" / self.version
        self.meta = json.loads((self.folder / "meta.json").read_text())

    def inputs(self):
        payload = (self.folder / "inputs.json").read_bytes()
        if digest(payload) != self.meta["inputs_sha256"]:
            raise ValueError("面板输入快照损坏")
        return json.loads(payload)

    @property
    def profiles(self):
        return self.inputs()["profiles"]

    def read(self, start=None, end=None, instruments=None, fields=None):
        frames = []
        for filename, sha in sorted(self.meta["chunks"].items()):
            path = self.folder / filename
            if digest(path.read_bytes()) != sha:
                raise ValueError(f"面板分块损坏: {filename}")
            frames.append(pl.read_parquet(path))
        frame = pl.concat(frames)
        if start is not None:
            frame = frame.filter(pl.col("time") >= millis(start))
        if end is not None:
            frame = frame.filter(pl.col("time") < millis(end))
        if instruments is not None:
            if set(instruments) - set(self.meta["instruments"]):
                raise KeyError("请求了数据版本中不存在的交易对")
            frame = frame.filter(pl.col("instrument").is_in(instruments))
        frame = frame.sort(["time", "instrument"])
        return (
            frame.select(["time", "instrument", *fields])
            if fields is not None
            else frame
        )

    def arrays(self, fields, *, instruments=None, start=None, end=None):
        frame = self.read(start, end, instruments)
        names = list(instruments or sorted(frame["instrument"].unique().to_list()))
        times = np.array(sorted(frame["time"].unique().to_list()), dtype=np.int64)
        result = {}
        for field in fields:
            columns = []
            for name in names:
                part = frame.filter(pl.col("instrument") == name).sort("time")
                if not np.array_equal(part["time"].to_numpy(), times):
                    raise ValueError("面板不是完整矩形")
                columns.append(part[field].to_numpy())
            result[field] = np.column_stack(columns)
        return dict(time=times, instruments=names, **result)

    def daily(self, trade_hour=0, *, invalid_prices="raise", **kwargs):
        """日线信号默认拒绝异常时刻；显式 carry 与小时回测一样按前值持平。

        原始异常价始终保留在 read()，不把质检原值直接交给日频信号。
        """
        frame = signal_frame(self.read(**kwargs), invalid_prices=invalid_prices)
        return aggregate_daily(frame, trade_hour=trade_hour)


def signal_frame(frame, *, invalid_prices="raise"):
    if invalid_prices not in ("raise", "carry"):
        raise ValueError("invalid_prices 只能为 raise/carry")
    invalid = (pl.col("lifecycle") == "active") & ((pl.col("quality") & 2048) != 0)
    if invalid_prices == "raise":
        if frame.select(invalid.any()).item():
            raise ValueError(
                "日线信号含异常时间戳价格；显式 invalid_prices='carry' 才可持平"
            )
        return frame
    frame = frame.sort(["instrument", "time"])
    missing = (pl.col("lifecycle") == "active") & (
        (pl.col("quality") & (1 | 2048)) != 0
    )
    frame = frame.with_columns(
        pl.when(missing)
        .then(None)
        .otherwise(pl.col("close"))
        .forward_fill()
        .over("instrument")
        .alias("_signal_carry")
    )
    if frame.select(
        (
            missing
            & (pl.col("_signal_carry").is_null() | ~pl.col("_signal_carry").is_finite())
        ).any()
    ).item():
        raise ValueError("异常日线信号缺少此前估值")
    return frame.with_columns(
        *[
            pl.when(missing).then(pl.col("_signal_carry")).otherwise(pl.col(f)).alias(f)
            for f in ("open", "high", "low", "close")
        ],
        *[
            pl.when(missing).then(0.0).otherwise(pl.col(f)).alias(f)
            for f in ("volume", "quote_volume", "taker_buy_quote_volume")
        ],
    ).drop("_signal_carry")


def aggregate_daily(frame, *, trade_hour=0):
    """只输出完整 24 小时的日区间，available_at 为右端；不在这里移位信号。"""
    if not isinstance(trade_hour, int) or not 0 <= trade_hour < 24:
        raise ValueError("trade_hour 必须是 UTC 0..23")
    shifted = frame.sort("time").with_columns(
        (((pl.col("time") - trade_hour * HOUR) // DAY) * DAY + trade_hour * HOUR).alias(
            "day"
        )
    )
    return (
        shifted.group_by(["day", "instrument"])
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("quote_volume").sum(),
            pl.col("taker_buy_quote_volume").sum(),
            pl.col("funding").sum().alias("funding_report_only"),
            pl.col("time").n_unique().alias("hours"),
            pl.col("quality").bitwise_or(),
        )
        .filter(pl.col("hours") == 24)
        .with_columns((pl.col("day") + DAY).alias("available_at"))
        .sort(["day", "instrument"])
    )


def rank_coins(frame, profiles, at, *, n, lookback="30D", min_age="30D"):
    """给 C3 的数据核对辅助入口；只计 at 之前已完成的永续成交额。"""
    at = millis(at)

    def duration(value):
        if isinstance(value, int):
            return value
        unit = {"D": DAY, "h": HOUR}[value[-1]]
        return int(value[:-1]) * unit

    window, age = duration(lookback), duration(min_age)
    if n <= 0 or window <= 0 or age < 0:
        raise ValueError("排名参数非法")
    allowed = [
        p["instrument"]
        for p in profiles
        if p["contract_type"] == "PERPETUAL"
        and p["quote_asset"] == p["margin_asset"] == "USDT"
        and at - millis(p["listed_at"]) >= age
    ]
    return (
        frame.filter(
            pl.col("instrument").is_in(allowed)
            & (pl.col("time") >= at - window)
            & (pl.col("time") + HOUR <= at)
            & (pl.col("lifecycle") == "active")
        )
        .group_by("coin")
        .agg(pl.col("quote_volume").sum())
        .sort(["quote_volume", "coin"], descending=[True, False])
        .head(n)
    )


def open(root=None, version=None, *, allow_inferred_funding=False):
    return CryptoDataset(root, version, allow_inferred_funding=allow_inferred_funding)
