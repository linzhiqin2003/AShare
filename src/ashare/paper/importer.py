"""只读导入原始录制；保留发送、接收和证券自身的行情时间。"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from paperbroker.journal import canonical, durable_write

from .market import TZ, normalize, rows, suspension_rules, timestamp
from .schedule import Schedule
from .storage import ParquetStore


def import_raw(directory: Path, output: Path, day: str) -> dict:
    # 禁止向原始目录写入；两遍流式扫描，内存不保存全市场全天响应。
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("输出目录不能位于原始录制目录中")
    paths = sorted(directory.rglob("*.json.gz"))
    headers = []
    symbols = set()
    for path in paths:
        with gzip.open(path, "rt") as handle:
            raw = json.load(handle)
        received = timestamp(raw["t_recv"])
        if datetime.fromtimestamp(received / 1000, TZ).strftime("%Y%m%d") != day:
            continue
        headers.append((received, str(path), timestamp(raw["t_send"]), raw["api_name"]))
        if raw["response"].get("code") == 0 and raw["api_name"] in (
            "rt_k",
            "stk_limit",
        ):
            symbols.update(r["ts_code"] for r in rows(raw["response"]))
    if not headers or not symbols:
        raise ValueError("该日没有可导入的证券快照")
    symbols = sorted(symbols)
    store = ParquetStore(output / "quotes")
    counts = Counter()
    before_open = Counter()
    lag, network, updates = [], [], []
    prior: dict[str, tuple[int, int]] = {}
    limits: dict = {}
    suspended: set = set()
    intervals: dict = {}
    sequence = 0
    for received, path, sent, api in sorted(headers):
        with gzip.open(path, "rt") as handle:
            raw = json.load(handle)
        counts[f"files_{api}"] += 1
        network.append(received - sent)
        response = raw["response"]
        if response.get("code") != 0:
            counts[f"error_code_{response.get('code')}"] += 1
            continue
        records = rows(response)
        if api == "stk_limit":
            limits = {r["ts_code"]: (r["up_limit"], r["down_limit"]) for r in records}
        elif api == "suspend_d":
            suspended, intervals = suspension_rules(records)
        if api not in ("rt_k", "stk_auction"):
            continue
        early = (
            datetime.fromtimestamp(received / 1000, TZ).strftime("%H:%M:%S")
            < "09:30:00"
        )
        if api == "rt_k" and early:
            before_open["responses"] += 1
            before_open["rows"] += len(records)
            before_open["empty_responses"] += not records
        if api == "stk_auction":
            records = [
                r
                | dict(
                    trade_time=f"{datetime.strptime(day, '%Y%m%d').date()} 09:25:00",
                    close=r["price"],
                    open=r["price"],
                    auction_volume=r["vol"],
                )
                for r in records
            ]
        for r in records:
            try:
                trade = timestamp(r["trade_time"])
                date = datetime.fromtimestamp(trade / 1000, TZ).strftime("%Y%m%d")
            except (KeyError, TypeError, ValueError, OverflowError):
                counts["invalid_trade_time"] += 1
                continue
            if date != day:
                counts["old_date_rows"] += 1
                if early:
                    before_open["old_date_rows"] += 1
                continue
            if api == "rt_k":
                lag.append(received - trade)
                vol = round(float(r.get("vol") or 0))
                previous = prior.get(r["ts_code"])
                if previous:
                    counts["unchanged_trade_time"] += trade == previous[0]
                    counts["volume_reversal"] += vol < previous[1]
                    if trade > previous[0]:
                        updates.append(trade - previous[0])
                prior[r["ts_code"]] = trade, vol
        quotes = normalize(
            records,
            symbols,
            day=day,
            limits=limits,
            suspended=suspended,
            suspension_intervals=intervals,
            auction_kind=1 if api == "stk_auction" else 0,
        )
        sequence += 1
        store.received_at, store.sent_at, store.source = received, sent, api
        store.put(sequence, quotes)
        counts["normalized_rows"] += len(quotes)

    def distribution(values: list[int]) -> dict:
        if not values:
            return {"count": 0}
        return dict(
            count=len(values),
            min=min(values),
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            max=max(values),
        )

    start = datetime.strptime(day, "%Y%m%d").replace(tzinfo=TZ)
    table = Schedule([dict(cal_date=day, is_open=1)]).table
    events = []
    for phase, clock in (
        ("preopen", table.preopen),
        ("open_auction", table.open_auction),
        ("continuous", table.continuous),
        ("break", table.lunch),
        ("continuous", table.afternoon),
        ("close_auction", table.close_auction),
        ("closed", table.close),
    ):
        events.append(
            dict(
                type="session",
                time=timestamp(datetime.combine(start.date(), clock, TZ)),
                day=int(day),
                phase=phase,
            )
        )
    # 回放脚本包含时段，不猜测未观察到的竞价量，也不凭盘中样本自动结算。
    from dataclasses import asdict

    from .profiles import configuration

    script = dict(
        version=2, symbols=symbols, config=asdict(configuration(symbols)), events=events
    )
    durable_write(output / "script.json", canonical(script))
    summary = dict(
        version=2,
        day=day,
        symbols=len(symbols),
        batches=sequence,
        counts=dict(counts),
        before_0930=dict(before_open),
        trade_lag_ms=distribution(lag),
        network_ms=distribution(network),
        update_interval_ms=distribution(updates),
    )
    durable_write(output / "import-report.json", canonical(summary))
    return summary
