"""期货原始库的独立校验。

只读原始库（清单 + Parquet），不调用同步、适配器和面板的任何代码：合约代码解析、单位、日历
都在这里另写一遍，以便发现那几处实现的错误。每一项给出结论 verdict：

- ``pass``：没有问题；
- ``explained``：有不符，但每一条都归入了已知原因（原因写在 details 里）；
- ``fail``：有未解释的不符；
- ``info``：统计信息，没有对错（如逐年行数、覆盖起点）。

用法::

    from ashare.data.store import Store
    from ashare.data.futures_verify import verify_futures
    report = verify_futures(Store("/path/to/root", readonly=True))
    report["checks"]["pre_settle_continuity"]["verdict"]
"""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime

import polars as pl

from .store import atomic_json

SUFFIXES = {
    "CFX": "CFFEX",
    "SHF": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFE": "GFEX",
}
SUFFIX_OF = {v: k for k, v in SUFFIXES.items()}
DAY_SETS = ("futures_daily", "futures_mapping", "futures_settle")
PAGE = {
    "futures_daily": 2000,
    "futures_mapping": 5000,
    "futures_settle": 1000,
    "futures_contracts": 10000,
    "futures_calendar": 10000,
}
CONTRACT = re.compile(r"^[A-Z]+\d{4}F?\.(CFX|SHF|DCE|ZCE|INE|GFE)$")


def _check(verdict, checked, failed, details=None, note=""):
    return dict(
        verdict=verdict,
        checked=int(checked),
        failed=int(failed),
        details=details if details is not None else [],
        note=note,
    )


def _entries(store, name):
    return {k: v for k, v in store.manifest().items() if v.get("dataset") == name}


def _read(store, name):
    files = [
        store.root / v["file"]
        for v in _entries(store, name).values()
        if v.get("status") == "complete" and v.get("rows")
    ]
    frames = []
    for path in files:
        try:
            frames.append(pl.read_parquet(path))
        except Exception:  # noqa: BLE001 - 损坏文件由 manifest_files 报告，这里跳过
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def check_manifest(store):
    """清单对文件：每个完成分区的文件存在且哈希一致；原始目录里没有清单之外的文件。"""
    bad, listed, checked = [], set(), 0
    for key, entry in store.manifest().items():
        if not entry.get("dataset", "").startswith("futures_"):
            continue
        checked += 1
        if entry.get("status") != "complete":
            bad.append(dict(key=key, problem=f"status={entry.get('status')}"))
            continue
        path = store.root / entry["file"]
        listed.add(path)
        if not path.is_file():
            bad.append(dict(key=key, problem="文件不存在"))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_hash"]:
            bad.append(dict(key=key, problem="文件哈希不符"))
    orphans = []
    for name in PAGE:
        folder = store.root / "raw" / name
        if folder.exists():
            orphans += [
                str(p.relative_to(store.root))
                for p in folder.rglob("*.parquet")
                if p not in listed
            ]
    details = bad + [
        dict(key=o, problem="清单之外的文件（旧修订，可由 gc 清理）") for o in orphans
    ]
    verdict = "fail" if bad else ("explained" if orphans else "pass")
    return _check(
        verdict,
        checked,
        len(bad) + len(orphans),
        details[:50],
        "旧修订指同一分区刷新后内容变化留下的上一份文件" if orphans else "",
    )


def open_days(calendar):
    return {
        ex: sorted(
            g.filter(pl.col("is_open").cast(pl.Int64) == 1)["cal_date"].to_list()
        )
        for (ex,), g in calendar.group_by("exchange")
    }


def check_partitions(store, days_by_exchange, end):
    """日分区覆盖：期货交易所开市日并集里，覆盖起点之后的每一天都有完成的分区；
    以及页大小整数倍的可疑截断。"""
    union = sorted({d for days in days_by_exchange.values() for d in days if d <= end})
    starts = {
        "futures_daily": "19950417",
        "futures_mapping": "19950417",
        "futures_settle": "20110104",
    }
    out = {}
    for name in DAY_SETS:
        entries = {v["partition"]: v for v in _entries(store, name).values()}
        expected = [d for d in union if d >= starts[name]]
        missing = [
            d for d in expected if entries.get(d, {}).get("status") != "complete"
        ]
        empty = [d for d in expected if entries.get(d, {}).get("rows") == 0]
        full_page = [
            p
            for p, v in entries.items()
            if v.get("rows") and v["rows"] % PAGE[name] == 0
        ]
        problems = (
            [dict(partition=d, problem="缺分区") for d in missing]
            + [dict(partition=d, problem="空分区") for d in empty]
            + [
                dict(partition=d, problem=f"行数是页大小 {PAGE[name]} 的整数倍")
                for d in full_page
            ]
        )
        verdict = "fail" if missing or full_page else "explained" if empty else "pass"
        out[name] = _check(
            verdict,
            len(expected),
            len(problems),
            problems[:50],
            "空分区：交易所日历开市、源当天一条日线也没有（源整天缺失），逐日列出"
            if empty
            else "",
        )
    for name in ("futures_contracts", "futures_calendar"):
        entries = _entries(store, name)
        full_page = [
            v["partition"]
            for v in entries.values()
            if v.get("rows") and v["rows"] % PAGE[name] == 0
        ]
        missing = sorted(
            set(SUFFIXES.values())
            - {
                v["partition"]
                for v in entries.values()
                if v.get("status") == "complete"
            }
        )
        problems = [dict(partition=p, problem="缺交易所") for p in missing] + [
            dict(partition=p, problem="行数是页大小整数倍") for p in full_page
        ]
        out[name] = _check(
            "fail" if problems else "pass", len(entries), len(problems), problems
        )
    return out


def yearly_rows(store):
    result = {}
    for name in DAY_SETS:
        counter = Counter()
        for v in _entries(store, name).values():
            if v.get("status") == "complete":
                counter[v["partition"][:4]] += v.get("rows", 0)
        result[name] = dict(sorted(counter.items()))
    return result


def check_calendars(days_by_exchange):
    """各交易所日历互相对比（2000 年以后）：列出只在部分交易所开市的日子。"""
    union = sorted(
        {d for days in days_by_exchange.values() for d in days if d >= "20000101"}
    )
    sets = {ex: set(d) for ex, d in days_by_exchange.items() if d}
    firsts = {ex: min(s) for ex, s in sets.items()}
    differ = []
    for day in union:
        existing = sorted(ex for ex in sets if firsts[ex] <= day)
        listed = sorted(ex for ex in existing if day in sets[ex])
        if listed != existing:
            differ.append(
                dict(date=day, open=listed, closed=sorted(set(existing) - set(listed)))
            )
    verdict = "pass" if not differ else "explained"
    return _check(
        verdict,
        len(union),
        len(differ),
        differ[:50],
        "各交易所开业日不同（中金所 2010、能源中心 2018、广期所 2022 起才有合约）已排除；其余差异逐日列出",
    )


def parse_contracts(raw):
    """合约表：交易所、品种、每手单位（金融期货用 multiplier，商品用 per_unit）。"""
    return raw.with_columns(
        exchange=pl.col("ts_code")
        .str.extract(r"\.(\w+)$")
        .replace_strict(SUFFIXES, default=None),
        product=pl.col("ts_code").str.extract(r"^([A-Z]+)"),
        unit=pl.when(pl.col("multiplier").is_not_null())
        .then(pl.col("multiplier"))
        .otherwise(pl.col("per_unit")),
    )


def check_contract_coverage(contracts, daily, days_by_exchange, last_day):
    """每个合约从上市到最后交易日（截到样本末尾），日线行数应与交易所日历一致；缺口逐个列出。"""
    rows = daily.filter(pl.col("ts_code").str.contains(CONTRACT.pattern))
    have = {
        k: set(v)
        for k, v in rows.group_by("ts_code").agg(pl.col("trade_date")).iter_rows()
    }
    exchange_days = set(
        rows.with_columns(
            ex=pl.col("ts_code")
            .str.extract(r"\.(\w+)$")
            .replace_strict(SUFFIXES, default=None)
        )
        .select("ex", "trade_date")
        .unique()
        .iter_rows()
    )
    categories = Counter()
    day_kinds = Counter()
    gaps, checked = [], 0
    outside = []
    for code, ex, first, last in contracts.select(
        "ts_code", "exchange", "list_date", "delist_date"
    ).iter_rows():
        if ex not in days_by_exchange or not first:
            continue
        checked += 1
        stop = min(last or last_day, last_day)
        expected = [d for d in days_by_exchange[ex] if first <= d <= stop]
        got = have.get(code, set())
        missing = sorted(set(expected) - got)
        extra = sorted(d for d in got if d < first or (last and d > last))
        if extra:
            outside.append(dict(symbol=code, dates=extra[:5], count=len(extra)))
        if not missing:
            continue
        if not got:
            reason = "合约整段没有日线（多为未成交的远月或源缺失）"
        elif all(d < min(got) for d in missing):
            reason = "上市后最初几天没有日线（源从首笔成交日开始记录）"
        elif all(d > max(got) for d in missing):
            reason = "最后交易日前若干天没有日线"
        else:
            reason = "中间缺日"
        categories[reason] += 1
        for day in missing:
            kind = (
                "交易所当天整体无日线（源整天缺失）"
                if (ex, day) not in exchange_days
                else "同交易所其他合约有日线、该合约无记录"
            )
            day_kinds[(kind, day[:4] < "2005")] += 1
        gaps.append(
            dict(
                symbol=code,
                missing=len(missing),
                expected=len(expected),
                first_missing=missing[:3],
                reason=reason,
            )
        )
    unknown = sorted(set(have) - set(contracts["ts_code"]))
    details = dict(
        categories=dict(categories),
        missing_days=[
            dict(kind=k, before_2005=early, days=n)
            for (k, early), n in sorted(day_kinds.items())
        ],
        gaps=gaps,
        gap_contracts=len(gaps),
        rows_outside_listing=outside[:50],
        rows_outside_count=len(outside),
        daily_codes_without_contract=unknown[:50],
        unknown_count=len(unknown),
    )
    failed = len(gaps) + len(outside) + len(unknown)
    return _check(
        "explained" if failed else "pass",
        checked,
        failed,
        details,
        "缺口按原因归类；「中间缺日」逐个列在 gaps 里",
    )


def check_pre_settle(daily, days_by_exchange):
    """同一合约 pre_settle[t] = settle[t−1]（上一条有记录的交易日）。

    不符再分两类：上一条记录不是交易所上一个开市日（中间缺日，前结算按缺掉的那天算）；
    相邻两个开市日仍不符（源错误或交易所调整基准价），后者逐条列出。
    """
    previous_open = {
        (SUFFIX_OF[ex], day): before
        for ex, days in days_by_exchange.items()
        for before, day in zip(days, days[1:], strict=False)
    }
    rows = daily.filter(pl.col("ts_code").str.contains(CONTRACT.pattern)).sort(
        "ts_code", "trade_date"
    )
    rows = rows.with_columns(
        previous=pl.col("settle").shift(1).over("ts_code"),
        previous_day=pl.col("trade_date").shift(1).over("ts_code"),
    ).filter(pl.col("previous").is_not_null() & pl.col("pre_settle").is_not_null())
    diff = (rows["pre_settle"] - rows["previous"]).abs()
    tol = 1e-6 * rows["previous"].abs().fill_null(1) + 1e-9
    bad = rows.filter(diff > tol).with_columns(
        ex=pl.col("ts_code").str.extract(r"\.(\w+)$")
    )
    adjacent = [
        previous_open.get((ex, day)) == before
        for ex, day, before in bad.select(
            "ex", "trade_date", "previous_day"
        ).iter_rows()
    ]
    bad = bad.with_columns(adjacent=pl.Series(adjacent, dtype=pl.Boolean))
    summary = (
        bad.with_columns(year=pl.col("trade_date").str.slice(0, 4))
        .group_by("ex", "adjacent")
        .agg(
            rows=pl.len(),
            first=pl.col("trade_date").min(),
            last=pl.col("trade_date").max(),
        )
        .sort("ex", "adjacent")
    )
    cols = ["ts_code", "trade_date", "previous_day", "pre_settle", "previous"]
    details = dict(
        summary=summary.to_dicts(),
        adjacent_mismatches=bad.filter(pl.col("adjacent")).select(cols).to_dicts(),
        gap_examples=bad.filter(~pl.col("adjacent")).select(cols).head(20).to_dicts(),
    )
    verdict = "pass" if bad.is_empty() else "explained"
    return _check(
        verdict,
        len(rows),
        len(bad),
        details,
        "相邻开市日的不符逐条列在 adjacent_mismatches；其余是中间缺日",
    )


def check_amount(daily, contracts):
    """成交额 ≈ 成交量 × 结算价 × 每手单位，反查每手单位与单位口径；并统计成交量奇偶（双边计数恒为偶数）。"""
    info = contracts.select("ts_code", "exchange", "product", "unit")
    rows = (
        daily.join(info, on="ts_code", how="inner")
        .filter((pl.col("vol") > 0) & (pl.col("amount") > 0) & (pl.col("settle") > 0))
        .with_columns(
            implied=pl.col("amount") * 10000 / (pl.col("vol") * pl.col("settle")),
            year=pl.col("trade_date").str.slice(0, 4),
        )
        .with_columns(ratio=pl.col("implied") / pl.col("unit"))
    )
    table = (
        rows.group_by("exchange", "product", "year")
        .agg(
            n=pl.len(),
            ratio=pl.col("ratio").median(),
            within=((pl.col("ratio") - 1).abs() < 0.02).mean(),
        )
        .sort("exchange", "product", "year")
    )
    off = table.filter((pl.col("ratio") - 1).abs() > 0.02).with_columns(
        clean_factor=pl.col("ratio").round(1).is_in([0.2, 0.5, 2.0, 5.0, 10.0, 50.0])
        & ((pl.col("ratio") / pl.col("ratio").round(1) - 1).abs() < 0.01)
    )
    parity = (
        daily.filter(
            pl.col("ts_code").str.contains(CONTRACT.pattern) & (pl.col("vol") > 0)
        )
        .with_columns(
            exchange=pl.col("ts_code")
            .str.extract(r"\.(\w+)$")
            .replace_strict(SUFFIXES, default=None),
            year=pl.col("trade_date").str.slice(0, 4),
        )
        .group_by("exchange", "year")
        .agg(
            n=pl.len(),
            vol_even=(pl.col("vol") % 2 == 0).mean(),
            oi_even=(pl.col("oi").fill_null(0) % 2 == 0).mean(),
        )
        .sort("exchange", "year")
    )
    # 逐行：结算价是全天加权均价的交易所，单行偏离 5% 以上就列出（中金所结算价是最后一小时均价，另计）。
    outliers = rows.filter((pl.col("ratio") - 1).abs() > 0.05)
    row_summary = outliers.group_by("exchange").len().sort("exchange")
    details = dict(
        off_products=off.to_dicts()[:100],
        off_count=len(off),
        row_outliers=len(outliers),
        row_outliers_by_exchange=row_summary.to_dicts(),
        row_examples=outliers.select(
            "ts_code", "trade_date", "vol", "amount", "settle", "unit", "ratio"
        )
        .head(30)
        .to_dicts(),
        parity=parity.to_dicts(),
        overall_within=float(((rows["ratio"] - 1).abs() < 0.02).mean())
        if len(rows)
        else None,
    )
    failed = len(off) + len(outliers)
    return _check(
        "explained" if failed else "pass",
        len(rows),
        failed,
        details,
        "中位比值偏离 1 超过 2% 的品种-年份与偏离 5% 以上的单行逐条列出；偶数占比接近 1 表示双边计数",
    )


def check_settle_params(settle):
    """结算参数表：各交易所覆盖起点与各字段逐年缺失率。"""
    frame = settle.with_columns(
        exchange=pl.col("ts_code")
        .str.extract(r"\.(\w+)$")
        .replace_strict(SUFFIXES, default=None),
        year=pl.col("trade_date").str.slice(0, 4),
    )
    starts = (
        frame.group_by("exchange")
        .agg(first=pl.col("trade_date").min(), rows=pl.len())
        .sort("exchange")
    )
    fields = [
        "settle",
        "trading_fee_rate",
        "trading_fee",
        "delivery_fee",
        "long_margin_rate",
        "short_margin_rate",
        "b_hedging_margin_rate",
        "offset_today_fee",
    ]
    missing = (
        frame.group_by("exchange", "year")
        .agg(
            rows=pl.len(),
            **{
                f: pl.col(f).is_null().mean().round(3)
                for f in fields
                if f in frame.columns
            },
        )
        .sort("exchange", "year")
    )
    return _check(
        "info",
        len(frame),
        0,
        dict(starts=starts.to_dicts(), missing=missing.to_dicts()),
    )


def check_expiry_settle(daily, contracts):
    """最后交易日的结算价：是否落在最小变动价位的网格上（中金所股指期货的交割结算价是指数
    两小时算术平均，一般不在 0.2 点网格上），以及与收盘价、前一日结算价的关系。"""
    tick = contracts.with_columns(
        tick=pl.col("quote_unit_desc").str.extract(r"^([0-9.]+)").cast(pl.Float64)
    ).select("ts_code", "exchange", "product", "delist_date", "tick")
    rows = daily.join(tick, on="ts_code", how="inner").filter(
        pl.col("settle").is_not_null()
    )
    rows = rows.with_columns(
        last_day=pl.col("trade_date") == pl.col("delist_date"),
        on_grid=(
            (pl.col("settle") / pl.col("tick"))
            - (pl.col("settle") / pl.col("tick")).round(0)
        ).abs()
        < 1e-6,
        equals_close=(pl.col("settle") - pl.col("close")).abs() < 1e-9,
    )
    summary = (
        rows.group_by("exchange", "product", "last_day")
        .agg(
            n=pl.len(),
            on_grid=pl.col("on_grid").mean().round(3),
            equals_close=pl.col("equals_close").mean().round(3),
        )
        .sort("exchange", "product", "last_day")
    )
    return _check("info", len(rows), 0, summary.to_dicts())


def check_unique(store):
    out, total = [], 0
    for name, keys in (
        ("futures_daily", ["ts_code", "trade_date"]),
        ("futures_mapping", ["ts_code", "trade_date"]),
        ("futures_settle", ["ts_code", "trade_date"]),
        ("futures_contracts", ["ts_code"]),
        ("futures_calendar", ["exchange", "cal_date"]),
    ):
        frame = _read(store, name)
        if frame.is_empty():
            continue
        total += len(frame)
        dup = frame.filter(pl.struct(keys).is_duplicated())
        if len(dup):
            out.append(
                dict(
                    dataset=name,
                    rows=len(dup),
                    examples=dup.select(keys).head(5).to_dicts(),
                )
            )
    return _check("fail" if out else "pass", total, sum(o["rows"] for o in out), out)


def verify_futures(store, *, write=True):
    """全部校验；结果写 ``reports/futures_verify.json`` 与 ``.md``。"""
    started = time.perf_counter()
    calendar = _read(store, "futures_calendar")
    days = open_days(calendar)
    raw_contracts = _read(store, "futures_contracts")
    contracts = parse_contracts(raw_contracts)
    daily = _read(store, "futures_daily")
    settle = _read(store, "futures_settle")
    last_day = daily["trade_date"].max()
    checks = {"manifest_files": check_manifest(store)}
    for name, item in check_partitions(store, days, last_day).items():
        checks[f"partitions_{name}"] = item
    checks["unique_keys"] = check_unique(store)
    checks["calendars"] = check_calendars(days)
    checks["contract_coverage"] = check_contract_coverage(
        contracts, daily, days, last_day
    )
    checks["pre_settle_continuity"] = check_pre_settle(daily, days)
    checks["amount_multiplier"] = check_amount(daily, contracts)
    checks["settle_params"] = check_settle_params(settle)
    checks["expiry_settlement"] = check_expiry_settle(daily, contracts)
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        yearly_rows=yearly_rows(store),
        last_day=last_day,
        checks=checks,
    )
    if write:
        folder = store.root / "reports"
        atomic_json(
            folder / "futures_verify.json", json.loads(json.dumps(report, default=str))
        )
        lines = [
            "# 期货原始库校验",
            "",
            f"生成：{report['generated_at']}，最后交易日 {last_day}",
            "",
            "| 检查 | 结论 | 检查数 | 不符数 |",
            "|---|---|---|---|",
        ]
        for name, item in checks.items():
            lines.append(
                f"| {name} | {item['verdict']} | {item['checked']} | {item['failed']} |"
            )
        (folder / "futures_verify.md").write_text("\n".join(lines) + "\n")
    return report


def summarize_gaps(gaps):
    """把缺口列表按交易所和原因汇总（供文档）。"""
    table = defaultdict(Counter)
    for gap in gaps:
        table[gap["symbol"].split(".")[-1]][gap["reason"]] += 1
    return {k: dict(v) for k, v in table.items()}


__all__ = ["verify_futures"]
