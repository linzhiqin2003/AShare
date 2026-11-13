"""期货原始库与期货面板的独立校验。

原始库部分只读清单与 Parquet，合约代码、单位、日历都在这里另写一遍，不调用同步和适配器；
面板部分读构建好的统一字段表（``futures_panel/<版本>/*.parquet``），检查适配器的产出。

每一项的结论 verdict：

- ``pass``：没有不符；
- ``explained``：有不符，但每一条都在已知例外清单里（``futures_known_exceptions.json``，
  逐条写明合约、日期和原因）；
- ``fail``：有清单之外的不符，或清单不允许例外的检查（统一字段的取值范围、引用完整性）有不符；
- ``info``：统计信息，没有对错。

总结论 ``verdict`` 为 fail 当且仅当有一项 fail；命令行 ``verify-futures`` 此时以非零状态退出。
清单里有、本次没出现的条目列在 ``stale_exceptions``（数据被修正了，清单该删条目）。
清单的 ``cause_unknown`` 列出「不符已确认、原因未查明」的原因编号；用到这些原因的条目在
``known_exceptions.cause_unknown`` 里单独计数（仍算已解释，但不和有确切原因的混在一起）。

用法::

    from ashare.data.store import Store
    from ashare.data.futures_verify import verify_futures
    report = verify_futures(Store("/path/to/root", readonly=True))
    report["verdict"], report["checks"]["pre_settle_continuity"]["verdict"]
"""

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

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
STARTS = {
    "futures_daily": "19950417",
    "futures_mapping": "19950417",
    "futures_settle": "20110104",
}
CONTRACT = re.compile(r"^[A-Z]+\d{4}F?\.(CFX|SHF|DCE|ZCE|INE|GFE)$")
# 2020-01-02 起上期所、能源中心、大商所、郑商所改为单边统计（此前成交量、持仓量恒为偶数）。
SINGLE_SIDED_FROM = {
    "SHFE": "20200101",
    "INE": "20200101",
    "DCE": "20200101",
    "CZCE": "20200101",
}
# 单边以后某交易所某天有至少这么多成交行、成交量却全是偶数，判为双边复发（偶然发生的概率 2^-n）。
PARITY_MIN_ROWS = 12
# 交易单位不同：合约某月反查值与合约表相差超过这个倍数。
SPEC_FACTOR = 1.5
# 逐行成交额核对的容差（结算价是全天加权均价；中金所是最后一小时均价，另有偏离，按行登记）。
ROW_TOLERANCE = 0.05
PLATEAU_MIN_ROWS = 100
KNOWN_PATH = Path(__file__).with_name("futures_known_exceptions.json")


def load_known(path=KNOWN_PATH):
    """读已知例外清单；返回 dict(reasons=..., entries=[...])。"""
    if path is None or not Path(path).exists():
        return dict(reasons={}, entries=[], cause_unknown=[])
    return json.loads(Path(path).read_text())


class Register:
    """已知例外清单的查询：按检查名和键精确匹配，规则型条目（raw_range）按区间匹配。"""

    KEYS = {
        "partition_empty": ("dataset", "partition"),
        "calendar_difference": ("date",),
        "daily_on_closed_day": ("exchange", "date"),
        "coverage_gap": ("symbol", "missing"),
        "outside_listing": ("symbol", "count"),
        "code_without_contract": ("symbol",),
        "pre_settle": ("symbol", "date"),
        "daily_range": ("symbol", "date", "field"),
        "parity_day": ("exchange", "date"),
        "parity_odd": ("symbol", "date"),
        "spec_history": ("symbol",),
        "multiplier_row": ("symbol", "date"),
        "unresolved_code": ("dataset", "code", "date"),
        "zero_fee_day": ("exchange", "date"),
        "page_plateau": ("dataset", "rows"),
    }

    def __init__(self, known):
        self.reasons = known.get("reasons", {})
        self.exact = {}
        self.rules = []
        for k, entry in enumerate(known.get("entries", [])):
            check = entry["check"]
            if check == "raw_range":
                self.rules.append((k, entry))
                continue
            key = (check, *(entry[f] for f in self.KEYS[check]))
            self.exact[key] = k
        self.used = set()

    def reason(self, check, *key):
        k = self.exact.get((check, *key))
        if k is None:
            return None
        self.used.add(k)
        return k

    def rule(self, dataset, field, exchange, date, value):
        for k, entry in self.rules:
            if (
                entry["dataset"] == dataset
                and entry["field"] == field
                and entry["exchange"] == exchange
                and entry["start"] <= date <= entry["end"]
                and value in entry["values"]
            ):
                self.used.add(k)
                return k
        return None

    def cause_unknown(self, known):
        """用到的条目里，原因属于「未查明」的：按原因、按检查计数。"""
        unknown = set(known.get("cause_unknown", []))
        entries = known.get("entries", [])
        hits = [
            entries[k] for k in sorted(self.used) if entries[k]["reason"] in unknown
        ]
        return dict(
            count=len(hits),
            by_reason=dict(Counter(e["reason"] for e in hits)),
            by_check=dict(Counter(e["check"] for e in hits)),
        )

    def stale(self, known):
        entries = known.get("entries", [])
        return [entries[k] for k in range(len(entries)) if k not in self.used]


def _check(verdict, checked, failed, details=None, note="", explained=0):
    return dict(
        verdict=verdict,
        checked=int(checked),
        failed=int(failed),
        explained=int(explained),
        details=details if details is not None else [],
        note=note,
    )


def _judge(problems, register, check, key_fields, checked, note="", extra=None):
    """problems：dict 列表；能按 key_fields 在清单里找到的算已解释，其余失败。"""
    unexplained = [
        p
        for p in problems
        if register.reason(check, *(p[f] for f in key_fields)) is None
    ]
    verdict = "fail" if unexplained else ("explained" if problems else "pass")
    details = dict(unexplained=unexplained[:50], unexplained_count=len(unexplained))
    if extra:
        details.update(extra)
    return _check(
        verdict,
        checked,
        len(unexplained),
        details,
        note,
        explained=len(problems) - len(unexplained),
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


def _exchange(column="ts_code"):
    return (
        pl.col(column).str.extract(r"\.(\w+)$").replace_strict(SUFFIXES, default=None)
    )


# ---------------------------------------------------------------- 原始库：文件与分区


def check_manifest(store):
    """清单对文件：每个期货分区状态为完成、文件存在且哈希一致。清单之外的旧修订文件只列出。"""
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
        if not folder.exists():
            continue
        for path in folder.rglob("*.parquet"):
            if path in listed:
                continue
            # 同一分区刷新后留下的上一份修订：可读、文件名是内容哈希，只列出；其余都是不符。
            try:
                pl.read_parquet_schema(path)
                orphans.append(str(path.relative_to(store.root)))
            except Exception:  # noqa: BLE001 - 读不了就是坏文件
                bad.append(
                    dict(
                        key=str(path.relative_to(store.root)),
                        problem="清单之外的坏文件",
                    )
                )
    return _check(
        "fail" if bad else "pass",
        checked,
        len(bad),
        dict(
            problems=bad[:50],
            old_revisions=orphans[:50],
            old_revision_count=len(orphans),
        ),
        "old_revisions 是同一分区刷新后留下的上一份文件，可由 gc 清理，不算不符",
    )


def open_days(calendar):
    return {
        ex: sorted(
            g.filter(pl.col("is_open").cast(pl.Int64) == 1)["cal_date"].to_list()
        )
        for (ex,), g in calendar.group_by("exchange")
    }


def check_partitions(store, days_by_exchange, end, register):
    """日分区：覆盖起点后交易所开市日并集的每一天都有完成分区；空分区须在清单里；
    行数是页大小的整数倍，或多个分区恰好等于观测到的最大行数（真实单页上限可能比声明的小），判截断。"""
    union = sorted({d for days in days_by_exchange.values() for d in days if d <= end})
    out = {}
    for name in DAY_SETS:
        entries = {v["partition"]: v for v in _entries(store, name).values()}
        expected = [d for d in union if d >= STARTS[name]]
        missing = [
            d for d in expected if entries.get(d, {}).get("status") != "complete"
        ]
        empty = [
            dict(dataset=name, partition=d)
            for d in expected
            if entries.get(d, {}).get("status") == "complete"
            and entries[d].get("rows") == 0
        ]
        rows = [
            v.get("rows", 0) for v in entries.values() if v.get("status") == "complete"
        ]
        top = max(rows, default=0)
        at_top = sum(r == top for r in rows) if top else 0
        page = [
            p
            for p, v in entries.items()
            if v.get("rows") and v["rows"] % PAGE[name] == 0
        ]
        # 真实单页上限若比声明的小，截断的分区会齐刷刷停在同一个行数上：多个分区恰好等于观测到
        # 的最大行数时判可疑，除非清单说明这是自然的平台（如映射表每天列全同一批连续代码）。
        # 行数很少时并列最大是常态，不当截断（真实上限不会低到 100 行以内：单页 1000 起）。
        plateau = at_top > 1 and top >= PLATEAU_MIN_ROWS
        if plateau and register.reason("page_plateau", name, top) is None:
            page += [p for p, v in entries.items() if v.get("rows") == top]
        item = _judge(
            empty,
            register,
            "partition_empty",
            ("dataset", "partition"),
            len(expected),
            "空分区：交易所日历开市、源当天一行也没有，逐日在清单里",
            extra=dict(max_rows=top, partitions_at_max=at_top),
        )
        if missing or page:
            item["verdict"] = "fail"
            item["failed"] += len(missing) + len(page)
            item["details"]["missing"] = missing[:50]
            item["details"]["page_limit"] = page[:50]
        out[name] = item
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


# ---------------------------------------------------------------- 原始库：日历


def check_calendars(days_by_exchange, register):
    """各交易所日历互相对比（2000 年以后、各自开业以后）：只在部分交易所开市的日子须在清单里。"""
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
    return _judge(differ, register, "calendar_difference", ("date",), len(union))


def check_daily_on_closed(daily, days_by_exchange, register):
    """日线所在交易日必须是该交易所日历上的开市日。"""
    rows = (
        daily.filter(pl.col("ts_code").str.contains(CONTRACT.pattern))
        .select(exchange=_exchange(), date=pl.col("trade_date"))
        .unique()
    )
    sets = {ex: set(d) for ex, d in days_by_exchange.items()}
    problems = [
        dict(exchange=ex, date=day)
        for ex, day in rows.iter_rows()
        if day not in sets.get(ex, set())
    ]
    return _judge(
        problems, register, "daily_on_closed_day", ("exchange", "date"), len(rows)
    )


# ---------------------------------------------------------------- 原始库：覆盖与连续性


def parse_contracts(raw):
    """合约表：交易所、品种、每手单位（金融期货用 multiplier，商品用 per_unit；报价按 500 千克的 ×2）。"""
    return raw.with_columns(
        exchange=_exchange(),
        product=pl.col("ts_code").str.extract(r"^([A-Z]+)"),
        unit=pl.coalesce("multiplier", "per_unit").cast(pl.Float64)
        * pl.when(
            pl.col("quote_unit").str.contains("500千克")
            & (pl.col("trade_unit") == "吨")
        )
        .then(2.0)
        .otherwise(1.0),
    )


def check_contract_coverage(contracts, daily, days_by_exchange, last_day, register):
    """每个合约从上市到最后交易日（截到样本末尾）每个开市日都应有日线；缺口按合约和缺日数登记。
    上市窗口之外的行、合约表里没有的代码同样逐个登记。"""
    rows = daily.filter(pl.col("ts_code").str.contains(CONTRACT.pattern))
    have = {
        k: set(v)
        for k, v in rows.group_by("ts_code").agg(pl.col("trade_date")).iter_rows()
    }
    gaps, outside, checked = [], [], 0
    categories = Counter()
    for code, ex, first, last in contracts.select(
        "ts_code", "exchange", "list_date", "delist_date"
    ).iter_rows():
        if ex not in days_by_exchange or not first:
            continue
        checked += 1
        stop = min(last or last_day, last_day)
        expected = [d for d in days_by_exchange[ex] if first <= d <= stop]
        got = have.get(code, set())
        extra = sorted(d for d in got if d < first or (last and d > last))
        if extra:
            outside.append(dict(symbol=code, count=len(extra), first=extra[0]))
        missing = sorted(set(expected) - got)
        if not missing:
            continue
        if not got:
            reason = "整段没有日线"
        elif all(d < min(got) for d in missing):
            reason = "上市后最初几天没有日线"
        elif all(d > max(got) for d in missing):
            reason = "最后交易日前若干天没有日线"
        else:
            reason = "中间缺日"
        categories[reason] += 1
        gaps.append(
            dict(
                symbol=code, missing=len(missing), first_missing=missing[0], kind=reason
            )
        )
    unknown = [dict(symbol=s) for s in sorted(set(have) - set(contracts["ts_code"]))]
    a = _judge(
        gaps,
        register,
        "coverage_gap",
        ("symbol", "missing"),
        checked,
        extra=dict(categories=dict(categories)),
    )
    b = _judge(outside, register, "outside_listing", ("symbol", "count"), checked)
    c = _judge(unknown, register, "code_without_contract", ("symbol",), len(have))
    merged = dict(
        gaps=a["details"], outside_listing=b["details"], without_contract=c["details"]
    )
    verdicts = {a["verdict"], b["verdict"], c["verdict"]}
    verdict = (
        "fail"
        if "fail" in verdicts
        else ("explained" if "explained" in verdicts else "pass")
    )
    return _check(
        verdict,
        checked,
        a["failed"] + b["failed"] + c["failed"],
        merged,
        "合约的缺口、窗口外的行、未知代码都要在清单里",
        explained=a["explained"] + b["explained"] + c["explained"],
    )


def check_pre_settle(daily, days_by_exchange, register):
    """同一合约 pre_settle[t] = settle[t−1]。上一条记录不是交易所上一个开市日的（中间缺日，
    缺口已由 contract_coverage 登记）不在这里计；相邻开市日仍不符的逐条在清单里。"""
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
    problems = [
        dict(symbol=code, date=day, pre_settle=p, previous=q)
        for code, day, before, ex, p, q in bad.select(
            "ts_code", "trade_date", "previous_day", "ex", "pre_settle", "previous"
        ).iter_rows()
        if previous_open.get((ex, day)) == before
    ]
    return _judge(
        problems,
        register,
        "pre_settle",
        ("symbol", "date"),
        len(rows),
        extra=dict(gap_rows=len(bad) - len(problems)),
    )


# ---------------------------------------------------------------- 原始库：取值范围


PRICE_FIELDS = ("open", "high", "low", "close", "settle", "pre_settle", "pre_close")


def check_daily_ranges(daily, register):
    """日线：价格为正；low ≤ open、close ≤ high；成交量、成交额、持仓量非负。逐行逐字段登记。"""
    rows = daily.filter(pl.col("ts_code").str.contains(CONTRACT.pattern))
    # 开、高、低价写成 0 表示「没有连续竞价成交」，是源的写法（2026-04-01 起大商所、郑商所、
    # 广期所及中金所个别行；其中约 500 行成交量 > 0，是到期月份的非竞价成交），等同于空。
    idle = rows.filter(pl.col("open") == 0)
    rows = rows.with_columns(
        [
            pl.when(pl.col(f) == 0).then(None).otherwise(pl.col(f)).alias(f)
            for f in ("open", "high", "low")
        ]
    )
    tests = [(f, pl.col(f) <= 0) for f in PRICE_FIELDS if f in rows.columns]
    tests += [
        ("high_low", pl.col("high") < pl.col("low")),
        (
            "open_band",
            (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low")),
        ),
        (
            "close_band",
            (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low")),
        ),
    ]
    tests += [(f, pl.col(f) < 0) for f in ("vol", "amount", "oi") if f in rows.columns]
    problems = []
    for field, bad in tests:
        hit = rows.filter(bad.fill_null(False)).select("ts_code", "trade_date")
        problems += [dict(symbol=c, date=d, field=field) for c, d in hit.iter_rows()]
    return _judge(
        problems,
        register,
        "daily_range",
        ("symbol", "date", "field"),
        len(rows),
        extra=dict(
            zero_open_rows=len(idle),
            zero_open_with_volume=len(idle.filter(pl.col("vol") > 0)),
        ),
    )


SETTLE_RANGES = (
    # 字段, 合法下界(含), 上界(不含), 换算：中金所保证金为百分数
    ("long_margin_rate", 0.0, 1.0),
    ("short_margin_rate", 0.0, 1.0),
    ("b_hedging_margin_rate", 0.0, 1.0),
    ("s_hedging_margin_rate", 0.0, 1.0),
    ("trading_fee_rate", 0.0, None),
    ("trading_fee", 0.0, None),
    ("offset_today_fee", 0.0, None),
    ("delivery_fee", 0.0, None),
)


def check_settle_ranges(settle, register):
    """结算参数原始值：保证金率（中金所百分数先 ÷100）落在 (0, 1)；各项费用非负。
    源里成片的异常（郑商所平今费 -1/-2）按「数据集、字段、交易所、日期区间、取值」登记为规则。"""
    frame = settle.filter(
        pl.col("ts_code").str.contains(CONTRACT.pattern)
    ).with_columns(ex=_exchange())
    problems = Counter()
    unexplained = []
    for field, low, high in SETTLE_RANGES:
        if field not in frame.columns:
            continue
        value = (
            pl.when(pl.col("ex") == "CFFEX")
            .then(pl.col(field) / 100)
            .otherwise(pl.col(field))
            if "margin" in field
            else pl.col(field)
        )
        bad = value < low if "fee" in field else (value <= low) | (value >= high)
        hit = frame.filter(bad.fill_null(False)).select(
            "ex", "trade_date", "ts_code", pl.col(field).alias("value")
        )
        for ex, day, code, raw in hit.iter_rows():
            if register.rule("futures_settle", field, ex, day, raw) is None:
                unexplained.append(
                    dict(field=field, exchange=ex, date=day, code=code, value=raw)
                )
            else:
                problems[(field, ex)] += 1
    explained = sum(problems.values())
    verdict = "fail" if unexplained else ("explained" if explained else "pass")
    return _check(
        verdict,
        len(frame),
        len(unexplained),
        dict(
            unexplained=unexplained[:50],
            unexplained_count=len(unexplained),
            explained_by_field=[
                dict(field=f, exchange=e, rows=n)
                for (f, e), n in sorted(problems.items())
            ],
        ),
        explained=explained,
    )


def check_zero_fee_days(settle, register):
    """某交易所某天有费用的行里九成以上开仓手续费两列都是 0 —— 源错误，逐日在清单里。"""
    frame = settle.filter(
        pl.col("ts_code").str.contains(CONTRACT.pattern)
    ).with_columns(ex=_exchange())
    present = (
        pl.col("trading_fee_rate").is_not_null() | pl.col("trading_fee").is_not_null()
    )
    zero = (
        present
        & (pl.col("trading_fee_rate").fill_null(0) == 0)
        & (pl.col("trading_fee").fill_null(0) == 0)
    )
    days = (
        frame.group_by("ex", "trade_date")
        .agg(share=zero.sum() / present.sum())
        .filter(pl.col("share").is_not_nan() & (pl.col("share") >= 0.9))
        .sort("ex", "trade_date")
    )
    problems = [
        dict(exchange=e, date=d) for e, d in days.select("ex", "trade_date").iter_rows()
    ]
    return _judge(
        problems,
        register,
        "zero_fee_day",
        ("exchange", "date"),
        frame.select("ex", "trade_date").n_unique(),
    )


# ---------------------------------------------------------------- 原始库：单位


def check_parity(daily, register):
    """单边/双边：切换日前成交量、持仓量必须全为偶数；切换后某交易所某天成交行 ≥ PARITY_MIN_ROWS
    且成交量全为偶数，判为双边复发。"""
    rows = (
        daily.filter(
            pl.col("ts_code").str.contains(CONTRACT.pattern) & (pl.col("vol") > 0)
        )
        .with_columns(ex=_exchange())
        .filter(pl.col("ex").is_in(list(SINGLE_SIDED_FROM)))
    )
    switch = pl.col("ex").replace_strict(SINGLE_SIDED_FROM, default=None)
    before = rows.filter(pl.col("trade_date") < switch)
    odd = before.filter((pl.col("vol") % 2 != 0) | (pl.col("oi").fill_null(0) % 2 != 0))
    a = _judge(
        [
            dict(symbol=c, date=d)
            for c, d in odd.select("ts_code", "trade_date").iter_rows()
        ],
        register,
        "parity_odd",
        ("symbol", "date"),
        len(before),
    )
    after = (
        rows.filter(pl.col("trade_date") >= switch)
        .group_by("ex", "trade_date")
        .agg(n=pl.len(), even=(pl.col("vol") % 2 == 0).all())
        .filter((pl.col("n") >= PARITY_MIN_ROWS) & pl.col("even"))
    )
    b = _judge(
        [
            dict(exchange=e, date=d)
            for e, d in after.select("ex", "trade_date").iter_rows()
        ],
        register,
        "parity_day",
        ("exchange", "date"),
        len(rows) - len(before),
    )
    verdicts = {a["verdict"], b["verdict"]}
    verdict = (
        "fail"
        if "fail" in verdicts
        else ("explained" if "explained" in verdicts else "pass")
    )
    return _check(
        verdict,
        len(rows),
        a["failed"] + b["failed"],
        dict(before_switch=a["details"], after_switch=b["details"]),
        explained=a["explained"] + b["explained"],
    )


def check_spec_history(daily, contracts, register):
    """合约表的每手单位与成交额反查对照：某个月中位比值偏离超过 SPEC_FACTOR 倍的合约（交易单位
    历史上不同）逐个在清单里，面板对它们按日取乘数。合约表被改错（如乘数 ×10）会在这里失败。"""
    rows = (
        daily.join(
            contracts.select("ts_code", "unit", "list_date", "delist_date"),
            on="ts_code",
        )
        .filter(
            (pl.col("vol") > 0)
            & (pl.col("amount") > 0)
            & (pl.col("settle") > 0)
            & (pl.col("trade_date") >= pl.col("list_date"))
            & (pl.col("trade_date") <= pl.col("delist_date"))
        )
        .with_columns(
            ratio=pl.col("amount")
            * 10000
            / (pl.col("vol") * pl.col("settle") * pl.col("unit"))
        )
    )
    months = rows.group_by(
        "ts_code", pl.col("trade_date").str.slice(0, 6).alias("month")
    ).agg(ratio=pl.col("ratio").median())
    off = (
        months.filter(
            (pl.col("ratio") >= SPEC_FACTOR) | (pl.col("ratio") <= 1 / SPEC_FACTOR)
        )
        .group_by("ts_code")
        .agg(months=pl.len(), ratio=pl.col("ratio").median())
        .sort("ts_code")
    )
    problems = [
        dict(symbol=c, months=m, ratio=round(r, 3)) for c, m, r in off.iter_rows()
    ]
    return _judge(
        problems, register, "spec_history", ("symbol",), rows["ts_code"].n_unique()
    )


# ---------------------------------------------------------------- 统一字段（面板）


def _panel(store):
    current = store.root / "futures_panel/current.json"
    if not current.exists():
        return None
    path = current.parent / json.loads(current.read_text())["version"]
    return dict(
        path=path,
        meta=json.loads((path / "metadata.json").read_text()),
        **{
            name: pl.read_parquet(path / f"{name}.parquet")
            for name in ("contracts", "daily", "params", "mapping", "multipliers")
            if (path / f"{name}.parquet").exists()
        },
    )


def check_multiplier_rows(panel, register):
    """统一日线逐行：成交额 ≈ 成交量 × 结算价 × 当天乘数（按 multipliers 分段取）。
    偏离超过 ROW_TOLERANCE 的行逐条在清单里。乘数被改错、按日取值错位都会在这里失败。"""
    daily = panel["daily"].filter(
        (pl.col("volume") > 0) & (pl.col("amount") > 0) & (pl.col("settlement") > 0)
    )
    segments = panel["multipliers"].sort("start_date").rename({"start_date": "date"})
    joined = (
        daily.sort("date")
        .join_asof(
            segments.select("symbol", "date", "multiplier"),
            on="date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .join(
            segments.group_by("symbol").agg(
                head=pl.col("multiplier").sort_by("date").first()
            ),
            on="symbol",
            how="left",
        )
        .with_columns(multiplier=pl.coalesce("multiplier", "head"))
        .with_columns(
            ratio=pl.col("amount")
            / (pl.col("volume") * pl.col("settlement") * pl.col("multiplier"))
        )
    )
    bad = joined.filter(
        ((pl.col("ratio") - 1).abs() > ROW_TOLERANCE) | pl.col("ratio").is_null()
    )
    problems = [
        dict(symbol=s, date=d, ratio=None if r is None else round(r, 4))
        for s, d, r in bad.select("symbol", "date", "ratio")
        .sort("symbol", "date")
        .iter_rows()
    ]
    by_exchange = (
        bad.group_by(pl.col("symbol").str.extract(r"\.(\w+)$").alias("exchange"))
        .len()
        .sort("exchange")
        .to_dicts()
    )
    return _judge(
        problems,
        register,
        "multiplier_row",
        ("symbol", "date"),
        len(joined),
        extra=dict(by_exchange=by_exchange),
    )


def check_multiplier_sources(panel):
    """乘数来源统计；manual 必须写依据；以「未核实」开头的依据与无法核对的合约单独列出。"""
    table = panel["multipliers"]
    counts = table.group_by("source").len().sort("source").to_dicts()
    manual = table.filter(pl.col("source") == "manual")
    missing_basis = manual.filter(pl.col("basis").is_null() | (pl.col("basis") == ""))
    unverified = (
        table.filter(pl.col("source") == "contract_table_unverified")
        .group_by(
            pl.col("symbol").str.extract(r"^([A-Z_]+)\d").alias("product"),
            pl.col("symbol").str.extract(r"\.(\w+)$").alias("exchange"),
        )
        .agg(contracts=pl.len(), multiplier=pl.col("multiplier").unique().sort())
        .sort("exchange", "product")
    )
    changes = table.group_by("symbol").len().filter(pl.col("len") > 1)
    details = dict(
        sources=counts,
        manual_rules=manual.group_by("basis")
        .agg(contracts=pl.len(), multiplier=pl.col("multiplier").unique())
        .to_dicts(),
        unverifiable_by_product=unverified.to_dicts(),
        contracts_with_mid_life_change=table.filter(
            pl.col("symbol").is_in(changes["symbol"])
        ).to_dicts(),
    )
    return _check(
        "fail" if len(missing_basis) else "info",
        len(table),
        len(missing_basis),
        details,
    )


def _listed(contracts):
    return contracts.select("symbol", "listing_date", "last_trade_date")


def check_unified_refs(panel):
    """统一字段的引用完整性（不允许例外）：参数表、映射表的合约都在合约表里且当天在上市期内；
    乘数表每个合约都有、首段从上市日开始。"""
    contracts = _listed(panel["contracts"])
    problems = []
    for name in ("params", "mapping"):
        frame = panel.get(name)
        if frame is None:
            continue
        joined = frame.select("date", "symbol").join(contracts, on="symbol", how="left")
        bad = joined.filter(
            pl.col("listing_date").is_null()
            | (pl.col("date") < pl.col("listing_date"))
            | (pl.col("date") > pl.col("last_trade_date"))
        )
        problems += [
            dict(table=name, symbol=s, date=d)
            for d, s in bad.select("date", "symbol").head(50).iter_rows()
        ]
        if len(bad) > 50:
            problems.append(dict(table=name, more=len(bad) - 50))
    multipliers = panel.get("multipliers")
    if multipliers is not None:
        heads = multipliers.group_by("symbol").agg(start=pl.col("start_date").min())
        joined = contracts.join(heads, on="symbol", how="left")
        bad = joined.filter(
            pl.col("start").is_null() | (pl.col("start") != pl.col("listing_date"))
        )
        problems += [
            dict(table="multipliers", symbol=s) for s in bad["symbol"].head(50)
        ]
    checked = sum(len(panel[n]) for n in ("params", "mapping") if n in panel)
    return _check("fail" if problems else "pass", checked, len(problems), problems)


def check_unified_ranges(panel):
    """统一参数（不允许例外）：保证金率在 (0, 1)，各项费用非负，没有整个交易所整天手续费全为 0。"""
    params = panel["params"]
    problems = []
    for field in (
        "margin_long",
        "margin_short",
        "hedge_margin_long",
        "hedge_margin_short",
    ):
        bad = params.filter(
            ((pl.col(field) <= 0) | (pl.col(field) >= 1)).fill_null(False)
        )
        if len(bad):
            problems.append(
                dict(
                    field=field,
                    rows=len(bad),
                    example=bad.select("date", "symbol", field).head(3).to_dicts(),
                )
            )
    fees = [c for c in params.columns if c.startswith(("fee_", "delivery_fee"))]
    for field in fees:
        bad = params.filter((pl.col(field) < 0).fill_null(False))
        if len(bad):
            problems.append(
                dict(
                    field=field,
                    rows=len(bad),
                    example=bad.select("date", "symbol", field).head(3).to_dicts(),
                )
            )
    total = sum(
        pl.col(f).fill_null(0) for f in ("fee_open_rate", "fee_open_per_contract")
    )
    zero = (
        params.with_columns(
            ex=pl.col("symbol").str.extract(r"\.(\w+)$"),
            total=total,
            known=pl.any_horizontal(
                pl.col("fee_open_rate", "fee_open_per_contract").is_not_null()
            ),
        )
        .filter(pl.col("known"))
        .group_by("ex", "date")
        .agg(all_zero=(pl.col("total") == 0).all())
        .filter("all_zero")
    )
    if len(zero):
        problems.append(
            dict(
                field="fee_open",
                zero_days=zero.select("ex", "date").head(20).to_dicts(),
            )
        )
    return _check(
        "fail" if problems else "pass",
        len(params),
        sum(p.get("rows", 1) for p in problems),
        problems,
    )


def check_code_resolution(store, panel, register):
    """源代码 → 合约：结算参数与映射的原始行里，字面代码不是当天上市合约的，按本模块自己的规则
    判能否找回（郑商所同品种同年份末位同月份、或同品种同年月换交易所后缀，当天上市且唯一）；
    找不回的逐条在清单里。另核对统一参数行数 = 原始行 − 连续代码行 − 找不回的行 − 同日重复行。"""
    contracts = _read(store, "futures_contracts").select(
        "ts_code",
        "list_date",
        "delist_date",
        product=pl.col("ts_code").str.extract(r"^([A-Z]+)"),
        digits=pl.col("ts_code").str.extract(r"(\d{4})F?\."),
        exchange=_exchange(),
    )
    out, unresolved_total = {}, []
    for dataset, column in (
        ("futures_settle", "ts_code"),
        ("futures_mapping", "mapping_ts_code"),
    ):
        raw = _read(store, dataset)
        if raw.is_empty():
            continue
        rows = (
            raw.filter(pl.col(column).str.contains(CONTRACT.pattern))
            .with_row_index("_row")
            .select(
                "_row",
                code=pl.col(column),
                date=pl.col("trade_date"),
                product=pl.col(column).str.extract(r"^([A-Z]+)"),
                digits=pl.col(column).str.extract(r"(\d{4})F?\."),
                exchange=_exchange(column),
            )
            .filter(~pl.col("product").is_in(["SCTAS"]))
        )
        literal = rows.join(
            contracts.select("ts_code", "list_date", "delist_date"),
            left_on="code",
            right_on="ts_code",
        ).filter(
            (pl.col("date") >= pl.col("list_date"))
            & (pl.col("date") <= pl.col("delist_date"))
        )
        rest = rows.join(literal.select("_row"), on="_row", how="anti")
        cand = rest.join(
            contracts.rename(
                {"ts_code": "candidate", "exchange": "cex", "digits": "cdigits"}
            ),
            on="product",
        ).filter(
            (pl.col("date") >= pl.col("list_date"))
            & (pl.col("date") <= pl.col("delist_date"))
            & (
                (
                    (pl.col("exchange") == "CZCE")
                    & (pl.col("cex") == "CZCE")
                    & (pl.col("cdigits").str.slice(1) == pl.col("digits").str.slice(1))
                )
                | (
                    (pl.col("cex") != pl.col("exchange"))
                    & (pl.col("cdigits") == pl.col("digits"))
                )
            )
        )
        unique = cand.group_by("_row").agg(n=pl.len()).filter(pl.col("n") == 1)
        lost = rest.join(unique.select("_row"), on="_row", how="anti")
        problems = [
            dict(dataset=dataset, code=c, date=d)
            for c, d in lost.select("code", "date").sort("date", "code").iter_rows()
        ]
        item = _judge(
            problems,
            register,
            "unresolved_code",
            ("dataset", "code", "date"),
            len(rows),
            extra=dict(literal=len(literal), recovered=len(unique), lost=len(lost)),
        )
        out[dataset] = item
        unresolved_total += problems
        if dataset == "futures_settle" and panel is not None:
            recovered = rest.join(unique.select("_row"), on="_row").with_columns(
                _r=pl.lit(1)
            )
            resolved = pl.concat(
                [literal.select("_row", "date"), recovered.select("_row", "date")]
            )
            # 同一天同一合约被两行指到（错后缀与正确后缀并存）只算一行：按找回的合约去重。
            mapped = pl.concat(
                [
                    literal.select("date", symbol=pl.col("code")),
                    cand.join(unique.select("_row"), on="_row").select(
                        "date", symbol=pl.col("candidate")
                    ),
                ]
            ).unique()
            expected = len(mapped)
            got = len(panel["params"])
            item["details"]["unified_rows"] = dict(
                expected=expected, got=got, resolved_rows=len(resolved)
            )
            if expected != got:
                item["verdict"] = "fail"
                item["failed"] += abs(expected - got)
    verdicts = {v["verdict"] for v in out.values()}
    verdict = (
        "fail"
        if "fail" in verdicts
        else ("explained" if "explained" in verdicts else "pass")
    )
    return _check(
        verdict,
        sum(v["checked"] for v in out.values()),
        sum(v["failed"] for v in out.values()),
        {k: v["details"] for k, v in out.items()},
        explained=sum(v["explained"] for v in out.values()),
    )


def check_source_continuous(store, daily):
    """来源的连续日线（``CU.SHF``、``CUL.SHF``）与同日映射指向的合约日线（结算价、收盘价）对照。
    统计信息：两者都来自同一来源、互不约束，不符说明来源的连续价格不是照映射取的（多为成交
    稀少的品种），面板的连续合约只用映射和合约日线，不用来源的连续价格。"""
    mapping = _read(store, "futures_mapping")
    if mapping.is_empty():
        return _check("info", 0, 0, [], "没有映射表")
    continuous = daily.filter(pl.col("ts_code").str.contains(r"^[A-Z]+\.\w+$")).select(
        continuous=pl.col("ts_code"),
        date=pl.col("trade_date"),
        c_settle="settle",
        c_close="close",
    )
    target = mapping.select(
        continuous=pl.col("ts_code"),
        date=pl.col("trade_date"),
        symbol=pl.col("mapping_ts_code"),
    )
    contracts = daily.select(
        symbol=pl.col("ts_code"),
        date=pl.col("trade_date"),
        settle="settle",
        close="close",
    )
    joined = continuous.join(target, on=["continuous", "date"]).join(
        contracts, on=["symbol", "date"], how="left"
    )
    bad = joined.filter(
        (pl.col("c_settle") != pl.col("settle")).fill_null(False)
        | (pl.col("c_close") != pl.col("close")).fill_null(False)
        | (pl.col("settle").is_null() & pl.col("c_settle").is_not_null())
    )
    summary = (
        bad.group_by("continuous")
        .agg(rows=pl.len(), first=pl.col("date").min(), last=pl.col("date").max())
        .sort("rows", descending=True)
    )
    return _check(
        "info",
        len(joined),
        0,
        dict(
            mismatched_rows=len(bad),
            by_continuous=summary.to_dicts(),
            examples=bad.sort("date").head(20).to_dicts(),
        ),
    )


def check_oi_only(daily):
    """只有持仓量、没有开盘价/结算价/成交量的行：必须是零成交（面板 has_quote 为假）。"""
    rows = daily.filter(
        pl.col("ts_code").str.contains(CONTRACT.pattern)
        & pl.col("open").is_null()
        & pl.col("settle").is_null()
        & pl.col("oi").is_not_null()
    )
    traded = rows.filter(pl.col("vol").fill_null(0) > 0)
    return _check(
        "fail" if len(traded) else "info",
        len(rows),
        len(traded),
        dict(
            rows=len(rows),
            volume_values=rows["vol"].unique().to_list(),
            by_exchange=rows.group_by(_exchange().alias("exchange"))
            .len()
            .sort("exchange")
            .to_dicts(),
            traded=traded.select("ts_code", "trade_date").head(20).to_dicts(),
        ),
    )


def check_settle_params(settle):
    """结算参数表：各交易所覆盖起点与各字段逐年缺失率（统计）。"""
    frame = settle.with_columns(
        exchange=_exchange(), year=pl.col("trade_date").str.slice(0, 4)
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
    """最后交易日的结算价是否落在最小变动价位网格上（统计；中金所股指是交割结算价）。"""
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


# ---------------------------------------------------------------- 汇总


def verify_futures(store, *, write=True, known=None):
    """全部校验；``known`` 为已知例外清单（默认读包内 JSON，传 ``{}`` 表示没有例外）。
    结果写 ``reports/futures_verify.json`` 与 ``.md``，返回报告 dict。"""
    started = time.perf_counter()
    known = load_known() if known is None else known
    register = Register(known)
    calendar = _read(store, "futures_calendar")
    days = open_days(calendar)
    # SCTAS（原油 TAS 指令）价格是相对结算价的价差、可以为负，不是可持有的合约，整体不校验。
    tas = ~pl.col("ts_code").str.starts_with("SCTAS")
    contracts = parse_contracts(_read(store, "futures_contracts")).filter(tas)
    daily = _read(store, "futures_daily").filter(tas)
    settle = _read(store, "futures_settle")
    settle = settle.filter(tas) if len(settle) else settle
    last_day = daily["trade_date"].max()
    checks = {"manifest_files": check_manifest(store)}
    for name, item in check_partitions(store, days, last_day, register).items():
        checks[f"partitions_{name}"] = item
    checks["unique_keys"] = check_unique(store)
    checks["calendars"] = check_calendars(days, register)
    checks["daily_on_closed_day"] = check_daily_on_closed(daily, days, register)
    checks["contract_coverage"] = check_contract_coverage(
        contracts, daily, days, last_day, register
    )
    checks["pre_settle_continuity"] = check_pre_settle(daily, days, register)
    checks["daily_ranges"] = check_daily_ranges(daily, register)
    checks["settle_ranges"] = check_settle_ranges(settle, register)
    checks["zero_fee_days"] = check_zero_fee_days(settle, register)
    checks["sidedness_parity"] = check_parity(daily, register)
    checks["contract_table_units"] = check_spec_history(daily, contracts, register)
    panel = _panel(store)
    checks["code_resolution"] = check_code_resolution(store, panel, register)
    if panel is None:
        checks["panel"] = _check(
            "fail", 0, 1, [], "期货面板尚未构建：先 build-futures 再校验"
        )
    else:
        checks["multiplier_rows"] = check_multiplier_rows(panel, register)
        checks["multiplier_sources"] = check_multiplier_sources(panel)
        checks["unified_references"] = check_unified_refs(panel)
        checks["unified_ranges"] = check_unified_ranges(panel)
    checks["source_continuous"] = check_source_continuous(store, daily)
    checks["oi_only_rows"] = check_oi_only(daily)
    checks["settle_params"] = check_settle_params(settle)
    checks["expiry_settlement"] = check_expiry_settle(daily, contracts)
    verdicts = Counter(item["verdict"] for item in checks.values())
    report = dict(
        verdict="fail" if verdicts["fail"] else "pass",
        verdicts=dict(verdicts),
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        panel_version=None if panel is None else panel["meta"]["version"],
        raw_version=store.futures_version,
        yearly_rows=yearly_rows(store),
        last_day=last_day,
        known_exceptions=dict(
            entries=len(known.get("entries", [])),
            used=len(register.used),
            by_check=dict(Counter(e["check"] for e in known.get("entries", []))),
            cause_unknown=register.cause_unknown(known),
        ),
        stale_exceptions=register.stale(known)[:100],
        checks=checks,
    )
    if write:
        folder = store.root / "reports"
        atomic_json(
            folder / "futures_verify.json", json.loads(json.dumps(report, default=str))
        )
        lines = [
            "# 期货原始库与面板校验",
            "",
            f"总结论：**{report['verdict']}**；生成 {report['generated_at']}，最后交易日 {last_day}，"
            f"已知例外 {report['known_exceptions']['used']} / {report['known_exceptions']['entries']} 条用到，"
            f"其中原因未查明 {report['known_exceptions']['cause_unknown']['count']} 条",
            "",
            "| 检查 | 结论 | 检查数 | 未解释 | 已解释 |",
            "|---|---|---|---|---|",
        ]
        for name, item in checks.items():
            lines.append(
                f"| {name} | {item['verdict']} | {item['checked']} | {item['failed']} | {item['explained']} |"
            )
        (folder / "futures_verify.md").write_text("\n".join(lines) + "\n")
    return report


__all__ = ["Register", "load_known", "verify_futures"]
