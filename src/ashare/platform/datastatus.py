"""数据页：本地数据层的只读状态（数据集、字段覆盖、原始表、更新记录、校验报告）。

只读：不打开写句柄、不联网、不触发同步。逐字段覆盖统计要读完整面板（约 6 GB），
每个面板版本只算一次：后台线程计算，结果写到平台目录 ``cache/datastatus-<版本>.json``，
重启后直接读。清单、报告等小文件按文件签名（mtime、大小）缓存。
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import APIRouter, HTTPException

LOG = logging.getLogger("ashare.platform")
SHANGHAI = ZoneInfo("Asia/Shanghai")
# 统计口径变了就加一，旧缓存自动失效。
STATS_FORMAT = 1
# 表格里的覆盖迷你图点数；详情图用逐月数据。
SPARK_POINTS = 60
HISTOGRAM_BINS = 40

CATEGORIES = (
    (
        "行情",
        (
            "open",
            "high",
            "low",
            "close",
            "previous_close",
            "volume",
            "amount",
            "vwap",
            "turnover",
            "adj_factor",
        ),
    ),
    (
        "股本市值",
        (
            "total_market_cap",
            "float_market_cap",
            "total_shares",
            "float_shares",
            "free_shares",
        ),
    ),
    (
        "估值",
        ("pe", "pe_ttm", "pb", "ps", "ps_ttm", "dividend_yield", "dividend_yield_ttm"),
    ),
    (
        "交易约束",
        ("upper_limit", "lower_limit", "tradable", "can_buy_open", "can_sell_open"),
    ),
    (
        "状态",
        (
            "is_listed",
            "listing_age",
            "is_st",
            "is_suspended",
            "industry",
            "industry_source",
            "has_quote",
        ),
    ),
    ("公司行为", ("split", "cash_dividend", "delist", "action_source")),
)
# 这些整数字段以 0 表示缺失（行业编码、行业来源）；其余整数、布尔字段没有缺失值。
ZERO_MISSING = frozenset({"industry", "industry_source"})
# 编码或枚举，画分布没有意义。
CATEGORICAL = frozenset({"industry", "industry_source", "action_source"})

# 数据页上的中文名：补研究台补全表里没有的字段，指数成分用指数的通行名称。
LABELS = {
    "can_buy_open": "开盘可买",
    "can_sell_open": "开盘可卖",
    "previous_close": "前收盘价",
    "industry": "申万一级行业",
    **{
        f"member_csi{c}": f"{n} 成分"
        for c, n in (
            (300, "沪深 300"),
            (500, "中证 500"),
            (1000, "中证 1000"),
            (2000, "中证 2000"),
        )
    },
    **{
        f"weight_csi{c}": f"{n} 权重"
        for c, n in (
            (300, "沪深 300"),
            (500, "中证 500"),
            (1000, "中证 1000"),
            (2000, "中证 2000"),
        )
    },
    "member_csi_all": "中证全指成分",
    "weight_csi_all": "中证全指权重",
}

GROUPS = (
    ("stock", "股票日频面板"),
    ("futures", "境内期货"),
    ("fund", "场内基金"),
    ("cb", "可转债"),
    ("index", "指数行情"),
)


def category(name):
    for label, names in CATEGORIES:
        if name in names:
            return label
    if name.startswith(("member_", "weight_")):
        return "指数成分"
    return "其他"


def _order(name):
    for index, (_, names) in enumerate(CATEGORIES):
        if name in names:
            return (index, names.index(name), name)
    if name.startswith(("member_", "weight_")):
        # 同一指数的成分与权重相邻，按指数规模排列。
        code = name.split("_", 1)[1]
        rank = ("csi300", "csi500", "csi1000", "csi2000", "csi_all")
        position = rank.index(code) if code in rank else len(rank)
        return (len(CATEGORIES), position, name.startswith("weight_"), name)
    return (len(CATEGORIES) + 1, 0, name)


def group_of(dataset):
    if dataset.startswith("index_"):
        return "index"
    if dataset.startswith("fut"):
        return "futures"
    if dataset.startswith("fund_"):
        return "fund"
    if dataset.startswith("cb_"):
        return "cb"
    return "stock"


def _day(value):
    value = str(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}" if len(value) >= 8 else value


def _finite(value, digits=6):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _local(stamp):
    try:
        moment = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        return None
    return moment.astimezone(SHANGHAI)


def _size(folder):
    total = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _size(entry.path)
    except OSError:
        return total
    return total


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def field_stats(array, name, dates, listed):
    """一个面板字段的覆盖与最新分布。``listed`` 是同形状的上市掩码。

    覆盖率 = 当天在上市股票中有值的比例；浮点以有限值为有值，``ZERO_MISSING`` 的整数以非零为有值，
    其余布尔、整数字段没有缺失值。分布只取最新一天上市且有值的股票。
    """
    rows = array.shape[0]
    counts = np.zeros(rows, dtype=np.int64)
    kind = array.dtype.kind
    latest = None
    for start in range(0, rows, 256):
        block = np.asarray(array[start : start + 256])
        mask = listed[start : start + 256]
        if kind == "f":
            present = np.isfinite(block) & mask
        elif name in ZERO_MISSING:
            present = (block != 0) & mask
        else:
            present = mask
        counts[start : start + len(block)] = present.sum(axis=1)
        if start + len(block) == rows:
            latest = (block[-1], present[-1])
    universe = listed.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(universe > 0, counts / np.maximum(universe, 1), np.nan)
    first = np.flatnonzero(counts > 0)
    months = np.array([str(d)[:6] for d in dates])
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    monthly = []
    for begin, end in zip(starts, np.r_[starts[1:], rows], strict=True):
        part = coverage[begin:end]
        part = part[np.isfinite(part)]
        monthly.append(
            [str(months[begin]), _finite(part.mean(), 4) if len(part) else None]
        )
    total = universe.sum()
    result = dict(
        name=name,
        dtype=str(array.dtype),
        start=_day(dates[first[0]]) if len(first) else None,
        latest=_finite(coverage[-1], 4) if rows else None,
        overall=_finite(counts.sum() / total, 4) if total else None,
        latest_count=int(counts[-1]) if rows else 0,
        monthly=monthly,
        distribution=None,
        true_share=None,
    )
    if latest is None:
        return result
    values, present = latest
    if kind == "b":
        listed_today = int(listed[-1].sum())
        result["true_share"] = (
            _finite(values[listed[-1]].sum() / listed_today, 4)
            if listed_today
            else None
        )
    elif name not in CATEGORICAL:
        values = np.asarray(values[present], dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) >= 2:
            low, high = np.percentile(values, [1, 99])
            if high <= low:
                low, high = float(values.min()), float(values.max())
            # 两端各 1% 的极值并入边上的格子，而不是丢掉：少量股票时不会整张图为空。
            clipped = np.clip(values, low, high)
            bins = np.histogram(clipped, bins=HISTOGRAM_BINS, range=(low, high))[0]
            result["distribution"] = dict(
                low=_finite(low, 8),
                high=_finite(high, 8),
                counts=bins.tolist(),
                quantiles=[
                    _finite(q, 8) for q in np.percentile(values, [5, 25, 50, 75, 95])
                ],
                n=int(len(values)),
            )
    return result


def _spark(monthly, points=SPARK_POINTS):
    values = [v for _, v in monthly]
    if len(values) <= points:
        return values
    edges = np.linspace(0, len(values), points + 1).astype(int)
    output = []
    for begin, end in zip(edges[:-1], edges[1:], strict=True):
        part = [v for v in values[begin:end] if v is not None]
        output.append(round(sum(part) / len(part), 4) if part else None)
    return output


class DataStatus:
    # 概览最多每隔这么多秒看一次文件签名（测试里可设为 0）。
    CHECK_INTERVAL = 2.0

    def __init__(self, root=None, cache_dir=None, platform_store=None):
        self._root_option = root
        self._root = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.platform_store = platform_store
        self.lock = threading.RLock()
        self._overview = (None, None)
        self._checked = 0.0
        self._signature = None
        self._stats = {}
        self._computing = set()
        self._errors = {}
        self._references = (None, None)
        self._references_checked = 0.0
        self._parsed = {}

    @property
    def root(self):
        # 延迟解析：创建应用时还不读数据目录（测试与未配置数据目录时都能启动）。
        if self._root is None:
            from ashare.data.store import data_root

            self._root = data_root(self._root_option) / "cn"
        return self._root

    # ---- 签名与小文件 -------------------------------------------------

    def signature(self):
        names = (
            "manifest.sqlite3",
            "manifest.sqlite3-wal",
            "manifest.json",
            "panel/current.json",
            "futures_panel/current.json",
            "fund_panel/fund/current.json",
            "fund_panel/cb/current.json",
            "reports/verify.json",
            "reports/futures_verify.json",
            "reports/funds_verify.json",
        )
        state = []
        for name in names:
            try:
                stat = (self.root / name).stat()
                state.append((name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                state.append((name, None, None))
        return tuple(state)

    def _current(self, folder):
        current = _read_json(self.root / folder / "current.json")
        if not current or "version" not in current:
            return None, None
        path = self.root / folder / current["version"]
        if not (path / "metadata.json").exists():
            return None, None
        return current["version"], path

    def panel_version(self):
        return self._current("panel")[0]

    # ---- 概览 ---------------------------------------------------------

    def overview(self):
        with self.lock:
            now = time.monotonic()
            cached, signature = self._overview
            if cached is not None and now - self._checked < self.CHECK_INTERVAL:
                return cached
            self._checked = now
            current = self.signature()
            if cached is not None and current == signature:
                return cached
            started = time.perf_counter()
            result = self._build_overview()
            self._overview = (result, current)
            LOG.info("数据概览重建 elapsed=%.3fs", time.perf_counter() - started)
            return result

    def _manifest(self):
        if not (
            (self.root / "manifest.sqlite3").exists()
            or (self.root / "manifest.json").exists()
        ):
            return {}
        from ashare.data.store import Store

        return Store(self.root.parent, "cn", readonly=True).manifest()

    def _build_overview(self):
        manifest = self._manifest()
        tables = {}
        history = {}
        index_months = {}
        for entry in manifest.values():
            name = entry.get("dataset")
            if not name:
                continue
            row = tables.setdefault(
                name,
                dict(
                    name=name,
                    group=group_of(name),
                    partitions=0,
                    rows=0,
                    failed=0,
                    first=None,
                    last=None,
                    fetched=None,
                ),
            )
            row["partitions"] += 1
            row["rows"] += int(entry.get("rows") or 0)
            row["failed"] += entry.get("status") != "complete"
            partition = str(entry.get("partition", ""))
            if name == "index_daily" and "_" in partition:
                code, _, month = partition.rpartition("_")
                if entry.get("status") == "complete" and entry.get("rows"):
                    index_months.setdefault(code, []).append((month, entry["file"]))
                partition = month
            if row["first"] is None or partition < row["first"]:
                row["first"] = partition
            if row["last"] is None or partition > row["last"]:
                row["last"] = partition
            local = _local(entry.get("fetched_at"))
            if local is None:
                continue
            stamp = local.isoformat(timespec="seconds")
            if row["fetched"] is None or stamp > row["fetched"]:
                row["fetched"] = stamp
            day = history.setdefault(
                (local.date().isoformat(), group_of(name)),
                dict(
                    date=local.date().isoformat(),
                    group=group_of(name),
                    first=stamp,
                    last=stamp,
                    datasets={},
                    partitions=0,
                    rows=0,
                    calls=0,
                ),
            )
            day["first"] = min(day["first"], stamp)
            day["last"] = max(day["last"], stamp)
            day["datasets"][name] = day["datasets"].get(name, 0) + 1
            day["partitions"] += 1
            day["rows"] += int(entry.get("rows") or 0)
            day["calls"] += int(entry.get("calls") or 0)
        records = sorted(
            history.values(), key=lambda r: (r["date"], r["last"]), reverse=True
        )
        for record in records:
            record["datasets"] = [
                name
                for name, _ in sorted(record["datasets"].items(), key=lambda i: -i[1])
            ]
        datasets = []
        stock = self._stock_panel()
        if stock:
            datasets.append(stock)
        for key, folder in (
            ("futures", "futures_panel"),
            ("fund", "fund_panel/fund"),
            ("cb", "fund_panel/cb"),
        ):
            built = self._other_panel(key, folder)
            has_raw = any(t["group"] == key for t in tables.values())
            if built or has_raw:
                datasets.append(
                    built or dict(key=key, name=dict(GROUPS)[key], panel=None)
                )
        indexes = self._indexes(index_months)
        if indexes:
            datasets.append(
                dict(key="index", name="指数行情", panel=None, indexes=indexes)
            )
        for dataset in datasets:
            key = dataset["key"]
            dataset["tables"] = sorted(
                (t for t in tables.values() if t["group"] == key),
                key=lambda t: t["name"],
            )
            dataset["history"] = [r for r in records if r["group"] == key][:30]
            dataset.setdefault("verify", None)
        return dict(datasets=datasets)

    def _stock_panel(self):
        version, path = self._current("panel")
        if version is None:
            return None
        meta = _read_json(path / "metadata.json") or {}
        dates = meta.get("dates") or []
        symbols = meta.get("symbols") or []
        fields = [
            name for name in meta.get("fields", {}) if (path / f"{name}.npy").exists()
        ]
        return dict(
            key="stock",
            name="股票日频面板",
            panel=dict(
                version=version,
                start=_day(dates[0]) if dates else None,
                end=_day(dates[-1]) if dates else None,
                days=len(dates),
                symbols=len(symbols),
                fields=len(fields),
                size=_size(path),
                built_at=_local(meta.get("built_at")).isoformat(timespec="seconds")
                if _local(meta.get("built_at"))
                else None,
            ),
            verify=self._stock_verify(version),
        )

    def _stock_verify(self, version):
        report = _read_json(self.root / "reports/verify.json")
        if not report or not isinstance(report.get("checks"), dict):
            return None
        checks = [
            dict(
                name=name,
                checked=check.get("checked"),
                failed=check.get("failed"),
                pass_rate=check.get("pass_rate"),
                note=check.get("note") or "",
            )
            for name, check in sorted(report["checks"].items())
            if isinstance(check, dict)
        ]
        generated = _local(report.get("generated_at"))
        return dict(
            generated_at=generated.isoformat(timespec="seconds") if generated else None,
            current=report.get("panel_version") == version,
            verdict=None,
            checks=checks,
        )

    def _other_panel(self, key, folder):
        version, path = self._current(folder)
        if version is None:
            return None
        meta = _read_json(path / "metadata.json") or {}
        dates = meta.get("dates") or []
        shape = meta.get("shape") or [len(dates), None]
        report_name = "futures_verify.json" if key == "futures" else "funds_verify.json"
        report = _read_json(self.root / "reports" / report_name)
        verify = None
        if report and isinstance(report.get("checks"), dict):
            generated = _local(report.get("generated_at"))
            verify = dict(
                generated_at=generated.isoformat(timespec="seconds")
                if generated
                else None,
                current=report.get("panel_version") in (version, None)
                if key == "futures"
                else True,
                verdict=report.get("verdict"),
                checks=[
                    dict(
                        name=name,
                        checked=check.get("checked"),
                        failed=check.get("failed"),
                        pass_rate=None,
                        note=check.get("note") or "",
                        verdict=check.get("verdict"),
                    )
                    for name, check in sorted(report["checks"].items())
                    if isinstance(check, dict)
                ],
            )
        built = _local(meta.get("built_at"))
        return dict(
            key=key,
            name=dict(GROUPS)[key],
            panel=dict(
                version=version,
                start=_day(dates[0]) if dates else None,
                end=_day(dates[-1]) if dates else None,
                days=int(shape[0]) if shape and shape[0] is not None else len(dates),
                symbols=int(shape[1]) if len(shape) > 1 and shape[1] else None,
                fields=len(meta.get("fields") or {}),
                size=_size(path),
                built_at=built.isoformat(timespec="seconds") if built else None,
            ),
            verify=verify,
        )

    def _indexes(self, months):
        from ashare.data.catalog import BENCHMARKS, INDEXES

        if not months:
            return []
        names = dict(BENCHMARKS)
        for alias, code in INDEXES.items():
            names.setdefault(code, alias)
        import polars as pl

        rows = []
        for code, found in sorted(months.items()):
            found.sort()
            try:
                dates = (
                    pl.scan_parquet([self.root / f for _, f in {found[0], found[-1]}])
                    .filter(pl.col("ts_code") == code)
                    .select(pl.col("trade_date").cast(pl.String))
                    .collect()["trade_date"]
                )
            except (OSError, pl.exceptions.PolarsError):
                LOG.warning("指数分区读取失败 code=%s", code)
                continue
            if len(dates):
                rows.append(
                    dict(
                        symbol=code,
                        name=names.get(code, code),
                        start=_day(dates.min()),
                        end=_day(dates.max()),
                    )
                )
        return rows

    # ---- 字段统计 -----------------------------------------------------

    def _cache_path(self, version):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"datastatus-{STATS_FORMAT}-{version}.json"

    def stats(self, wait=False):
        """返回当前面板版本的逐字段统计；未算完时返回 None 并在后台开算。"""
        version, path = self._current("panel")
        if version is None:
            return version, None
        with self.lock:
            if version in self._stats:
                return version, self._stats[version]
            cached = self._cache_path(version)
            if cached is not None and cached.exists():
                value = _read_json(cached)
                if value and value.get("version") == version:
                    self._stats = {version: value}
                    return version, value
            if version not in self._computing:
                self._computing.add(version)
                thread = threading.Thread(
                    target=self._compute,
                    args=(version, path),
                    name="datastatus",
                    daemon=True,
                )
                thread.start()
            else:
                thread = None
        if wait and thread is not None:
            thread.join()
            return version, self._stats.get(version)
        return version, None

    def _compute(self, version, path):
        started = time.perf_counter()
        try:
            meta = _read_json(path / "metadata.json") or {}
            dates = np.array(meta.get("dates") or [], dtype="U8")
            names = [n for n in meta.get("fields", {}) if (path / f"{n}.npy").exists()]
            listed_path = path / "is_listed.npy"
            if listed_path.exists():
                listed = np.load(listed_path).astype(bool, copy=False)
            else:
                shape = np.load(path / f"{names[0]}.npy", mmap_mode="r").shape
                listed = np.ones(shape, dtype=bool)

            def one(name):
                array = np.load(path / f"{name}.npy", mmap_mode="r")
                return field_stats(array, name, dates, listed)

            with ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="fieldstats"
            ) as pool:
                results = list(pool.map(one, names))
            fields_meta = meta.get("fields", {})
            for result in results:
                info = fields_meta.get(result["name"]) or {}
                result["unit"] = info.get("unit") or _catalog_unit(result["name"])
                result["category"] = category(result["name"])
            results.sort(key=lambda r: _order(r["name"]))
            value = dict(version=version, fields=results)
            cached = self._cache_path(version)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                temporary = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(value, ensure_ascii=False))
                os.replace(temporary, cached)
            with self.lock:
                self._stats = {version: value}
            LOG.info(
                "字段统计完成 version=%s fields=%d elapsed=%.2fs",
                version[:12],
                len(results),
                time.perf_counter() - started,
            )
        except Exception:
            LOG.exception("字段统计失败 version=%s", version[:12])
            with self.lock:
                self._errors[version] = time.monotonic()
        finally:
            with self.lock:
                self._computing.discard(version)

    def failed(self, version):
        with self.lock:
            return version in self._errors

    # ---- 因子引用 -----------------------------------------------------

    def references(self):
        """字段 → 引用它的因子（库内按表达式去重，加上登记因子）；按库签名缓存。"""
        rows = []
        signature = None
        with self.lock:
            # 打开平台库查签名约 10–20 ms；5 秒内复用上次结果，新跑的因子最多晚 5 秒计入。
            if (
                self._references[0] is not None
                and time.monotonic() - self._references_checked < 5
            ):
                return self._references[0]
            self._references_checked = time.monotonic()
        if self.platform_store is not None:
            with self.platform_store.connect() as db:
                signature = tuple(
                    db.execute("SELECT count(*), max(updated) FROM entries").fetchone()
                )
                if signature != self._references[1]:
                    rows = db.execute(
                        "SELECT expression_id, name, source, max(sharpe) AS sharpe "
                        "FROM entries GROUP BY expression_id"
                    ).fetchall()
        with self.lock:
            if self._references[0] is not None and signature == self._references[1]:
                return self._references[0]
        from .language import parse, registry

        output = {}

        def add(fields, name, sharpe):
            for field in fields:
                bucket = output.setdefault(field, {})
                previous = bucket.get(name)
                if previous is None or (
                    sharpe is not None
                    and (previous["sharpe"] is None or sharpe > previous["sharpe"])
                ):
                    bucket[name] = dict(name=name, sharpe=sharpe)

        for row in rows:
            key = row["expression_id"]
            if key not in self._parsed:
                try:
                    self._parsed[key] = frozenset().union(
                        *(v.expression.fields for v in parse(row["source"]))
                    )
                except ValueError:
                    self._parsed[key] = frozenset()
            add(self._parsed[key], row["name"], _sharpe(row["sharpe"]))
        best = self.platform_store.best() if self.platform_store is not None else {}
        for factor in registry().values():
            add(factor.fields, factor.name, _sharpe(best.get(factor.name)))
        result = {
            field: sorted(
                items.values(),
                key=lambda r: (r["sharpe"] is None, -(r["sharpe"] or 0), r["name"]),
            )
            for field, items in output.items()
        }
        with self.lock:
            self._references = (result, signature)
        return result


def _sharpe(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def _catalog_unit(name):
    from ashare.data.catalog import FIELDS

    spec = FIELDS.get(name)
    return spec.unit if spec else ""


def routes(data_root=None, platform_store=None):
    """数据页接口；``create_app`` 用一行 ``include_router`` 注册。"""
    from .language import FIELD_LABELS

    labels = FIELD_LABELS | LABELS
    cache = Path(platform_store.root) / "cache" if platform_store is not None else None
    status = DataStatus(data_root, cache, platform_store)
    router = APIRouter(prefix="/api/data")

    @router.get("/overview")
    def overview():
        return status.overview()

    def table(version, stats, refs):
        return [
            dict(
                name=item["name"],
                label=labels.get(item["name"], item["name"]),
                category=item["category"],
                dtype=item["dtype"],
                unit=item["unit"],
                start=item["start"],
                latest=item["latest"],
                spark=_spark(item["monthly"]),
                references=len(refs.get(item["name"], ())),
            )
            for item in stats["fields"]
        ]

    @router.get("/fields")
    def fields():
        version, stats = status.stats()
        if version is None:
            return dict(state="missing", version=None, fields=[])
        if stats is None:
            return dict(
                state="failed" if status.failed(version) else "pending",
                version=version,
                fields=[],
            )
        return dict(
            state="ready",
            version=version,
            fields=table(version, stats, status.references()),
        )

    @router.get("/fields/{name}")
    def field_detail(name: str):
        version, stats = status.stats()
        if stats is None:
            raise HTTPException(404, "字段统计尚未就绪")
        item = next((f for f in stats["fields"] if f["name"] == name), None)
        if item is None:
            raise HTTPException(404, "没有这个字段")
        return dict(
            item,
            label=labels.get(name, name),
            factors=status.references().get(name, [])[:12],
        )

    router.status = status
    return router
