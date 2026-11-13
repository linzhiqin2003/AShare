"""不可变 Parquet 修订版与原子清单；未完成分区永远不当作完成。"""

import copy
import hashlib
import io
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from .catalog import ALL_DATASETS, FUND_DATASETS, FUTURES_DATASETS
from .manifest import Manifest


def data_root(root=None):
    resolved = Path(
        root
        or os.environ.get("ASHARE_DATA_DIR")
        or Path(__file__).resolve().parents[3] / "data"
    ).resolve()
    if "pytest" in sys.modules:
        repository = Path(__file__).resolve().parents[3]
        forbidden = {p / "data" for p in (repository, *repository.parents)}
        if any(resolved == p or resolved.is_relative_to(p) for p in forbidden):
            raise RuntimeError(
                "测试禁止访问真实数据根目录；请传 tmp_path 或设置临时 ASHARE_DATA_DIR"
            )
    return resolved


def assert_no_secret(value):
    secret = os.environ.get("TUSHARE_TOKEN")
    if secret and secret in value:
        raise ValueError("拒绝写入含鉴权信息的内容")


def atomic_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    assert_no_secret(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def lock(path, *, shared=False, blocking=True):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        yield


class Store:
    def __init__(self, root=None, market="cn", *, readonly=False):
        if market != "cn":
            raise ValueError(f"尚不支持市场 {market!r}，可用市场：cn")
        self.root = data_root(root) / market
        self._manifest = Manifest(self.root, readonly=readonly)
        self.manifest_path = self._manifest.path
        self._snapshot = None

    def snapshot(self):
        frozen = copy.copy(self)
        frozen._snapshot = dict(self.manifest())
        return frozen

    def manifest(self):
        if self._snapshot is not None:
            return self._snapshot
        return self._manifest.read()

    def entries(self, name=None):
        return {
            k: v
            for k, v in self.manifest().items()
            if name is None or v["dataset"] == name
        }

    def complete(self, key, *, verify_hash=True):
        entry = self.manifest().get(key)
        if not entry or entry["status"] != "complete":
            return False
        path = self.root / entry["file"]
        return path.exists() and (
            not verify_hash
            or hashlib.sha256(path.read_bytes()).hexdigest() == entry["file_hash"]
        )

    def mark(self, key, entry):
        assert_no_secret(json.dumps(entry, ensure_ascii=False))
        return self._manifest.put(key, entry)

    def export_manifest(self, path):
        """导出兼容旧版本的清单；迁移前的 manifest.json 始终保留。"""
        atomic_json(path, self.manifest())

    def write(self, name, partition, params, result):
        # 多个写者可以并发，GC 只有拿到独占锁才清理遗留临时文件。
        with lock(self.root / ".raw.lock", shared=True):
            return self._write(name, partition, params, result)

    def _write(self, name, partition, params, result):
        spec = ALL_DATASETS[name]
        table = result.table
        key = f"{name}/{partition}"
        previous = self.manifest().get(key, {})
        if spec.partition == "fut_day":
            self._guard_futures_refresh(key, spec, previous, len(table))
        suspicious = (
            spec.partition == "day"
            and not spec.empty_allowed
            and not len(table)
            and (not spec.coverage_start or partition >= spec.coverage_start)
        )
        prior = (
            self._manifest.predecessor(name, partition)
            if spec.partition == "day"
            else {}
        )
        collapsed = (
            not spec.empty_allowed
            and prior.get("rows", 0) >= spec.drop_check_min_rows
            and len(table) > 0
            and len(table) < prior["rows"] * spec.drop_check_ratio
        )
        suspicious = suspicious or collapsed
        if previous.get("quality_state") == "confirmed" and previous.get("rows") == len(
            table
        ):
            suspicious = False
        if suspicious and previous.get("status") == "complete":
            raise ValueError(
                f"{name}/{partition} 刷新返回可疑空表或行数骤降，保留旧修订"
            )
        if not table.columns:
            table = pl.DataFrame(
                schema={c: pl.String for c in spec.columns.split(",") if c}
            )
        assert_no_secret(table.write_json())
        # 对原始值排序后求哈希，抓取时间不参与内容版本。
        ordered = table.sort(
            [c for c in spec.keys if c in table.columns], nulls_last=True
        )
        buffer = io.BytesIO()
        ordered.write_ipc(buffer)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        key = f"{name}/{partition}"
        previous = self.manifest().get(key, {})
        if previous.get("content_hash") == digest and self.complete(key):
            return self.mark(
                key,
                dict(
                    params=params,
                    fetched_at=result.fetched_at,
                    refresh_error=None,
                    calls=result.calls,
                    retries=result.retries,
                ),
            )
        path = (
            self.root
            / "raw"
            / (spec.raw_dir or spec.api)
            / partition
            / f"{digest}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table = table.with_columns(pl.lit(result.fetched_at).alias("_fetched_at"))
        if not path.exists():
            temp = path.with_name(f".{uuid.uuid4().hex}.tmp")
            try:
                table.write_parquet(temp)
                with temp.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temp, path)
                fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                temp.unlink(missing_ok=True)
        key = f"{name}/{partition}"
        entry = dict(
            dataset=name,
            partition=partition,
            params=params,
            rows=len(table),
            fetched_at=result.fetched_at,
            status="suspect" if suspicious else "complete",
            quality_state="row_collapse"
            if collapsed
            else "empty"
            if suspicious
            else "unchecked",
            content_hash=digest,
            file=str(path.relative_to(self.root)),
            file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            calls=result.calls,
            retries=result.retries,
            refresh_error=None,
        )
        self.mark(key, entry)
        return entry

    @staticmethod
    def _guard_futures_refresh(key, spec, previous, rows):
        """期货日分区的刷新保护。

        期货各交易日的行数随交易所休市差异、品种上市而大幅变化，与前一交易日比没有意义；
        危险的是「重抓同一天时源偶发空返回或截断，把已完成的好修订覆盖掉」。所以只和
        同一分区上一个完成修订比：变成空表、或行数降到 drop_check_ratio 以下（上一修订
        至少 drop_check_min_rows 行时）一律拒绝写入，旧修订保留，调用方记 refresh_error。
        人工确认过的修订（confirm-partition）同样受保护。首次写入的空分区照常记完成，
        由 verify-futures 按已知例外清单逐条判定。
        """
        if previous.get("status") != "complete" or not previous.get("rows"):
            return
        before = previous["rows"]
        if rows == 0 or (
            before >= spec.drop_check_min_rows and rows < before * spec.drop_check_ratio
        ):
            raise ValueError(
                f"{key} 刷新返回 {rows} 行（上一修订 {before} 行），疑似空返回或截断，保留旧修订"
            )

    def confirm(self, key, reason):
        """人工确认异常分区，只解除质量隔离，不跳过文件完整性检查。"""
        entry = self.manifest().get(key, {})
        path = self.root / entry.get("file", "missing")
        if (
            not reason.strip()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("file_hash")
        ):
            raise ValueError("确认必须有说明且分区文件完整")
        return self.mark(
            key,
            dict(
                status="complete", quality_state="confirmed", confirmation_reason=reason
            ),
        )

    def validate_partitions(self, names):
        """批次完成后按日期重核，消除并发完成顺序对质量状态的影响。"""
        for name in names:
            spec = ALL_DATASETS[name]
            if spec.partition != "day" or spec.empty_allowed:
                continue
            previous = None
            for key, entry in sorted(self.entries(name).items()):
                if not entry.get("file") or entry.get("status") not in (
                    "complete",
                    "suspect",
                ):
                    continue
                if spec.coverage_start and entry["partition"] < spec.coverage_start:
                    continue
                collapsed = (
                    previous is not None
                    and previous["rows"] >= spec.drop_check_min_rows
                    and entry["rows"] < previous["rows"] * spec.drop_check_ratio
                )
                if entry.get("quality_state") != "confirmed":
                    state = (
                        "empty"
                        if entry["rows"] == 0
                        else "row_collapse"
                        if collapsed
                        else "checked"
                    )
                    status = "complete" if state == "checked" else "suspect"
                    if entry.get("quality_state") != state or entry["status"] != status:
                        self.mark(key, dict(status=status, quality_state=state))
                previous = entry

    def scan(self, name, start=None, end=None, codes=None):
        """codes 只对「代码_年月」分区（指数日线、权重）生效：只打开这些代码的分区。"""

        def in_range(entry):
            kind = ALL_DATASETS[name].partition
            part = entry["partition"]
            if kind in ("day", "fut_day"):
                return (start is None or part >= start) and (end is None or part <= end)
            if kind in ("index_month", "index_month_daily") and "_" in part:
                code, month = part.rsplit("_", 1)
                return (
                    (codes is None or code in codes)
                    and (start is None or month >= start[:6])
                    and (end is None or month <= end[:6])
                )
            return True

        entries = [
            v
            for v in self.entries(name).values()
            if v["status"] == "complete" and in_range(v)
        ]
        if not entries and (start is not None or end is not None or codes is not None):
            candidates = [
                v for v in self.entries(name).values() if v["status"] == "complete"
            ]
            if candidates:
                return pl.scan_parquet(self.root / candidates[0]["file"]).head(0)
        if not entries:
            raise FileNotFoundError(
                f"缺少数据集 {name}；请运行 python -m ashare.data sync --datasets {name}"
            )
        nonempty = [e for e in entries if e["rows"]]
        entries = nonempty or entries[:1]
        return pl.concat(
            [pl.scan_parquet(self.root / e["file"]) for e in entries],
            how="diagonal_relaxed",
        )

    def read(self, name, start=None, end=None):
        return self.scan(name, start, end).collect()

    def _version(self, futures):
        values = {
            k: v["content_hash"]
            for k, v in self.manifest().items()
            if v["status"] == "complete"
            and (v.get("dataset") in FUTURES_DATASETS) == futures
            and v.get("dataset") not in FUND_DATASETS  # 基金与可转债另算版本
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    @property
    def version(self):
        """股票原始库的内容版本；共用清单里的期货条目不参与，加不加期货都不变。"""
        return self._version(False)

    @property
    def futures_version(self):
        """期货原始库（FUTURES_DATASETS）的内容版本，与股票版本互不影响。"""
        return self._version(True)

    @property
    def fund_version(self):
        """场内基金与可转债原始库（FUND_DATASETS）的内容版本，与股票、期货版本互不影响。"""
        values = {
            k: v["content_hash"]
            for k, v in self.manifest().items()
            if v["status"] == "complete" and v.get("dataset") in FUND_DATASETS
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
