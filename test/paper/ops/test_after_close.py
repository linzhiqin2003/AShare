"""盘后阶段（2026-11-14 新增，模拟盘每日运行）：失败不继续、数据没齐返回 NOT_READY、幂等、
阶段锁、校验门禁；另有一条在合成数据目录上用真实 sync / build / verify 加假数据源的集成测试。
"""

import json
from contextlib import asynccontextmanager

import polars as pl
import pytest

from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.paper.ops import after_close as stage
from ashare.paper.ops.context import DATA_LOCK, Exit, stage_lock

from .helpers import calendar, context

DAY = "20261113"  # 周五
NOW = "2026-11-13T18:30:00+08:00"


class Fakes:
    """注入的 sync / build / verify；记录调用顺序，按场景标记当天分区。"""

    def __init__(self, ctx, *, ready=stage.REQUIRED, fail=None, checks=None):
        self.ctx, self.ready, self.fail = ctx, ready, fail
        self.calls = []
        self.checks = checks or {"manifest_files": 0, "ohlc_high": 0}
        self.version = "v1"
        self.built = None  # 面板当前版本：建过才有

    @asynccontextmanager
    async def source(self, store):
        self.calls.append("open_source")
        yield object()

    async def sync(self, source, store, start, end, names=None):
        self.calls.append(("sync", tuple(names or ()), end))
        if self.fail == "sync" and names is None:
            raise RuntimeError("boom")
        if names is None:
            for name in self.ready:
                store.mark(
                    f"{name}/{DAY}",
                    dict(dataset=name, partition=DAY, status="complete"),
                )

    def build(self, store, incremental):
        self.calls.append("build")
        if self.fail == "build":
            raise RuntimeError("boom")
        self.built = self.version
        return {"version": self.version}

    def verify(self, store):
        self.calls.append("verify")
        return {"checks": {k: {"failed": v} for k, v in self.checks.items()}}

    def run(self, **options):
        return stage.after_close(
            self.ctx,
            source_factory=self.source,
            sync_fn=self.sync,
            build_fn=self.build,
            verify_fn=self.verify,
            **options,
        )


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    ctx = context(tmp_path, NOW)
    cal = calendar(["20261112", DAY, "20261116"], "20261231")
    monkeypatch.setattr(
        stage,
        "TradingCalendar",
        type("C", (), {"from_store": staticmethod(lambda d: cal)}),
    )
    f = Fakes(ctx)
    monkeypatch.setattr(stage, "_panel_version", lambda c: f.built)
    monkeypatch.setattr(stage, "_panel_last_day", lambda c: DAY)
    return f


def state(ctx, name):
    return json.loads((ctx.ops / "state" / f"{name}.json").read_text())


def test_success_then_idempotent(fakes):
    result = fakes.run()
    assert result.code == Exit.OK and result.day == DAY
    # 先只同步日历（终点在未来），再同步数据（终点是当天）。
    syncs = [c for c in fakes.calls if c[0] == "sync"]
    assert syncs[0][1] == ("trade_cal",) and syncs[0][2] > "2026-12-01"
    assert syncs[-1] == ("sync", (), "2026-11-13")
    assert fakes.calls.index("build") < fakes.calls.index("verify")
    done = state(fakes.ctx, "after-close-done")
    assert done["day"] == DAY and done["panel_version"] == "v1"
    assert state(fakes.ctx, "after-close")["status"] == "ok"
    # 重跑：只同步日历，不再同步数据、不重建。
    fakes.calls.clear()
    again = fakes.run()
    assert again.code == Exit.OK and again.details == {"skipped": True}
    assert "build" not in fakes.calls
    assert [c for c in fakes.calls if c[0] == "sync"] == [
        ("sync", ("trade_cal",), syncs[0][2])
    ]
    fakes.calls.clear()
    assert fakes.run(force=True).code == Exit.OK and "build" in fakes.calls


def test_not_ready_does_not_build(fakes):
    fakes.ready = ("daily", "adj_factor")  # 每日指标、涨跌停价、ST 还没发布
    result = fakes.run()
    assert result.code == Exit.NOT_READY
    assert result.details["missing"] == ["daily_basic", "stk_limit", "stock_st"]
    assert "build" not in fakes.calls and "verify" not in fakes.calls
    assert not (fakes.ctx.ops / "state" / "after-close-done.json").exists()
    assert state(fakes.ctx, "after-close")["status"] == "not_ready"


@pytest.mark.parametrize("step", ["sync", "build"])
def test_failure_stops_before_next_step(fakes, step):
    fakes.fail = step
    result = fakes.run()
    assert result.code == Exit.FAILED
    if step == "sync":
        assert "build" not in fakes.calls
    assert "verify" not in fakes.calls
    assert not (fakes.ctx.ops / "state" / "after-close-done.json").exists()


def test_panel_must_end_on_the_day(fakes, monkeypatch):
    monkeypatch.setattr(stage, "_panel_last_day", lambda c: "20261112")
    result = fakes.run()
    assert result.code == Exit.FAILED and "verify" not in fakes.calls


def test_verify_gate_hard_checks_and_accept(fakes):
    assert fakes.run().code == Exit.OK  # 第一次：当前计数成为基线
    fakes.version = "v2"
    fakes.checks = {"manifest_files": 0, "ohlc_high": 5}  # 只有软检查变多：警告
    result = fakes.run(force=True)
    assert result.code == Exit.OK and result.details["warnings"] == ["ohlc_high: 0 → 5"]
    fakes.version = "v3"
    fakes.checks = {"manifest_files": 2, "ohlc_high": 5, "partition_coverage_daily": 0}
    result = fakes.run(force=True)
    assert result.code == Exit.FAILED and result.details["increased"] == [
        "manifest_files: 0 → 2"
    ]
    assert state(fakes.ctx, "after-close-done")["panel_version"] == "v2"  # 没前进
    accepted = fakes.run(accept_verify=True)
    assert accepted.code == Exit.OK
    baseline = state(fakes.ctx, "verify-baseline")
    assert baseline["failed"]["manifest_files"] == 2 and baseline["accepted"] is True


def test_compare_checks_prefixes():
    hard, soft = stage.compare_checks(
        {"partition_coverage_x": 1, "unique_keys": 0, "price_limits": 3},
        {
            "partition_coverage_x": {"failed": 2},
            "unique_keys": {"failed": 0},
            "price_limits": {"failed": 4},
            "brand_new": {"failed": 9},  # 新出现的软检查带不一致：警告
            "brand_new_clean": {"failed": 0},
        },
    )
    assert hard == ["partition_coverage_x: 1 → 2"]
    assert soft == ["brand_new: 新出现 9", "price_limits: 3 → 4"]
    assert stage.compare_checks({}, {"unique_keys": {"failed": 3}}) == ([], [])


def test_lock_prevents_second_instance(fakes):
    """盘后和信号共用一把锁（2026-11-14 审查）：信号读的面板不能正被重建。"""
    with stage_lock(fakes.ctx, DATA_LOCK):
        result = fakes.run()
    assert result.code == Exit.BUSY and fakes.calls == []
    assert fakes.run().code == Exit.OK  # 锁释放后照常


def test_hard_check_that_stops_running_fails_the_gate(fakes):
    """2026-11-14 审查：基线里有的硬性检查这次没出现（verify 前提不满足时不写这一项）不能放行。"""
    fakes.checks = {"manifest_files": 0, "open_masks_match_panelbt": 0}
    assert fakes.run().code == Exit.OK  # 基线
    fakes.version = "v2"
    fakes.checks = {"manifest_files": 0}
    result = fakes.run(force=True)
    assert result.code == Exit.FAILED
    assert result.details["increased"] == ["open_masks_match_panelbt: 这次没有检查"]
    assert fakes.run(accept_verify=True).code == Exit.OK  # 人工核对后放行


def test_new_hard_check_with_failures_fails_the_gate(fakes):
    """2026-11-14 审查：第一次出现就带不一致的硬性检查不能被当作基线接受。"""
    assert fakes.run().code == Exit.OK
    fakes.version = "v2"
    fakes.checks = {"manifest_files": 0, "ohlc_high": 0, "partition_coverage_daily": 7}
    result = fakes.run(force=True)
    assert result.code == Exit.FAILED
    assert result.details["increased"] == ["partition_coverage_daily: 新出现 7"]
    # 新出现但没有不一致：照常通过，并进入基线。
    fakes.checks = {"manifest_files": 0, "ohlc_high": 0, "partition_coverage_daily": 0}
    assert fakes.run(force=True).code == Exit.OK
    assert (
        state(fakes.ctx, "verify-baseline")["failed"]["partition_coverage_daily"] == 0
    )


# ------------------------------------------------------------------ 集成


class SyntheticSource:
    """按参数从合成原始表里取行，像 Tushare 一样返回；``withhold`` 里的数据集在 ``day`` 返回空表
    （当天还没发布；更早的日子照常）。"""

    def __init__(self, tables, withhold=(), day=None):
        self.tables, self.withhold, self.day, self.calls = tables, set(withhold), day, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls += 1
        table = self.tables.get(api)
        if table is None:
            columns = [c for c in fields.split(",") if c]
            return FetchResult(
                pl.DataFrame(schema={c: pl.String for c in columns}), "t", 1
            )
        date = next(
            (c for c in ("trade_date", "cal_date", "ex_date") if c in table.columns),
            None,
        )
        for key, value in params.items():
            if key == "start_date" and date:
                table = table.filter(pl.col(date) >= value)
            elif key == "end_date" and date:
                table = table.filter(pl.col(date) <= value)
            elif key in table.columns:
                table = table.filter(pl.col(key).cast(pl.String) == str(value))
        if api in self.withhold and params.get("trade_date") == self.day:
            table = table.head(0)
        return FetchResult(table, "2026-02-02T00:00:00Z", 1)


def test_real_sync_build_verify_on_synthetic_datadir(tmp_path):
    """当天日线第一次没发布 → NOT_READY 且面板不动；发布后 → 重建、校验、记下完成状态。"""
    from test.fixtures.synthetic_datadir import cached_datadir, copy_datadir, raw_tables

    info = copy_datadir(cached_datadir(), tmp_path / "data")
    day = info.dates[-1]
    ctx = context(tmp_path, f"{day[:4]}-{day[4:6]}-{day[6:]}T18:30:00+08:00", info.root)
    tables = raw_tables()[0]
    # 合成库只有股票相关的表（没有期货、融资融券标的）；生产上用缺省的全部日表。
    datasets = [n for n in tables if n != "trade_cal"]
    store = Store(info.root)
    # 模拟当天日线还没抓到过（上次失败）：清单里不是 complete。
    store.mark(f"daily/{day}", {"status": "failed"})
    before = json.loads((info.root / "cn/panel/current.json").read_text())

    first = stage.after_close(
        ctx,
        datasets=datasets,
        source_factory=lambda s: SyntheticSource(tables, withhold={"daily"}, day=day),
    )
    assert first.code == Exit.NOT_READY and first.details["missing"] == ["daily"]
    assert store.manifest()[f"daily/{day}"]["status"] == "suspect"
    assert json.loads((info.root / "cn/panel/current.json").read_text()) == before

    second = stage.after_close(
        ctx, datasets=datasets, source_factory=lambda s: SyntheticSource(tables)
    )
    assert second.code == Exit.OK, second.message
    assert store.manifest()[f"daily/{day}"]["status"] == "complete"
    done = json.loads((ctx.ops / "state" / "after-close-done.json").read_text())
    import ashare.data

    assert done == dict(
        day=day,
        panel_version=str(ashare.data.open(info.root).panel([]).version),
        finished_at=done["finished_at"],
    )
    # 再跑一次：幂等（只同步日历）。
    third = stage.after_close(
        ctx, datasets=datasets, source_factory=lambda s: SyntheticSource(tables)
    )
    assert third.code == Exit.OK and third.details == {"skipped": True}
