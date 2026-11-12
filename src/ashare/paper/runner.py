"""模拟盘运行器；每一步先确认内核接受，再推进调度状态。"""

from __future__ import annotations

import json
import os
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from paperbroker import Broker
from paperbroker._core import BrokerCore
from paperbroker.journal import canonical, durable_write

from .market import (
    TZ,
    corporate_actions,
    normalize,
    shares,
    suspension_rules,
    timestamp,
)
from .schedule import Schedule
from .source import Batch, Clock, LiveSource, Monitor, SourceError, log
from .storage import ParquetStore


def load_targets(path: Path) -> dict:
    target = json.loads(path.read_text())
    if target.get("version") != 1 or not isinstance(target.get("weights"), dict):
        raise ValueError("目标组合必须为 version=1、weights 为证券到权重的映射")
    time.fromisoformat(target["at"])
    values = list(target["weights"].values())
    if (
        any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values)
        or sum((Decimal(str(v)) for v in values), Decimal(0)) > 1
    ):
        raise ValueError("权重必须非负且总和不超过 1")
    return target


def report(broker: Broker, events: list[dict], state: dict | None = None) -> dict:
    if state is not None and "account" in state:
        return state
    state = state or json.loads(broker.dump())
    account = json.loads(BrokerCore.restore(json.dumps(state)).account())
    names = [p["name"] for p in state["config"]["instruments"]]
    for symbol, p, mark in zip(
        names, account["positions"], state["marks"], strict=True
    ):
        p.update(
            symbol=symbol,
            mark=mark,
            market_value=p["total"] * mark,
            cost_price=p["cost"] / p["total"] if p["total"] else 0,
        )
    events = [e for e in events if e["day"] == state["day"]]
    order_ids = {e["id"] for e in events if e["type"] == "accepted"}
    return dict(
        version=2,
        money_scale=10_000,
        account=account,
        fills=[e for e in events if e["type"] == "fill"],
        orders=[o for o in state["orders"] if o["id"] in order_ids],
        events=events,
    )


class LiveRunner:
    def __init__(
        self,
        broker: Broker,
        source: LiveSource,
        schedule: Schedule,
        clock: Clock,
        store: ParquetStore | None,
        targets: dict | None = None,
        output: Path | None = None,
        settlement_deadline: time = time(15, 3),
        status_interval: float = 30,
    ):
        self.broker, self.source, self.schedule, self.clock, self.store = (
            broker,
            source,
            schedule,
            clock,
            store,
        )
        self.targets, self.output = targets, output
        if settlement_deadline < schedule.table.close:
            raise ValueError("结算截止时间不能早于收盘")
        self.settlement_deadline = settlement_deadline
        self.status_interval = max(1, status_interval)
        self.last_status = float("-inf")
        self.stale_close: dict[str, list[str]] = {}
        if targets and set(targets["weights"]) - broker.symbol_ids.keys():
            raise ValueError("目标组合含未注册证券")
        self.monitor = Monitor(clock)
        state = json.loads(broker.dump())
        self.phase = state["phase"]
        self.metadata_day = self.auction_day = self.target_day = ""
        self.settled_day = str(state["settled_day"])
        self.events: list[dict] = [
            e for e in broker.history if e["day"] == state["day"]
        ]
        self.close_baseline: dict[int, int] = {}
        self.baseline_day = ""
        self.last_checkpoint = 0
        self.last_target = self.next_finish = self.next_auction = float("-inf")
        self._report_days: set[str] = set()
        if broker.journal:
            path = broker.journal.directory / "runner.json"
            if path.exists():
                saved = json.loads(path.read_bytes())
                self.stale_close = saved.get("stale_close", {})
                self.baseline_day = saved["day"]
                self.close_baseline = {
                    int(k): v for k, v in saved["close_baseline"].items()
                }
        broker.events()  # 历史输出已经来自 history；不重复记入本次运行。

    def _checked(self, result: dict, operation: str) -> None:
        self._drain()
        if not result["accepted"]:
            reasons = sorted(
                {e["reason"] for e in result["events"] if e["type"] == "rejected"}
            )
            log(
                "engine_rejected",
                level=40,
                operation=operation,
                request_id=result["request_id"],
                reasons=reasons,
            )
            raise SourceError(f"内核拒绝 {operation}: {','.join(reasons)}")

    def _session(self, day: str, phase: str) -> None:
        state = json.loads(self.broker.dump())
        if state["day"] == int(day) and state["phase"] == phase:
            self.phase = phase
            return
        self._checked(
            self.broker.session(timestamp(self.clock.now()), int(day), phase), "session"
        )
        self.phase = phase

    def _drain(self) -> None:
        self.broker.events()
        day = self.broker.account()["day"]
        self.events = [e for e in self.broker.history if e["day"] == day]
        journal = self.broker.journal
        if journal is not None:
            if self.clock.monotonic() - self.last_status >= self.status_interval:
                durable_write(
                    journal.directory / "status.json",
                    canonical(report(self.broker, self.events)),
                )
                self.last_status = self.clock.monotonic()
            if journal.sequence - self.last_checkpoint >= 100:
                self.broker.checkpoint()
                self.last_checkpoint = journal.sequence

    def _save_context(self, day: str) -> None:
        if self.broker.journal:
            durable_write(
                self.broker.journal.directory / "runner.json",
                canonical(
                    dict(
                        day=day,
                        close_baseline=self.close_baseline,
                        stale_close=self.stale_close,
                    )
                ),
            )

    def _reports(self) -> None:
        if self.output is None:
            return
        summaries = self.broker.daily_states
        archive = (
            self.broker.journal.directory / "days" if self.broker.journal else None
        )
        days = {p.stem for p in archive.glob("*.json")} if archive else set(summaries)
        self.output.mkdir(parents=True, exist_ok=True)
        for day in sorted(days - self._report_days):
            destination = self.output / f"{day}.json"
            if destination.exists():
                self._report_days.add(day)
                continue
            source = archive / f"{day}.json" if archive else None
            stale = self.stale_close.get(day, [])
            if source and source.exists() and not stale:
                # 同一账户内归档与日报共用 inode，避免重复占用和重复刷盘。
                os.link(source, destination)
                fd = os.open(self.output, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            else:
                summary = (
                    json.loads(source.read_bytes())
                    if source and source.exists()
                    else summaries[day]
                )
                durable_write(
                    destination, canonical(summary | {"stale_close_symbols": stale})
                )
            self._report_days.add(day)

    def _catch_actions(self, day: str) -> None:
        state = json.loads(self.broker.dump())
        if state["phase"] != "preopen":
            return
        through = state["actions_through"]
        start = (
            (datetime.strptime(str(through), "%Y%m%d") + timedelta(days=1))
            if through
            else datetime.strptime(day, "%Y%m%d")
        )
        end = datetime.strptime(day, "%Y%m%d")
        required = {
            (start + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((end - start).days + 1)
        }
        if required - self.schedule.known_days:
            self.schedule.update(
                self.source.client.call(
                    "trade_cal",
                    dict(
                        exchange="SSE",
                        start_date=start.strftime("%Y%m%d"),
                        end_date=day,
                    ),
                )
            )
        if required - self.schedule.known_days:
            raise SourceError("公司行为补记所需日历不完整")
        for ex_day in sorted(required & self.schedule.open_days):
            dividends = self.source.client.call("dividend", {"ex_date": ex_day})
            state = json.loads(self.broker.dump())
            actions = [
                a
                for a in corporate_actions(dividends, self.broker.symbols, ex_day)
                if state["positions"][a["symbol"]]["total"]
                and state["corporate_dates"][a["symbol"]] < int(ex_day)
            ]
            self._checked(
                self.broker.apply(
                    dict(
                        type="corporate",
                        time=timestamp(self.clock.now()),
                        day=int(ex_day),
                        actions=actions,
                    )
                ),
                "corporate",
            )

    def metadata(self, day: str) -> list[dict]:
        limits = self.source.client.call("stk_limit", {"trade_date": day})
        suspended = self.source.client.call("suspend_d", {"trade_date": day})
        self.source.limits = {
            r["ts_code"]: (r["up_limit"], r["down_limit"]) for r in limits
        }
        self.source.suspended, self.source.suspension_intervals = suspension_rules(
            suspended
        )
        self.source.day = day
        missing = set(self.broker.symbols) - self.source.limits.keys()
        if missing:
            log("unavailable_parameters", level=30, day=day, count=len(missing))
        log("metadata_ready", day=day, symbols=len(self.broker.symbols))
        return limits

    def consume(self, batch: Batch) -> None:
        if batch.symbols and batch.symbols != self.broker.symbols:
            from .profiles import instrument

            if batch.symbols[: len(self.broker.symbols)] != self.broker.symbols:
                raise SourceError("行情证券 ID 与账户不一致")
            new = batch.symbols[len(self.broker.symbols) :]
            self.broker.config.instruments.update({s: instrument(s) for s in new})
            self._checked(self.broker.append_symbols(new, batch.received_at), "append")
        self.monitor.observe(batch)
        if self.store:
            self.store.received_at, self.store.source = batch.received_at, batch.source
            self.store.sent_at = (
                batch.sent_at if batch.sent_at is not None else batch.received_at
            )
        try:
            self.broker.feed(batch.quotes, received_at=batch.received_at)
        except (ValueError, OverflowError) as exc:
            # 原生错误仅含固定的本地校验信息，不含网络响应和密钥。
            raise SourceError(f"行情输入拒绝: {exc}") from None
        if self.store and not self.broker.journal:
            # 可选的独立录制出口；dry-run 默认不传 store。
            self.store.put(batch.received_at, batch.quotes)
        self._drain()

    def _prime(self, day: str, limits: list[dict]) -> None:
        # 只用当时已知的昨收/持仓标价；不得用尚未公布的竞价结果。
        if (
            not self.targets
            or time.fromisoformat(self.targets["at"]) >= self.schedule.table.continuous
        ):
            return
        state = json.loads(self.broker.dump())
        prices = {r["ts_code"]: r.get("pre_close") for r in limits}
        previous_days = sorted(d for d in self.schedule.open_days if d < day)
        if previous_days:
            previous = self.source.client.call(
                "daily", {"trade_date": previous_days[-1]}
            )
            prices.update({r["ts_code"]: r["close"] for r in previous})
        now = self.clock.now()
        records = [
            dict(
                ts_code=s,
                close=prices.get(s) or state["marks"][i] / 10_000,
                trade_time=now.isoformat(),
                vol=0,
            )
            for i, s in enumerate(self.broker.symbols)
        ]
        self.consume(
            Batch(
                timestamp(now),
                normalize(
                    records,
                    self.broker.symbols,
                    limits=self.source.limits,
                    suspended=self.source.suspended,
                    day=day,
                ),
                "preopen_reference",
            )
        )

    def finish(self, day: str, now: datetime, *, catch_up: bool = False) -> None:
        state = json.loads(self.broker.dump())
        if state["settled_day"] == int(day):
            self.settled_day = day
            self._reports()
            return
        if self.phase == "close_auction" and not catch_up:
            batch = self.source.poll()
            close_time = timestamp(
                datetime.combine(
                    datetime.strptime(day, "%Y%m%d").date(),
                    self.schedule.table.close,
                    TZ,
                )
            )
            observed = {i: q for i, q in enumerate(state["quotes"]) if q is not None}
            for row in batch.quotes:
                i = int(row[0])
                if i not in observed or row[1] >= observed[i][1]:
                    observed[i] = row.tolist()
            active = {
                i
                for i, name in enumerate(self.broker.symbols)
                if name in self.source.limits and name not in self.source.suspended
            }
            stale = {
                i for i in active if i not in observed or observed[i][1] < close_time
            }
            if stale and now.time().replace(tzinfo=None) < self.settlement_deadline:
                raise SourceError(f"尚未取得活跃证券的收盘快照: count={len(stale)}")
            self.stale_close = {day: [self.broker.symbols[i] for i in sorted(stale)]}
            self._save_context(day)
            import numpy as np

            final = []
            for i, row in sorted(observed.items()):
                row = list(row)
                baseline = (
                    self.close_baseline.get(i) if self.baseline_day == day else None
                )
                row[16] = (
                    max(0, int(row[7]) - baseline)
                    if baseline is not None and i not in stale
                    else 0
                )
                row[17] = 2
                final.append(row)
            batch.quotes = np.array(final, dtype=np.int64).reshape(-1, 18)
            batch.source = "rt_k_close_volume_proxy"
            self.consume(batch)
            log("closing_auction_proxy", day=day, symbols=len(self.close_baseline))
        elif catch_up:
            log(
                "settlement_catch_up",
                level=30,
                day=day,
                missing_auction="unobserved volume is not filled",
            )
        self._session(day, "closed")
        records = self.source.client.call("daily", {"trade_date": day})
        stale_names = set(self.stale_close.get(day, []))
        if (
            not records
            and set(self.broker.symbols) - stale_names - self.source.suspended
        ):
            raise SourceError("当日日线尚未发布，等待后重试结算")
        missing = set(self.broker.symbols) - {r["ts_code"] for r in records}
        active_missing = (
            missing & self.source.limits.keys() - self.source.suspended - stale_names
        )
        if active_missing:
            raise SourceError(f"活跃证券日线缺失: count={len(active_missing)}")
        now = self.clock.now().astimezone(TZ)
        quote_time = f"{datetime.strptime(day, '%Y%m%d').date()} 23:59:59"
        quotes = [
            r
            | dict(
                trade_time=quote_time,
                vol=shares(r["vol"], 100),
                amount=Decimal(str(r["amount"])) * 1000,
            )
            for r in records
            if r["ts_code"] not in stale_names
        ]
        self.consume(
            Batch(
                timestamp(now),
                normalize(
                    quotes,
                    self.broker.symbols,
                    limits=self.source.limits,
                    suspended=self.source.suspended,
                    day=day,
                ),
                "daily_close",
            )
        )
        self._checked(
            self.broker.settle(timestamp(self.clock.now()), int(day)), "settle"
        )
        if self.broker.journal:
            self.broker.checkpoint()
        self.settled_day = day
        self._reports()
        if self.broker.journal:
            self.broker.journal.gc()
        log("day_settled", day=day, equity=self.broker.account()["equity"])

    def _targets(self, day: str, now: datetime) -> None:
        if not self.targets:
            return
        local = now.time().replace(tzinfo=None)
        if local < time.fromisoformat(
            self.targets["at"]
        ) or local >= time.fromisoformat(self.targets.get("until", "14:56:50")):
            return
        if self.phase == "open_auction" and local >= time(9, 25):
            return
        if self.phase not in ("continuous", "open_auction"):
            return
        interval = max(1, float(self.targets.get("rebalance_seconds", 30)))
        if self.clock.monotonic() - self.last_target < interval:
            return
        self.last_target = self.clock.monotonic()
        results = self.broker.target_weights(
            timestamp(now),
            self.targets["weights"],
            protection_ticks=int(self.targets.get("protection_ticks", 2)),
        )
        for result in results:
            self._checked(result, "target_order")
        if results:
            self.target_day = day
        self.last_target = self.clock.monotonic()
        self._drain()

    def step(self) -> bool:
        self._reports()
        now = self.clock.now().astimezone(TZ)
        day = now.strftime("%Y%m%d")
        if day not in self.schedule.known_days:
            self.schedule.update(
                self.source.client.call(
                    "trade_cal",
                    dict(
                        exchange="SSE",
                        start_date=(now - timedelta(days=10)).strftime("%Y%m%d"),
                        end_date=(now + timedelta(days=40)).strftime("%Y%m%d"),
                    ),
                )
            )
            if day not in self.schedule.known_days:
                raise SourceError("交易日历未覆盖当前日期")
        if day not in self.schedule.open_days:
            return False
        phase = self.schedule.phase(now)
        if (
            self.metadata_day != day
            and now.time().replace(tzinfo=None) >= self.schedule.table.preopen
            and self.settled_day != day
        ):
            state = json.loads(self.broker.dump())
            if (
                state["day"] not in (0, int(day))
                and state["day"] != state["settled_day"]
            ):
                previous = str(state["day"])
                self.metadata(previous)
                self.finish(previous, now, catch_up=True)
            limits = self.metadata(day)
            new_day = json.loads(self.broker.dump())["day"] != int(day)
            if new_day:
                self._session(day, "preopen")
            self._catch_actions(day)
            self.metadata_day = day
            self.events = [e for e in self.broker.history if e["day"] == int(day)]
            self.auction_day = self.target_day = ""
            self.last_target = self.next_finish = self.next_auction = float("-inf")
            if self.baseline_day != day:
                self.close_baseline = {}
                self.baseline_day = day
            if new_day:
                self._prime(day, limits)
            now = self.clock.now().astimezone(TZ)
            phase = self.schedule.phase(now)
        if phase == "closed" and t_after_close(now, self.schedule):
            if self.settled_day != day and self.clock.monotonic() >= self.next_finish:
                self.next_finish = self.clock.monotonic() + 60
                self.finish(day, now)
            return False
        self._session(day, phase)
        t = now.time().replace(tzinfo=None)
        if (
            phase == "open_auction"
            and t >= self.schedule.table.auction_result
            and self.auction_day != day
            and self.clock.monotonic() >= self.next_auction
        ):
            self.next_auction = self.clock.monotonic() + 3
            auction = self.source.client.call("stk_auction", {"trade_date": day})
            records = [
                r
                | dict(
                    trade_time=f"{now.date()} 09:25:00",
                    close=r["price"],
                    open=r["price"],
                    auction_volume=r["vol"],
                )
                for r in auction
            ]
            quotes = normalize(
                records,
                self.broker.symbols,
                limits=self.source.limits,
                suspended=self.source.suspended,
                auction_kind=1,
                day=day,
            )
            if len(quotes):
                self.consume(Batch(timestamp(self.clock.now()), quotes, "stk_auction"))
                self.auction_day = day
        if phase in ("continuous", "close_auction"):
            batch = self.source.poll()
            if phase == "close_auction" and not self.close_baseline:
                self.close_baseline = {int(row[0]): int(row[7]) for row in batch.quotes}
                self._save_context(day)
            self.consume(batch)
            self._targets(day, self.clock.now().astimezone(TZ))
            log("heartbeat", day=day, phase=phase, symbols=len(self.broker.symbols))
            return True
        self._targets(day, now)
        return False

    def run(self, duration: float | None = None) -> None:
        start = self.clock.monotonic()
        while duration is None or self.clock.monotonic() - start < duration:
            before = self.clock.monotonic()
            try:
                self.step()
            except SourceError as exc:
                # SourceError 只包含本模块生成的原因/枚举，不包含服务端 msg。
                log("runner_source_failure", level=40, reason=str(exc))
            self.clock.sleep(
                max(0.1, self.source.interval - (self.clock.monotonic() - before))
            )


def t_after_close(now: datetime, schedule: Schedule) -> bool:
    return now.time().replace(tzinfo=None) >= schedule.table.close
