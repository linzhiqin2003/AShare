"""单标的择时研究入口；真实研究的名单、设定、台账由调用方显式传入。"""

import numpy as np

from ashare.data.crypto import metrics, minute
from ashare.data.crypto.storage import digest, encoded
from ashare.factor.stats import adjust

from . import backtest
from .definition import require_factor
from .inference import estimate, labels
from .ledger import StageLedger
from .settings import Settings, read_list
from .view import View, lookahead_check


class Study:
    def __init__(
        self,
        settings,
        dataset,
        *,
        ledger=None,
        minute_root=None,
        minute_version=None,
        metrics_root=None,
        metrics_version=None,
    ):
        self._settings = (
            settings if isinstance(settings, Settings) else Settings.load(settings)
        )
        self._dataset = dataset
        self._minute = (
            None
            if minute_root is None
            else (
                minute.data_root(minute_root),
                minute.metadata(root=minute_root, version=minute_version)["version"],
            )
        )
        self._metrics = (
            None
            if metrics_root is None
            else (
                metrics.data_root(metrics_root),
                metrics.metadata(root=metrics_root, version=metrics_version)["version"],
            )
        )
        self.data_versions = dict(
            hourly=dataset.version,
            minute=self._minute[1] if self._minute else None,
            metrics=self._metrics[1] if self._metrics else None,
        )
        self.data_version = digest(encoded(self.data_versions))
        self.ledger = StageLedger(
            self.settings.value["ledger"] if ledger is None else ledger,
            self.settings,
            self.data_version,
        )
        self.ledger.records()

    @property
    def settings(self):
        return self._settings

    def _view(self, stage):
        _, end = self.settings.bounds(stage)
        symbol = self.settings.value["symbol"]
        frame = self._dataset.read(end=end, instruments=["usdm:" + symbol])
        return View(
            frame,
            symbol=symbol,
            cutoff=end,
            minute_source=self._minute,
            metrics_source=self._metrics,
        )

    def view(self, stage="mining"):
        self.ledger.records()
        if stage != "mining":
            raise ValueError("验证/确认 view 只能在名单闸门预约后由评估入口提供")
        return self._view(stage)

    def _evaluate(self, factor, stage):
        values, cuts = lookahead_check(factor, self._view(stage))
        sample = labels(self._dataset, values, self.settings, stage)
        if stage == "mining":
            mining = sample
            mine_values = values
        else:
            mine_values, _ = lookahead_check(factor, self._view("mining"))
            mining = labels(self._dataset, mine_values, self.settings, "mining")
        result = estimate(
            sample, mining, self.settings, confirmation=stage == "confirmation"
        )
        lo, hi = self.settings.bounds("mining")
        at, x = mine_values
        cfg = self.settings.value
        use = (
            (at >= lo)
            & (at < hi)
            & (at % 86_400_000 == cfg["trade_hour"] * 3_600_000)
            & np.isfinite(x)
        )
        result["normalization"] = dict(
            mean=float(np.mean(x[use])) if use.any() else np.nan,
            std=float(np.std(x[use], ddof=1)) if use.sum() > 1 else np.nan,
            n=int(use.sum()),
        )
        result.update(
            definition_hash=factor.definition_hash,
            test_hash=self.test_hash(factor),
            family=factor.family,
            data_versions=self.data_versions,
            data_version=self.data_version,
            settings_hash=self.settings.hash,
            lookahead_checks=len(cuts),
        )
        return result, values

    def test_hash(self, factor):
        return self.settings.test_hash(require_factor(factor))

    def evaluate(self, factor, *, stage="mining", list_path=None):
        definition_hash = require_factor(factor)
        test_hash = self.settings.test_hash(definition_hash)
        if stage not in ("mining", "validation"):
            raise ValueError("确认期必须用 confirm 一次运行整份名单")
        with self.ledger.locked():
            self.ledger.records()
            if stage == "mining":
                cached = self.ledger.cached(test_hash, definition_hash)
                if cached is not None:
                    return cached
                previous = next(
                    (
                        r["data_version"]
                        for r in reversed(self.ledger.records())
                        if r["stage"] == "mining"
                        and r.get("test_hash") == test_hash
                        and r.get("definition_hash") == definition_hash
                        and (
                            r["event"] in ("started", "failed")
                            or (
                                r["event"] == "completed"
                                and self.ledger.valid_result_record(r)
                            )
                        )
                    ),
                    None,
                )
                recomputation = previous is not None and previous != self.data_version
                self.ledger.append(
                    stage,
                    "started",
                    recomputation=recomputation,
                    definition_hash=definition_hash,
                    test_hash=test_hash,
                    family=factor.family,
                )
            else:
                if list_path is None:
                    raise ValueError("验证期必须提供名单文件")
                expected = self.settings.value["validation_hash"]
                candidates = read_list(list_path, expected, 10)
                if test_hash not in candidates:
                    raise ValueError("候选不在验证名单")
                if self.ledger.cached(test_hash, definition_hash) is None:
                    raise ValueError("候选缺少同设定和数据版本的挖掘记录")
                self.ledger.reserve(stage, [test_hash], list_hash=expected)
            try:
                result, _ = self._evaluate(factor, stage)
                if stage == "mining":
                    result["recomputation"] = recomputation
            except Exception as exc:
                self.ledger.append(
                    stage,
                    "failed",
                    definition_hash=definition_hash,
                    test_hash=test_hash,
                    family=factor.family,
                    error=type(exc).__name__,
                )
                raise
            self.ledger.append(
                stage,
                "completed",
                definition_hash=definition_hash,
                test_hash=test_hash,
                family=factor.family,
                result=result,
            )
            return result

    def confirm(self, factors, *, list_path):
        factors = tuple(factors)
        identities = [self.test_hash(f) for f in factors]
        with self.ledger.locked():
            records = self.ledger.records()
            expected = self.settings.value["confirmation_hash"]
            candidates = read_list(list_path, expected, 5)
            lookup = dict(zip(identities, factors, strict=True))
            if len(lookup) != len(factors) or set(lookup) != set(candidates):
                raise ValueError("提供的定义与最终名单不一致")
            mining = []
            for key in candidates:
                previous = self.ledger.cached(key, lookup[key].definition_hash)
                if previous is None or not any(
                    r["stage"] == "validation"
                    and r["event"] == "completed"
                    and r.get("test_hash") == key
                    and r["data_version"] == self.data_version
                    and r["evaluation_hash"] == self.settings.evaluation_hash
                    for r in records
                ):
                    raise ValueError("最终候选缺少本数据版本的挖掘/验证记录")
                mining.append(previous)
            self.ledger.reserve("confirmation", candidates, list_hash=expected)
            try:
                results, series = [], []
                for key in candidates:
                    result, values = self._evaluate(lookup[key], "confirmation")
                    results.append(result)
                    series.append(values)
                adjusted = adjust([r["p"] for r in results], "holm")
                for row, p in zip(results, adjusted, strict=True):
                    row["holm_p"] = float(p)
                portfolio = backtest.run(
                    self._dataset,
                    series,
                    [r["normalization"] for r in mining],
                    self.settings,
                )
                result = dict(
                    candidates=results,
                    family_size=len(candidates),
                    strategies=portfolio,
                )
            except Exception as exc:
                self.ledger.append(
                    "confirmation",
                    "failed",
                    tests=candidates,
                    error=type(exc).__name__,
                )
                raise
            self.ledger.append(
                "confirmation", "completed", tests=candidates, result=result
            )
            return result

    def summary(self):
        return self.ledger.mining_summary()
