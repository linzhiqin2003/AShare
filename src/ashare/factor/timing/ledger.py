"""原子预约与追加台账：失败也消耗验证/确认名额，损坏记录拒绝放行。"""

import json
from pathlib import Path

import numpy as np

from ashare.data.crypto.storage import digest, encoded
from ashare.data.store import lock
from ashare.factor.stats import adjust
from ashare.ledger import Ledger


class StageLedger:
    def __init__(self, path, settings, data_version):
        path = Path(path).resolve()
        if str(path) != settings.value["ledger"]:
            raise ValueError("LEDGER_PATH: 台账位置与研究设定不一致")
        self.ledger = Ledger(path)
        self.settings = settings
        self.data_version = data_version
        self.lock_path = Path(str(path) + ".gate.lock")

    def records(self):
        records = self.ledger.records()
        if self.ledger.skipped_lines:
            raise ValueError("阶段台账损坏；拒绝绕过闸门")
        for r in records:
            if not isinstance(r, dict):
                raise ValueError("阶段台账记录不完整")
            # 挖掘完成项只是可丢弃缓存；闸门事件仍须结构完整、失败关闭。
            if r.get("stage") == "mining" and r.get("event") == "completed":
                if not r.get("test_hash") and not (
                    isinstance(r.get("result"), dict) and r["result"].get("test_hash")
                ):
                    raise ValueError(
                        "LEDGER_IDENTITY: 旧台账缺少检验身份，拒绝重置闸门"
                    )
                continue
            if (
                not {
                    "event",
                    "stage",
                    "settings_hash",
                    "dates_hash",
                    "evaluation_hash",
                    "data_version",
                }
                <= r.keys()
            ):
                raise ValueError("阶段台账记录不完整")
            if (
                r["stage"] in ("mining", "validation")
                and not r.get("test_hash")
                or r["stage"] == "confirmation"
                and not r.get("tests")
            ):
                raise ValueError("LEDGER_IDENTITY: 旧台账缺少检验身份，拒绝重置闸门")
        if any(
            r["stage"] in ("validation", "confirmation")
            and r["dates_hash"] != self.settings.dates_hash
            for r in records
        ):
            raise ValueError("已有验证/确认记录，不能修改阶段日期")
        return records

    def append(self, stage, event, **values):
        return self.ledger.append(
            dict(
                stage=stage,
                event=event,
                settings_hash=self.settings.hash,
                dates_hash=self.settings.dates_hash,
                evaluation_hash=self.settings.evaluation_hash,
                data_version=self.data_version,
                **values,
            )
        )

    def reserve(self, stage, tests, *, list_hash=None):
        """必须在取得结果之前预约；进程异常也不允许再次试验。"""
        records = self.records()
        if stage == "validation" and any(
            r["stage"] == stage and r.get("test_hash") in tests for r in records
        ):
            raise ValueError("验证候选只能运行一次")
        if stage == "confirmation" and any(r["stage"] == stage for r in records):
            raise ValueError("确认名单只能运行一次")
        return self.append(
            stage,
            "reserved",
            tests=list(tests),
            test_hash=tests[0] if len(tests) == 1 else None,
            list_hash=list_hash,
        )

    @staticmethod
    def valid_result_record(r):
        """检查缓存结构和内外身份；不提供抵抗私有文件主动篡改的认证。"""
        try:
            return StageLedger._valid_result_record(r)
        except (TypeError, ValueError, OverflowError):
            # JSON 数字也可能超出统计库的表示范围；坏缓存必须退化为未命中。
            return False

    @staticmethod
    def _valid_result_record(r):
        required = {
            "stage",
            "event",
            "settings_hash",
            "dates_hash",
            "evaluation_hash",
            "data_version",
            "definition_hash",
            "test_hash",
            "family",
            "result",
            "time",
        }
        if (
            not required <= r.keys()
            or r["stage"] != "mining"
            or r["event"] != "completed"
        ):
            return False
        result = r["result"]
        if not isinstance(result, dict):
            return False
        identity = ("definition_hash", "test_hash", "data_version", "settings_hash")
        for key in (*identity, "dates_hash", "evaluation_hash"):
            value = r[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                return False
        if any(result.get(key) != r[key] for key in identity):
            return False
        versions = result.get("data_versions")
        if not isinstance(versions, dict) or set(versions) != {
            "hourly",
            "minute",
            "metrics",
        }:
            return False
        if not isinstance(versions["hourly"], str) or any(
            v is not None and not isinstance(v, str) for v in versions.values()
        ):
            return False
        if digest(encoded(versions)) != r["data_version"]:
            return False
        if not isinstance(r["family"], str) or result.get("family") != r["family"]:
            return False
        integers = ("n", "meat_terms", "repetitions", "horizon", "lookahead_checks")
        if any(type(result.get(k)) is not int or result[k] < 0 for k in integers):
            return False

        def number(value):
            return type(value) in (int, float) and not np.isinf(value)

        if any(not number(result.get(k)) for k in ("alpha", "beta", "se", "t", "p")):
            return False
        if np.isfinite(result["p"]) and not 0 <= result["p"] <= 1:
            return False
        for key, numeric in (
            ("normalization", ("mean", "std")),
            ("descriptive", ("spearman",)),
        ):
            block = result.get(key)
            if (
                not isinstance(block, dict)
                or type(block.get("n")) is not int
                or block["n"] < 0
                or any(not number(block.get(k)) for k in numeric)
            ):
                return False
        return (
            result.get("inference") == "Hodrick-1B/Berger-Boos"
            and type(result.get("recomputation")) is bool
        )

    def cached(self, test_hash, definition_hash):
        records = self.records()
        warned = {
            r.get("record_hash") for r in records if r.get("event") == "cache_warning"
        }
        expected = dict(
            test_hash=test_hash,
            definition_hash=definition_hash,
            data_version=self.data_version,
            settings_hash=self.settings.hash,
        )
        for r in reversed(records):
            if r["stage"] != "mining" or r["event"] != "completed":
                continue
            inner = r.get("result")
            inner = inner if isinstance(inner, dict) else {}
            if test_hash not in (r.get("test_hash"), inner.get("test_hash")) and not (
                definition_hash
                in (r.get("definition_hash"), inner.get("definition_hash"))
                and r.get("evaluation_hash") == self.settings.evaluation_hash
            ):
                continue
            valid = self.valid_result_record(r) and all(
                r.get(k) == v and inner.get(k) == v for k, v in expected.items()
            )
            valid = (
                valid
                and r.get("dates_hash") == self.settings.dates_hash
                and r.get("evaluation_hash") == self.settings.evaluation_hash
            )
            if valid:
                return r["result"]
            record_hash = digest(json.dumps(r, sort_keys=True, allow_nan=True).encode())
            if record_hash not in warned:
                self.append(
                    "mining",
                    "cache_warning",
                    code="CACHE_INCONSISTENT",
                    test_hash=test_hash,
                    definition_hash=definition_hash,
                    record_hash=record_hash,
                )
                warned.add(record_hash)
            return None
        return None

    def mining_summary(self):
        # 每检验一个名额；修订数据后取该检验最近一次结果，不把重跑视作新尝试。
        latest = {}
        for r in self.records():
            if (
                r["stage"] == "mining"
                and r.get("test_hash")
                and r["event"] in ("started", "failed", "completed")
            ):
                if r["event"] == "completed" and not self.valid_result_record(r):
                    continue
                latest.setdefault(r["test_hash"], np.nan)
                if r["event"] in ("started", "failed"):
                    latest[r["test_hash"]] = np.nan
                if r["event"] == "completed":
                    latest[r["test_hash"]] = (
                        r["result"]["p"] if self.valid_result_record(r) else np.nan
                    )
        keys = sorted(latest)
        q = adjust([latest[k] for k in keys], "bh")
        return dict(attempts=len(keys), q=dict(zip(keys, q.tolist(), strict=True)))

    def locked(self):
        return lock(self.lock_path)
