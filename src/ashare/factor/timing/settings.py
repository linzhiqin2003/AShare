"""研究设定：规范化日期、完整内容哈希与阶段日期签名。"""

import json
from dataclasses import dataclass
from pathlib import Path

from ashare.data.crypto.storage import digest, encoded
from ashare.data.crypto.timeutil import HOUR, millis

STAGES = ("mining", "validation", "confirmation")


@dataclass(frozen=True)
class Settings:
    payload: bytes

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if "ledger" in value and not Path(value["ledger"]).is_absolute():
            value["ledger"] = str(Path(path).resolve().parent / value["ledger"])
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value):
        value = json.loads(encoded(value))
        required = {
            "ledger",
            "symbol",
            "trade_hour",
            "horizon",
            "stages",
            "validation_hash",
            "confirmation_hash",
        }
        if set(value) - required - {
            "repetitions",
            "seed",
            "assume",
            "initial_cash",
        } or required - set(value):
            raise ValueError("研究设定字段缺失或未知")
        if (
            not isinstance(value["ledger"], str)
            or not Path(value["ledger"]).is_absolute()
        ):
            raise ValueError("LEDGER_PATH: from_dict 的台账位置必须为绝对路径")
        value["ledger"] = str(Path(value["ledger"]).resolve())
        if not isinstance(value["symbol"], str) or not value["symbol"].isalnum():
            raise ValueError("标的非法")
        if (
            type(value["trade_hour"]) is not int
            or not 0 <= value["trade_hour"] < 24
            or type(value["horizon"]) is not int
            or value["horizon"] < 1
        ):
            raise ValueError("调仓时点或持有期非法")
        if set(value["stages"]) != set(STAGES):
            raise ValueError("必须设置挖掘/验证/确认三段")
        previous = None
        for stage in STAGES:
            dates = value["stages"][stage]
            if len(dates) != 2:
                raise ValueError("每段必须为 [开始,结束)")
            a, b = map(millis, dates)
            if (
                a >= b
                or a % HOUR
                or b % HOUR
                or (previous is not None and a < previous)
            ):
                raise ValueError("阶段必须按整点递增且不重叠")
            value["stages"][stage] = [a, b]
            previous = b
        for stage in ("validation", "confirmation"):
            sha = value[stage + "_hash"]
            if sha is not None and (
                not isinstance(sha, str)
                or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)
            ):
                raise ValueError("名单哈希非法")
        value.setdefault("repetitions", 4000)
        value.setdefault("seed", 6182026)
        value.setdefault("assume", None)
        value.setdefault("initial_cash", 10000.0)
        if (
            type(value["repetitions"]) is not int
            or value["repetitions"] < 1
            or type(value["seed"]) is not int
            or value["seed"] < 0
        ):
            raise ValueError("模拟设定非法")
        if value["assume"] not in (None, "conservative", "previous") or not 0 < value[
            "initial_cash"
        ] < float("inf"):
            raise ValueError("回测设定非法")
        return cls(encoded(value))

    @property
    def value(self):
        return json.loads(self.payload)

    @property
    def hash(self):
        return digest(self.payload)

    @property
    def dates_hash(self):
        return digest(encoded(self.value["stages"]))

    @property
    def evaluation_hash(self):
        return digest(
            encoded({k: v for k, v in self.value.items() if not k.endswith("_hash")})
        )

    def test_hash(self, definition_hash):
        from .inference import LABEL_DEFINITION

        return digest(
            encoded(
                dict(
                    definition_hash=definition_hash,
                    symbol=self.value["symbol"],
                    trade_hour=self.value["trade_hour"],
                    horizon=self.value["horizon"],
                    label=LABEL_DEFINITION,
                    stages=self.value["stages"],
                )
            )
        )

    def bounds(self, stage):
        if stage not in STAGES:
            raise ValueError("未知阶段")
        return tuple(self.value["stages"][stage])


def read_list(path, expected_hash, limit):
    payload = Path(path).read_bytes()
    if expected_hash is None or digest(payload) != expected_hash:
        raise ValueError("名单哈希不符")
    values = json.loads(payload)
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= limit
        or any(
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError("名单必须为不重复的检验哈希列表，且不超过名额")
    return values
