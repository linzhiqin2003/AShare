"""示例因子集的独立参考实现（pandas），输入是 ``inputs.reference_frames`` 的结果。"""

import numpy as np


def _finite(frame):
    return frame.where(np.isfinite(frame))


def example_range_mean(frames):
    ratio = _finite(
        (frames["high"] - frames["low"])
        / (0.5 * frames["open"] + 0.5 * frames["close"])
    )
    return ratio.rolling(9).mean()


def example_volume_turn(frames):
    change = _finite(np.log(frames["volume"])).diff(3)
    direction = np.sign(frames["close"].diff(3))
    return change.rank(axis=1, pct=True) * direction


REFERENCES = {
    "example_range_mean": example_range_mean,
    "example_volume_turn": example_volume_turn,
}
