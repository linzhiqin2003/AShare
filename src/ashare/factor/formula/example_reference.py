"""示例因子集的独立参考实现（pandas），输入是 ``inputs.reference_frames`` 的结果。"""

import numpy as np


def _finite(frame):
    return frame.where(np.isfinite(frame))


def example_spread_mean(frames):
    ratio = _finite(
        (frames["vwap"] - frames["open"]) / (frames["high"] + frames["low"])
    )
    return ratio.rolling(6).mean()


def example_amount_peak(frames):
    peak = frames["amount"].rolling(4).max()
    return peak.rank(axis=1, pct=True) * np.sign(frames["vwap"] - frames["open"])


REFERENCES = {
    "example_spread_mean": example_spread_mean,
    "example_amount_peak": example_amount_peak,
}
