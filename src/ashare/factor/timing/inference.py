"""阶段标签和统计：复用 C6，校准只取挖掘期，尾部跨界收益始终屏蔽。"""

import numpy as np

from ashare.data.crypto.timeutil import DAY, HOUR
from ashare.factor.crypto_eval import (
    SimulationBank,
    calibrate,
    conservative_test,
    crypto_labels,
    descriptive,
    hodrick_1b,
)

# 标签公式变更必须更换此版本，避免不同统计假设合并成一次尝试。
LABEL_DEFINITION = "c6-usdm-open-additive-log-return-with-funding-v1"


def stage_mask(entries, horizon, start, end):
    entries = np.asarray(entries)
    return (entries >= start) & (entries + horizon * DAY <= end)


class LabelSource:
    """标签专用的终点开盘价读取器；不交给因子函数。"""

    def __init__(self, dataset, end):
        self._dataset = dataset
        self._end = end

    def arrays(self, fields, *, instruments):
        names = [name for name in instruments if name.startswith("usdm:")]
        source = self._dataset.arrays(fields, instruments=names, end=self._end + HOUR)
        # C6 同时输出 spot/usdm；C8 只用永续，现货列明确为缺失，不额外读取现货。
        return dict(
            time=source["time"],
            **{
                field: np.column_stack(
                    [
                        source[field][:, names.index(name)]
                        if name in names
                        else np.full(
                            len(source["time"]),
                            0 if field in ("quality", "tradable") else np.nan,
                        )
                        for name in instruments
                    ]
                )
                for field in fields
            },
        )


def labels(dataset, values, settings, stage):
    start, end = settings.bounds(stage)
    cfg = settings.value
    at, x = values
    # 先建连续日历轴。缺信号日保留 NaN，不能压缩 g 造成虚假的邻接。
    daily = at % DAY == cfg["trade_hour"] * HOUR
    at, x = at[daily], x[daily]
    if not len(at):
        raise ValueError("没有日频信号")
    calendar = np.arange(at[0], end, DAY, dtype=np.int64)
    full_x = np.full(len(calendar), np.nan)
    indices = np.searchsorted(calendar, at)
    valid = indices < len(calendar)
    full_x[indices[valid]] = x[valid]
    result = crypto_labels(
        LabelSource(dataset, end),
        [cfg["symbol"]],
        (calendar, full_x[:, None]),
        trade_hour=cfg["trade_hour"],
        horizons=(1, cfg["horizon"]),
    )
    one_day = stage_mask(result["entry"], 1, start, end)
    daily_g = np.where(one_day, result["daily"]["usdm"][:, 0], np.nan)
    mask = stage_mask(result["entry"], cfg["horizon"], start, end)
    y = np.where(mask, result["descriptive"][cfg["horizon"]]["usdm"][:, 0], np.nan)
    return dict(entries=result["entry"], x=full_x, g=daily_g, y=y, mask=mask)


def calibration(sample, settings):
    # C6 保持 < boundary；C8 允许退出==阶段结束，故上界加 1ms。
    start, end = settings.bounds("mining")
    inside = stage_mask(sample["entries"], 1, start, end)
    return calibrate(
        np.where(inside, sample["x"], np.nan),
        np.where(inside, sample["g"], np.nan),
        sample["entries"],
        boundary=end + 1,
    )


def estimate(sample, calibration_sample, settings, *, confirmation=False):
    cfg = settings.value
    # g 已逐日按阶段范围屏蔽；不把 h 日 mask 作为 g mask 再裁一次尾部。
    x = np.where(sample["mask"], sample["x"], np.nan)
    observed = hodrick_1b(x, sample["g"], cfg["horizon"])
    result = dict(
        **observed,
        descriptive=descriptive(x, sample["y"]),
        p=np.nan,
        repetitions=cfg["repetitions"],
        horizon=cfg["horizon"],
        inference="Hodrick-1B/Berger-Boos",
    )
    # 定义变化/数据修订后重算，但不以验证期或确认期估 δ。
    try:
        cal = calibration(calibration_sample, settings)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    result["calibration"] = {
        k: np.asarray(v).tolist()
        for k, v in cal.items()
        if k not in ("eps", "delta_grid")
    }
    if observed["n"] < 100 or not np.isfinite(observed["t"]):
        result["reason"] = "有效样本不足 100 或统计量退化"
        return result
    # 仅取阶段单日收益域；尾部无完整 h 日标签的 x 必须在观察与模拟中都排除。
    domain = np.flatnonzero(np.isfinite(sample["g"]))
    if not len(domain):
        return result
    take = slice(domain[0], domain[-1] + 1)
    xe, ge = x[take], sample["g"][take]
    observed = hodrick_1b(xe, ge, cfg["horizon"])
    result.update(observed)
    bank = SimulationBank(
        cal["eps"], len(xe), repetitions=cfg["repetitions"], seed=cfg["seed"]
    )
    _, details = conservative_test(
        bank,
        np.isfinite(xe),
        np.isfinite(ge),
        cfg["horizon"],
        cal["delta_grid"],
        observed["t"],
        full_refinement=confirmation,
    )
    result.update(details)
    return result
