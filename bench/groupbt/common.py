"""分组回测逐位对照共用工具；真实数据与结果只能写入被忽略的 .cache。"""

from itertools import product

import numpy as np


class Panel:
    def __init__(self, arrays, dates, symbols):
        self.arrays, self.dates, self.symbols = arrays, dates, symbols

    def __getitem__(self, name):
        return self.arrays[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


def synthetic(seed=1701, n=139):
    rng = np.random.default_rng(seed)
    dates = np.concatenate(
        [
            np.arange("2015-01-01", "2015-09-08", dtype="datetime64[D]"),
            np.arange("2024-11-01", "2024-11-26", dtype="datetime64[D]"),
            np.arange("2025-01-01", "2025-01-26", dtype="datetime64[D]"),
        ]
    )
    dates = np.char.replace(dates.astype(str), "-", "")
    shape = (len(dates), n)
    close = 10 * np.exp(rng.normal(0, 0.025, shape).cumsum(axis=0))
    op = close * np.exp(rng.normal(0, 0.01, shape))
    listed = np.ones(shape, bool)
    listed[:260, -5:] = False
    susp = rng.random(shape) < 0.07
    quote = listed & ~susp
    close[~quote] = np.nan
    op[~quote] = np.nan
    adj = np.ones(shape)
    adj[260:, : n // 3] = 1.25
    cols = np.arange(n)[None, :]
    # 成分随日改变，停牌、涨跌停与池变动重叠，覆盖冻结旧持仓。
    moving = (cols + np.arange(shape[0])[:, None] // 7) % n
    arrays = dict(
        open=op,
        close=close,
        adj_factor=adj,
        total_market_cap=np.exp(rng.normal(23, 1.5, shape)),
        industry=rng.integers(1, 6, shape).astype(np.uint32),
        is_listed=listed,
        is_suspended=susp,
        has_quote=quote,
        upper_limit=np.where(rng.random(shape) < 0.1, close, close * 1.1),
        lower_limit=np.where(rng.random(shape) < 0.1, close, close * 0.9),
        can_buy_open=quote & (rng.random(shape) > 0.1),
        can_sell_open=quote & (rng.random(shape) > 0.1),
        member_csi300=moving < n // 4,
        member_csi500=(moving >= n // 4) & (moving < n // 2),
        member_csi1000=(moving >= n // 2) & (moving < 3 * n // 4),
    )
    factor = rng.normal(size=shape)
    factor[rng.random(shape) < 0.04] = np.nan
    factor[251] = np.nan
    factor[258, :12] = 0.125  # 稳定排序的并列
    factor[265, :3] = [np.inf, -np.inf, np.nan]
    arrays["industry"][263, :4] = 0
    arrays["total_market_cap"][266, :4] = 0
    return Panel(arrays, dates, np.array([f"S{i}" for i in range(n)])), factor


def cases(*, long_windows=False):
    from ashare.factor.groupbt import IS, VALIDATION

    keys = (
        "universe",
        "group_neutralize",
        "decay",
        "trade_at",
        "weights_portfolio",
        "sub_universes",
        "window",
    )
    configs = [
        dict(zip(keys, values, strict=True))
        for values in product(
            ("TOP800", "TOP1800", "ALL"),
            (None, "industry", "cap", "cap+industry"),
            (0, 3),
            ("open", "close"),
            (False, True),
            (False, True),
            (IS, VALIDATION),
        )
    ]
    if long_windows:
        # 193 = 3 * 64 + 1；300 行还覆盖 240 日风险历史与不完整末块。
        configs.extend(
            dict(
                universe="ALL",
                group_neutralize=mode,
                decay=3,
                trade_at=trade_at,
                weights_portfolio=weighted,
                sub_universes=True,
                window=window,
            )
            for mode, trade_at, weighted, window in product(
                (None, "industry", "cap", "cap+industry"),
                ("open", "close"),
                (False, True),
                (("20150101", "20150713"), ("20150101", "20260101")),
            )
        )
    return configs


def snapshot(result):
    """保存所有公开数据、派生表/曲线与文本；浮点保留 NaN 位和有符号零。"""
    out = {}

    def add(path, value):
        if isinstance(value, dict):
            out[path + "/keys"] = np.asarray(list(value), dtype="U")
            for key, v in value.items():
                add(path + "/" + str(key), v)
        elif isinstance(value, (list, tuple)):
            out[path + "/length"] = np.asarray(len(value))
            for i, v in enumerate(value):
                add(path + "/" + str(i), v)
        else:
            out[path] = np.asarray("<None>" if value is None else value)
            assert not out[path].dtype.hasobject

    def visit(r, prefix):
        for name in (
            "universe",
            "trade_at",
            "n_groups",
            "dates",
            "group_pnl",
            "pnl",
            "turnover",
            "exposure",
            "by",
        ):
            add(prefix + name, getattr(r, name))
        for name in r.pnl:
            for by in ("year", "quarter"):
                table = r.table(name, by=by)
                add(prefix + f"table/{name}/{by}", table.to_dicts())
            add(prefix + f"curves/{name}", r.curves(name))
        for name in ("exposure_mean", "to_dict", "text", "chart", "html"):
            add(prefix + name, getattr(r, name)())
        add(prefix + "subnames", list(r.subs))
        for name, sub in r.subs.items():
            visit(sub, prefix + "subs/" + name + "/")

    visit(result, "")
    return out


def compare(expected, got):
    if set(expected) != set(got):
        raise AssertionError("输出字段不同")
    for key, a in expected.items():
        b = got[key]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            error = None
            if a.shape == b.shape and a.dtype.kind == b.dtype.kind == "f":
                valid = np.isfinite(a) & np.isfinite(b)
                error = float(np.max(np.abs(a[valid] - b[valid]), initial=0))
            # 只给误差和字段，不打印输入或行情收益值。
            raise AssertionError(f"逐位不一致：{key}，最大绝对误差={error}")


def real():
    import panelops

    import ashare.data
    from ashare.factor.groupbt import FIELDS

    panel = ashare.data.open().panel(FIELDS)
    close = panel["close"] * panel["adj_factor"]
    returns = close / panelops.ts_delay(close, 1) - 1
    factor = panelops.ts_std(returns, 20)
    return panel, factor
