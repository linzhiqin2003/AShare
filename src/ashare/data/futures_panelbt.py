"""期货面板 → ``panelbt.Market`` / ``panelbt.Futures`` 的输入。

最短例子（纯期货组合）::

    import ashare.data as ad
    import panelbt as bt
    from ashare.data.futures_panelbt import panelbt_inputs

    fut = ad.open("/path/to/root").futures()
    inputs = panelbt_inputs(fut, products=["IF"], start="2024-01-02", end="2024-12-31")
    market = bt.Market(**inputs["market"], futures=inputs["futures"])
    # inputs["dates"], inputs["symbols"] 是行和列的标签

数值约定（与 panelbt SPEC 的 Futures 一节一致）：

- 执行价 = 开盘价（境内为夜盘开盘），没有行情为 NaN（不成交）；收盘价 = close。
- 买卖掩码 = 当天有行情且在上市期内；源里没有涨跌停幅度，停板不成交无法表达（见文档）。
- 结算价在合约内向前填补；挂牌前的占位行用首个结算价（panelbt 要求每行为正，只用于计数）。
- 到期行 = 最后交易日所在行；最后交易日晚于样本末尾时为 T。到期行的结算价就是源在最后交易日
  给的结算价：中金所股指期货是交割结算价，商品期货是普通结算价（实物交割品种不应持有到期）。
- 合约乘数取面板 ``multiplier`` 字段（按日期）。panelbt 每个合约只收一个乘数；区间内改过交易单位的
  合约（PB1309–1406、FB2001–2006 等）默认报错，``multiplier_change="split"`` 时在分界日拆成两列：
  前一列以分界日前一交易日为到期行（按结算价了结），后一列从分界日起可交易，代码写成
  ``PB1309.SHFE@20130902``。交易所实际是按比例折算持仓，拆列后需要策略自己重新开仓：分界日前一天
  结算价到分界日开盘价之间的跳空不归任何一列。返回的 ``splits`` 列出每个分界（同一合约的前后两列
  和分界日），调用方据此把持仓从前一列挪到后一列；不拆列时为空列表。
- 保证金率默认取交易所投机保证金率 max(买, 卖)；手续费默认取交易所的开仓手续费（比例与每手
  两部分，panelbt 不区分开平和平今），``fee_scale`` 用来乘期货公司加收倍数。
- 参数缺值不用以后的值回填（那是前视）。``param_fill="previous"``（默认）在合约内用此前最近的
  有效值向前填补，``"none"`` 不填。仍缺的上市日格子用 ``missing`` 里给的值（标量，或
  「品种 → 值」的字典，键为 margin_rate / fee_rate / fee_per_contract），否则报错并列出合约。
  不在上市期的格子不可能有持仓，填 0。
- 显式传 ``margin_rate`` / ``fee_rate`` / ``fee_per_contract`` 时整列替换源值（标量、长度 N 或 T×N）。
"""

import numpy as np
import polars as pl

FIELDS = (
    "open",
    "close",
    "settlement",
    "volume",
    "has_quote",
    "is_listed",
    "multiplier",
    "margin_long",
    "margin_short",
    "fee_open_rate",
    "fee_open_per_contract",
)
FILLS = ("previous", "none")


def _fill(column):
    """一维：只向前填补（用此前最近的有效值），开头的缺值保持为空。"""
    out = column.copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    index = np.where(valid, np.arange(len(out)), -1)
    np.maximum.accumulate(index, out=index)
    seen = index >= 0
    out[seen] = column[index[seen]]
    return out


def fill_columns(matrix):
    """逐列只向前填补；不回填开头。"""
    return (
        np.column_stack([_fill(matrix[:, j]) for j in range(matrix.shape[1])])
        if matrix.size
        else matrix.copy()
    )


def _settlement(matrix):
    """结算价向前填补；首个结算价之前的行（尚无行情，不可能有持仓）用首个结算价占位。"""
    out = fill_columns(matrix)
    for j in range(out.shape[1]):
        valid = np.isfinite(out[:, j])
        if valid.any():
            first = int(np.argmax(valid))
            out[:first, j] = out[first, j]
    return out


def _rate(name, value, panel_value, symbols, products, listed, fill, missing):
    if value is not None:
        return value
    matrix = fill_columns(panel_value) if fill == "previous" else panel_value.copy()
    fallback = (missing or {}).get(name)
    if fallback is not None:
        if isinstance(fallback, dict):
            column = np.array([fallback.get(p, np.nan) for p in products], dtype=float)
        else:
            column = np.full(len(products), float(fallback))
        matrix = np.where(np.isfinite(matrix), matrix, column[None, :])
    bad = (~np.isfinite(matrix) & listed).any(axis=0)
    if bad.any():
        raise ValueError(
            f"{name}: 这些合约在上市期内有缺值 {list(np.asarray(symbols)[bad][:10])}；"
            f"用 missing={{'{name}': ...}} 补缺（不会用以后的值回填），或显式传入 {name}"
        )
    # 不在上市期的格子由调用处统一填 0（拆列后的列按各自的段）。
    return matrix


def _segments(multiplier, listed):
    """每列按乘数变化切段：[(列, 起始行, 结束行(不含), 乘数)]；首段从第 0 行起。"""
    t, n = multiplier.shape
    out = []
    for j in range(n):
        rows = np.flatnonzero(listed[:, j] & np.isfinite(multiplier[:, j]))
        if not len(rows):
            raise ValueError("合约在区间内没有上市日")
        values = multiplier[rows, j]
        cuts = [int(rows[k]) for k in range(1, len(rows)) if values[k] != values[k - 1]]
        starts, ends = [0, *cuts], [*cuts, t]
        firsts = [values[0], *(multiplier[c, j] for c in cuts)]
        out += [
            (j, lo, hi, float(v))
            for lo, hi, v in zip(starts, ends, firsts, strict=True)
        ]
    return out


def panelbt_inputs(
    fut,
    *,
    products=None,
    symbols=None,
    start=None,
    end=None,
    margin_rate=None,
    fee_rate=None,
    fee_per_contract=None,
    fee_scale=1.0,
    delivery_fee_rate=0.0,
    delivery_fee_per_contract=0.0,
    use_volume=True,
    param_fill="previous",
    missing=None,
    multiplier_change="error",
):
    """返回 dict(market=Market 关键字参数, futures=panelbt.Futures, dates, symbols, contracts, splits)。

    ``splits``：``[dict(symbol, before, after, date, before_column, after_column), ...]``，
    ``before`` / ``after`` 是同一合约前后两段的列标签，``*_column`` 是列下标，``date`` 是分界日
    （后一列的首个可交易日，前一列在它前一个交易日到期）。

    参数约定见模块说明。例::

        inputs = panelbt_inputs(fut, products=["IF"], start="2011-01-04", end="2011-12-30",
                                missing={"margin_rate": 0.15, "fee_rate": 2.5e-5,
                                         "fee_per_contract": 0.0})
    """
    import panelbt as bt

    if param_fill not in FILLS:
        raise ValueError(f"param_fill 只能是 {FILLS}")
    if multiplier_change not in ("error", "split"):
        raise ValueError("multiplier_change 只能是 'error' 或 'split'")
    if symbols is None:
        # 只取与样本区间有交集的合约。
        probe = fut.panel(["is_listed"], products=products, start=start, end=end)
        symbols = list(
            np.asarray(probe.symbols)[np.asarray(probe["is_listed"]).any(axis=0)]
        )
    panel = fut.panel(list(FIELDS), symbols=symbols, start=start, end=end)
    dates, names = panel.dates, np.asarray(panel.symbols)
    t, n = panel.shape
    if t == 0 or n == 0:
        raise ValueError("所选区间或合约为空")
    info = fut.contracts().filter(pl.col("symbol").is_in(list(names))).collect()
    lookup = {r["symbol"]: r for r in info.iter_rows(named=True)}
    last = np.array(
        [lookup[s]["last_trade_date"] or "99999999" for s in names], dtype="U8"
    )
    product = np.array([lookup[s]["product"] for s in names])
    if (last < dates[0]).any():
        raise ValueError(f"合约在样本开始前已到期：{list(names[last < dates[0]][:10])}")
    listed = np.asarray(panel["is_listed"])
    # 最后交易日所在行；不是交易日时取其前最近的交易日；晚于样本末尾为 T。
    expiry = np.where(
        last <= dates[-1], np.searchsorted(dates, last, side="right") - 1, t
    ).astype(np.int64)
    settlement = _settlement(np.asarray(panel["settlement"], dtype=float))
    missing_settle = ~np.isfinite(settlement).all(axis=0)
    if missing_settle.any():
        raise ValueError(
            f"这些合约在区间内没有结算价：{list(names[missing_settle][:10])}"
        )
    tradable = np.asarray(panel["has_quote"]) & listed
    rates = {
        name: _rate(name, value, matrix, names, product, listed, param_fill, missing)
        for name, value, matrix in (
            (
                "margin_rate",
                margin_rate,
                np.fmax(
                    np.asarray(panel["margin_long"]), np.asarray(panel["margin_short"])
                ),
            ),
            ("fee_rate", fee_rate, np.asarray(panel["fee_open_rate"])),
            (
                "fee_per_contract",
                fee_per_contract,
                np.asarray(panel["fee_open_per_contract"]),
            ),
        )
    }
    segments = _segments(np.asarray(panel["multiplier"], dtype=float), listed)
    if len(segments) > n and multiplier_change == "error":
        changed = sorted({names[j] for j, lo, _, _ in segments if lo > 0})
        raise ValueError(
            f"这些合约在区间内改过交易单位：{changed[:10]}；"
            "传 multiplier_change='split' 在分界日拆列，或缩短区间"
        )
    columns = np.array([j for j, _, _, _ in segments])
    keep = np.zeros((t, len(segments)), dtype=bool)
    labels, multiplier, ends = [], [], expiry[columns].copy()
    for k, (j, lo, hi, value) in enumerate(segments):
        keep[lo:hi, k] = True
        labels.append(names[j] if lo == 0 else f"{names[j]}@{dates[lo]}")
        multiplier.append(value)
        if hi < t:
            ends[k] = hi - 1  # 分界日前一行按结算价了结
    splits = [
        dict(
            symbol=str(names[j]),
            before=str(labels[k - 1]),
            after=str(labels[k]),
            date=str(dates[lo]),
            before_column=k - 1,
            after_column=k,
        )
        for k, (j, lo, _, _) in enumerate(segments)
        if lo > 0
    ]
    labels = np.array(labels)

    def take(matrix):
        matrix = np.asarray(matrix)
        return matrix[:, columns] if matrix.ndim == 2 else matrix[columns]

    def rate(value):
        return value if np.isscalar(value) else take(value)

    listed_k = take(listed) & keep
    tradable_k = take(tradable) & keep
    rate_k = {
        k: np.where(listed_k, take(v), 0.0)
        if not np.isscalar(v) and np.ndim(v) == 2
        else rate(v)
        for k, v in rates.items()
    }
    futures = bt.Futures(
        is_future=np.ones(len(segments), dtype=bool),
        multiplier=np.array(multiplier),
        margin_rate=rate_k["margin_rate"],
        settlement=take(settlement),
        expiry=ends.astype(np.int64),
        fee_rate=rate_k["fee_rate"] * fee_scale,
        fee_per_contract=rate_k["fee_per_contract"] * fee_scale,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
    )
    market = dict(
        exec_price=np.where(tradable_k, take(panel["open"]), np.nan),
        close=take(np.asarray(panel["close"], dtype=float)),
        can_buy=tradable_k,
        can_sell=tradable_k,
        volume=np.nan_to_num(take(panel["volume"]), nan=0.0) if use_volume else None,
    )
    if len(segments) > n:
        split = pl.DataFrame({"label": labels, "symbol": names[columns]})
        info = split.join(info, on="symbol").drop("symbol").rename({"label": "symbol"})
    return dict(
        market=market,
        futures=futures,
        dates=dates,
        symbols=labels,
        contracts=info,
        splits=splits,
    )


__all__ = ["fill_columns", "panelbt_inputs"]
