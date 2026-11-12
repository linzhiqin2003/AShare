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
- 保证金率默认取交易所投机保证金率 max(买, 卖)，在合约内前后填补；源无值时须显式传入。
- 手续费默认取交易所的开仓手续费（比例与每手两部分，panelbt 不区分开平和平今）；
  ``fee_scale`` 用来乘期货公司加收倍数。源无值的格子须用 ``fee_rate`` / ``fee_per_contract`` 显式给。
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
    "margin_long",
    "margin_short",
    "fee_open_rate",
    "fee_open_per_contract",
)


def _fill(column):
    """一维：向前填补，再用首个有效值回填开头。"""
    out = column.copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    index = np.where(valid, np.arange(len(out)), 0)
    np.maximum.accumulate(index, out=index)
    out = out[index]
    first = int(np.argmax(valid))
    out[:first] = column[first]
    return out


def fill_columns(matrix):
    return (
        np.column_stack([_fill(matrix[:, j]) for j in range(matrix.shape[1])])
        if matrix.size
        else matrix.copy()
    )


def _rate(value, panel_value, name, symbols):
    if value is not None:
        return value
    filled = fill_columns(panel_value)
    bad = ~np.isfinite(filled).all(axis=0)
    if bad.any():
        raise ValueError(
            f"{name}: 源没有这些合约的值 {list(np.asarray(symbols)[bad][:10])}；请显式传入 {name}"
        )
    return filled


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
):
    """返回 dict(market=Market 关键字参数, futures=panelbt.Futures, dates, symbols, contracts)。"""
    import panelbt as bt

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
    multiplier = np.array([lookup[s]["multiplier"] for s in names], dtype=float)
    if (last < dates[0]).any():
        raise ValueError(f"合约在样本开始前已到期：{list(names[last < dates[0]][:10])}")
    # 最后交易日所在行；不是交易日时取其前最近的交易日；晚于样本末尾为 T。
    expiry = np.where(
        last <= dates[-1], np.searchsorted(dates, last, side="right") - 1, t
    ).astype(np.int64)
    settlement = fill_columns(np.asarray(panel["settlement"], dtype=float))
    missing = ~np.isfinite(settlement).all(axis=0)
    if missing.any():
        raise ValueError(f"这些合约在区间内没有结算价：{list(names[missing][:10])}")
    tradable = np.asarray(panel["has_quote"]) & np.asarray(panel["is_listed"])
    margin = _rate(
        margin_rate,
        np.fmax(np.asarray(panel["margin_long"]), np.asarray(panel["margin_short"])),
        "margin_rate",
        names,
    )
    rate = _rate(fee_rate, np.asarray(panel["fee_open_rate"]), "fee_rate", names)
    unit = _rate(
        fee_per_contract,
        np.asarray(panel["fee_open_per_contract"]),
        "fee_per_contract",
        names,
    )
    futures = bt.Futures(
        is_future=np.ones(n, dtype=bool),
        multiplier=multiplier,
        margin_rate=margin,
        settlement=settlement,
        expiry=expiry,
        fee_rate=np.asarray(rate, dtype=float) * fee_scale
        if not np.isscalar(rate)
        else rate * fee_scale,
        fee_per_contract=np.asarray(unit, dtype=float) * fee_scale
        if not np.isscalar(unit)
        else unit * fee_scale,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
    )
    market = dict(
        exec_price=np.where(tradable, np.asarray(panel["open"]), np.nan),
        close=np.asarray(panel["close"], dtype=float),
        can_buy=tradable,
        can_sell=tradable,
        volume=np.nan_to_num(np.asarray(panel["volume"]), nan=0.0)
        if use_volume
        else None,
    )
    return dict(
        market=market, futures=futures, dates=dates, symbols=names, contracts=info
    )


__all__ = ["fill_columns", "panelbt_inputs"]
