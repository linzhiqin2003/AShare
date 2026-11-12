"""用中金所股指期货对比两条数据路径喂给 panelbt 的结果：

- 现有路径：股票数据根里给对冲腿用的 ``fut_basic`` / ``fut_daily`` 长表（``Data.table``）；
- 新路径：期货面板 + ``futures_panelbt.panelbt_inputs``。

两边用同一套显式的保证金和手续费，按 ``panelbt.roll_targets`` 做一个滚动空头，逐位比较输入数组
和回测净值。用法::

    python -m ashare.data.futures_hedge_check --stock-root <股票数据根> --futures-root <期货数据根> \\
        --product IF --start 2024-01-02 --end 2024-12-31
"""

import argparse
import json

import numpy as np
import polars as pl

import ashare.data as ad

from .futures_panelbt import fill_columns, panelbt_inputs
from .sync import compact

MARGIN, FEE = 0.12, 0.000023


def legacy_inputs(stock_root, product, start, end):
    """现有做法：直接从中金所长表按合约逐列取结算价、开盘价、到期行和乘数。"""
    import panelbt as bt

    data = ad.open(stock_root)
    basic = data.table("fut_basic").filter(pl.col("underlying") == product).collect()
    daily = data.table("fut_daily", start=start, end=end).collect()
    dates = sorted(daily["date"].unique().to_list())
    basic = basic.filter(
        (pl.col("listing_date") <= dates[-1]) & (pl.col("expiry_date") >= dates[0])
    ).sort("expiry_date", "symbol")
    codes = basic["symbol"].to_list()
    t, n = len(dates), len(codes)
    row = {d: i for i, d in enumerate(dates)}
    col = {c: j for j, c in enumerate(codes)}
    settle, open_, close = (np.full((t, n), np.nan) for _ in range(3))
    for code, day, s, o, c in (
        daily.filter(pl.col("symbol").is_in(codes))
        .select("symbol", "date", "settlement", "open", "close")
        .iter_rows()
    ):
        settle[row[day], col[code]] = s
        open_[row[day], col[code]] = o
        close[row[day], col[code]] = c
    last = np.array(basic["expiry_date"].to_list(), dtype="U8")
    expiry = np.where(
        last <= dates[-1], np.searchsorted(dates, last, side="right") - 1, t
    )
    tradable = np.isfinite(open_) | np.isfinite(settle) | np.isfinite(close)
    listed = (
        np.array(dates)[:, None] >= np.array(basic["listing_date"].to_list())[None, :]
    ) & (np.array(dates)[:, None] <= last[None, :])
    tradable &= listed
    futures = bt.Futures(
        is_future=np.ones(n, bool),
        multiplier=basic["contract_multiplier"].to_numpy(),
        margin_rate=MARGIN,
        settlement=fill_columns(settle),
        expiry=expiry.astype(np.int64),
        fee_rate=FEE,
    )
    market = dict(
        exec_price=np.where(tradable, open_, np.nan),
        close=close,
        can_buy=tradable,
        can_sell=tradable,
    )
    return dict(market=market, futures=futures, dates=np.array(dates), symbols=codes)


def compare(
    stock_root, futures_root, product="IF", start="2024-01-02", end="2024-12-31"
):
    import panelbt as bt

    start, end = compact(start), compact(end)
    old = legacy_inputs(stock_root, product, start, end)
    fut = ad.open(futures_root).futures()
    new = panelbt_inputs(
        fut,
        products=[product],
        start=start,
        end=end,
        margin_rate=MARGIN,
        fee_rate=FEE,
        fee_per_contract=0.0,
        use_volume=False,
    )
    to_old = [s.replace(".CFFEX", ".CFX") for s in new["symbols"]]
    report = dict(
        dates_equal=bool(np.array_equal(old["dates"], new["dates"])),
        symbols_equal=to_old == old["symbols"],
        settlement_equal=bool(
            np.array_equal(old["futures"].settlement, new["futures"].settlement)
        ),
        expiry_equal=bool(np.array_equal(old["futures"].expiry, new["futures"].expiry)),
        multiplier_equal=bool(
            np.array_equal(old["futures"].multiplier, new["futures"].multiplier)
        ),
        exec_equal=bool(
            np.array_equal(
                old["market"]["exec_price"], new["market"]["exec_price"], equal_nan=True
            )
        ),
        close_equal=bool(
            np.array_equal(
                old["market"]["close"], new["market"]["close"], equal_nan=True
            )
        ),
        contracts=len(new["symbols"]),
        rows=len(new["dates"]),
    )
    from panelbt.futures import roll_targets

    # 合约按最后交易日排序；到期晚于样本末尾的只留第一个，使到期行严格递增。
    expiry = np.asarray(new["futures"].expiry)
    prefix = int(np.searchsorted(expiry, len(new["dates"]))) + 1
    targets = roll_targets(expiry[:prefix], len(new["dates"]), 5, -1.0)
    config = bt.Config(initial_cash=10_000_000.0)
    runs = []
    for inputs in (old, new):
        market = bt.Market(
            **{k: v for k, v in inputs["market"].items() if k != "volume"},
            futures=inputs["futures"],
        )
        runs.append(bt.run(market, targets, config=config))
    report["nav_equal"] = bool(np.array_equal(runs[0].nav, runs[1].nav))
    report["final_nav"] = float(runs[1].nav[-1])
    report["futures_pnl_total"] = float(np.sum(runs[1].futures_pnl))
    report["days_with_position"] = int(np.count_nonzero(runs[1].futures_margin))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-root", required=True)
    parser.add_argument("--futures-root", required=True)
    parser.add_argument("--product", default="IF")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.stock_root, args.futures_root, args.product, args.start, args.end
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
