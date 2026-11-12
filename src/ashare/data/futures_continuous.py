"""主力 / 次主力连续合约与换月日程表。

连续价格只用来算信号；真正回测按 ``roll_schedule`` 逐个合约持有和换月。

换月规则（``rule``）：

- ``"source"``：照来源给的主力映射（Tushare ``fut_mapping``），只有主力（rank=1）。
- ``"open_interest"`` / ``"volume"``：自己判定，只用前一交易日及以前的数据，防前视：
  1. 第 t 日可选合约 = 第 t−1 日有行情、且最后交易日晚于第 t 日的合约（不在最后交易日当天新进入，
     也不持有到最后交易日当天）。
  2. 主力：第 t−1 日该指标最大的可选合约记为候选。当前持有的合约仍可选时，只有候选到期更晚、
     且候选的指标严格大于当前合约，才换到候选（只向后换，不回切）；当前合约不再可选时，
     强制换到「到期晚于当前合约的可选合约」里指标最大的一个。
  3. 次主力：在到期晚于当天主力的可选合约里，用同样的规则（粘住、只向后、严格大于才换）。

复权（``adjust``）：在换月日 r 同时取旧、新两个合约当天的价格（缺一个就用两者都有的最近一天）。

- ``"none"``：不调整，直接拼接。
- ``"difference"``：差值后复权，第一段为真实价格，之后每段加上累计价差 Σ(旧 − 新)。
- ``"ratio"``：比例后复权，之后每段乘以累计比值 Π(旧 / 新)。

两种复权都使换月日的连续价格等于旧合约当天价格，即换月按换月日结算价完成，当天涨跌归旧合约。
"""

import numpy as np
import polars as pl

RULES = ("source", "open_interest", "volume")
ADJUSTS = ("none", "difference", "ratio")


def select_held(metric, eligible, expiry_order, *, previous=None):
    """按「前一日指标」逐日选持有列；纯 numpy，返回每行的列号（-1 表示无）。

    metric[t] 为第 t 日收盘后才知道的指标；第 t 行的选择只读 metric[t-1] 与 eligible[t]。
    ``expiry_order`` 为各列的到期先后名次（越大越晚）。``previous`` 给定时（次主力），第 t 行只考虑
    名次严格晚于 previous[t] 的列。
    """
    t_count, n = metric.shape
    held = np.full(t_count, -1, dtype=np.int64)
    current = -1
    for t in range(1, t_count):
        allowed = eligible[t].copy()
        if previous is not None:
            anchor = previous[t]
            if anchor < 0:
                allowed[:] = False
            else:
                allowed &= expiry_order > expiry_order[anchor]
        score = np.where(allowed, np.nan_to_num(metric[t - 1], nan=-np.inf), -np.inf)
        if current >= 0 and allowed[current]:
            best = int(np.argmax(score))
            if (
                allowed[best]
                and expiry_order[best] > expiry_order[current]
                and score[best] > score[current]
            ):
                current = best
        else:
            if current >= 0:
                later = allowed & (expiry_order > expiry_order[current])
                score = np.where(later, score, -np.inf)
            best = int(np.argmax(score)) if allowed.any() else -1
            current = best if best >= 0 and np.isfinite(score[best]) else -1
        held[t] = current
    return held


def adjust_prices(held, price, mode):
    """由逐日持有列和价格矩阵生成 (原始价, 复权价, 换月记录)。纯 numpy。"""
    if mode not in ADJUSTS:
        raise ValueError(f"adjust 只能是 {ADJUSTS}")
    t_count = len(held)
    raw = np.array([price[t, h] if h >= 0 else np.nan for t, h in enumerate(held)])
    adjusted = raw.copy()
    offset, factor = 0.0, 1.0
    rolls = []
    for t in range(1, t_count):
        old, new = held[t - 1], held[t]
        if old >= 0 and new >= 0 and old != new:
            day = t
            while day >= 0 and not (
                np.isfinite(price[day, old]) and np.isfinite(price[day, new])
            ):
                day -= 1
            if day < 0:
                old_price = new_price = np.nan
            else:
                old_price, new_price = price[day, old], price[day, new]
            rolls.append((t, old, new, old_price, new_price, day))
            if np.isfinite(old_price) and np.isfinite(new_price):
                offset += old_price - new_price
                if new_price != 0:
                    factor *= old_price / new_price
        if new >= 0:
            if mode == "difference":
                adjusted[t] = raw[t] + offset
            elif mode == "ratio":
                adjusted[t] = raw[t] * factor
    return raw, adjusted, rolls


def _inputs(fut, product, field, start, end):
    names = sorted({field, "open_interest", "volume", "has_quote"})
    panel = fut.panel(names, products=[product], start=start, end=end)
    info = (
        fut.contracts(products=[product]).select("symbol", "last_trade_date").collect()
    )
    last = dict(zip(info["symbol"], info["last_trade_date"], strict=False))
    symbols = list(panel.symbols)
    keys = [(last[s] or "99999999") + s for s in symbols]
    order = np.argsort(np.argsort(keys))
    expiry = np.array([last[s] or "99999999" for s in symbols], dtype="U8")
    return panel, order, expiry


def held_columns(
    fut,
    product,
    *,
    rule="open_interest",
    rank=1,
    field="settlement",
    start=None,
    end=None,
):
    """返回 (panel, 每行持有的列号)；rank=2 时为次主力。"""
    if rule not in RULES:
        raise ValueError(f"rule 只能是 {RULES}")
    if rank not in (1, 2):
        raise ValueError("rank 只能是 1（主力）或 2（次主力）")
    panel, order, expiry = _inputs(fut, product, field, start, end)
    dates = panel.dates
    if rule == "source":
        if rank != 1:
            raise ValueError(
                "来源映射只有主力；次主力请用 open_interest 或 volume 规则"
            )
        mapping = (
            fut.table("mapping", start=start, end=end)
            .filter(
                (pl.col("kind") == "main")
                & (pl.col("continuous").str.extract(r"^([A-Z_]+)\.") == product)
            )
            .select("date", "symbol")
            .collect()
        )
        column = {s: j for j, s in enumerate(panel.symbols)}
        row = {d: i for i, d in enumerate(dates)}
        held = np.full(len(dates), -1, dtype=np.int64)
        for day, symbol in mapping.iter_rows():
            if day in row and symbol in column:
                held[row[day]] = column[symbol]
        return panel, held
    eligible = np.zeros(panel.shape, dtype=bool)
    has_quote = np.asarray(panel["has_quote"])
    eligible[1:] = has_quote[:-1]
    eligible &= expiry[None, :] > dates[:, None]
    metric = np.asarray(panel["open_interest" if rule == "open_interest" else "volume"])
    main = select_held(metric, eligible, order)
    if rank == 1:
        return panel, main
    return panel, select_held(metric, eligible, order, previous=main)


def continuous(
    fut,
    product,
    *,
    rule="open_interest",
    rank=1,
    adjust="none",
    field="settlement",
    start=None,
    end=None,
):
    """连续价格长表：date、symbol（当天对应的合约）、price（原始）、adjusted（复权后）。"""
    if adjust not in ADJUSTS:
        raise ValueError(f"adjust 只能是 {ADJUSTS}")
    panel, held = held_columns(
        fut, product, rule=rule, rank=rank, field=field, start=start, end=end
    )
    price = np.asarray(panel[field])
    raw, adjusted, _ = adjust_prices(held, price, adjust)
    symbols = np.asarray(panel.symbols)
    return pl.DataFrame(
        dict(
            date=panel.dates,
            symbol=[symbols[h] if h >= 0 else None for h in held],
            price=raw,
            adjusted=adjusted,
        )
    ).filter(pl.col("symbol").is_not_null())


def roll_schedule(
    fut,
    product,
    *,
    rule="open_interest",
    rank=1,
    field="settlement",
    start=None,
    end=None,
):
    """换月日程表：date（新合约开始作为连续合约的日子）、from_symbol、to_symbol、
    price_date（取价格的日子，通常等于 date）、from_price、to_price、spread（新 − 旧）。"""
    panel, held = held_columns(
        fut, product, rule=rule, rank=rank, field=field, start=start, end=end
    )
    _, _, rolls = adjust_prices(held, np.asarray(panel[field]), "none")
    symbols, dates = np.asarray(panel.symbols), panel.dates
    return pl.DataFrame(
        [
            dict(
                date=dates[t],
                from_symbol=symbols[old],
                to_symbol=symbols[new],
                price_date=dates[day] if day >= 0 else None,
                from_price=old_price,
                to_price=new_price,
                spread=new_price - old_price,
            )
            for t, old, new, old_price, new_price, day in rolls
        ],
        schema=dict(
            date=pl.String,
            from_symbol=pl.String,
            to_symbol=pl.String,
            price_date=pl.String,
            from_price=pl.Float64,
            to_price=pl.Float64,
            spread=pl.Float64,
        ),
    )


__all__ = [
    "adjust_prices",
    "continuous",
    "held_columns",
    "roll_schedule",
    "select_held",
]
