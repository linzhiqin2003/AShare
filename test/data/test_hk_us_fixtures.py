"""港股、美股的合成原始数据（Tushare 接口形状，不含真实行情）和一个假的 Tushare 源。

本文件只放夹具，不含测试。事件都是手写的已知答案：

港股（30 个交易日，2026-01-05 起的工作日）
- 00001.HK：第 10 天 5 拆 1，第 20 天每股派 0.5；港股通名单一直在。
- 00002.HK：第 15 天退市（日线到第 14 天）。
- 00003.HK：代码复用。旧公司 00003!.HK 第 1 天退市，第 0 天的日线写在 00003.HK 下；
  新公司第 3 天上市。
- 08001.HK：低价股，第 5 天日线收盘价被四舍五入（0.05），因子表 close_price 是 0.045。
- 00005.HK：第 8–12 天停牌（没有日线，因子表照常有行），停牌期内第 9 天 10 合 1、第 11 天
  因子比 1.02（派息）：等效现金要按合股后的价位算；第 25 天起调出港股通名单，
  南向持股到第 27 天 → 第 25–28 天只能卖出（看上一交易日的持股）。
- 00003.HK 新公司第 20 天因子比 1.37（供股类大比值，按股数调整）。
- 08009.HK：第 12 天 20 合 1（因子比 0.05），合股当天开盘、收盘都只有隐含价位的 0.385 倍（市场同时
  打压，价格印证不了），但交易所前收已按 20 合 1 调整：以前收为准，执行。
- 00005.HK 停牌后复牌那天的前收是停牌前收盘按窗口合计比值调整后的价位（交易所口径）。
港股通名单从第 20 天起才有（之前的分区为空）。

美股
- AAA：第 12 天 4 拆 1（因子比 4，前收不除权）。
- BBB：证券表两行（代码复用）；第 7 天日线两行、数值不同（冲突，整组丢弃）。
- CCC：第 4 天后没有行情（推断退市）。
- DDD：第 15 天因子比 1.07（非整数比，分不出现金与股数）。
- EEE：第 18 天 10 合 1（因子比 0.1）。
- FFF：第 9 天日线两行完全相同（保留一行）。
- MRG：并购退市。第 10 天后没有成交，第 11 天因子 ×80（源在退市证券上把最后一天写成 1），
  第 12 天一行成交量为 0 的陈旧日线：不执行（action_source=7）。
- TOP：两条交易所序列。NAS 到第 8 天；ASE 全程，第 9 天因子错误 ×10、价格不动：不执行（6）。
- ZZZ：第 6 天因子 1 合 5（0.2）而价格不动：不执行（6）。
- TRF：转板。NAS 到第 14 天，NYS 第 15 天起，第 20 天 2 拆 1（在 NYS 序列里）：执行，拼接处不算事件。
- WMX：两条全程重叠的序列，NYS 到第 22 天结束且第 10 天含 1.01 的派息，NAS 只含拆股：
  第 22 天前取 NYS、之后取 NAS，因子含分红为「部分」。
- GAP：第 6–20 天没有任何行情，第 21 天起价格 ×50 且因子不变：长缺口两侧不连续，拆成两段。
- CZO：第 8 天价格 ×80（源漏记了合股），因子只给出 1.097 的小上调：价格不印证（6），而且价格跳出
  [1/3, 3]：从第 8 天起拆成 CZO#2，引擎不会把这次跳变当成收益。
- SHZ：与 GAP 同样的长缺口，复牌后价格只剩 0.1 倍（真实暴跌）：拆段时旧列在复牌日退市，按复牌价清算。
- HLT：第 4–26 天停牌（没有日线，因子照常有行），停牌期内第 15 天 10 合 1，复牌价已按合股调整：执行。
  截到停牌期内构建时这次合股还看不到复牌价（记 7），满 20 天后记退市，复牌后不再记（截断不变性用）。
- HOL：第 10–12 天因子表没有行（日线照常）：行情不按「截至当天的最后一个因子日」截断（截断不变性用）。
- SPX：第 10 天因子 ×0.85（非整数下调，小窗口）而价格不动：不执行（5）。
- STK：第 14 天 6 拆 5（因子比 1.2，小窗口里的整数比）：照常执行。
- SMC：第 16 天因子比 1.15、价格同步调整：小窗口里的非整数比被价格印证，执行（4）。
- PRF（类别 PF，优先股）、WRNTW（无类别、第 5 位 W 的权证）、OTCX（只有场外因子）：不进面板。
"""

from datetime import date, timedelta

import polars as pl

from ashare.data.sources.base import FetchResult


def weekdays(first, count):
    out, day = [], date.fromisoformat(first)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


DAYS = weekdays("2026-01-05", 30)
US_HOLIDAY = "20260119"
US_DAYS = [d for d in DAYS if d != US_HOLIDAY]


def calendar(open_days):
    first = date(2025, 12, 1)
    rows, previous = [], None
    for k in range(120):
        day = (first + timedelta(days=k)).strftime("%Y%m%d")
        is_open = int(
            day in open_days
            or (day < DAYS[0] and (first + timedelta(days=k)).weekday() < 5)
        )
        rows.append(dict(cal_date=day, is_open=is_open, pretrade_date=previous))
        if is_open:
            previous = day
    return pl.DataFrame(rows)


# 代码 → (起始价, {第几天: 因子比 r=F[t]/F[t-1]})
HK_EVENTS = {
    "00001.HK": (50.0, {10: 5.0}),
    "00002.HK": (8.0, {}),
    "00003.HK": (12.0, {20: 1.37}),
    "08001.HK": (0.045, {}),
    "00005.HK": (60.0, {9: 0.1, 11: 1.02}),
    "08009.HK": (0.2, {12: 0.05}),
}
# 代码 → (第几天, 当天开盘、收盘相对隐含价位的倍数)；前收不受影响。
HK_SHOCK = {"08009.HK": (12, 0.385)}
HK_DIVIDEND = ("00001.HK", 20, 0.5)
US_EVENTS = {
    "AAA": (200.0, {12: 4.0}),
    "BBB": (30.0, {}),
    "CCC": (5.0, {}),
    "DDD": (40.0, {15: 1.07}),
    "EEE": (0.8, {18: 0.1}),
    "FFF": (15.0, {}),
    "MRG": (58.0, {}),
    "TOP": (12.0, {}),
    "ZZZ": (10.0, {}),
    "TRF": (30.0, {20: 2.0}),
    "WMX": (60.0, {10: 1.01}),
    "GAP": (2.0, {}),
    "CZO": (0.41, {}),
    "SHZ": (6.0, {}),
    "HLT": (8.0, {15: 0.1}),
    "HOL": (14.0, {}),
    "SPX": (35.0, {}),
    "STK": (20.0, {14: 1.2}),
    "SMC": (25.0, {16: 1.15}),
    "PRF": (25.0, {}),
    "WRNTW": (0.5, {}),
    "OTCX": (3.0, {}),
}


def _path(start, events, n, *, market, shock=None):
    """价格路径：每天 +0.5%，事件日价格按因子比调整；返回 [(close, pre_close, r)]。

    ``shock``：(第几天, 倍数)，那天收盘价再乘这个倍数（前收不变）。
    """
    out = []
    close = start
    for t in range(n):
        r = events.get(t, 1.0)
        previous = close
        ex_reference = previous / r
        close = round(ex_reference * 1.005, 6)
        if shock and t == shock[0]:
            close = round(close * shock[1], 6)
        pre_close = ex_reference if market == "hk" else previous
        out.append((close, pre_close, r))
    return out


def hk_tables():
    rows, factor = [], []
    n = len(DAYS)
    for code, (start, events) in HK_EVENTS.items():
        events = dict(events)
        if code == HK_DIVIDEND[0]:
            # 派息 0.5：r = 前收 / (前收 − 0.5)，用路径里那天之前的收盘价算。
            path = _path(start, events, n, market="hk")
            prev = path[HK_DIVIDEND[1] - 1][0]
            events[HK_DIVIDEND[1]] = prev / (prev - HK_DIVIDEND[2])
        shock = HK_SHOCK.get(code)
        path = _path(start, events, n, market="hk", shock=shock)
        backward = 1.0
        window = 1.0  # 停牌以来累计的因子比
        last_shown = None
        for t, day in enumerate(DAYS):
            close, pre, r = path[t]
            backward *= r
            window *= r
            listed = True
            if code == "00002.HK" and t >= 15:
                listed = False
            suspended = code == "00005.HK" and 8 <= t <= 12
            if listed and not suspended:
                if last_shown is not None:
                    # 交易所的前收：上一个成交日收盘按其后全部除权调整（停牌复牌时跨整个缺口）。
                    pre = last_shown / window
                opening = pre * shock[1] if shock and t == shock[0] else pre
                shown = 0.05 if (code == "08001.HK" and t == 5) else round(close, 3)
                last_shown, window = close, 1.0
                rows.append(
                    dict(
                        ts_code=code,
                        trade_date=day,
                        open=round(opening, 3),
                        high=round(max(pre, close) * 1.01, 3),
                        low=round(min(pre, close) * 0.99, 3),
                        close=shown,
                        pre_close=round(pre, 6),
                        change=None,
                        pct_chg=None,
                        vol=1000.0 * (t + 1),
                        amount=1000.0 * (t + 1) * close,
                    )
                )
            if listed and not (code == "00003.HK" and t < 3):
                factor.append(
                    dict(
                        ts_code=code,
                        trade_date=day,
                        _b=backward,
                        close_price=round(close, 3),
                    )
                )
    # 00003.HK 第 0 天是旧公司的行情（同一代码）。
    rows = [
        r
        for r in rows
        if not (r["ts_code"] == "00003.HK" and r["trade_date"] in DAYS[1:3])
    ]
    securities = pl.DataFrame(
        [
            dict(
                ts_code="00001.HK",
                name="甲",
                list_status="L",
                list_date="19720101",
                delist_date=None,
                trade_unit=500.0,
                curr_type="HKD",
                market="主板",
            ),
            dict(
                ts_code="00002.HK",
                name="乙",
                list_status="D",
                list_date="20000101",
                delist_date=DAYS[15],
                trade_unit=1000.0,
                curr_type="HKD",
                market="主板",
            ),
            dict(
                ts_code="00003.HK",
                name="丙新",
                list_status="L",
                list_date=DAYS[3],
                delist_date=None,
                trade_unit=200.0,
                curr_type="HKD",
                market="主板",
            ),
            dict(
                ts_code="00003!.HK",
                name="丙旧",
                list_status="D",
                list_date="19900101",
                delist_date=DAYS[1],
                trade_unit=2000.0,
                curr_type="HKD",
                market="主板",
            ),
            dict(
                ts_code="08001.HK",
                name="丁",
                list_status="L",
                list_date="20100101",
                delist_date=None,
                trade_unit=None,
                curr_type="HKD",
                market="创业板",
            ),
            dict(
                ts_code="08009.HK",
                name="己",
                list_status="L",
                list_date="20100101",
                delist_date=None,
                trade_unit=10000.0,
                curr_type="HKD",
                market="创业板",
            ),
            dict(
                ts_code="00005.HK",
                name="戊",
                list_status="L",
                list_date="19800101",
                delist_date=None,
                trade_unit=400.0,
                curr_type="HKD",
                market="主板",
            ),
        ]
    ).with_columns(
        fullname=pl.col("name"),
        enname=pl.col("name"),
        cn_spell=pl.lit("X"),
        isin=pl.lit("HK0"),
    )
    connect = []
    hold = []
    for t, day in enumerate(DAYS):
        for code in ("00001.HK", "00005.HK"):
            if code == "00005.HK" and t > 27:
                continue  # 南向第 27 天后清仓
            hold.append(
                dict(
                    code=code[:5],
                    trade_date=day,
                    ts_code=code,
                    name=code,
                    vol=1e6,
                    ratio=1.0,
                    exchange="HK",
                )
            )
        if t >= 20:
            for kind in ("SH_HK", "SZ_HK"):
                connect.append(
                    dict(
                        ts_code="00001.HK",
                        trade_date=day,
                        type=kind,
                        name="甲",
                        type_name=kind,
                    )
                )
                if t < 25:
                    connect.append(
                        dict(
                            ts_code="00005.HK",
                            trade_date=day,
                            type=kind,
                            name="戊",
                            type_name=kind,
                        )
                    )
    return dict(
        daily=pl.DataFrame(rows),
        factor=pl.DataFrame(factor),
        securities=securities,
        connect=pl.DataFrame(connect),
        hold=pl.DataFrame(hold),
    )


def _us_series(code, t):
    """(交易所, 因子比修正) 列表：一个代码当天在哪些交易所序列里有因子行。"""
    if code == "OTCX":
        return [("OTC", 1.0)]
    if code == "TOP":
        return ([("NAS", 1.0)] if t <= 8 else []) + [("ASE", 10.0 if t >= 9 else 1.0)]
    if code == "TRF":
        return [("NAS", 1.0)] if t <= 14 else [("NYS", 1.0)]
    if code == "WMX":
        # NAS 序列不含第 10 天的派息（口径只含拆股）。
        return ([("NYS", 1.0)] if t <= 22 else []) + [
            ("NAS", 1 / 1.01 if t >= 10 else 1.0)
        ]
    return [("NAS", 1.0)]


def us_tables():
    rows, factor = [], []
    n = len(US_DAYS)
    for code, (start, events) in US_EVENTS.items():
        path = _path(start, events, n, market="us")
        backward = 1.0
        for t, day in enumerate(US_DAYS):
            close, pre, r = path[t]
            backward *= r
            if code == "CCC" and t > 4:
                continue
            factor_b = backward
            if code == "MRG" and t >= 11:
                if t > 12:
                    continue
                factor_b = backward * 80
            if code == "ZZZ" and t >= 6:
                factor_b = backward * 0.2
            if code == "CZO" and t >= 8:
                factor_b = backward * 1.097
            if code == "SPX" and t >= 10:
                factor_b = backward * 0.85
            gap = code in ("GAP", "SHZ") and 6 <= t <= 20
            scale = 50.0 if code == "GAP" and t > 20 else 1.0
            if code == "SHZ" and t > 20:
                scale = 0.1
            if code == "CZO" and t >= 8:
                scale = 80.0
            halted = code == "HLT" and 4 <= t <= 26
            if not gap and not halted and not (code == "MRG" and t == 11):
                row = dict(
                    ts_code=code,
                    trade_date=day,
                    close=round(close * scale, 4),
                    open=round(pre / r * scale, 4),
                    high=round(max(pre / r, close) * scale * 1.01, 4),
                    low=round(min(pre / r, close) * scale * 0.99, 4),
                    pre_close=round(pre * scale, 4),
                    change=None,
                    pct_change=None,
                    vol=0.0 if (code == "MRG" and t == 12) else 1e5 * (t + 1),
                    amount=1e5 * (t + 1) * close,
                    vwap=close,
                    turnover_ratio=None,
                    total_mv=close * 1e9,
                    pe=None,
                    pb=None,
                )
                rows.append(row)
                if code == "BBB" and t == 7:
                    rows.append({**row, "close": row["close"] * 2})
                if code == "FFF" and t == 9:
                    rows.append(dict(row))
            if gap or (code == "HOL" and 10 <= t <= 12):
                continue
            for exchange, fudge in _us_series(code, t):
                factor.append(
                    dict(
                        ts_code=code,
                        trade_date=day,
                        exchange=exchange,
                        _b=factor_b * fudge,
                        close_price=round(close * scale, 4),
                    )
                )
    securities = pl.DataFrame(
        [
            dict(
                ts_code=c,
                name=c,
                enname=c,
                classify={"PRF": "PF", "WRNTW": None, "GAP": None, "SHZ": None}.get(
                    c, "EQ"
                ),
                list_date="20000101",
                delist_date=None,
            )
            for c in US_EVENTS
        ]
        + [
            dict(
                ts_code="BBB",
                name="BBB old",
                enname="BBB OLD",
                classify="EQ",
                list_date=None,
                delist_date=None,
            )
        ]
    )
    return dict(
        daily=pl.DataFrame(rows), factor=pl.DataFrame(factor), securities=securities
    )


class FakeOverseas:
    """按 (接口, 参数) 返回合成数据；``asof`` 模拟前复权因子的快照（以 asof 当天为 1）。"""

    def __init__(self, asof=None, fail=()):
        self.calls = []
        self.fail = set(fail)
        self.hk = hk_tables()
        self.us = us_tables()
        self.asof = asof or DAYS[-1]

    def _factor(self, table, params):
        visible = table.filter(pl.col("trade_date") <= self.asof)
        # 每条序列（代码 × 交易所）各自以最后一天为 1。
        keys = [c for c in ("ts_code", "exchange") if c in visible.columns]
        anchor = (
            visible.sort("trade_date")
            .group_by(keys)
            .agg(pl.col("_b").last().alias("_anchor"))
        )
        visible = visible.join(anchor, on=keys).with_columns(
            cum_adjfactor=(pl.col("_b") / pl.col("_anchor")).round(6)
        )
        return self._slice(visible.drop("_b", "_anchor"), params)

    @staticmethod
    def _slice(table, params):
        if params.get("trade_date"):
            table = table.filter(pl.col("trade_date") == params["trade_date"])
        if params.get("ts_code"):
            table = table.filter(pl.col("ts_code") == params["ts_code"])
        if params.get("start_date"):
            table = table.filter(pl.col("trade_date") >= params["start_date"])
        if params.get("end_date"):
            table = table.filter(pl.col("trade_date") <= params["end_date"])
        return table

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls.append((api, dict(params)))
        key = (api, params.get("trade_date") or params.get("ts_code"))
        if key in self.fail:
            self.fail.discard(key)
            raise RuntimeError("故意失败")
        if api in ("hk_tradecal", "us_tradecal"):
            table = calendar(DAYS if api == "hk_tradecal" else US_DAYS)
            table = table.filter(
                pl.col("cal_date").is_between(
                    pl.lit(params["start_date"]), pl.lit(params["end_date"])
                )
            )
        elif api == "hk_basic":
            table = self.hk["securities"].filter(
                pl.col("list_status") == params["list_status"]
            )
        elif api == "us_basic":
            table = self.us["securities"]
        elif api == "hk_daily":
            table = self._slice(self.hk["daily"], params)
        elif api == "us_daily":
            table = self._slice(self.us["daily"], params)
        elif api == "hk_adjfactor":
            table = self._factor(self.hk["factor"], params)
        elif api == "us_adjfactor":
            table = self._factor(self.us["factor"], params)
        elif api == "stock_hsgt":
            table = self._slice(self.hk["connect"], params).filter(
                pl.col("type") == params["type"]
            )
        elif api == "hk_hold":
            table = self._slice(self.hk["hold"], params)
        elif api == "hk_daily_adj":
            table = self._slice(self.hk["daily"], params).select(
                "ts_code", (pl.col("close") * 1e9).alias("total_mv")
            )
        elif api == "fx_daily":
            table = pl.DataFrame(
                [
                    dict(
                        ts_code=params["ts_code"],
                        trade_date=d,
                        bid_close=7.8 + i / 1000,
                        ask_close=7.8 + i / 1000 + 0.002,
                        bid_open=7.8,
                        ask_open=7.8,
                        bid_high=7.9,
                        bid_low=7.7,
                        ask_high=7.9,
                        ask_low=7.7,
                        tick_qty=100,
                    )
                    for i, d in enumerate(DAYS)
                ]
            )
        elif api == "fx_midrate":
            table = pl.DataFrame(
                [
                    dict(trade_date=d, ts_code=params["ts_code"], mid_rate=91.5)
                    for d in DAYS
                ]
            )
        else:
            raise AssertionError(f"夹具没有接口 {api}")
        return FetchResult(table, "2026-02-14T00:00:00+00:00", 1)
