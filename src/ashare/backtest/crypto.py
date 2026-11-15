"""币安 USDT 小时市场：现货现金账户与 U 本位全仓账户各自独立。"""

import copy
from dataclasses import dataclass, replace

import numpy as np
import panelbt as bt
import polars as pl
from panelbt import markets

from ashare.data.crypto.events import asof
from ashare.data.crypto.timeutil import DAY, HOUR, millis

from .crypto_schedule import (
    DailyTargets,
    daily_targets,
    quarter_chain,
    roll_time,
    top_n,
)
from .crypto_wallets import MarginBand, TwoWallet, TwoWalletResult, two_wallet

__all__ = [
    "MarginBand",
    "TwoWallet",
    "TwoWalletResult",
    "two_wallet",
    "crypto_market",
    "daily_targets",
    "top_n",
    "roll_time",
    "quarter_chain",
    "CryptoMarket",
    "DailyTargets",
]


def intervals(times, mask, symbols, **extra):
    records = []
    for j, name in enumerate(symbols):
        ids = np.flatnonzero(mask[:, j])
        for group in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
            if len(group):
                records.append(
                    dict(
                        symbol=str(name),
                        start=str(np.datetime64(int(times[group[0]]), "ms")),
                        end=str(np.datetime64(int(times[group[-1]] + HOUR), "ms")),
                        **extra,
                    )
                )
    return records


def _arrays(frame, names):
    times = np.array(sorted(frame["time"].unique().to_list()), dtype=np.int64)
    parts = [frame.filter(pl.col("instrument") == name).sort("time") for name in names]
    if any(not np.array_equal(p["time"].to_numpy(), times) for p in parts):
        raise ValueError("小时面板缺行，不是完整矩形")
    return times, {
        f: np.column_stack([p[f].to_numpy() for p in parts])
        for f in frame.columns
        if f not in ("time", "instrument")
    }


@dataclass
class CryptoMarket:
    market: bt.Market
    times: np.ndarray
    symbols: tuple
    signal_prices: np.ndarray
    costs: bt.Costs
    config: bt.Config
    diagnostics: dict
    arrays: dict
    profiles: list
    events: list
    kind: str
    dataset: object

    @property
    def futures(self):
        return self.market.futures

    def daily(self, trade_hour=0):
        """日线仍由数据层聚合；day 是左端，available_at 是已完成时间。"""
        return self.dataset.daily(
            trade_hour=trade_hour,
            invalid_prices="carry",
            start=int(self.times[0]),
            end=int(self.times[-1] + HOUR),
            instruments=list(self.symbols),
        )

    def run(
        self,
        targets,
        *,
        kind="weights",
        allow_tier_exceeded=False,
        record_trades=None,
        roll_before="3D",
    ):
        from .crypto_run import simulate

        return simulate(
            self,
            targets,
            kind=kind,
            allow_tier_exceeded=allow_tier_exceeded,
            record_trades=record_trades,
            roll_before=roll_before,
        )

    def daily_result(self, result, trade_hour=0):
        """按 trade_hour 的完整区间报告；年化因子为 365，不改变小时记账。"""
        from .crypto_schedule import hour

        labels = (self.times - hour(trade_hour)) // DAY * DAY + hour(trade_hour)
        frame = pl.DataFrame(dict(day=labels, nav=result.nav, cash=result.cash))
        return (
            frame.group_by("day", maintain_order=True)
            .agg(pl.col("nav").last(), pl.col("cash").last(), pl.len().alias("hours"))
            .filter(pl.col("hours") == 24)
        )


def crypto_market(
    ds,
    symbols,
    start,
    end,
    kind="spot",
    *,
    assume=None,
    leverage=1.0,
    allow_missing=False,
    initial_cash=10000.0,
    max_participation=0.0,
    slippage=0.0,
    fee_switch=None,
    config=None,
):
    """[start,end) 小时面板 → 市场与信号视图，缺数据与规则缺口默认报错。

    ``assume='conservative'`` 在风险规则缺口取前后观测中更不利的一个，``assume='previous'``
    沿用前一个观测；未知交易对需有来源的规则映射。
    已证实的停机可以持平；其他缺口须 allow_missing=True，资金费缺值不能放行。
    信号视图第 t 行只含第 t-1 小时已完成的成交收盘；不会使用上市前 settlement 占位。
    """
    if kind not in ("spot", "usdm"):
        raise ValueError("kind 只能是 spot/usdm；两钱包不在本接口范围")
    names = tuple(str(s) if ":" in str(s) else f"{kind}:{s}" for s in symbols)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not s.startswith(kind + ":") for s in names)
    ):
        raise ValueError("symbols 必须非空、不重复且属于同一账户市场")
    start, end = millis(start), millis(end)
    if start % HOUR or end % HOUR or start >= end:
        raise ValueError("起止必须为递增 UTC 整点")
    inputs = ds.inputs()
    profiles_by = {p["instrument"]: p for p in inputs["profiles"]}
    profiles = [profiles_by[s] for s in names]
    if any(
        p["quote_asset"] != "USDT"
        or (
            kind == "usdm"
            and (
                p.get("margin_asset") != "USDT"
                or p["contract_type"]
                not in ("PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER")
                or p["underlying_type"] != "COIN"
            )
        )
        for p in profiles
    ):
        raise ValueError("只接受 USDT 报价/保证金的现货与 COIN 永续、季度")
    frame = ds.read(start=start, end=end, instruments=list(names))
    if frame.is_empty():
        raise ValueError("请求时段没有小时数据")
    times, a = _arrays(frame, names)
    if not np.array_equal(times, np.arange(start, end, HOUR)):
        raise ValueError(f"请求时段小时缺失或被数据层截断: {names} [{start}, {end})")
    events = inputs.get("events", [])
    migrations = []
    execution = np.tile(np.asarray(names, dtype=object), (len(times), 1))
    if kind == "spot" and any(e["kind"] == "migration" for e in events):
        from .crypto_migration import splice

        migrations, execution = splice(ds, names, times, a, inputs["profiles"], events)
    a["execution_symbol"] = execution
    active = a["lifecycle"] == "active"
    quality = a["quality"].astype(np.int64)
    missing = active & ((quality & (1 | 2048)) != 0)
    # 公告暂停与半截线自证停机有独立证据；其他缺口不得暗中变成停机。
    explained = ((quality & (8 | 1024)) != 0) & ((quality & 2048) == 0)
    required = ("open", "high", "low", "close") + (
        ("mark_open", "mark_high", "mark_low", "mark_close", "settlement")
        if kind == "usdm"
        else ()
    )
    for field in required:
        missing |= active & (~np.isfinite(a[field]) | (a[field] <= 0))
    missing_notes = intervals(times, missing & ~explained, names)
    if missing_notes and not allow_missing:
        raise ValueError(
            "生命周期内缺数据: "
            + "; ".join(
                f"{r['symbol']} [{r['start']}, {r['end']})" for r in missing_notes
            )
        )
    if kind == "usdm" and (not np.isfinite(a["funding"]).all()):
        raise ValueError("资金费缺数据，不能当成无事件的 0")
    if kind == "usdm":
        proven_expiry = (
            np.isfinite(a["settlement"])
            & (a["settlement"] > 0)
            & np.isin(
                a["settlement_source"],
                ["delivery_rest", "announcement", "mark_open_approximation"],
            )
            & (a["event_version"] != None)  # noqa: E711
        )
        if np.any(a["expires"] & ~proven_expiry):
            raise ValueError("到期结算缺有来源的结算价，allow_missing 不能替代结算事件")
    # 显式允许的异常行持平，首行无过去价格则仍拒绝；资金费保持真实事件。
    for j in range(len(names)):
        for i in np.flatnonzero(missing[:, j]):
            if i == 0 or not np.isfinite(a["close"][i - 1, j]):
                raise ValueError(f"{names[j]} 缺失行没有此前估值")
            for f in ("open", "high", "low", "close"):
                a[f][i, j] = a["close"][i - 1, j]
            if kind == "usdm":
                for f in (
                    "mark_open",
                    "mark_high",
                    "mark_low",
                    "mark_close",
                ):
                    a[f][i, j] = a["settlement"][i - 1, j]
                if not a["expires"][i, j]:
                    a["settlement"][i, j] = a["settlement"][i - 1, j]
            a["volume_engine"][i, j] = 0
            a["tradable"][i, j] = False
    events = inputs.get("events", [])
    if kind == "spot":
        for i, j in np.argwhere(a["lifecycle"] == "ended"):
            name = execution[i, j]
            if not any(
                e["kind"] == "delist" and e["effective_at"] <= times[i]
                for e in asof(events, int(times[i]), instrument=name)
            ):
                raise ValueError(
                    f"{name} 下架缺当时可知的公告: {np.datetime64(int(times[i]), 'ms')}"
                )
    profile = markets.get("binance_" + kind)
    dates = times.astype("datetime64[ms]")
    fees = profile.fee_terms(dates, fee_switch=fee_switch)
    delivery_fees = profile.fee_terms(
        (times + HOUR).astype("datetime64[ms]"), fee_switch=fee_switch
    )
    diagnostics = dict(
        migrations=migrations,
        rule_assumptions=[],
        rule_provenance=[],
        fee_provenance=[
            dict(r, parameter="taker", time_basis="execution_left_boundary")
            for r in fees["provenance"]
        ],
        delivery_fee_provenance=[
            dict(r, parameter="delivery", time_basis="settlement_right_boundary")
            for r in delivery_fees["provenance"]
        ]
        if kind == "usdm"
        else [],
        missing_intervals=missing_notes,
        halt_intervals=intervals(times, explained, names),
        inferred_funding_intervals=intervals(times, (quality & 4096) != 0, names),
        partial_halt_cells=int(np.count_nonzero(quality & 512)),
        funding_price_approximation_cells=int(np.count_nonzero(quality & 16)),
        settlement_price_approximation_cells=int(np.count_nonzero(quality & 32)),
        rule_approximation_intervals=intervals(
            times, a["rule_approximation"].astype(bool) & active, names
        ),
    )
    for j, _name in enumerate(names):
        for source in np.unique(a["rule_source_url"][:, j]):
            mask = np.zeros(active.shape, bool)
            mask[:, j] = (
                active[:, j]
                & (a["rule_source_url"][:, j] == source)
                & ((quality[:, j] & 64) != 0)
            )
            diagnostics["rule_assumptions"] += intervals(
                times,
                mask,
                names,
                parameter="order_filters",
                source_url=str(source),
                reason="数据层显式假设/最早快照沿用",
            )
    config = config or bt.Config(
        initial_cash=initial_cash,
        max_participation=max_participation,
        carry_blocked=False,
    )
    if (
        config.margin.allow_short
        or config.margin.allow_financing
        or config.rules is not None
    ):
        raise ValueError("crypto 使用独立现金/合约账户；不接现货融资或通用退出规则")
    futures = None
    if kind == "usdm":
        rules, notes = profile.rules(
            names, dates, assume=assume, leverage=leverage, active=active
        )
        diagnostics["rule_provenance"] = notes
        diagnostics["rule_approximation_intervals"] += [
            dict(r, rule_family="risk") for r in notes if r.get("approximation")
        ]
        diagnostics["rule_assumptions"] += [
            r for r in notes if r["confidence"] == "assumption"
        ]
        expiry = np.full(len(names), len(times), dtype=np.int64)
        for j in range(len(names)):
            hits = np.flatnonzero(a["expires"][:, j])
            if len(hits) > 1:
                raise ValueError("单合约不能有多个到期行")
            if len(hits):
                expiry[j] = hits[0]
            elif np.all(a["lifecycle"][:, j] == "ended"):
                expiry[j] = -1
        futures = bt.Futures(
            is_future=np.ones(len(names), bool),
            multiplier=[p["multiplier"] for p in profiles],
            margin_rate=rules["margin_rate"],
            maint_rate=rules["maint_rate"],
            settlement=a["settlement"],
            expiry=expiry,
            fee_rate=fees["taker"][:, None],
            delivery_fee_rate=delivery_fees["delivery"][:, None],
            funding=a["funding"],
            exec_at_open=True,
            qty_step=a["qty_step"],
            min_qty=a["min_qty"],
            min_notional=a["min_notional"],
            liquidation=bt.Liquidation(
                a["mark_open"],
                a["mark_high"],
                a["mark_low"],
                rules["clearance_fee_rate"],
                rules["tier_cap"],
            ),
        )
        config = replace(config, min_qty=0.0, qty_step=0.0)
    else:
        # 第一期现货只用固定 N 向量；快照来源及整个使用范围均显式报告。
        minimum, step = [], []
        for j, p in enumerate(profiles):
            snapshot = min(p["rules"], key=lambda r: r["known_at"])
            minimum.append(snapshot["min_qty"])
            step.append(snapshot["qty_step"])
            mask = np.zeros(active.shape, bool)
            mask[:, j] = active[:, j]
            diagnostics["rule_assumptions"] += intervals(
                times,
                mask,
                names,
                parameter="spot_quantity_snapshot",
                source_url=snapshot["source_url"],
                reason="现货固定数量快照；最小名义不执行",
            )
        config = replace(config, min_qty=minimum, qty_step=step)
    market = bt.Market(
        exec_price=a["open"],
        close=a["close"],
        can_buy=a["tradable"],
        can_sell=a["tradable"],
        volume=a["volume_engine"],
        futures=futures,
        crypto_spot=kind == "spot",
    )
    signals = np.full(a["close"].shape, np.nan)
    signals[1:] = np.where(active[:-1], a["close"][:-1], np.nan)
    return CryptoMarket(
        market,
        times,
        names,
        signals,
        profile.costs(dates, fee_switch=fee_switch, slippage=slippage),
        config,
        copy.deepcopy(diagnostics),
        a,
        profiles,
        events,
        kind,
        ds,
    )
