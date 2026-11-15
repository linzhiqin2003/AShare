"""C4 真实归档例子与计时，只向 results 写汇总数字，不写逐行行情。"""

from __future__ import annotations

import argparse
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import panelbt as bt

from ashare.backtest.crypto import MarginBand, TwoWallet, crypto_market, two_wallet
from ashare.data import crypto
from ashare.data.crypto.timeutil import HOUR, millis

ROOT = Path(__file__).resolve().parents[1]
# Script execution puts bench/profile.py ahead of the stdlib profiler.
if str(ROOT / "bench") in sys.path:
    sys.path.remove(str(ROOT / "bench"))
import cProfile  # noqa: E402


def pair(ds, symbols, start=None, end=None):
    args = (ds, symbols, start or ds.meta["start"], end or ds.meta["end"])
    spot = crypto_market(*args, kind="spot", initial_cash=1000, allow_missing=True)
    usdm = crypto_market(
        *args,
        kind="usdm",
        initial_cash=1000,
        leverage=5,
        assume="previous",
        allow_missing=True,
    )
    return spot, usdm


def summary(r):
    fee = sum(
        float(np.sum(getattr(leg, k)))
        for leg in (r.spot, r.usdm)
        for k in ("commission", "tax", "other_fee", "slippage_cost", "impact_cost")
    )
    peak = np.maximum.accumulate(np.r_[1, r.nav])
    return dict(
        final_nav=float(r.nav[-1]),
        annualized=float(r.nav[-1] ** (8760 / len(r.nav)) - 1),
        funding_income=-float(r.usdm.funding.sum()),
        fees=fee,
        max_drawdown=float(np.max(1 - np.r_[1, r.nav] / peak)),
        transfers=r.diagnostics["transfers"],
        rescue_failures=r.diagnostics["rescue_failed_count"],
        liquidations=sum(r.diagnostics["usdm"]["liquidation_counts"].values()),
        residual_max_usdt=r.diagnostics["residual_max"],
        rounding_residual_max_usdt=r.diagnostics["rounding_residual_max"],
        underfill_residual_max_usdt=r.diagnostics["underfill_residual_max"],
        residual_rows=r.diagnostics["residual_rows"],
        tier_exceeded=r.diagnostics["usdm"]["tier_cap_exceeded"],
        valid=r.diagnostics["valid"],
    )


def strategy_run(spot, usdm, factor=1.0):
    # 总资金 30,000；每腿名义为权益的 .4，合约初始分配 .5 倍名义。
    # 剩余现货现金等于一腿名义，5 倍杠杆仅用于降低最低 IM，目标保证金率仍为 .5。
    reserve, margin, capital = 1.0, 0.5, 30000.0
    h = 1 / (1 + margin + reserve)
    initial = ((1 + reserve) * h * capital, margin * h * capital)
    stats = dict(small_order_deferred=0, max_target_per_leg_per_coin=0.0)

    def plan(t, s, u):
        if not t:
            return None
        total = s.equity + u.wallet_balance + u.unrealized
        nominal = h * total / len(spot.profiles)
        coins = nominal / spot.signal_prices[t]
        contracts = np.floor(coins / usdm.futures.multiplier + 1e-9)
        opening = contracts - np.abs(u.shares)
        # 日频决策允许不交易：新增整张名义不足最小下单额的 1.5 倍时，整组延后。
        # 只用上一结算价；1.5 给开盘价跳变留空间，既不提交受阻整张，也不产生新现货敞口。
        new_nominal = opening * usdm.futures.multiplier * usdm.futures.settlement[t - 1]
        minimum = usdm.futures.min_notional[t]
        if np.any((opening > 0) & (new_nominal < 1.5 * minimum)):
            stats["small_order_deferred"] += 1
            return None
        stats["max_target_per_leg_per_coin"] = max(
            stats["max_target_per_leg_per_coin"], nominal
        )
        return np.tile(
            [h / len(spot.profiles), -h / len(spot.profiles)], (len(spot.profiles), 1)
        )

    start = time.perf_counter()
    r = two_wallet(
        spot,
        usdm,
        plan,
        initial=initial,
        kind="weights",
        trade_hour=0,
        hedge="follow_spot",
        transfer=MarginBand(0.5, 0.25, 1),
        transfer_margin_factor=factor,
        residual_limit=250,
        record_trades=False,
    )
    elapsed = time.perf_counter() - start
    assert r.diagnostics["underfill_residual_max"] < 1e-7
    bounds = np.nan_to_num(spot.signal_prices * usdm.futures.multiplier)
    assert np.all(np.abs(r.rounding_residual_notional) <= bounds + 1e-7)
    assert r.diagnostics["valid"]
    assert stats["max_target_per_leg_per_coin"] <= 50000
    return (
        r,
        elapsed,
        dict(
            capital=capital,
            initial=initial,
            h=h,
            cash_reserve_ratio=reserve,
            target_margin_ratio=margin,
            rebalance_hours=24,
            trade_hour_utc=0,
            leverage=5,
            pre_fund=True,
            pre_fund_buffer=0.02,
            min_order_safety_factor=1.5,
            **stats,
        ),
    )


def raw_funding(raw, start, end, units):
    associated = {
        round(int(x["fundingTime"]) / HOUR) * HOUR: x for x in raw["funding_prices"]
    }
    mark_open = {x["time"]: x["open"] for x in raw["markPriceKlines"]}
    records = [x for x in raw["fundingRate"] if start < x["time"] <= end]
    values = {}
    for x in records:
        a = associated.get(x["time"])
        price = float(a["markPrice"]) if a and a["markPrice"] else mark_open[x["time"]]
        values[x["time"]] = units * price * x["rate"]
    return values


def hand_checks(ds):
    inputs = ds.inputs()
    raw = inputs["data"]["usdm:BTCUSDT"]
    spot_raw = inputs["data"]["spot:BTCUSDT"]
    # 独立的一次现金储备救援：固定币数、不再平衡；从原始开盘、结算与费用重算。
    start, end = millis("2021-01-01"), millis("2021-01-15")
    spot, usdm = pair(ds, ["BTCUSDT"], start, end)
    entry_time = start + HOUR
    entry = next(x["open"] for x in raw["klines"] if x["time"] == entry_time)
    spot_entry = next(x["open"] for x in spot_raw["klines"] if x["time"] == entry_time)
    qty = 0.01
    previous_mark = next(
        x["close"] for x in raw["markPriceKlines"] if x["time"] == start
    )
    pre_initial = qty * previous_mark * 0.1
    pre = TwoWallet(
        spot,
        usdm,
        initial=(1000.0, pre_initial),
        transfer=MarginBand(0.5, 0.25, 1),
        hedge="follow_spot",
        allow_residual=True,
    )
    pre.step()
    pre.step({"BTC": (qty, -qty)})
    pre_expected = qty * previous_mark * (0.2 + 0.0004 + 0.02) - pre_initial
    pre_actual = pre.transfers[0]["actual"]
    np.testing.assert_allclose(pre_actual, pre_expected, atol=1e-10, rtol=1e-12)
    prefund_check = dict(
        source="raw previous mark close; new-target IM, fee and buffer",
        initial_wallet=pre_initial,
        expected_amount=pre_expected,
        engine_amount=pre_actual,
        difference=pre_actual - pre_expected,
    )
    initial = (1000.0, qty * entry * 0.5 + qty * entry * 0.0004)
    tw = TwoWallet(
        spot,
        usdm,
        initial=initial,
        transfer=MarginBand(0.5, 0.25, 1, pre_fund=False),
        hedge="follow_spot",
        record_trades=True,
        allow_residual=True,
        allow_tier_exceeded=True,
    )
    # 原始记录独立钱包余额；开仓后固定空 .01 BTC，资金费按右边界输入计算。
    funding = raw_funding(raw, entry_time, end, -qty)
    closes = {x["time"]: x["close"] for x in raw["markPriceKlines"]}
    wallet = initial[1] - qty * entry * 0.0004
    expected = None
    for t, at in enumerate(spot.times):
        if t > 1:
            previous = closes[int(at - HOUR)]
            unrealized = -qty * (previous - entry)
            nominal = qty * previous
            eq = wallet + unrealized
            request = (
                0.5 * nominal - eq if eq / nominal < 0.25 or eq / nominal > 1 else 0
            )
            if request > 0:
                available = initial[0] - qty * spot_entry * (1 + 0.001)
                expected = dict(
                    row=t,
                    requested=request,
                    actual=min(request, available),
                    pre_wallet=wallet,
                    pre_unrealized=unrealized,
                    pre_nominal=nominal,
                )
        tw.step({"BTC": (qty, -qty)} if t == 1 else None)
        if expected is not None:
            observed = tw.transfers[-1]
            np.testing.assert_allclose(
                observed["requested"], expected["requested"], rtol=1e-12, atol=1e-10
            )
            np.testing.assert_allclose(
                observed["actual"], expected["actual"], rtol=1e-12, atol=1e-10
            )
            assert tw.state[1].shares[0] * usdm.futures.multiplier[0] == -qty
            rescue = dict(
                source="independent raw trade opens, mark closes, funding events; fixed coin position",
                expected_amount=expected["actual"],
                pre_wallet=expected["pre_wallet"],
                pre_unrealized=expected["pre_unrealized"],
                pre_nominal=expected["pre_nominal"],
                engine_amount=observed["actual"],
                difference=observed["actual"] - expected["actual"],
                pre_margin_ratio=(expected["pre_wallet"] + expected["pre_unrealized"])
                / expected["pre_nominal"],
            )
            break
        if t >= 1:
            wallet -= funding.get(int(at + HOUR), 0)
    assert expected is not None, "固定窗口必须出现一次救援"

    start, end = millis("2024-05-31T23:00"), millis("2024-06-02")
    spot, usdm = pair(ds, ["BTCUSDT"], start, end)
    plan = np.full((len(spot.times), 1, 2), np.nan)
    plan[1, 0] = qty, -qty
    r = two_wallet(
        spot,
        usdm,
        plan,
        initial=(1000, 1000),
        hedge="follow_spot",
        record_trades=True,
        allow_residual=True,
    )
    events = raw_funding(raw, start + HOUR, end, -qty)
    assert sorted(events) == [start + 9 * HOUR, start + 17 * HOUR, start + 25 * HOUR]
    expected_funding = sum(events.values())
    for boundary, value in events.items():
        np.testing.assert_allclose(
            r.usdm.funding[(boundary - start) // HOUR - 1],
            value,
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        r.usdm.positions[1:, 0] * usdm.futures.multiplier[0], -qty, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        r.usdm.funding.sum(), expected_funding, rtol=1e-12, atol=1e-12
    )
    funding_check = dict(
        source="independent raw fundingRate and event-associated mark prices",
        events=len(events),
        expected_income=-expected_funding,
        engine_income=-float(r.usdm.funding.sum()),
    )
    return dict(prefund=prefund_check, rescue=rescue, funding_day=funding_check)


def performance(ds):
    spot, usdm = pair(ds, ["BTCUSDT", "ETHUSDT"])
    profiler = cProfile.Profile()
    profiler.enable()
    r, profiled, params = strategy_run(spot, usdm)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(18)
    # 只保存函数名和时间，不保存含本机路径的原始 profile。
    top = []
    for (file, _line, name), (_cc, nc, own, cumulative, _callers) in sorted(
        pstats.Stats(profiler).stats.items(), key=lambda x: x[1][2], reverse=True
    )[:18]:
        top.append(
            dict(
                function=name,
                file=Path(file).name,
                calls=nc,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    elapsed = [strategy_run(spot, usdm)[1] for _ in range(3)]
    w = np.full(usdm.market.shape, np.nan)
    w[np.arange(len(w)) % 24 == 1] = -0.25
    single = []
    stepped = []
    for _ in range(3):
        start = time.perf_counter()
        usdm.run(w, allow_tier_exceeded=True, record_trades=False)
        single.append(time.perf_counter() - start)
        sim = bt.Simulator(usdm.market, usdm.costs, usdm.config)
        start = time.perf_counter()
        for row in w:
            sim.step(row)
        stepped.append(time.perf_counter() - start)
    return dict(
        hours=len(spot.times),
        symbols=2,
        two_wallet_seconds=elapsed,
        single_crypto_run_seconds=single,
        single_simulator_loop_seconds=stepped,
        median_two_wallet_seconds=float(np.median(elapsed)),
        median_single_crypto_run_seconds=float(np.median(single)),
        median_single_loop_seconds=float(np.median(stepped)),
        coordinator_extra_us_per_row=float(
            (np.median(elapsed) - 2 * np.median(stepped)) / len(w) * 1e6
        ),
        gross_us_per_row=float(np.median(elapsed) / len(w) * 1e6),
        ratio_to_single_run=float(np.median(elapsed) / np.median(single)),
        profiled_seconds=profiled,
        profile_top=top,
        summary=summary(r),
        parameters=params,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()
    from ashare.paths import dataset_dir

    ds = crypto.open(dataset_dir("crypto", args.data, cwd_default=True))
    perf = performance(ds)
    (ROOT / "bench/results/crypto_wallets_performance.json").write_text(
        json.dumps(perf, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in perf.items() if k != "profile_top"}, indent=2),
        flush=True,
    )
    if args.performance_only:
        return
    spot, usdm = pair(ds, ["BTCUSDT"])
    results = {}
    for factor in (1, 1.5):
        r, elapsed, parameters = strategy_run(spot, usdm, factor)
        results[str(factor)] = summary(r) | dict(seconds=elapsed)
    report = dict(
        version=ds.version,
        start_ms=int(ds.meta["start"]),
        end_exclusive_ms=int(ds.meta["end"]),
        hours=len(spot.times),
        parameters=parameters,
        sensitivity=results,
        hand_checks=hand_checks(ds),
        interpretation="调仓前补资与现金储备救援；分别报告取整与成交不足残余",
    )
    (ROOT / "bench/results/crypto_wallets_example.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
