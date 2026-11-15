"""C3 可复现例子：只写汇总数字；输入目录与 Quant 源文件由参数提供。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ashare.backtest.crypto import crypto_market, daily_targets
from ashare.data import crypto
from ashare.data.crypto.dataset import write_panel
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import DAY, HOUR, millis

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (20, 60, 120)
VOL_WINDOW = 30
VOL_TARGET = 0.20


def signals(ds, kind, *, trade_hour=0, symbols=("BTCUSDT", "ETHUSDT"), gross_cap=None):
    """日线来自 C2；收盘动量投票 × 20% 年波动预算 / 币数，最后截断总名义。"""
    names = [kind + ":" + s for s in symbols]
    frame = ds.daily(trade_hour=trade_hour, instruments=names, invalid_prices="carry")
    wide = frame.pivot(on="instrument", index="day", values="close").sort("day")
    px = wide.select(names).to_numpy()
    votes = np.zeros_like(px)
    for lookback in WINDOWS:
        votes[lookback:] += np.sign(px[lookback:] / px[:-lookback] - 1) / len(WINDOWS)
    ret = np.full_like(px, np.nan)
    ret[1:] = px[1:] / px[:-1] - 1
    vol = np.full_like(px, np.nan)
    for i in range(VOL_WINDOW, len(px)):
        vol[i] = np.std(ret[i - VOL_WINDOW + 1 : i + 1], axis=0, ddof=1) * np.sqrt(365)
    weights = np.divide(
        votes * VOL_TARGET / len(symbols),
        vol,
        out=np.zeros_like(px),
        where=np.isfinite(vol) & (vol > 0),
    )
    weights[: max(WINDOWS)] = 0
    weights[~np.isfinite(px)] = 0
    if kind == "spot":
        weights = np.maximum(weights, 0)
    cap = gross_cap or (1.0 if kind == "spot" else 2.0)
    weights /= np.maximum(1.0, np.abs(weights).sum(axis=1) / cap)[:, None]
    return wide["day"].to_numpy(), weights


def summary(result, initial_cash, hours):
    nav = result.nav
    peak = np.maximum.accumulate(np.r_[1.0, nav])
    cost = sum(
        float(np.sum(getattr(result, k)))
        for k in (
            "commission",
            "tax",
            "other_fee",
            "delivery_fee",
            "slippage_cost",
            "impact_cost",
        )
        if getattr(result, k) is not None
    )
    funding = float(np.sum(result.funding)) if result.funding is not None else 0.0
    diag = result.diagnostics["crypto"]
    return dict(
        final_nav=float(nav[-1]),
        annualized=float(nav[-1] ** (8760 / hours) - 1),
        max_drawdown=float(np.max(1 - np.r_[1.0, nav] / peak)),
        funding_paid=funding,
        funding_pct_initial=funding / initial_cash,
        fees=cost,
        fees_pct_initial=cost / initial_cash,
        clearance_fee=float(np.sum(result.clearance_fee))
        if result.clearance_fee is not None
        else 0.0,
        liquidation_count=sum(diag["liquidation_counts"].values()),
        liquidation_by_stage=diag["liquidation_counts"],
        tier_cap_exceeded=diag["tier_cap_exceeded"],
        valid=diag["valid"],
        funding_price_approximation_cells=diag["funding_price_approximation_cells"],
        rule_assumption_intervals=len(diag["rule_assumptions"]),
        missing_intervals=len(diag["missing_intervals"]),
        halt_intervals=len(diag["halt_intervals"]),
    )


def trend(ds, kind):
    capital = 1000.0
    m = crypto_market(
        ds,
        ["BTCUSDT", "ETHUSDT"],
        ds.meta["start"],
        ds.meta["end"],
        kind=kind,
        initial_cash=capital,
        leverage=2.0,
        assume="conservative" if kind == "usdm" else None,
        allow_missing=True,
    )
    days, w = signals(ds, kind)
    result = m.run(daily_targets((days, w)))
    return summary(result, capital, len(m.times))


def initial_margin_cash(entry, mark, units, leverage, fee):
    """足额开仓所需资金，供手算用例复核。"""
    return float(
        np.nextafter(units * (mark / leverage + entry * fee + entry - mark), np.inf)
    )


def liquidation_check(ds):
    start, end = "2021-05-19", "2021-05-20"
    data = ds.arrays(
        ["open", "mark_open", "mark_high", "mark_low", "settlement", "funding"],
        start=start,
        end=end,
        instruments=["usdm:BTCUSDT"],
    )
    entry = float(data["open"][0, 0])
    fee = 0.0004
    mmr = 0.004
    clearance_rate = (
        0.02  # BTCUSDT 2021-02-07 04:00 – 2021-05-20 09:00 UTC (announcement aeeccb6d…)
    )
    units = 1.0
    capital = initial_margin_cash(entry, float(data["mark_open"][0, 0]), units, 10, fee)
    m = crypto_market(
        ds,
        ["BTCUSDT"],
        start,
        end,
        kind="usdm",
        leverage=10.0,
        initial_cash=capital,
        assume="conservative",
    )
    contracts = units / m.market.futures.multiplier[0]
    targets = np.full((24, 1), np.nan)
    targets[0, 0] = contracts
    result = m.run(targets, kind="shares", record_trades=True, allow_tier_exceeded=True)
    cash = capital - entry * fee
    basis = entry
    paid = 0.0
    expected = None
    for i in range(24):
        op = float(data["mark_open"][i, 0])
        low = float(data["mark_low"][i, 0])
        line = (units * basis - cash) / (units * (1 - mmr))
        if op <= line or low <= line:
            price = op if op <= line else line
            normal_fee = price * units * fee
            remainder = cash + units * (price - basis) - normal_fee
            clearance = min(clearance_rate * units * price, max(remainder, 0.0))
            expected = dict(
                hour=i,
                liquidation_price=price,
                closing_fee=normal_fee,
                clearance_fee=clearance,
                cash=max(0.0, remainder - clearance),
                bankruptcy_deficit=max(0.0, -remainder),
                funding_paid_before=paid,
                mark_low=float(data["mark_low"][:, 0].min()),
                opening_fee=entry * fee,
            )
            break
        close = float(data["settlement"][i, 0])
        cash += units * (close - basis)
        basis = close
        fund = float(data["funding"][i, 0]) * contracts
        cash -= fund
        paid += fund
    assert expected is not None
    closing = [t for t in result.trades if t.qty < 0]
    assert len(closing) == 1
    observed = dict(
        hour=closing[0].day,
        liquidation_price=closing[0].price,
        closing_fee=closing[0].commission,
        clearance_fee=float(result.clearance_fee.sum()),
        cash=float(result.cash[-1]),
        bankruptcy_deficit=float(result.bankruptcy_deficit.sum()),
        funding_paid_before=float(result.funding.sum()),
    )
    for key in observed:
        np.testing.assert_allclose(
            observed[key], expected[key], rtol=1e-11, atol=1e-8, err_msg=key
        )
    return dict(
        source="real mark and trade hours",
        assumptions="MMR 0.4% (observed continuity), clearance 2.00% (announcement 2021-02-07); first-tier uncertainty explicitly allowed",
        initial_cash=capital,
        units=units,
        expected=expected,
        observed=observed,
        tier_cap_exceeded=result.coverage["tier_cap_exceeded"],
        opening_notional=entry,
        actual_opening_contracts=result.trades[0].qty,
    )


def funding_check(ds):
    m = crypto_market(
        ds,
        ["BTCUSDT"],
        "2024-06-01",
        "2024-06-02",
        kind="usdm",
        assume="conservative",
        initial_cash=1000,
        leverage=2.0,
    )
    # 期望值只读冻结原始结算记录和档案，不能累加被测的 funding 面板。
    inputs = ds.inputs()
    profile = next(p for p in inputs["profiles"] if p["instrument"] == "usdm:BTCUSDT")
    raw = inputs["data"]["usdm:BTCUSDT"]
    unit = profile["multiplier"]
    contracts = 0.01 / unit
    assert m.market.futures.multiplier[0] == unit
    start, end = millis("2024-06-01"), millis("2024-06-02")
    records = sorted(
        (x for x in raw["fundingRate"] if start < x["time"] <= end),
        key=lambda x: x["time"],
    )
    expected_times = [start + h * HOUR for h in (8, 16, 24)]
    assert [x["time"] for x in records] == expected_times
    assert all(x["interval"] == 8 for x in records)
    associated = {}
    for x in raw["funding_prices"]:
        observed = int(x["fundingTime"])
        boundary = round(observed / HOUR) * HOUR
        if start < boundary <= end:
            assert abs(observed - boundary) <= 1000
            assert boundary not in associated
            associated[boundary] = x
    expected_by_hour = np.zeros(24)
    for x in records:
        price = associated[x["time"]]
        assert float(price["fundingRate"]) == x["rate"]
        assert float(price["markPrice"]) > 0
        row = (x["time"] - start) // HOUR - 1
        expected_by_hour[row] = contracts * unit * float(price["markPrice"]) * x["rate"]
    w = np.full((24, 1), np.nan)
    w[0, 0] = contracts
    r = m.run(w, kind="shares", record_trades=True)
    # t=00:00 开仓不付当时的费；三次后续结算前后均持有同一张数，没有中途退出。
    assert len(r.trades) == 1 and r.trades[0].day == 0
    assert r.trades[0].qty == contracts
    np.testing.assert_array_equal(r.positions[:, 0], np.full(24, contracts))
    np.testing.assert_allclose(r.funding, expected_by_hour, atol=1e-12, rtol=1e-12)
    expected_funding = float(expected_by_hour.sum())
    entry = next(x["open"] for x in raw["klines"] if x["time"] == start)
    final_mark = next(
        x["close"] for x in raw["markPriceKlines"] if x["time"] == end - HOUR
    )
    expected_cash = (
        1000
        + contracts * unit * (final_mark - entry)
        - contracts * unit * entry * 0.0005
        - expected_funding
    )
    np.testing.assert_allclose(r.cash[-1], expected_cash, atol=1e-10, rtol=1e-12)
    return dict(
        source="independent frozen raw funding records and associated mark prices",
        date="2024-06-01",
        events=len(records),
        settlement_hours_utc=[8, 16, 24],
        contracts=contracts,
        multiplier=unit,
        positions_verified=True,
        expected_funding=expected_funding,
        engine_funding=float(r.funding.sum()),
        expected_cash=float(expected_cash),
        engine_cash=float(r.cash[-1]),
    )


def delivery_check(folder):
    inputs = json.loads((ROOT / "test/fixtures/crypto/inputs.json").read_text())
    frame, notes = build_frame(**inputs)
    ds = write_panel(
        folder,
        frame,
        profiles=inputs["profiles"],
        events=inputs["events"],
        data=inputs["data"],
        notes=notes,
    )
    symbol = "usdm:FUTUSDT_240628"
    risk = dict(
        maint_rate=0.004,
        tier_cap=50000,
        max_leverage=125,
        clearance_fee_rate=0.0125,
        reason="synthetic hand-check only",
        source_url="https://example.test/synthetic",
    )
    m = crypto_market(
        ds,
        [symbol],
        inputs["start"],
        inputs["end"],
        kind="usdm",
        assume=dict(rules={symbol: risk}),
        initial_cash=100.0,
        leverage=10.0,
    )
    w = np.full((12, 1), np.nan)
    w[0, 0] = 2
    r = m.run(w, kind="shares", roll_before=None, record_trades=True)
    entry = m.arrays["open"][0, 0]
    settle = m.arrays["settlement"][7, 0]
    unit = m.futures.multiplier[0]
    expected = 100 + 2 * unit * (settle - entry) - 2 * unit * (entry + settle) * 0.0005
    np.testing.assert_allclose(r.cash[-1], expected, atol=1e-12)
    assert len(r.expiry_events) == 1 and r.expiry_events[0].day == 7
    return dict(
        source="synthetic C2 shared fixture; copied real dataset has no quarter contracts",
        expected_cash=float(expected),
        engine_cash=float(r.cash[-1]),
        expiry_row=7,
        delivery_fee=float(r.delivery_fee.sum()),
        funding=float(r.funding.sum()),
    )


def quant_compare(ds, account_path):
    """只读加载指定 account.py，禁止给外部项目写 pyc。最后一次共同清仓对齐估值边界。"""
    account_path = Path(account_path).resolve()
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(account_path.parents[2]))
    try:
        spec = importlib.util.spec_from_file_location(
            "quant.backtest.account", account_path
        )
        quant = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quant
        spec.loader.exec_module(quant)
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old
    days, w = signals(ds, "usdm", trade_hour=1, symbols=("BTCUSDT",), gross_cap=1.0)
    times = days + DAY
    use = (times >= ds.meta["start"]) & (times + HOUR <= ds.meta["end"])
    days, w, times = days[use], w[use], times[use]
    w[-1] = 0.0
    raw = ds.read(instruments=["usdm:BTCUSDT"])
    lookup = {r["time"]: r for r in raw.select("time", "open").iter_rows(named=True)}
    prices = np.array([lookup[int(t)]["open"] for t in times])
    hour_frame = pd.DataFrame(
        {k: raw[k].to_numpy() for k in ("open", "high", "low")},
        index=pd.to_datetime(raw["time"].to_numpy(), unit="ms"),
    )
    f = raw.filter(pl.col("funding_rate").is_finite())
    fund = pd.Series(
        f["funding_rate"].to_numpy(),
        index=pd.to_datetime(f["time"].to_numpy() + HOUR, unit="ms"),
    )
    qmarket = quant.Market(
        pd.to_datetime(days - HOUR, unit="ms"), prices, prices, hour_frame, fund
    )
    q = quant.run(
        qmarket,
        {"btc": "perp"},
        lambda state: {"btc": float(w[state.i, 0])},
        fee_perp_bp=5.0,
        margin=1.0,
        maint=0.004,
    )
    capital = 1000.0
    m = crypto_market(
        ds,
        ["BTCUSDT"],
        int(times[0]),
        int(times[-1] + HOUR),
        kind="usdm",
        assume="conservative",
        initial_cash=capital,
        leverage=1.0,
    )
    c = m.run(daily_targets((days, w), trade_hour=1))
    expected_tau = pd.to_datetime(times, unit="ms")
    assert qmarket.tau.equals(expected_tau)
    return dict(
        strategy="BTC 20/60/120-day votes, 30-day vol, 20% annual target, gross cap 1, 01:00 UTC; final common liquidation to cash",
        periods=len(days),
        crypto_final_nav=float(c.nav[-1]),
        quant_final_nav=float(q.equity.iloc[-1]),
        nav_difference=float(c.nav[-1] - q.equity.iloc[-1]),
        crypto_funding_pct_initial=float(c.funding.sum() / capital),
        quant_funding_pct_initial=float(q.funding_paid),
        crypto_fees_pct_initial=float(c.commission.sum() / capital),
        quant_fees_pct_initial=float(q.fees),
        crypto_liquidation_count=sum(
            c.diagnostics["crypto"]["liquidation_counts"].values()
        ),
        quant_liquidated=q.liquidated_at is not None,
        quant_capital_violations=q.capital_violations,
        quant_min_futures_buffer=float(q.min_futures_buffer),
        differences=[
            dict(
                item="funding",
                crypto="official associated mark, early boundary mark open",
                quant="trade-kline open; missing-hour fallback uses decision execution price",
            ),
            dict(
                item="liquidation",
                crypto="mark OHLC crossing price; normal closing fee plus separate capped clearance",
                quant="trade OHLC adverse extreme; maintenance rate used as clearance proxy",
            ),
            dict(
                item="deficit",
                crypto="explicit bankruptcy_deficit and bankruptcy latch",
                quant="negative futures cash clipped to zero without deficit ledger",
            ),
            dict(
                item="wallets",
                crypto="single standalone futures equity collateral",
                quant="separate spot cash and futures wallet, reset to nominal * margin after decisions",
            ),
            dict(
                item="fees",
                crypto="4 bp until assumed 2023-06-01; then 5 bp",
                quant="constant 5 bp in this comparison",
            ),
            dict(
                item="sizing",
                crypto="pre-fee equity, integer contracts, dated minimum-order filters",
                quant="continuous quantities, iterative post-fee equity, no minimum filters",
            ),
            dict(
                item="valuation",
                crypto="hourly mark settlement and mark-open sizing",
                quant="trade execution prices between daily decisions",
            ),
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.environ.get("CRYPTO_DATA_ROOT", "data/crypto"))
    p.add_argument("--quant-account", default=os.environ.get("QUANT_ACCOUNT_PATH"))
    p.add_argument("--version")
    p.add_argument("--allow-inferred-funding", action="store_true")
    p.add_argument("--output", default="bench/results/crypto_examples.json")
    p.add_argument("--synthetic-dir", default="tmp/crypto-quarter-handcheck")
    args = p.parse_args()
    ds = crypto.open(
        args.data, args.version, allow_inferred_funding=args.allow_inferred_funding
    )
    results = dict(
        data_version=ds.version,
        start=str(np.datetime64(ds.meta["start"], "ms")),
        end=str(np.datetime64(ds.meta["end"], "ms")),
        rows=ds.meta["rows"],
        parameters=dict(
            momentum_days=WINDOWS,
            vol_days=VOL_WINDOW,
            annual_vol_target=VOL_TARGET,
            capital_usdt=1000.0,
            usdm_target_gross_cap=2.0,
            spot_target_gross_cap=1.0,
            trade_hour=0,
            fee_switch="2023-06-01",
            risk_assume="conservative",
            missing_policy="explicit flat/untradable on diagnosed spot archive gaps",
        ),
        strategies={},
    )
    for kind in ("usdm", "spot"):
        results["strategies"][kind] = trend(ds, kind)
        print(kind, results["strategies"][kind], flush=True)
    results["hand_checks"] = dict(
        liquidation=liquidation_check(ds),
        funding=funding_check(ds),
        delivery=delivery_check(args.synthetic_dir),
    )
    if args.quant_account:
        results["quant_comparison"] = quant_compare(ds, args.quant_account)
    else:
        results["quant_comparison"] = dict(
            status="not_run", reason="pass --quant-account or QUANT_ACCOUNT_PATH"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
