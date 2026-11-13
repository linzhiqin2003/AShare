"""Hand check of `FuturesBook` against exchange rules on real data (read-only).

    python bench/futures_hand_check.py --root /path/to/data

RB2310.SHFE (rebar, 10 t/lot), exchange fees and margins of each day, no
broker add-on and no slippage, so every number follows the SHFE settlement
rules directly: open 10 long at the open of 2023-06-01, add 5 at the open of
2023-06-05, close 15 at the open of 2023-06-08. Prints each quantity from the
hand calculation and from the engine.
"""

import argparse

import numpy as np
import panelbt as bt

import ashare.data as ad

SYMBOL, START, END = "RB2310.SHFE", "2023-05-31", "2023-06-09"
PLAN = {"20230601": 10, "20230605": 15, "20230608": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    fut = ad.open(root).futures()
    inputs = fut.panelbt_inputs(symbols=[SYMBOL], start=START, end=END)
    f, m, dates = inputs["futures"], inputs["market"], inputs["dates"]
    raw = fut.panel(
        [
            "open",
            "settlement",
            "multiplier",
            "margin_long",
            "margin_short",
            "fee_open_rate",
            "fee_open_per_contract",
        ],
        symbols=[SYMBOL],
        start=START,
        end=END,
    )
    book = bt.FuturesBook(
        settlement=f.settlement,
        close=m["close"],
        exec_price=m["exec_price"],
        multiplier=f.multiplier,
        expiry=f.expiry,
        margin_rate=f.margin_rate,
        fee_rate=f.fee_rate,
        fee_per_contract=f.fee_per_contract,
        config=bt.Config(initial_cash=1e6, record_trades=True),
    )
    plan = np.full((len(dates), 1), np.nan)
    for day, lots in PLAN.items():
        plan[list(dates).index(day), 0] = lots
    r = book.run(plan, "lots")
    # ------------------------------------------------------- by hand
    o = np.asarray(raw["open"])[:, 0]
    s = np.asarray(raw["settlement"])[:, 0]
    mult = float(np.asarray(raw["multiplier"])[0, 0])
    rate = np.asarray(raw["fee_open_rate"])[:, 0]
    fixed = np.asarray(raw["fee_open_per_contract"])[:, 0]
    margin = np.fmax(np.asarray(raw["margin_long"]), np.asarray(raw["margin_short"]))[
        :, 0
    ]
    cash, lots, basis = 1e6, 0, 0.0
    rows = []
    for t, day in enumerate(dates):
        fee = 0.0
        if day in PLAN:
            q = PLAN[day] - lots
            fee = abs(q) * (mult * o[t] * rate[t] + fixed[t])
            if q > 0:  # open / add: basis becomes the weighted price
                basis = (lots * basis + q * o[t]) / (lots + q)
            else:  # close at the open: realized against the last settlement
                cash += -q * mult * (o[t] - basis)
            cash -= fee
            lots += q
        cash += lots * mult * (s[t] - basis)  # mark to today's settlement
        basis = s[t]
        held_margin = lots * mult * s[t] * margin[t]
        engine_fee = r.commission[t]
        rows.append(
            (
                day,
                o[t],
                s[t],
                rate[t],
                margin[t],
                lots,
                fee,
                engine_fee,
                cash,
                r.equity[t],
                held_margin,
                r.futures_margin[t],
            )
        )
    print(
        "| 日期 | 开盘 | 结算价 | 费率 | 保证金率 | 持仓 | 手续费（手算） | 手续费（引擎） | "
        "权益（手算） | 权益（引擎） | 占用保证金（手算） | 占用保证金（引擎） |"
    )
    print("|" + "---|" * 12)
    for row in rows:
        print(
            "| "
            + " | ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row)
            + " |"
        )
    worst = max(abs(a[8] - a[9]) + abs(a[6] - a[7]) + abs(a[10] - a[11]) for a in rows)
    print(f"\nmax abs difference: {worst:.3g} (multiplier {mult:g})")


if __name__ == "__main__":
    main()
