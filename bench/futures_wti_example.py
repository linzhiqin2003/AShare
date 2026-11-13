"""Synthetic WTI-style monthly contract chain through `book_from_table` (no data).

    python bench/futures_wti_example.py

Long 5 front-month contracts, rolled 5 trading rows before each last trading
day. Prices are synthetic (a random-walk spot with contango), so the numbers
only show the mechanics: P&L comes from each contract's settlement, rolls pay
fees and slippage twice, and the continuous-price return differs.
"""

import numpy as np
import polars as pl
from panelbt.futures import roll_targets

from ashare.backtest.futures import book_from_table


def chain(seed=7):
    """Long tables `daily` and `contracts` in the unified field names."""
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64("2024-01-02"), np.datetime64("2025-12-31"))
    days = days[np.is_busday(days)]
    dates = np.array([str(d).replace("-", "") for d in days])
    spot = 75 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    rows, contracts = [], []
    for year in (2024, 2025, 2026):
        for month in range(1, 13):
            # Last trade: 3 business days before the 25th of the prior month.
            prior = np.datetime64(f"{year}-{month:02}") - np.timedelta64(1, "M")
            ltd = np.busday_offset(np.datetime64(f"{prior}-25"), -3, roll="backward")
            listed = ltd - np.timedelta64(365, "D")
            if ltd < days[0] or listed > days[-1]:
                continue
            symbol = f"CL{year % 100:02}{month:02}.NYMEX"
            contracts.append(
                dict(
                    symbol=symbol,
                    multiplier=1000.0,
                    last_trade_date=str(ltd).replace("-", ""),
                )
            )
            for k in np.flatnonzero((days >= listed) & (days <= ltd)):
                tau = (ltd - days[k]).astype(int) / 365
                s = spot[k] * np.exp(0.05 * tau + rng.normal(0, 0.002))
                o = s * np.exp(rng.normal(0, 0.005))
                rows.append(
                    dict(
                        date=dates[k],
                        symbol=symbol,
                        open=o,
                        close=s,
                        settlement=round(s, 2),
                        volume=1000.0,
                    )
                )
    return pl.DataFrame(rows), pl.DataFrame(contracts)


def run_example():
    daily, contracts = chain()
    ext = book_from_table(
        daily,
        contracts,
        margin_rate=0.10,
        fee_per_contract=1.5,
        slippage=1e-4,
        initial_cash=1e6,
    )
    # Contracts are in last-trade order; those expiring after the sample all
    # read T, so keep the first of them (roll_targets needs increasing rows).
    keep = int(np.searchsorted(ext.expiry, len(ext.dates))) + 1
    plan = roll_targets(ext.expiry[:keep], len(ext.dates), 5, 5.0)
    result = ext.book.run(plan, kind="lots")
    return ext, result


if __name__ == "__main__":
    ext, r = run_example()
    print(
        f"rows {len(ext.dates)}, contracts {len(ext.symbols)}, trades {r.diagnostics['trades']}"
    )
    print(
        f"final equity {r.equity[-1]:.2f}, fees {r.commission.sum():.2f}, "
        f"slippage {r.slippage_cost.sum():.2f}, forced closes {len(r.expiry_events)}"
    )
