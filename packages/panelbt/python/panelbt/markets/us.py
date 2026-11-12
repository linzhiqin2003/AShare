"""US equity profile (NYSE/Nasdaq listed common stock and ADRs), in US dollars.

The viewpoint is a mainland-China individual with an overseas broker
(non-resident, W-8BEN filed). Every dated charge cites the section of
``docs/markets/us_rules.md`` (AShare repository); the full table, unverified
items and approximations are in ``docs/markets/PROFILES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..models import MarginModel
from .hk import RegionalProfile, _days, _flag, build_schedule

# Dividend withholding presets (us_rules.md 6.1).
US_WITHHOLDING = MappingProxyType(
    {
        "china_treaty": 0.10,  # mainland resident with W-8BEN, treaty art. 9(2)
        "no_treaty": 0.30,  # non-resident without treaty or W-8BEN (incl. HK)
        "us_person": 0.0,  # W-9; taxed in the investor's return, not withheld
    }
)

# ---------------------------------------------------------------------------
# Sell-side regulatory charges (us_rules.md 1.1, 1.2). No stamp duty (1.3).
# SEC Section 31 fee in $ per $1 of sales; FINRA TAF in $ per share sold with a
# per-trade cap (the cap is not expressible: ignored, an overcharge).
#
# SEC rates, effective dates (1.1 table):
#   2010-01-01 25.70/M  table start; rate in force since 2009-04-10
#   2010-01-15 12.70  2010-04-01 16.90  2011-01-21 19.20  2012-02-21 18.00
#   2012-04-01 22.40  2013-05-25 17.40  2014-03-18 22.10
#              (2009-04-10 to 2014-03-18: secondary compilation only, UNVERIFIED)
#   2015-02-14 18.40  (two sources)
#   2016-02-16 21.80  2017-07-04 23.10  2018-05-22 13.00 (official)
#   2019-04-16 20.70  (date single-source)
#   2020-02-18 22.10  2021-02-25 5.10  2022-05-14 22.90  2023-02-27 8.00
#   2024-05-22 27.80  2025-05-14 0.00  2026-04-04 20.60 (official)
#   The 2026 rate holds until 60 days after the FY2027 appropriation: recheck.
# FINRA TAF per share / cap (1.2 table):
#   2010-01-01 0.000075/3.75 (UNVERIFIED, 2011 SEC notice summary)
#   2011-07-01 0.000090/4.50  2012-03-01 0.000095/4.75  2012-07-01 0.000119/5.95
#   2022-01-01 0.000130/6.49  2023-01-01 0.000145/7.27  2024-01-01 0.000166/8.30
#   2026-01-01 0.000195/9.79  2026-10-01 0 (three-month pause, SR-FINRA-2026-021)
#   2027-01-01 0.000195/9.79 (resumes; later increases deferred, single source)
# ---------------------------------------------------------------------------

_M = 1e-6

US_ITEMS = (
    ("2010-01-01", {"sec_fee": 25.70 * _M, "taf": 0.000075, "taf_cap": 3.75}),
    ("2010-01-15", {"sec_fee": 12.70 * _M}),
    ("2010-04-01", {"sec_fee": 16.90 * _M}),
    ("2011-01-21", {"sec_fee": 19.20 * _M}),
    ("2011-07-01", {"taf": 0.000090, "taf_cap": 4.50}),
    ("2012-02-21", {"sec_fee": 18.00 * _M}),
    ("2012-03-01", {"taf": 0.000095, "taf_cap": 4.75}),
    ("2012-04-01", {"sec_fee": 22.40 * _M}),
    ("2012-07-01", {"taf": 0.000119, "taf_cap": 5.95}),
    ("2013-05-25", {"sec_fee": 17.40 * _M}),
    ("2014-03-18", {"sec_fee": 22.10 * _M}),
    ("2015-02-14", {"sec_fee": 18.40 * _M}),
    ("2016-02-16", {"sec_fee": 21.80 * _M}),
    ("2017-07-04", {"sec_fee": 23.10 * _M}),
    ("2018-05-22", {"sec_fee": 13.00 * _M}),
    ("2019-04-16", {"sec_fee": 20.70 * _M}),
    ("2020-02-18", {"sec_fee": 22.10 * _M}),
    ("2021-02-25", {"sec_fee": 5.10 * _M}),
    ("2022-01-01", {"taf": 0.000130, "taf_cap": 6.49}),
    ("2022-05-14", {"sec_fee": 22.90 * _M}),
    ("2023-01-01", {"taf": 0.000145, "taf_cap": 7.27}),
    ("2023-02-27", {"sec_fee": 8.00 * _M}),
    ("2024-01-01", {"taf": 0.000166, "taf_cap": 8.30}),
    ("2024-05-22", {"sec_fee": 27.80 * _M}),
    ("2025-05-14", {"sec_fee": 0.0}),
    ("2026-01-01", {"taf": 0.000195, "taf_cap": 9.79}),
    ("2026-04-04", {"sec_fee": 20.60 * _M}),
    ("2026-10-01", {"taf": 0.0, "taf_cap": 0.0}),
    ("2027-01-01", {"taf": 0.000195, "taf_cap": 9.79}),
)

_US_MAPPING = {
    "tax_buy": lambda s: 0.0,
    "tax_sell": lambda s: 0.0,
    "fee_rate_buy": lambda s: 0.0,
    "fee_rate_sell": lambda s: s["sec_fee"],
    "fee_per_share_buy": lambda s: 0.0,
    "fee_per_share_sell": lambda s: s["taf"],
}

# Brokers round each fee up to the cent per fill (us_rules.md 1.1, 1.2; broker
# practice, unverified): at most one cent per nonzero charge.
CENT = 0.01


@dataclass(frozen=True)
class USProfile(RegionalProfile):
    """US equities: whole shares by default, sell-side SEC fee and FINRA TAF."""

    def _fill_extras(self, it, fill_value):
        cents = np.where(it["sec_fee"] > 0, CENT, 0.0) + np.where(
            it["taf"] > 0, CENT, 0.0
        )
        return {"fee_rate_sell": cents}

    def quantities(self, *, board=None, lot_sizes=None, n=None, fractional=False):
        if fractional:
            if n is None:
                raise ValueError("provide n (number of securities)")
            return np.zeros(n), np.zeros(n)
        return super().quantities(board=board, lot_sizes=lot_sizes, n=n)

    def config(self, *, fractional=False, **kwargs):
        """As `RegionalProfile.config`; `fractional=True` allows fractional shares.

        Dividend flags: `withholding` (default 10%, see `US_WITHHOLDING`),
        `adr` (bool N) and `adr_withholding` (default 10%, an assumption for
        Chinese issuers: ADR dividends are foreign-source and withheld by the
        issuer's country, us_rules.md 6.1).
        """
        if fractional:
            n = kwargs.pop("n", None)
            if n is None:
                raise ValueError("fractional=True requires n")
            kwargs["lot_sizes"] = np.zeros(n)
        return super().config(**kwargs)

    def _flag_names(self):
        return ("withholding", "adr", "adr_withholding")

    def withholding(
        self, n, dates=None, *, withholding=None, adr=None, adr_withholding=None
    ):
        """Dividend withholding per security (length N, or T×N with dates)."""
        w = self.dividend_rate if withholding is None else float(withholding)
        a = 0.10 if adr_withholding is None else float(adr_withholding)
        if not (0 <= w <= 1 and 0 <= a <= 1):
            raise ValueError("withholding rates: expected [0, 1]")
        rate = np.full(n, w)
        if adr is not None:
            rate = np.where(_flag(adr, n, "adr"), a, rate)
        return rate if dates is None else np.tile(rate, (len(_days(dates)), 1))


def settlement_days(dates) -> np.ndarray:
    """Settlement cycle by trade date: T+3, T+2 from 2017-09-05, T+1 from 2024-05-28.

    Metadata only (us_rules.md 4.2): the kernel has no settlement delay and
    lets sale proceeds fund same-day buys, i.e. margin-account availability.
    The T+3 start (1995-06-07) is unverified.
    """
    d = _days(dates)
    return np.where(
        d >= np.datetime64("2024-05-28"),
        1,
        np.where(d >= np.datetime64("2017-09-05"), 2, 3),
    )


US = USProfile(
    "US equities",
    "USD",
    252,
    build_schedule(US_ITEMS, _US_MAPPING),
    items=US_ITEMS,
    # Assumption: IBKR Pro Fixed, $0.005 per share, $1 minimum, 1% of trade
    # value maximum, both sides (us_rules.md 1.4). Regulatory fees are charged
    # on top (whether Fixed includes them is unverified: an overcharge).
    commission=(
        ("commission_rate", 0.0),
        ("commission_per_share", 0.005),
        ("commission_min", 1.0),
        ("commission_max_rate", 0.01),
    ),
    fill_value=10_000.0,
    dividend_rate=US_WITHHOLDING["china_treaty"],
    whole_shares=True,
)


def reg_t_margin(
    *,
    financing_rate,
    borrow_rate=0.003,
    periods_per_year=252,
    short_maintenance=0.30,
    margin_call="liquidate",
) -> USProfile:
    """US with a Reg T margin account (us_rules.md 5.3, 5.4).

    Initial 50% long and short (Reg T 150% short = proceeds + 50%); maintenance
    25% long and `short_maintenance` (30%) short. FINRA's per-share floors
    ($5/share, and 100% or $2.50/share under $5) cannot be expressed: 30% is
    optimistic for shorts under about $16.7. Annual rates are converted per
    period; `financing_rate` (the broker's margin loan rate) is required, and
    the 0.3% a year borrow fee is an assumption (easy-to-borrow level, 5.4).
    Its `config()` requires an explicit `delisting=` (see `config`).
    """
    margin = MarginModel(
        allow_short=True,
        long_initial=0.5,
        short_initial=0.5,
        long_maintenance=0.25,
        short_maintenance=short_maintenance,
        allow_financing=True,
        short_proceeds_usable=False,
        financing_rate=np.asarray(financing_rate, dtype=float) / periods_per_year,
        borrow_rate=np.asarray(borrow_rate, dtype=float) / periods_per_year,
        margin_call=margin_call,
    )
    return replace(US, name="US equities (Reg T margin account)", margin=margin)


__all__ = [
    "US",
    "USProfile",
    "US_ITEMS",
    "US_WITHHOLDING",
    "reg_t_margin",
    "settlement_days",
]
