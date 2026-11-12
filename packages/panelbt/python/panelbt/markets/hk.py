"""Hong Kong market profiles: a local HK account and Southbound Stock Connect.

Both profiles backtest in Hong Kong dollars. Every dated charge below cites the
section of ``docs/markets/hk_rules.md`` (AShare repository) it comes from; the
full table, the unverified items and the approximations are documented in
``docs/markets/PROFILES.md``.

This module also holds the machinery shared with :mod:`panelbt.markets.us`:
itemised fee tables, the reference-fill approximation for per-fill fixed
charges, per-security dividend withholding and home-currency conversion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from ..models import Config, Costs, MarginModel, Market
from . import FeeSchedule, MarketProfile

_DEFAULT = object()


def _days(dates: ArrayLike) -> np.ndarray:
    try:
        d = np.asarray(dates, dtype="datetime64[D]")
    except (TypeError, ValueError) as exc:
        raise ValueError("dates: expected an array convertible to dates") from exc
    if d.ndim != 1 or np.isnat(d).any():
        raise ValueError("dates: expected one-dimensional valid dates")
    return d


def _patches(periods: tuple) -> tuple:
    out = tuple(
        (np.datetime64(date, "D"), MappingProxyType(dict(terms)))
        for date, terms in periods
    )
    if not out or any(a[0] >= b[0] for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("items: expected strictly increasing effective dates")
    return out


def expand_items(periods: tuple, dates: ArrayLike) -> dict[str, np.ndarray]:
    """Expand cumulative dated patches of named charges to one array per item.

    Dates before the first effective date fail, exactly like `FeeSchedule`.
    """
    d = _days(dates)
    periods = _patches(periods)
    if (d < periods[0][0]).any():
        raise ValueError(f"dates precede the fee table (starts {periods[0][0]})")
    names = periods[0][1].keys()
    if any(set(terms) - set(names) for _, terms in periods):
        raise ValueError("items: every item must be defined in the first period")
    out = {k: np.zeros(len(d)) for k in names}
    for start, terms in periods:
        for key, value in terms.items():
            out[key][d >= start] = value
    return out


def _states(periods: tuple) -> list[tuple[str, dict]]:
    """Full item values in force at each effective date."""
    state: dict = {}
    rows = []
    for date, terms in periods:
        state.update(terms)
        rows.append((date, dict(state)))
    return rows


@dataclass(frozen=True)
class RegionalProfile(MarketProfile):
    """A market profile with itemised dated charges and investor-level defaults.

    `items` holds cumulative dated patches of named charges (the audit trail);
    `fees` is the engine-key schedule derived from them. `commission` holds the
    default broker commission, an assumption the caller should override.
    `fill_value` is the reference value of one fill used to turn per-fill fixed
    charges, floors and rounding (which the kernel cannot express) into an
    equivalent proportional rate; None disables that approximation.
    """

    items: tuple = ()
    commission: tuple = ()
    fill_value: float | None = None
    dividend_rate: float = 0.0
    delisting: str = "zero"
    connect: bool = False
    whole_shares: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.fill_value is not None and not (
            np.isfinite(self.fill_value) and self.fill_value > 0
        ):
            raise ValueError("fill_value: expected a positive finite value or None")
        if self.delisting not in ("zero", "last"):
            raise ValueError("delisting: expected 'zero' or 'last'")

    # -- fees -------------------------------------------------------------

    def charges(self, dates: ArrayLike) -> dict[str, np.ndarray]:
        """Each statutory charge by name, as in force on each date."""
        return expand_items(self.items, dates)

    def _fill_extras(self, it: dict, fill_value: float) -> dict[str, np.ndarray]:
        """Per-fill fixed amounts (currency per fill) by Costs key."""
        return {}

    def costs(self, dates, *, fill_value=_DEFAULT, **overrides) -> Costs:
        """Expand the dated fee table into `Costs`.

        Proportional statutory charges are exact. Per-fill fixed charges,
        per-fill floors and round-ups are added as `amount / fill_value`
        (pessimistic for fills larger than `fill_value`; pass the typical fill
        size of the strategy, or None to leave them out). Per-fill caps are
        ignored (an overcharge). Broker commission defaults to the profile's
        assumption; any keyword replaces the corresponding final value.
        """
        d = _days(dates)
        values = self.fees.expand(d)
        fv = self.fill_value if fill_value is _DEFAULT else fill_value
        if fv is not None:
            if not (np.isscalar(fv) and np.isfinite(fv) and fv > 0):
                raise ValueError("fill_value: expected a positive finite value or None")
            for key, amount in self._fill_extras(self.charges(d), float(fv)).items():
                values[key] = values[key] + amount / fv
        values.update(dict(self.commission))
        unknown = set(overrides) - set(Costs.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown cost fields: {', '.join(sorted(unknown))}")
        values.update(overrides)
        return Costs(**values)

    # -- trading units, dividends, delisting -----------------------------

    def quantities(self, *, board=None, lot_sizes=None, n=None):
        """Per-security minimum quantity and step.

        `lot_sizes` is length N, or T×N for board lots that change over time.
        The kernel takes one lot per security, so a T×N input is reduced to
        each security's largest lot (coarser rounding: pessimistic) with a
        warning when lots actually change.
        """
        if board is not None:
            raise ValueError("board: only the A-share profile uses boards")
        if lot_sizes is not None:
            a = np.asarray(lot_sizes, dtype=float)
            if a.ndim == 2:
                if not np.isfinite(a).all() or (a < 0).any() or 0 in a.shape:
                    raise ValueError("lot_sizes: expected finite nonnegative T×N")
                changing = (a != a[:1]).any(axis=0)
                if changing.any():
                    warnings.warn(
                        f"lot_sizes change over time for {int(changing.sum())} "
                        "securities; the kernel takes one lot per security, so "
                        "each security uses its largest lot (pessimistic)",
                        UserWarning,
                        stacklevel=3,
                    )
                a = a.max(axis=0)
            return super().quantities(lot_sizes=a, n=n)
        if self.whole_shares:
            if n is None:
                raise ValueError("provide n (number of securities) or lot_sizes")
            return np.ones(n), np.ones(n)
        raise ValueError("lot_sizes: board lots differ by security; pass them")

    def withholding(self, n: int, dates=None, **flags) -> np.ndarray:
        """Dividend withholding rate per security (length N, or T×N with dates)."""
        rate = np.full(n, float(self.dividend_rate))
        return rate if dates is None else np.tile(rate, (len(_days(dates)), 1))

    def _dividend_tax(self, n, dates, flags, cash_div=None):
        rates = self.withholding(n, dates, **flags)
        if cash_div is not None:
            div = np.asarray(cash_div, dtype=float)
            if div.ndim != 2 or div.shape[1] != n:
                raise ValueError("cash_div: expected T×N with N securities")
            if rates.ndim == 2 and rates.shape != div.shape:
                raise ValueError("cash_div and dates: expected the same T")
            full = np.broadcast_to(rates, div.shape)
            paying = np.nan_to_num(div) > 0
            # Only securities paying on a day matter for that day's rate.
            per_day = np.where(paying, full, -1.0).max(axis=1)
            per_day = np.where(per_day < 0, full.min(axis=1), per_day)
            if (paying & (full != per_day[:, None])).any():
                _warn_dividend()
            return per_day
        if rates.ndim == 1:
            if (rates != rates[0]).any():
                _warn_dividend()
            return float(rates.max())
        per_day = rates.max(axis=1)
        if (rates != per_day[:, None]).any():
            _warn_dividend()
        return per_day

    def config(
        self,
        *,
        board=None,
        lot_sizes=None,
        n=None,
        dates=None,
        delisting=None,
        cash_div=None,
        **overrides,
    ) -> Config:
        """Build `Config` with lots, dividend withholding and delisting.

        Delisting: prices are frozen at the last valid close until the delist
        row, which settles the position. `"zero"` settles at 0: pessimistic
        for longs, optimistic for shorts (a short is bought back for nothing).
        `"last"` settles at the last close: optimistic for longs, neutral for
        shorts. Long-only accounts default to `"zero"`; when the margin model
        allows shorting, `delisting` must be given explicitly.

        Per-security flags (see `withholding`) select dividend withholding;
        the kernel has one rate per day, so differing securities use the
        highest rate of the day with a warning. Passing the market's
        `cash_div` (T×N) restricts that maximum to securities actually paying
        on the day, which removes most of the overcharge; `net_dividends` is
        an exact alternative for long-only accounts.
        """
        flags = {k: overrides.pop(k) for k in self._flag_names() if k in overrides}
        if lot_sizes is None and n is None:
            n = next(
                (len(v) for v in flags.values() if v is not None and np.ndim(v) == 1),
                None,
            )
        minimum, step = self.quantities(board=board, lot_sizes=lot_sizes, n=n)
        margin = overrides.get("margin", self.margin)
        if self.connect and (margin.allow_short or margin.allow_financing):
            raise ValueError(
                "Southbound Stock Connect investors can neither short nor borrow"
            )
        explicit = delisting is not None or "delist_recovery" in overrides
        if margin.allow_short and not explicit:
            raise ValueError(
                "delisting: pass it explicitly when shorting is allowed. "
                "'zero' settles a delisted holding at 0: pessimistic for longs, "
                "optimistic for shorts (bought back for nothing). 'last' settles "
                "at the last close: optimistic for longs, neutral for shorts. "
                "For books with material short exposure prefer 'last' and "
                "report the long-side bias, or run both"
            )
        mode = self.delisting if delisting is None else delisting
        if mode not in ("zero", "last"):
            raise ValueError("delisting: expected 'zero' or 'last'")
        values = dict(
            min_qty=minimum,
            qty_step=step,
            margin=self.margin,
            dividend_tax_rate=self._dividend_tax(len(minimum), dates, flags, cash_div),
            delist_recovery=0.0 if mode == "zero" else 1.0,
        )
        values.update(overrides)
        rules = values.get("rules")
        if rules is not None and rules.sell_lock and self.sell_lock == 0:
            raise ValueError(
                f"{self.name} allows same-day round trips: pass Rules(sell_lock=False)"
            )
        return Config(**values)

    def _flag_names(self) -> tuple:
        return ()

    # -- tradability ------------------------------------------------------

    def market(self, exec_price, close, *, connect_eligible=None, **kwargs) -> Market:
        """Convert prices and trading rules to a validated Market.

        For Southbound Stock Connect, `connect_eligible` (bool T×N, point in
        time) marks securities that may be bought; ineligible holdings
        (sell-only) may still be sold. Omitted means every security is
        eligible.
        """
        keys = ("suspended", "high", "low", "up_limit", "down_limit")
        rules = {k: kwargs.pop(k) for k in keys if k in kwargs}
        buy, sell = self.tradability(exec_price, **rules)
        if connect_eligible is not None:
            if not self.connect:
                raise ValueError("connect_eligible applies to Stock Connect only")
            e = np.asarray(connect_eligible, dtype=bool)
            if e.shape != buy.shape:
                raise ValueError("connect_eligible: expected the shape of exec_price")
            buy = buy & e
        return Market(exec_price, close, buy, sell, **kwargs)


def _warn_dividend():
    warnings.warn(
        "dividend withholding differs by security; the kernel applies one rate "
        "per day, so the highest rate is used (pessimistic). For exact long-only "
        "cash use net_dividends(cash_div, rates) with dividend_tax_rate=0",
        UserWarning,
        stacklevel=4,
    )


def build_schedule(items: tuple, mapping) -> FeeSchedule:
    """Derive an engine-key `FeeSchedule` from itemised dated patches."""
    rows = []
    for date, state in _states(items):
        rows.append((date, {k: float(f(state)) for k, f in mapping.items()}))
    return FeeSchedule(tuple(rows))


def net_dividends(
    cash_div: ArrayLike, rates: ArrayLike, account, *, allow_short=False
) -> np.ndarray:
    """Dividends net of per-security withholding: `cash_div × (1 − rates)`.

    Exact alternative to the per-day `dividend_tax_rate` for long-only
    accounts: pass the result as `Market(cash_div=...)` with
    `Config(dividend_tax_rate=0)`. Reported dividends are then net and a
    NaN close after the ex-date is marked down by the net amount. A short
    would pay only the net amount instead of the gross (optimistic), so
    `account` (the `Config` or `MarginModel` of the run) must not allow
    shorting unless `allow_short=True` is passed deliberately. Positions
    are only known inside the kernel; the account is the check available
    here. `rates` is length N or T×N.
    """
    margin = getattr(account, "margin", account)
    if not isinstance(margin, MarginModel):
        raise ValueError("account: expected the run's Config or MarginModel")
    if margin.allow_short and not allow_short:
        raise ValueError(
            "net_dividends: shorts would pay net instead of gross dividends "
            "(optimistic); use dividend_tax_rate, or pass allow_short=True"
        )
    div = np.asarray(cash_div, dtype=float)
    r = np.asarray(rates, dtype=float)
    if div.ndim != 2 or r.ndim not in (1, 2):
        raise ValueError("cash_div: expected T×N; rates: length N or T×N")
    if not ((r >= 0) & (r <= 1)).all():
        raise ValueError("rates: expected values in [0, 1]")
    return div * (1.0 - np.broadcast_to(r, div.shape))


def home_currency_nav(nav: ArrayLike, fx: ArrayLike, *, base_fx=None) -> np.ndarray:
    """Convert a NAV series in the security currency to the home currency.

    `fx` is home currency per unit of security currency, one value per row
    of `nav` (same length, same row order): row t's NAV is converted at row
    t's rate, i.e. the rate observed on that row's date at the time the
    closing mark is taken. `base_fx` is the rate at which the initial
    capital was converted (default `fx[0]`). Missing or nonpositive rates
    raise ValueError; fill gaps yourself (e.g. carry the last rate forward)
    before calling. Assumes all cash stays in the security currency (a local
    HK or US account). Conversion costs are not included.
    """
    v = np.asarray(nav, dtype=float)
    f = np.asarray(fx, dtype=float)
    if v.ndim != 1 or f.shape != v.shape or not (np.isfinite(f) & (f > 0)).all():
        raise ValueError("nav/fx: expected equal-length 1-D series, fx positive")
    b = f[0] if base_fx is None else float(base_fx)
    if not (np.isfinite(b) and b > 0):
        raise ValueError("base_fx: expected a positive finite value")
    return v * f / b


def connect_home_equity(
    equity: ArrayLike, cash: ArrayLike, fx: ArrayLike, initial_cash: float
) -> np.ndarray:
    """Home-currency (CNY) equity of a Southbound backtest run in HKD.

    A Southbound investor keeps cash in CNY and converts only what each day's
    trades, fees and dividends move. This converts each day's HKD cash change
    at that day's rate and the HKD securities value at the day's rate:
    ``home_cash[t] = home_cash[t-1] + (cash[t] - cash[t-1]) × fx[t]`` with
    ``home_cash[-1] = initial_cash × fx[0]``, and returns
    ``home_cash + (equity - cash) × fx``. `equity` and `cash` are the
    Result fields of a long-only, unlevered run; `fx` is CNY per HKD at
    mid rate (the settlement spread is the `fx_cost` in `costs`), one value
    per row, aligned with `equity` and `cash`: row t uses row t's rate.
    Missing or nonpositive rates raise ValueError; fill gaps before calling.
    """
    e, c, f = (np.asarray(x, dtype=float) for x in (equity, cash, fx))
    if e.ndim != 1 or c.shape != e.shape or f.shape != e.shape:
        raise ValueError("equity/cash/fx: expected equal-length 1-D series")
    if not (np.isfinite(f) & (f > 0)).all() or not initial_cash > 0:
        raise ValueError("fx and initial_cash: expected finite positive values")
    delta = np.diff(np.concatenate(([float(initial_cash)], c)))
    home_cash = initial_cash * f[0] + np.cumsum(delta * f)
    return home_cash + (e - c) * f


# ---------------------------------------------------------------------------
# Hong Kong charges (docs/markets/hk_rules.md section 1). Rates are per side,
# on the traded amount, identical for buyer and seller. Patches are cumulative.
#
# 2010-01-01  table start. Stamp duty 0.1% (1.1, IRD rate table), rounded up to
#             HK$1 per contract note; SFC levy 0.004% (1.1, HKEX fee page
#             history, 2006-12-01 to 2010-09-30); no AFRC levy yet; HKEX
#             trading fee 0.005% (1.1, HKEX news 2022-11-01); trading tariff
#             HK$0.50 per transaction (1.1; "per side per trade" is the broker
#             practice, unverified, 1.1 note 1); HKSCC stock settlement fee
#             0.002% with a HK$2 floor and HK$100 cap per side per trade (1.1,
#             HKSCC circular PT/OES/HKSCC/008/2025 of 2025-02-21, "Existing"; unchanged 2010-2025 is
#             unverified, 1.1 note 2).
# 2010-10-01  SFC levy 0.003% (1.1, government press release 2010-05-20).
# 2014-11-01  SFC levy 0.0027% (1.1, L.N. 74 of 2014).
# 2021-08-01  stamp duty 0.13% (1.1, Ord. No. 16 of 2021).
# 2022-01-01  AFRC levy 0.00015% (1.1, HKEX fee page).
# 2023-01-01  trading fee 0.00565%, trading tariff abolished (1.1, HKEX news
#             2022-11-01). Southbound stopped collecting the tariff on
#             2023-01-03 (1.2, ChinaClear Shenzhen guide revision note).
# 2023-11-17  stamp duty 0.1% (1.1, Ord. No. 29 of 2023).
# 2025-06-30  settlement fee 0.0042%, floor and cap removed (1.1, HKSCC
#             circular PT/SCS/CCASS/022/2025 of 2025-05-29; by trade
#             date). ETP market-making trades pay 0.0020%: not modelled
#             (the full rate is charged, an overcharge).
# The investor compensation levy has been suspended since 2005-12-19 (1.1).
# ---------------------------------------------------------------------------

_HK_START = {
    "stamp": 0.001,
    "sfc_levy": 0.00004,
    "afrc_levy": 0.0,
    "trading_fee": 0.00005,
    "settlement": 0.00002,
    "settlement_min": 2.0,
    "settlement_max": 100.0,
    "tariff": 0.5,
}

HK_ITEMS = (
    ("2010-01-01", _HK_START),
    ("2010-10-01", {"sfc_levy": 0.00003}),
    ("2014-11-01", {"sfc_levy": 0.000027}),
    ("2021-08-01", {"stamp": 0.0013}),
    ("2022-01-01", {"afrc_levy": 0.0000015}),
    ("2023-01-01", {"trading_fee": 0.0000565, "tariff": 0.0}),
    ("2023-11-17", {"stamp": 0.001}),
    (
        "2025-06-30",
        {"settlement": 0.000042, "settlement_min": 0.0, "settlement_max": np.inf},
    ),
)

# Southbound starts with the Shanghai link on 2014-11-17 (hk_rules.md 0, 6.1);
# the same statutory charges (1.2), the tariff until 2023-01-02.
SOUTHBOUND_ITEMS = (
    ("2014-11-17", {**_HK_START, "sfc_levy": 0.000027}),
    ("2021-08-01", {"stamp": 0.0013}),
    ("2022-01-01", {"afrc_levy": 0.0000015}),
    ("2023-01-01", {"trading_fee": 0.0000565}),
    ("2023-01-03", {"tariff": 0.0}),
    ("2023-11-17", {"stamp": 0.001}),
    (
        "2025-06-30",
        {"settlement": 0.000042, "settlement_min": 0.0, "settlement_max": np.inf},
    ),
)


def _hk_rate(s):
    # Fixed summation order; the hand-computed tests use the same order.
    return s["sfc_levy"] + s["afrc_levy"] + s["trading_fee"] + s["settlement"]


_HK_MAPPING = {
    "tax_buy": lambda s: s["stamp"],
    "tax_sell": lambda s: s["stamp"],
    "fee_rate_buy": _hk_rate,
    "fee_rate_sell": _hk_rate,
}

# Rounding allowance for stamp duty rounded up to whole HK$ per fill.
STAMP_ROUNDING = 1.0


@dataclass(frozen=True)
class HKProfile(RegionalProfile):
    """Hong Kong profile; `fx_cost` is the Southbound CNY/HKD settlement spread.

    `fx_cost` is charged as a proportional fee on both sides (the settlement
    exchange ratios differ for buys and sells around the mid rate).
    """

    fx_cost: float = 0.0

    def costs(self, dates, *, fill_value=_DEFAULT, fx_cost=None, **overrides):
        """As `RegionalProfile.costs`; `fx_cost` is added to both fee rates.

        The exchange spread is added after overrides, so overriding
        `fee_rate_buy/sell` replaces the statutory charges but keeps the
        spread; pass `fx_cost=0` to drop it.
        """
        fx = self.fx_cost if fx_cost is None else fx_cost
        if fx and not self.connect:
            raise ValueError("fx_cost applies to Stock Connect only")
        if not (np.isscalar(fx) and np.isfinite(fx) and 0 <= fx < 1):
            raise ValueError("fx_cost: expected a finite rate in [0, 1)")
        c = super().costs(dates, fill_value=fill_value, **overrides)
        values = {k: getattr(c, k) for k in Costs.__dataclass_fields__}
        for key in ("fee_rate_buy", "fee_rate_sell"):
            values[key] = np.asarray(values[key], dtype=float) + fx
        return Costs(**values)

    def _fill_extras(self, it, fill_value):
        # Tariff per fill, the settlement fee floor in excess of the rate, and
        # the stamp duty round-up. The HK$100 settlement cap is ignored.
        floor = np.maximum(it["settlement_min"] - it["settlement"] * fill_value, 0.0)
        other = it["tariff"] + floor
        stamp = np.where(it["stamp"] > 0, STAMP_ROUNDING, 0.0)
        return {
            "fee_rate_buy": other,
            "fee_rate_sell": other,
            "tax_buy": stamp,
            "tax_sell": stamp,
        }

    def _flag_names(self):
        return ("h_share", "fie_h_share", "resident_enterprise")

    def withholding(
        self,
        n,
        dates=None,
        *,
        h_share=None,
        fie_h_share=None,
        resident_enterprise=None,
    ):
        """Dividend withholding per security.

        Local account (hk_rules.md 6.1): 0% for non-H shares, 10% for H
        shares; foreign-invested-enterprise H shares (`fie_h_share`, opt-in)
        0% before 2026-09-01 and 20% from then (needs `dates`).
        `resident_enterprise` (opt-in) marks non-H shares of Chinese resident
        enterprises (red chips) withheld at 10% when held through HKSCC
        nominees (6.1 table; 11.1 says 0%; no company notice verified).
        Without `h_share` every security is treated as an H share (a
        pessimistic fallback, with a warning). Southbound: 20% for every
        security (6.1).
        """
        if self.connect:
            return super().withholding(n, dates)
        if h_share is None:
            warnings.warn(
                "h_share not given: every security is withheld at 10% as if it "
                "were an H share (pessimistic fallback; with 1 H share among 50 "
                "equal-weight names at a 4% yield the overcharge is about 0.39% "
                "of NAV a year). Pass h_share=<bool per security>",
                UserWarning,
                stacklevel=4,
            )
        h = np.ones(n, bool) if h_share is None else _flag(h_share, n, "h_share")
        rate = np.where(h, 0.10, 0.0)
        if resident_enterprise is not None:
            rate = np.where(
                _flag(resident_enterprise, n, "resident_enterprise"), 0.10, rate
            )
        if fie_h_share is None:
            return rate if dates is None else np.tile(rate, (len(_days(dates)), 1))
        if dates is None:
            raise ValueError("fie_h_share: the 2026-09-01 change requires dates")
        fie = _flag(fie_h_share, n, "fie_h_share")
        if (fie & ~h).any():
            raise ValueError("fie_h_share: flagged securities must be H shares")
        d = _days(dates)
        after = (d >= np.datetime64("2026-09-01"))[:, None]
        out = np.tile(rate, (len(d), 1))
        return np.where(fie[None, :], np.where(after, 0.20, 0.0), out)


def _flag(value, n, name):
    a = np.asarray(value)
    if a.shape != (n,) or a.dtype.kind not in "bui":
        raise ValueError(f"{name}: expected a boolean vector of length N")
    return a.astype(bool)


def hk_margin(
    *,
    long_initial,
    long_maintenance,
    short_initial,
    short_maintenance,
    financing_rate,
    borrow_rate=0.003,
    periods_per_year=252,
    margin_call="liquidate",
) -> HKProfile:
    """HK_DIRECT with a margin account; ratios must come from the broker.

    Hong Kong sets no statutory margin ratios (hk_rules.md 5.1), so all four
    are required. Rates are annual and converted per period. The default
    `borrow_rate` of 0.3% a year is an assumption (no published HK level;
    the easy-to-borrow US level from us_rules.md 5.4). Pass `can_short`
    (designated securities for short selling, point in time) to `Market`.
    Its `config()` requires an explicit `delisting=` (see `config`).
    """
    from dataclasses import replace

    margin = MarginModel(
        allow_short=True,
        long_initial=long_initial,
        short_initial=short_initial,
        long_maintenance=long_maintenance,
        short_maintenance=short_maintenance,
        allow_financing=True,
        short_proceeds_usable=False,
        financing_rate=np.asarray(financing_rate, dtype=float) / periods_per_year,
        borrow_rate=np.asarray(borrow_rate, dtype=float) / periods_per_year,
        margin_call=margin_call,
    )
    return replace(HK_DIRECT, name="Hong Kong local margin account", margin=margin)


HK_DIRECT = HKProfile(
    "Hong Kong (local account)",
    "HKD",
    252,
    build_schedule(HK_ITEMS, _HK_MAPPING),
    items=HK_ITEMS,
    # Assumption: Futu HK standard pricing, no commission and a HK$15 platform
    # fee per order (hk_rules.md 1.3, broker page, not verified).
    commission=(("commission_rate", 0.0), ("commission_min", 15.0)),
    fill_value=20_000.0,
    dividend_rate=0.10,
)

HK_SOUTHBOUND = HKProfile(
    "Hong Kong (Southbound Stock Connect)",
    "HKD",
    252,
    build_schedule(SOUTHBOUND_ITEMS, _HK_MAPPING),
    items=SOUTHBOUND_ITEMS,
    # Assumption: 0.03% with no minimum, the top of the 0.025%-0.03% range
    # quoted by third parties for mainland brokers (hk_rules.md 1.3, unverified).
    commission=(("commission_rate", 0.0003), ("commission_min", None)),
    fill_value=20_000.0,
    dividend_rate=0.20,
    connect=True,
    # Assumption: the settlement exchange ratio spread per side; the research
    # gives no figure (hk_rules.md 1.2, 11.2 item 4).
    fx_cost=0.0005,
)


__all__ = [
    "HK_DIRECT",
    "HK_ITEMS",
    "HK_SOUTHBOUND",
    "HKProfile",
    "RegionalProfile",
    "SOUTHBOUND_ITEMS",
    "connect_home_equity",
    "expand_items",
    "hk_margin",
    "home_currency_nav",
    "net_dividends",
]
