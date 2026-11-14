"""Binance USDT profiles. Evidence: AShare docs/markets/BINANCE.md, 2026-11-15.

Observation windows are deliberately bounded. A snapshot does not establish an
undocumented historical transition. Risk extrapolation requires explicit consent.
Only NumPy and the Python standard library are runtime dependencies.
"""

from dataclasses import dataclass

import numpy as np

from ..models import Costs
from . import MarketProfile

HOUR = np.timedelta64(1, "h")
FEE_SOURCE = "https://www.binance.com/en/fee/schedule"
OLD_FEE_SOURCE = "https://web.archive.org/web/20230531224744/https://www.binance.com/en/fee/futureFee"
R2021 = (
    "https://www.binance.com/en/support/announcement/f55d25695ca8453ab4c132c1fd4dc36a"
)
R2023 = "https://www.binance.com/es/support/announcement/detail/d75c5ca94f704e96a6a4e55ffddfd65d"
R2025 = "https://www.binance.com/en/support/announcement/detail/8e428625ebeb4cc7ae8678026846095c"
SNAPSHOT = "https://fapi.binance.com/fapi/v1/exchangeInfo"
FIELDS = ("maint_rate", "tier_cap", "max_leverage", "clearance_fee_rate")


def _times(dates):
    dates = np.asarray(dates, dtype="datetime64[ms]")
    if dates.ndim != 1 or not len(dates) or np.isnat(dates).any():
        raise ValueError("dates: expected nonempty finite UTC timestamps")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates: expected strictly increasing timestamps")
    return dates


def _segments(dates, mask, **detail):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(dates[indices]) != HOUR) + 1)
    return [
        dict(start=str(dates[g[0]]), end=str(dates[g[-1]] + HOUR), **detail)
        for g in groups
    ]


@dataclass(frozen=True)
class BinanceFees:
    """Hourly fee schedule, including an explicitly assumed 2023 transition."""

    kind: str
    required_overrides_before: tuple = ()
    components: tuple = ()

    def terms(self, dates, fee_switch=None):
        dates = _times(dates)
        inception = np.datetime64(
            "2019-12-20T08" if self.kind == "usdm" else "2017-07-01", "ms"
        )
        if (dates < inception).any():
            raise ValueError(f"fee evidence unavailable before {inception}")
        if self.kind == "spot":
            taker = np.full(len(dates), 0.001)
            notes = _segments(
                dates,
                np.ones(len(dates), bool),
                parameter="taker",
                value=0.001,
                source_url=FEE_SOURCE,
                confidence="inferred_between_observations",
                reason="VIP0; BTC zero-fee promotion omitted",
            )
            return dict(
                maker=taker.copy(),
                taker=taker,
                delivery=np.zeros(len(dates)),
                provenance=notes,
            )
        switch = np.datetime64(fee_switch or "2023-06-01T00", "ms")
        if np.isnat(switch) or switch < np.datetime64("2023-06-01"):
            raise ValueError("fee_switch must respect the confirmed 2023-05-31 fee")
        later = dates >= switch
        taker = np.where(later, 0.0005, 0.0004)
        notes = []
        for mask, value, source, confidence, reason in (
            (
                ~later,
                0.0004,
                OLD_FEE_SOURCE,
                "supported_endpoints",
                "2019-12-20 announcement and 2023-05-31 fee page",
            ),
            (
                later,
                0.0005,
                FEE_SOURCE,
                "assumption",
                "unverified switch date; assumed " + str(switch),
            ),
        ):
            # Moving the switch also makes the post-May old-rate interval an assumption.
            for assumed in (False, True):
                uncertain = (
                    (dates >= np.datetime64("2023-06-01")) if fee_switch else later
                )
                selected = mask & (uncertain if assumed else ~uncertain)
                notes += _segments(
                    dates,
                    selected,
                    parameter="taker_and_delivery",
                    value=value,
                    source_url=source,
                    confidence="assumption" if assumed else confidence,
                    reason=reason,
                )
        return dict(
            maker=np.full(len(dates), 0.0002),
            taker=taker,
            delivery=taker.copy(),
            provenance=notes,
        )

    def expand(self, dates):
        rate = self.terms(dates)["taker"]
        return dict(
            fee_rate_buy=rate,
            fee_rate_sell=rate.copy(),
            tax_buy=0.0,
            tax_sell=0.0,
            fee_per_share_buy=0.0,
            fee_per_share_sell=0.0,
        )


# Field-specific evidence intervals. Point observations cover that UTC hour;
# the last exchangeInfo observation covers its stated observation day only.
# MMR continuity is explicitly supported by BINANCE.md 5.2. Tier-cap change
# dates between the 2023 and 2025 observations are NOT invented.
def _observations(symbol):
    result = {f: [] for f in FIELDS}

    def add(field, start, end, value, source, confidence="official"):
        original = {
            R2021: "2021-06-22T03:30",
            R2023: "2023-12-24T09:35",
            R2025: "2025-08-19T06:30",
        }.get(source, start)
        effective = np.datetime64(original, "ms")
        floor = effective.astype("datetime64[h]").astype("datetime64[ms]")
        applied = floor + HOUR if effective != floor else floor
        timing = dict(
            effective_at=str(effective),
            applied_at=str(applied),
            approximation=bool(effective != applied),
            rounding="ceil_to_hour" if effective != applied else "none",
        )
        result[field].append(
            (
                np.datetime64(start, "ms"),
                np.datetime64(end, "ms"),
                value,
                source,
                confidence,
                timing,
            )
        )

    if symbol in ("BTCUSDT", "ETHUSDT"):
        btc = symbol == "BTCUSDT"
        if btc:
            add(
                "maint_rate",
                "2021-06-22T04",
                "2025-08-19T08",
                0.004,
                R2021,
                "documented_continuity",
            )
        else:
            add("maint_rate", "2023-12-24T09", "2023-12-24T10", 0.005, R2023)
            add(
                "maint_rate",
                "2023-12-24T10",
                "2025-08-19T08",
                0.004,
                R2023,
                "documented_continuity",
            )
        observations = [
            (
                "2023-12-24T09",
                "2023-12-24T10",
                50000 if btc else 200000,
                125 if btc else 100,
                R2023,
            ),
            ("2023-12-24T10", "2023-12-24T11", 50000, 125, R2023),
            ("2025-08-19T06", "2025-08-19T07", 300000, 125, R2025),
            ("2025-08-19T07", "2025-08-19T08", 300000, 150, R2025),
        ]
        if btc:
            observations.insert(
                0, ("2021-06-22T04", "2021-06-22T05", 50000, 125, R2021)
            )
        for start, end, cap, leverage, source in observations:
            add("tier_cap", start, end, cap, source)
            add("max_leverage", start, end, leverage, source)
    if symbol in ("BTCUSDT", "ETHUSDT", "BTCUSDT_261225"):
        add(
            "clearance_fee_rate",
            "2026-11-15",
            "2026-11-16",
            0.0125,
            SNAPSHOT,
            "dated_snapshot",
        )
    return result


@dataclass(frozen=True)
class BinanceProfile(MarketProfile):
    kind: str = "spot"

    def fee_terms(self, dates, *, fee_switch=None):
        return self.fees.terms(dates, fee_switch)

    def costs(self, dates, *, fee_switch=None, **overrides):
        """Spot fees go in Costs; futures fees belong exclusively in Futures."""
        terms = self.fee_terms(dates, fee_switch=fee_switch)
        fee = terms["taker"] if self.kind == "spot" else 0.0
        values = dict(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=fee,
            fee_rate_sell=fee,
            slippage=0.0,
        )
        values.update(overrides)
        return Costs(**values)

    def rules(self, symbols, dates, *, assume=None, leverage=1.0, active=None):
        """Return risk arrays and provenance, rejecting gaps unless assumed.

        ``assume='previous'`` extends the previous observation (the earliest one
        before coverage). A mapping may specify ``policy='previous'`` and
        ``rules={symbol: {field: value, reason: ..., source_url: ...}}`` for gaps.
        Unknown contracts never borrow the underlying perpetual's rules silently.
        """
        if self.kind != "usdm":
            raise ValueError("risk rules are only available for USD-M futures")
        dates = _times(dates)
        names = [str(s).removeprefix("usdm:") for s in symbols]
        if not names or len(set(names)) != len(names):
            raise ValueError("symbols must be nonempty and unique")
        shape = (len(dates), len(names))
        active = np.ones(shape, bool) if active is None else np.asarray(active, bool)
        if active.shape != shape:
            raise ValueError("active: expected T x N")
        if assume is None:
            policy, custom = None, {}
        elif isinstance(assume, str):
            policy, custom = assume, {}
        elif isinstance(assume, dict):
            policy, custom = assume.get("policy"), assume.get("rules", {})
        else:
            raise ValueError("assume: expected 'previous' or explicit rule mapping")
        if policy not in (None, "previous"):
            raise ValueError("assume policy: expected 'previous'")
        output = {
            f: np.full(shape, 1.0 if f in ("tier_cap", "max_leverage") else 0.0)
            for f in FIELDS
        }
        notes, errors = [], []
        for j, symbol in enumerate(names):
            for field, records in _observations(symbol).items():
                known = np.zeros(len(dates), bool)
                for lo, hi, value, source, confidence, timing in records:
                    mask = active[:, j] & (dates >= lo) & (dates < hi)
                    output[field][mask, j] = value
                    known |= mask
                    notes += _segments(
                        dates,
                        mask,
                        symbol=symbol,
                        parameter=field,
                        value=value,
                        source_url=source,
                        confidence=confidence,
                        **timing,
                    )
                missing = active[:, j] & ~known
                override = custom.get(symbol, custom.get("usdm:" + symbol, {}))
                if missing.any() and field in override:
                    if not override.get("reason") or not override.get("source_url"):
                        raise ValueError(
                            "explicit assumptions require reason and source_url"
                        )
                    value = float(override[field])
                    if (
                        not np.isfinite(value)
                        or value < 0
                        or (field in ("tier_cap", "max_leverage") and value <= 0)
                    ):
                        raise ValueError(f"invalid assumed {field}")
                    output[field][missing, j] = value
                    notes += _segments(
                        dates,
                        missing,
                        symbol=symbol,
                        parameter=field,
                        value=value,
                        source_url=override["source_url"],
                        confidence="assumption",
                        reason=override["reason"],
                    )
                elif missing.any() and policy == "previous" and records:
                    starts = np.array([r[0] for r in records])
                    pick = np.maximum(
                        0, np.searchsorted(starts, dates, side="right") - 1
                    )
                    for k, record in enumerate(records):
                        mask = missing & (pick == k)
                        output[field][mask, j] = record[2]
                        notes += _segments(
                            dates,
                            mask,
                            symbol=symbol,
                            parameter=field,
                            value=record[2],
                            source_url=record[3],
                            confidence="assumption",
                            reason="previous observation, or earliest before coverage",
                            **record[5],
                        )
                elif missing.any():
                    errors.extend(
                        _segments(dates, missing, symbol=symbol, parameter=field)
                    )
        if errors:
            detail = "; ".join(
                f"{e['symbol']} {e['parameter']} [{e['start']}, {e['end']})"
                for e in errors
            )
            raise ValueError(
                "risk rule evidence missing; pass explicit assume: " + detail
            )
        lev = np.asarray(leverage, dtype=float)
        try:
            lev = np.broadcast_to(lev, shape)
        except ValueError as exc:
            raise ValueError("leverage: expected scalar or T x N") from exc
        if (
            not np.isfinite(lev).all()
            or (lev <= 0).any()
            or ((lev > output["max_leverage"]) & active).any()
        ):
            raise ValueError(
                "leverage exceeds the dated first-tier maximum or is invalid"
            )
        output["margin_rate"] = 1 / lev
        return output, notes


BINANCE_SPOT = BinanceProfile(
    "Binance spot", "USDT", 8760, BinanceFees("spot"), kind="spot"
)
BINANCE_USDM = BinanceProfile(
    "Binance USD-M", "USDT", 8760, BinanceFees("usdm"), kind="usdm"
)
