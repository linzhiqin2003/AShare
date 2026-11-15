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


# Field-specific evidence intervals (BINANCE.md 5.2, 5.3, researched 2026-11-15).
# Each record is (field, start, end, value, source, confidence) with UTC times;
# a non-hourly effective time applies from the next whole hour. Intervals only
# cover what the evidence shows: an announcement covers its effective hour and,
# when the next announcement states the same "before" value, the time up to it;
# a dated snapshot covers its hour; "observed_continuity" means the same value
# was observed at both ends and at points between (not proof of no change).
# Everything else is a gap: rules() rejects it unless an assumption is passed.
WB = "https://web.archive.org/web/"
FAQ_LEVERAGE_2019 = (
    WB
    + "20191114222312/https://binance.zendesk.com/hc/en-us/articles/360033162192-Leverage-Overview"
)
FAQ_LEVERAGE_2021 = (
    WB + "20210124091911/https://www.binance.com/en/support/faq/360033162192"
)
FAQ_TIERS_2020 = (
    WB
    + "20200927195452im_/https://public.bnbstatic.com/image/cms/content/body/202009/a480eab7aa90d625b1dbd94a7301b789.png"
)
FAQ_LIQ_2019 = (
    WB
    + "20190918185837/https://binance.zendesk.com/hc/en-us/articles/360033525271-Liquidation"
)
FAQ_LIQ_2020 = (
    WB
    + "20200520125129/https://binance.zendesk.com/hc/en-us/articles/360033525271-Liquidation"
)
ANN = "https://www.binance.com/en/support/announcement/detail/"
A_125X = (
    WB
    + "20200513174225/https://binance.zendesk.com/hc/en-us/articles/360035327011-Binance-Futures-Increases-Maximum-Leverage-to-125x-and-Launches-Leverage-Adjustment-Function"
)
A_ETH75 = (
    WB
    + "20200513174416/https://binance.zendesk.com/hc/en-us/articles/360037729012-Binance-Futures-Increases-the-Maximum-Leverage-of-the-ETH-USDT-Perpetual-Contract-to-75x"
)
A2020_11 = ANN + "41d8a353103c4fe18bf25c403bf7af3b"
A2021_02 = ANN + "aeeccb6d332a4e74a51c2ac05f9078e5"
A2021_05 = ANN + "839a865d251f435f95fba499d2ad95c9"
A2021_0608 = ANN + "03cac798a8454f42848f4467a434ba5d"
A2021_0610 = ANN + "6bd7a166980648469ab771b5835847c3"
R2025_06 = ANN + "e778988edec446038ad536bcb0c8d460"
BRACKETS = (
    WB + "{}id_/https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets"
)
LIVE_BRACKETS = (
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets"
)
EXCHANGE_INFO = WB + "{}id_/https://fapi.binance.com/fapi/v1/exchangeInfo"
EXCHANGE_INFO_WWW = (
    WB + "{}id_/https://www.binance.com/fapi/v1/exchangeInfo?showall=true"
)
LIVE_END = "2026-11-16T00"  # live snapshots taken on 2026-11-15 cover that UTC day

# exchangeInfo liquidationFee snapshots (Wayback timestamp, BTCUSDT, ETHUSDT).
CLEARANCE_SNAPSHOTS = (
    ("20210716214352", 0.012, 0.005),
    ("20210815214444", 0.012, 0.005),
    ("20210914214535", 0.006, 0.010),
    ("20211014214606", 0.020, 0.020),
    ("20211113214732", 0.030, 0.030),
    ("20211213214732", 0.012, 0.012),
    ("20220102134838", 0.015, 0.015),
    ("20220318130953", 0.005, 0.005),
    ("20220908123350", 0.015, 0.015),
    ("20220916203339", 0.0175, 0.015),
    ("20221116205542", 0.0175, 0.015),
    ("20221127174756", 0.0175, 0.015),
    ("20231102093209", 0.0125, 0.0125),
    ("w20231206233545", 0.0125, 0.0125),
    ("w20240408103545", 0.0125, 0.0125),
    ("w20250414160908", 0.0125, 0.0125),
)

FAQ_LEVERAGE_2019_09 = (
    WB
    + "20190918184207/https://binance.zendesk.com/hc/en-us/articles/360033162192-Leverage-Overview"
)
FAQ_TIERS_2020_05 = (
    WB
    + "20200519164655im_/https://binance.zendesk.com/hc/article_attachments/360054347892/mceclip2.png"
)
NOW = "2026-11-15T00"  # live snapshots were taken on this UTC day

# Evidence classes. EVIDENCE records are used by default; INFERRED spans (the
# same value seen at both ends, or a backfill from the undocumented brackets
# ``updateTime``) are only used when the caller passes an assumption policy,
# and are then reported as assumptions.
EVIDENCE = {"official", "official_prior_value", "dated_snapshot", "upper_bound"}
INFERRED = {"observed_continuity", "bracket_update_time"}

# (field, start, end or None for a one-hour point, value, source, confidence)
_RISK = {
    "BTCUSDT": (
        # maintenance rate of the first tier
        (
            "maint_rate",
            "2019-09-18T18:42:07",
            None,
            0.025,
            FAQ_LEVERAGE_2019_09,
            "dated_snapshot",
        ),
        (
            "maint_rate",
            "2019-11-14T22:23:12",
            None,
            0.004,
            FAQ_LEVERAGE_2019,
            "dated_snapshot",
        ),
        (
            "maint_rate",
            "2019-11-15T00",
            NOW,
            0.004,
            FAQ_LEVERAGE_2019,
            "observed_continuity",
        ),
        ("maint_rate", NOW, LIVE_END, 0.004, LIVE_BRACKETS, "dated_snapshot"),
        # first-tier notional cap
        (
            "tier_cap",
            "2019-09-18T18:42:07",
            None,
            500000,
            FAQ_LEVERAGE_2019_09,
            "dated_snapshot",
        ),
        (
            "tier_cap",
            "2019-11-14T22:23:12",
            None,
            50000,
            FAQ_LEVERAGE_2019,
            "dated_snapshot",
        ),
        (
            "tier_cap",
            "2019-11-15T00",
            "2025-06-17T06:30",
            50000,
            FAQ_LEVERAGE_2019,
            "observed_continuity",
        ),
        (
            "tier_cap",
            "2025-06-17T06:30",
            "2025-08-19T06:30",
            300000,
            R2025_06,
            "official",
        ),
        ("tier_cap", "2025-08-19T06:30", None, 300000, R2025, "official"),
        (
            "tier_cap",
            "2025-08-19T08",
            NOW,
            300000,
            LIVE_BRACKETS,
            "bracket_update_time",
        ),
        ("tier_cap", NOW, LIVE_END, 300000, LIVE_BRACKETS, "dated_snapshot"),
        # maximum leverage
        (
            "max_leverage",
            "2019-09-18T18:42:07",
            None,
            20,
            FAQ_LEVERAGE_2019_09,
            "dated_snapshot",
        ),
        ("max_leverage", "2019-10-18T09:12:09", None, 125, A_125X, "official"),
        (
            "max_leverage",
            "2019-10-18T11",
            "2025-08-19T06:30",
            125,
            A_125X,
            "observed_continuity",
        ),
        ("max_leverage", "2025-08-19T06:30", None, 150, R2025, "official"),
        (
            "max_leverage",
            "2025-08-19T08",
            NOW,
            150,
            LIVE_BRACKETS,
            "bracket_update_time",
        ),
        ("max_leverage", NOW, LIVE_END, 150, LIVE_BRACKETS, "dated_snapshot"),
        # liquidation (insurance) clearance fee
        (
            "clearance_fee_rate",
            "2019-09-18T18:58:37",
            None,
            0.005,
            FAQ_LIQ_2019,
            "upper_bound",
        ),
        (
            "clearance_fee_rate",
            "2020-05-20T12:51:29",
            None,
            0.003,
            FAQ_LIQ_2020,
            "dated_snapshot",
        ),
        (
            "clearance_fee_rate",
            "2020-05-20T14",
            "2020-11-23T07",
            0.003,
            FAQ_LIQ_2020,
            "observed_continuity",
        ),
        ("clearance_fee_rate", "2020-11-23T07", None, 0.005, A2020_11, "official"),
        (
            "clearance_fee_rate",
            "2021-02-07T03",
            None,
            0.010,
            A2021_02,
            "official_prior_value",
        ),
        (
            "clearance_fee_rate",
            "2021-02-07T04",
            "2021-05-20T09",
            0.020,
            A2021_02,
            "official",
        ),
        (
            "clearance_fee_rate",
            "2021-05-20T09",
            "2021-06-08T07",
            0.003,
            A2021_05,
            "official",
        ),
        (
            "clearance_fee_rate",
            "2021-06-08T07",
            "2021-06-10T04",
            0.006,
            A2021_0608,
            "official",
        ),
        ("clearance_fee_rate", "2021-06-10T04", None, 0.012, A2021_0610, "official"),
        # announcement / snapshot points inside the inferred spans above
        ("maint_rate", "2021-06-22T03:30", None, 0.004, R2021, "official"),
        ("maint_rate", "2023-12-24T09:35", None, 0.004, R2023, "official"),
        ("maint_rate", "2025-06-17T06:30", None, 0.004, R2025_06, "official"),
        ("maint_rate", "2025-08-19T06:30", None, 0.004, R2025, "official"),
        ("tier_cap", "2021-06-22T03:30", None, 50000, R2021, "official"),
        ("tier_cap", "2023-12-24T09:35", None, 50000, R2023, "official"),
        ("tier_cap", "2025-06-17T06", None, 50000, R2025_06, "official_prior_value"),
        ("max_leverage", "2021-06-22T03:30", None, 125, R2021, "official"),
        ("max_leverage", "2023-12-24T09:35", None, 125, R2023, "official"),
        ("max_leverage", "2025-06-17T06:30", None, 125, R2025_06, "official"),
        ("max_leverage", "2025-08-19T06", None, 125, R2025, "official_prior_value"),
        # prior values stated by the same announcements (the hour before each change)
        ("maint_rate", "2021-06-22T03", None, 0.004, R2021, "official_prior_value"),
        ("maint_rate", "2023-12-24T09", None, 0.004, R2023, "official_prior_value"),
        ("maint_rate", "2025-06-17T06", None, 0.004, R2025_06, "official_prior_value"),
        ("maint_rate", "2025-08-19T06", None, 0.004, R2025, "official_prior_value"),
        ("tier_cap", "2021-06-22T03", None, 50000, R2021, "official_prior_value"),
        ("tier_cap", "2023-12-24T09", None, 50000, R2023, "official_prior_value"),
        ("max_leverage", "2021-06-22T03", None, 125, R2021, "official_prior_value"),
        ("max_leverage", "2023-12-24T09", None, 125, R2023, "official_prior_value"),
        ("max_leverage", "2025-06-17T06", None, 125, R2025_06, "official_prior_value"),
    ),
    "ETHUSDT": (
        (
            "maint_rate",
            "2020-05-19T16:46:55",
            None,
            0.0065,
            FAQ_TIERS_2020_05,
            "dated_snapshot",
        ),
        (
            "maint_rate",
            "2020-05-19T18",
            "2020-09-27T19:54:52",
            0.0065,
            FAQ_TIERS_2020,
            "observed_continuity",
        ),
        (
            "maint_rate",
            "2020-09-27T19:54:52",
            None,
            0.0065,
            FAQ_TIERS_2020,
            "dated_snapshot",
        ),
        (
            "maint_rate",
            "2021-01-24T09:19:11",
            None,
            0.005,
            FAQ_LEVERAGE_2021,
            "dated_snapshot",
        ),
        (
            "maint_rate",
            "2021-01-24T11",
            "2023-12-24T09:35",
            0.005,
            FAQ_LEVERAGE_2021,
            "observed_continuity",
        ),
        ("maint_rate", "2023-12-24T09:35", None, 0.004, R2023, "official"),
        ("maint_rate", "2023-12-24T11", NOW, 0.004, R2023, "observed_continuity"),
        ("maint_rate", NOW, LIVE_END, 0.004, LIVE_BRACKETS, "dated_snapshot"),
        (
            "tier_cap",
            "2020-05-19T16:46:55",
            None,
            10000,
            FAQ_TIERS_2020_05,
            "dated_snapshot",
        ),
        (
            "tier_cap",
            "2020-05-19T18",
            "2020-09-27T19:54:52",
            10000,
            FAQ_TIERS_2020,
            "observed_continuity",
        ),
        (
            "tier_cap",
            "2020-09-27T19:54:52",
            None,
            10000,
            FAQ_TIERS_2020,
            "dated_snapshot",
        ),
        (
            "tier_cap",
            "2021-01-24T09:19:11",
            None,
            10000,
            FAQ_LEVERAGE_2021,
            "dated_snapshot",
        ),
        (
            "tier_cap",
            "2022-11-17T06:31:03",
            "2023-12-12T04:43:26",
            100000,
            BRACKETS.format("20230109204158"),
            "bracket_update_time",
        ),
        (
            "tier_cap",
            "2023-12-12T04:43:26",
            "2023-12-24T09:35",
            200000,
            BRACKETS.format("20231212142739"),
            "bracket_update_time",
        ),
        ("tier_cap", "2023-12-24T09:35", None, 50000, R2023, "official"),
        ("tier_cap", "2025-08-19T06", None, 300000, R2025, "official_prior_value"),
        ("tier_cap", "2025-08-19T06:30", None, 300000, R2025, "official"),
        (
            "tier_cap",
            "2025-08-19T08",
            NOW,
            300000,
            LIVE_BRACKETS,
            "bracket_update_time",
        ),
        ("tier_cap", NOW, LIVE_END, 300000, LIVE_BRACKETS, "dated_snapshot"),
        ("max_leverage", "2019-12-19T10:45:42", None, 75, A_ETH75, "official"),
        (
            "max_leverage",
            "2019-12-19T12",
            "2020-09-27T19:54:52",
            75,
            A_ETH75,
            "observed_continuity",
        ),
        (
            "max_leverage",
            "2020-09-27T19:54:52",
            None,
            75,
            FAQ_TIERS_2020,
            "dated_snapshot",
        ),
        (
            "max_leverage",
            "2021-01-24T09:19:11",
            None,
            100,
            FAQ_LEVERAGE_2021,
            "dated_snapshot",
        ),
        (
            "max_leverage",
            "2021-01-24T11",
            "2023-12-24T09:35",
            100,
            FAQ_LEVERAGE_2021,
            "observed_continuity",
        ),
        (
            "max_leverage",
            "2023-12-24T09:35",
            "2025-08-19T06:30",
            125,
            R2023,
            "official",
        ),
        ("max_leverage", "2025-08-19T06:30", None, 150, R2025, "official"),
        (
            "max_leverage",
            "2025-08-19T08",
            NOW,
            150,
            LIVE_BRACKETS,
            "bracket_update_time",
        ),
        ("max_leverage", NOW, LIVE_END, 150, LIVE_BRACKETS, "dated_snapshot"),
        (
            "clearance_fee_rate",
            "2019-09-18T18:58:37",
            None,
            0.005,
            FAQ_LIQ_2019,
            "upper_bound",
        ),
        (
            "clearance_fee_rate",
            "2020-05-20T12:51:29",
            None,
            0.005,
            FAQ_LIQ_2020,
            "dated_snapshot",
        ),
        (
            "clearance_fee_rate",
            "2020-05-20T14",
            "2020-11-23T07",
            0.005,
            FAQ_LIQ_2020,
            "observed_continuity",
        ),
        ("clearance_fee_rate", "2020-11-23T07", None, 0.0075, A2020_11, "official"),
        (
            "clearance_fee_rate",
            "2021-02-07T03",
            None,
            0.015,
            A2021_02,
            "official_prior_value",
        ),
        (
            "clearance_fee_rate",
            "2021-02-07T04",
            "2021-05-20T09",
            0.030,
            A2021_02,
            "official",
        ),
        ("clearance_fee_rate", "2021-05-20T09", None, 0.005, A2021_05, "official"),
        # announcement / snapshot points inside the inferred spans above
        ("maint_rate", "2023-12-24T09", None, 0.005, R2023, "official_prior_value"),
        ("maint_rate", "2025-08-19T06:30", None, 0.004, R2025, "official"),
        ("tier_cap", "2023-12-24T09", None, 200000, R2023, "official_prior_value"),
        ("max_leverage", "2023-12-24T09", None, 100, R2023, "official_prior_value"),
        ("maint_rate", "2025-08-19T06", None, 0.004, R2025, "official_prior_value"),
    ),
}

# Brackets endpoint snapshots (Wayback timestamp; BTCUSDT and ETHUSDT first tier
# as (maintenance rate, notional cap, max leverage)), read from the archived JSON.
BRACKET_SNAPSHOTS = (
    ("20230109204158", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230112100642", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230117130848", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230123011658", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230123173245", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230125172711", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230126122316", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230130111032", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230201104251", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230202105414", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230202121809", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230203115212", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230206134006", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230208171928", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230209224855", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230214095447", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230217132942", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230220164732", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230221192504", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230222115333", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230223133306", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230301055208", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230301173353", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230307161316", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230309152624", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230310132447", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230312221217", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230313100500", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230313162159", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230314151252", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230316101735", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230317100433", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230320122408", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230322121208", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230323161729", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230324125455", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230327075932", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20230328074159", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20231002133915", (0.004, 50000, 125), (0.005, 100000, 100)),
    ("20231212142739", (0.004, 50000, 125), (0.005, 200000, 100)),
    ("20260826180437", (0.004, 300000, 150), (0.004, 300000, 150)),
)

# Leverage FAQ observations (Wayback timestamp, symbol, maintenance, cap, leverage, source).
FAQ_TIER_POINTS = (
    ("20200519164655", "BTCUSDT", 0.004, 50000, 125, FAQ_TIERS_2020_05),
    ("20200927195452", "BTCUSDT", 0.004, 50000, 125, FAQ_TIERS_2020),
    ("20210124091911", "BTCUSDT", 0.004, 50000, 125, FAQ_LEVERAGE_2021),
    ("20200519164655", "ETHUSDT", None, None, 75, FAQ_TIERS_2020_05),
)
FAQ_LIQ_2020_09 = (
    WB + "20200927210138/https://www.binance.com/en/support/faq/360033525271"
)
FAQ_CLEARANCE_POINTS = (("20200927210138", 0.003, 0.005, FAQ_LIQ_2020_09),)

# Adverse direction for the "conservative" gap policy.
ADVERSE = dict(maint_rate=max, clearance_fee_rate=max, tier_cap=min, max_leverage=min)


def _hour_ceil(value):
    t = np.datetime64(value, "ms")
    floor = t.astype("datetime64[h]").astype("datetime64[ms]")
    return t, (floor + HOUR if t != floor else floor)


def _observations(symbol):
    rows = list(_RISK.get(symbol, ()))
    if symbol in _RISK:
        column = 1 if symbol == "BTCUSDT" else 2
        for stamp, *values in CLEARANCE_SNAPSHOTS:
            ts = stamp.lstrip("w")
            url = (
                EXCHANGE_INFO_WWW if stamp.startswith("w") else EXCHANGE_INFO
            ).format(ts)
            at = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
            rows.append(
                (
                    "clearance_fee_rate",
                    at,
                    None,
                    values[column - 1],
                    url,
                    "dated_snapshot",
                )
            )

    def stamp_time(ts):
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}"

    if symbol in _RISK:
        fields = ("maint_rate", "tier_cap", "max_leverage")
        for ts, btc, eth in BRACKET_SNAPSHOTS:
            values = btc if symbol == "BTCUSDT" else eth
            for field, value in zip(fields, values, strict=True):
                rows.append(
                    (
                        field,
                        stamp_time(ts),
                        None,
                        value,
                        BRACKETS.format(ts),
                        "dated_snapshot",
                    )
                )
        for ts, name, *values, url in FAQ_TIER_POINTS:
            if name != symbol:
                continue
            for field, value in zip(fields, values, strict=True):
                if value is not None:
                    rows.append(
                        (field, stamp_time(ts), None, value, url, "dated_snapshot")
                    )
        for ts, btc, eth, url in FAQ_CLEARANCE_POINTS:
            value = btc if symbol == "BTCUSDT" else eth
            rows.append(
                (
                    "clearance_fee_rate",
                    stamp_time(ts),
                    None,
                    value,
                    url,
                    "dated_snapshot",
                )
            )
    if symbol in ("BTCUSDT", "ETHUSDT", "BTCUSDT_261225"):
        rows.append(
            ("clearance_fee_rate", NOW, LIVE_END, 0.0125, SNAPSHOT, "dated_snapshot")
        )
    result = {f: [] for f in FIELDS}
    for field, start, end, value, source, confidence in rows:
        if confidence not in EVIDENCE | INFERRED:
            raise AssertionError(f"unknown confidence {confidence}")
        effective, lo = _hour_ceil(start)
        hi = lo + HOUR if end is None else _hour_ceil(end)[1]
        if hi <= lo:
            raise AssertionError(f"empty {symbol} {field} record at {start}")
        timing = dict(
            effective_at=str(effective),
            applied_at=str(lo),
            approximation=bool(effective != lo),
            rounding="ceil_to_hour" if effective != lo else "none",
        )
        result[field].append((lo, hi, value, source, confidence, timing))
    for field in result:
        # Evidence points may sit inside inferred spans (evidence wins there);
        # records of the same class must not overlap.
        result[field].sort(key=lambda r: (r[0], r[4] in INFERRED))
        for group in (EVIDENCE, INFERRED):
            same = [r for r in result[field] if r[4] in group]
            for a, b in zip(same, same[1:], strict=False):
                if b[0] < a[1]:
                    raise AssertionError(f"overlapping {symbol} {field} records")
    return result


def _describe(record):
    lo, hi, value, source, confidence, timing = record
    return dict(
        value=value,
        source_url=source,
        confidence=confidence,
        start=str(lo),
        end=str(hi),
        effective_at=timing["effective_at"],
    )


def _fill_gaps(column, j, dates, missing, records, policy, symbol, field):
    """Fill hours no record covers. Neighbours include inferred records."""
    notes = []
    if not missing.any():
        return notes
    starts = np.array([r[0] for r in records])
    after = np.searchsorted(starts, dates, side="right")
    for k in np.unique(after[missing]):
        mask = missing & (after == k)
        before = records[k - 1] if k > 0 else None
        later = records[k] if k < len(records) else None
        side = (
            "between"
            if before and later
            else "after_last"
            if before
            else "before_first"
        )
        if policy == "previous":
            chosen = before or later
            near = [chosen]
            reason = "previous observation, or earliest before coverage"
        else:
            near = [r for r in (before, later) if r is not None]
            value = ADVERSE[field](r[2] for r in near)
            chosen = next(r for r in near if r[2] == value)
            reason = "conservative: most adverse neighbouring observation"
        column[mask, j] = chosen[2]
        notes += _segments(
            dates,
            mask,
            symbol=symbol,
            parameter=field,
            value=chosen[2],
            source_url=chosen[3],
            confidence="assumption",
            reason=reason,
            policy=policy,
            side=side,
            **chosen[5],
            chosen=_describe(chosen),
            candidates=[_describe(r) for r in near],
        )
    return notes


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

        By default only EVIDENCE records are used. With a policy, INFERRED spans
        are used and reported as assumptions, then remaining gaps are filled:
        ``assume='previous'`` extends the previous record (the earliest one
        before coverage); ``assume='conservative'`` takes the most adverse of
        the two neighbouring records (higher maintenance and clearance rates,
        lower tier cap and leverage). A mapping may specify ``policy=...`` and
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
            raise ValueError(
                "assume: expected 'previous', 'conservative' or explicit rule mapping"
            )
        if policy not in (None, "previous", "conservative"):
            raise ValueError("assume policy: expected 'previous' or 'conservative'")
        output = {
            f: np.full(shape, 1.0 if f in ("tier_cap", "max_leverage") else 0.0)
            for f in FIELDS
        }
        notes, errors = [], []
        for j, symbol in enumerate(names):
            for field, records in _observations(symbol).items():
                known = np.zeros(len(dates), bool)
                inferred = []
                for record in records:
                    lo, hi, value, source, confidence, timing = record
                    mask = active[:, j] & (dates >= lo) & (dates < hi)
                    if confidence in INFERRED:
                        inferred.append((mask, record))
                        continue
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
                    # An explicit mapping beats inferred spans and policies.
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
                elif missing.any() and policy is not None and records:
                    for mask, (_lo, _hi, value, source, confidence, timing) in inferred:
                        mask = mask & missing
                        output[field][mask, j] = value
                        missing &= ~mask
                        notes += _segments(
                            dates,
                            mask,
                            symbol=symbol,
                            parameter=field,
                            value=value,
                            source_url=source,
                            confidence="assumption",
                            reason=f"inferred span ({confidence}), not proof of no change",
                            **timing,
                        )
                    notes += _fill_gaps(
                        output[field], j, dates, missing, records, policy, symbol, field
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
