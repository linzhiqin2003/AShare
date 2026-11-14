"""独立重建生命周期和规则预期，不调用生产查询函数。"""

import math

HOUR = 3_600_000


def verify_contract(frame, inputs):
    errors = []
    profiles = {p["instrument"]: p for p in inputs["profiles"]}
    previous = {}
    for r in frame.sort(["time", "instrument"]).iter_rows(named=True):
        name, t = r["instrument"], r["time"]
        p = profiles[name]

        def fail(check, name=name, t=t):
            errors.append(dict(check=check, instrument=name, time=t))

        latest = {}
        for e in inputs["events"]:
            if e["instrument"] != name or e["known_at"] > t:
                continue
            if (
                e["event_id"] not in latest
                or latest[e["event_id"]]["known_at"] < e["known_at"]
            ):
                latest[e["event_id"]] = e
        live = sorted(
            (e for e in latest.values() if e.get("status", "active") == "active"),
            key=lambda e: e["effective_at"],
        )
        terminal = [
            e
            for e in live
            if e["kind"] in ("delist", "settlement", "delivery")
            and e["effective_at"] <= t
        ]
        expected_life = (
            "pre_listing" if t < p["listed_at"] else "ended" if terminal else "active"
        )
        if r["lifecycle"] != expected_life:
            fail("lifecycle")
        rates = [
            x for x in inputs["data"][name].get("fundingRate", []) if x["time"] <= t
        ]
        expected_rate = (
            max(rates, key=lambda x: x["time"])["rate"]
            if rates and expected_life == "active"
            else math.nan
        )
        if (
            not (math.isnan(expected_rate) and math.isnan(r["last_funding_rate"]))
            and r["last_funding_rate"] != expected_rate
        ):
            fail("last_known_funding_rate")
        if expected_life == "pre_listing" and p["market"] == "usdm":
            marks = sorted(
                (
                    x
                    for x in inputs["data"][name]["markPriceKlines"]
                    if x["time"] >= p["listed_at"]
                ),
                key=lambda x: x["time"],
            )
            if not marks or r["settlement"] != marks[0]["close"]:
                fail("prelisting_placeholder")
        if expected_life == "ended" and name in previous and p["market"] == "usdm":
            if r["settlement"] != previous[name]["settlement"]:
                fail("terminal_carry")
        suspend = False
        for e in live:
            if e["effective_at"] <= t and e["kind"] in ("suspend", "resume"):
                suspend = e["kind"] == "suspend"
        expected_tradable = (
            expected_life == "active"
            and not suspend
            and not (r["quality"] & (1 | 2048))
            and r["volume"] > 0
        )
        if r["tradable"] != expected_tradable:
            fail("tradable")
        eligible = [
            x
            for x in p["rules"]
            if x["known_at"] <= t
            and ((x["effective_at"] + HOUR - 1) // HOUR) * HOUR <= t
        ]
        assumed = False
        if eligible:
            rule = sorted(eligible, key=lambda x: (x["effective_at"], x["known_at"]))[
                -1
            ]
            assumed = rule["source"] == "assumption"
        else:
            candidates = sorted(
                (x for x in p["rules"] if x["source"] in ("snapshot", "assumption")),
                key=lambda x: x["known_at"],
            )
            if not candidates:
                fail("missing_rule")
                continue
            rule, assumed = candidates[0], True
        unit = p["multiplier"] if p["market"] == "usdm" else 1
        for field in ("qty_step", "min_qty", "min_notional"):
            expected = rule[field] / unit if field != "min_notional" else rule[field]
            if not math.isclose(r[field], expected, rel_tol=1e-12, abs_tol=1e-15):
                fail("rule_" + field)
        if (
            bool(r["quality"] & 64) != assumed
            or r["rule_source_url"] != rule["source_url"]
        ):
            fail("rule_provenance")
        if r["multiplier"] != p["multiplier"] or r["coin_units"] != p["coin_units"]:
            fail("unit_mapping")
        previous[name] = r
        if (
            r["rule_known_at"] != rule["known_at"]
            or r["rule_effective_at"] != rule["effective_at"]
        ):
            fail("rule_time")
    return errors
