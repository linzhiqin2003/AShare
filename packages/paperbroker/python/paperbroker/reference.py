"""Slow independent Python specification, intended for event-by-event differential tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .api import Instrument, MarketConfig, QuoteArray, money


def fee_zero() -> dict[str, int]:
    return dict(commission=0, tax=0, other=0)


class ReferenceBroker:
    def __init__(
        self,
        symbols: list[str],
        initial_cash: float = 1_000_000,
        config: MarketConfig | None = None,
    ):
        config = config or MarketConfig()
        c = asdict(config) | dict(
            symbols=len(symbols), initial_cash=money(initial_cash)
        )
        c["instruments"] = [
            asdict(
                config.instruments.get(name)
                or Instrument(
                    tick=config.tick,
                    t_plus_one=config.t_plus_one,
                    costs=config.costs,
                    lots=config.lots,
                )
            )
            | {"name": name}
            for name in symbols
        ]
        self.state = dict(
            version=3,
            config=c,
            day=0,
            time=0,
            phase="closed",
            cash=c["initial_cash"],
            frozen=0,
            positions=[dict(total=0, sellable=0, frozen=0, cost=0) for _ in symbols],
            quotes=[None for _ in symbols],
            quote_received=[0 for _ in symbols],
            corporate_done=[False for _ in symbols],
            corporate_dates=[0 for _ in symbols],
            actions_through=0,
            event_sequence=0,
            order_offset=0,
            auction_done=[[False, False] for _ in symbols],
            marks=[0 for _ in symbols],
            orders=[],
            fees=fee_zero(),
            day_fees=fee_zero(),
            flows=c["initial_cash"],
            settled_day=0,
        )
        self.output: list[dict[str, Any]] = []

    def emit(self, event_type: str, **values: Any) -> None:
        self.state["event_sequence"] += 1
        self.output.append(
            deepcopy(
                dict(
                    type=event_type,
                    seq=self.state["event_sequence"],
                    time=self.state["time"],
                    **({"day": self.state["day"]} | values),
                )
            )
        )

    def changed(self, symbol: int | None = None) -> None:
        s = self.state
        self.emit(
            "account",
            cash=s["cash"],
            frozen=s["frozen"],
            symbol=symbol,
            position=None if symbol is None else s["positions"][symbol],
        )

    def reject(self, reason: str, order_id: int | None = None) -> None:
        self.emit("rejected", reason=reason, id=order_id)

    def fees(self, side: str, value: int, symbol: int = 0) -> dict[str, int]:
        c = self.state["config"]["instruments"][symbol]["costs"]
        if not value:
            return fee_zero()
        commission = max(c["minimum"], -(-value * c["commission_ppm"] // 1_000_000))
        tax = -(-value * c["sell_tax_ppm"] // 1_000_000) if side == "sell" else 0
        return dict(commission=commission, tax=tax, other=0)

    def reserve(self, order: dict[str, Any]) -> int:
        if order["side"] == "sell" or not order["remaining"]:
            return 0
        value = order["remaining"] * order["price"]
        return (
            value
            + sum(
                self.fees(
                    order["side"], order["notional"] + value, order["symbol"]
                ).values()
            )
            - sum(order["fees"].values())
        )

    def account(self) -> dict[str, Any]:
        s = self.state
        value = sum(
            p["total"] * mark
            for p, mark in zip(s["positions"], s["marks"], strict=True)
        )
        return deepcopy(
            dict(
                cash=s["cash"],
                frozen=s["frozen"],
                market_value=value,
                equity=s["cash"] + s["frozen"] + value,
                positions=s["positions"],
                fees=s["fees"],
                day_fees=s["day_fees"],
                flows=s["flows"],
                day=s["day"],
                phase=s["phase"],
            )
        )

    def events(self) -> list[dict[str, Any]]:
        result, self.output = self.output, []
        return result

    def finish(self, order: dict[str, Any], status: str) -> None:
        s = self.state
        s["cash"] += order["reserve"]
        s["frozen"] -= order["reserve"]
        order["reserve"] = 0
        order["status"] = status
        p = s["positions"][order["symbol"]]
        if order["side"] == "sell":
            p["sellable"] += order["remaining"]
            p["frozen"] -= order["remaining"]
        self.emit("ended", id=order["id"], status=status, remaining=order["remaining"])
        self.changed(order["symbol"])

    def apply(self, event: dict[str, Any]) -> None:
        s, c = self.state, self.state["config"]
        if event["time"] < s["time"]:
            self.reject("time")
            return
        s["time"] = event["time"]
        kind = event["type"]
        if kind == "session":
            day, phase = event["day"], event["phase"]
            if day < s["day"] or (day != s["day"] and s["day"] != s["settled_day"]):
                self.reject("session")
                return
            if day != s["day"]:
                s.update(
                    day=day,
                    order_offset=s["order_offset"] + len(s["orders"]),
                    orders=[],
                    day_fees=fee_zero(),
                    quotes=[None] * c["symbols"],
                    quote_received=[0] * c["symbols"],
                    corporate_done=[False for _ in range(c["symbols"])],
                    auction_done=[[False, False] for _ in range(c["symbols"])],
                )
            if phase != s["phase"] and s["phase"] in ("open_auction", "close_auction"):
                auction_kind = 1 if s["phase"] == "open_auction" else 2
                for q in s["quotes"]:
                    if (
                        q
                        and q[17] == auction_kind
                        and not s["auction_done"][q[0]][auction_kind - 1]
                    ):
                        self.match(q, q[16], True)
                        s["auction_done"][q[0]][auction_kind - 1] = True
            s["phase"] = phase
            self.emit("session", day=day, phase=phase)
        elif kind == "order":
            symbol, side, qty, price = (
                event[k] for k in ("symbol", "side", "qty", "price")
            )
            reason = None
            if (
                s["phase"] not in ("continuous", "open_auction", "close_auction")
                or s["day"] == s["settled_day"]
            ):
                reason = "session"
            elif symbol >= c["symbols"]:
                reason = "unknown_symbol"
            elif s["quotes"][symbol] is None:
                reason = "no_quote"
            else:
                q, p = s["quotes"][symbol], s["positions"][symbol]
                profile = c["instruments"][symbol]
                if q[15]:
                    reason = "suspended"
                elif (
                    qty <= 0
                    or qty > 1_000_000_000
                    or (
                        (
                            qty % profile["lots"]["step"]
                            or qty
                            < max(profile["lots"]["minimum"], profile["lots"]["step"])
                        )
                        and not (
                            side == "sell"
                            and (
                                qty == p["sellable"]
                                if profile["lots"]["step"] == 1
                                else qty % profile["lots"]["step"]
                                == p["sellable"] % profile["lots"]["step"]
                            )
                        )
                    )
                ):
                    reason = "quantity"
                elif price <= 0 or price > 1_000_000_000 or price % profile["tick"]:
                    reason = "tick"
                elif c["enforce_bands"] and (
                    (q[13] > 0 and price > q[13]) or (q[14] > 0 and price < q[14])
                ):
                    reason = "price_band"
                elif side == "sell" and qty > p["sellable"]:
                    reason = "sellable"
            if reason:
                self.reject(reason)
                return
            o = dict(
                id=s["order_offset"] + len(s["orders"]) + 1,
                symbol=symbol,
                side=side,
                kind=event["kind"],
                qty=qty,
                remaining=qty,
                price=price,
                time=s["time"],
                reserve=0,
                notional=0,
                fees=fee_zero(),
                status="open",
            )
            o["reserve"] = self.reserve(o)
            if o["reserve"] > s["cash"]:
                self.reject("cash")
                return
            s["cash"] -= o["reserve"]
            s["frozen"] += o["reserve"]
            if side == "sell":
                s["positions"][symbol]["sellable"] -= qty
                s["positions"][symbol]["frozen"] += qty
            s["orders"].append(o)
            self.emit("accepted", id=o["id"])
            self.changed(symbol)
        elif kind == "cancel":
            order_id = event["id"]
            if (
                not s["order_offset"] < order_id <= s["order_offset"] + len(s["orders"])
                or s["orders"][order_id - s["order_offset"] - 1]["status"] != "open"
            ):
                self.reject("unknown_order", order_id)
            else:
                self.finish(s["orders"][order_id - s["order_offset"] - 1], "cancelled")
        elif kind == "transfer":
            amount = event["amount"]
            if not 0 <= s["cash"] + amount <= 10**18:
                self.reject("cash")
            else:
                s["cash"] += amount
                s["flows"] += amount
                self.changed()
        elif kind in ("settle", "corporate"):
            actions = event["actions"]
            action_day = event.get("day", s["day"])
            valid = len({a["symbol"] for a in actions}) == len(actions) and all(
                0 <= a["symbol"] < c["symbols"]
                and s["corporate_dates"][a["symbol"]] < action_day
                and 0 <= a["cash_per_share"] <= 10**9
                and 1 <= a["split_num"] <= 1000
                and 1 <= a["split_den"] <= 1000
                and (a["reference_price"] is None or 1 <= a["reference_price"] <= 10**9)
                and s["positions"][a["symbol"]]["total"]
                * a["split_num"]
                // a["split_den"]
                <= 10**9
                for a in actions
            )
            if kind == "corporate":
                if (
                    s["phase"] != "preopen"
                    or any(o["status"] == "open" for o in s["orders"])
                    or not s["actions_through"] <= action_day <= s["day"]
                    or not valid
                ):
                    self.reject("invalid_action")
                    return
            else:
                if event["day"] == s["settled_day"]:
                    self.reject("already_settled")
                    return
                if event["day"] != s["day"] or s["phase"] != "closed":
                    self.reject("session")
                    return
                if (
                    not valid
                    or abs(event["interest"]) > 10**18 // 4
                    or s["cash"] + s["frozen"] + event["interest"] < 0
                ):
                    self.reject("invalid_action")
                    return
                for o in s["orders"]:
                    if o["status"] == "open":
                        self.finish(o, "expired")
            for a in actions:
                s["corporate_done"][a["symbol"]] = action_day == s["day"]
                s["corporate_dates"][a["symbol"]] = action_day
                p = s["positions"][a["symbol"]]
                old = p["total"]
                dividend = old * a["cash_per_share"]
                s["cash"] += dividend
                p["total"] = old * a["split_num"] // a["split_den"]
                p["sellable"] = p["sellable"] * a["split_num"] // a["split_den"]
                p["cost"] = max(0, p["cost"] - dividend)
                q = s["quotes"][a["symbol"]]
                s["marks"][a["symbol"]] = (
                    a["reference_price"]
                    if a["reference_price"] is not None
                    else max(0, s["marks"][a["symbol"]] - a["cash_per_share"])
                    * a["split_den"]
                    // a["split_num"]
                )
                if q:
                    q[2] = s["marks"][a["symbol"]]
                self.emit(
                    "corporate",
                    ex_day=action_day,
                    symbol=a["symbol"],
                    dividend=dividend,
                    before=old,
                    after=p["total"],
                )
                self.changed(a["symbol"])
            s["actions_through"] = max(s["actions_through"], action_day)
            if kind == "settle":
                s["cash"] += event["interest"]
                for p in s["positions"]:
                    p["sellable"] = p["total"]
                s["settled_day"] = event["day"]
                self.changed()
                self.emit("settled", day=event["day"], account=self.account())

    def feed(self, quotes: QuoteArray, *, received_at: int | None = None) -> None:
        from datetime import datetime, timedelta, timezone

        s = self.state
        if received_at is not None and received_at < s["time"]:
            raise ValueError("receipt time precedes engine time")
        for q in quotes.tolist():
            if q[1] >= 946_684_800_000:
                zone = timezone(timedelta(seconds=s["config"]["utc_offset_seconds"]))
                date = int(datetime.fromtimestamp(q[1] / 1000, zone).strftime("%Y%m%d"))
                if date != s["day"]:
                    continue
            old = s["quotes"][q[0]]
            if old and (
                q[1] < old[1]
                or (
                    q[1] == old[1]
                    and not (
                        q[17] > 0
                        and old[17] == 0
                        and q[2:16] == old[2:16]
                        and not s["auction_done"][q[0]][q[17] - 1]
                        and s["phase"] in ("open_auction", "close_auction")
                    )
                )
                or (q[7] < old[7] and s["phase"] != "closed")
            ):
                continue
            increment = q[7] - old[7] if old else 0
            s["time"] = received_at if received_at is not None else s["time"]
            s["quotes"][q[0]] = q
            s["marks"][q[0]] = q[2]
            if s["phase"] == "continuous" and q[17] == 0:
                self.match(q, increment, False)
            s["quote_received"][q[0]] = s["time"]
        if received_at is not None:
            s["time"] = received_at

    def match(self, q: list[int], volume: int, auction: bool) -> None:
        s, c = self.state, self.state["config"]
        if q[15]:
            return
        budget = volume * c["participation_ppm"] // 1_000_000
        available = {"buy": max(0, q[10]), "sell": max(0, q[12])}
        for o in s["orders"]:
            if (
                o["status"] != "open"
                or o["symbol"] != q[0]
                or q[1] <= o["time"]
                or (not auction and s["quote_received"][q[0]] < o["time"])
            ):
                continue
            buy = o["side"] == "buy"
            book = q[9] if buy else q[11]
            auction_price = q[3] if auction and q[17] == 1 else q[2]
            locked = (buy and q[13] > 0 and auction_price >= q[13]) or (
                not buy and q[14] > 0 and auction_price <= q[14]
            )
            if locked and (book <= 0 or available[o["side"]] == 0):
                continue
            if not auction and book == 0:
                continue
            profile = c["instruments"][q[0]]
            base = (
                (q[3] if q[17] == 1 else q[2])
                if auction
                else (book if book > 0 else q[2])
            )
            slip = c["slippage_ticks"] * profile["tick"] + -(
                -base * c["slippage_ppm"] // 1_000_000
            )
            price = (
                -(-(base + slip) // profile["tick"]) * profile["tick"]
                if buy
                else (base - slip) // profile["tick"] * profile["tick"]
            )
            if (
                price <= 0
                or (buy and price > o["price"])
                or (not buy and price < o["price"])
            ):
                continue
            if c["enforce_bands"] and (
                (q[13] > 0 and price > q[13]) or (q[14] > 0 and price < q[14])
            ):
                continue
            qty = min(o["remaining"], budget)
            if not auction and book >= 0:
                qty = min(qty, available[o["side"]])
            if qty <= 0:
                continue
            value = qty * price
            total_fees = self.fees(o["side"], o["notional"] + value, o["symbol"])
            fees = {k: total_fees[k] - o["fees"][k] for k in total_fees}
            if not buy and s["cash"] + value < sum(fees.values()):
                self.reject("cash", o["id"])
                self.finish(o, "unaffordable_fees")
                continue
            old_reserve = o["reserve"]
            o["notional"] += value
            o["fees"] = total_fees
            o["remaining"] -= qty
            o["reserve"] = self.reserve(o)
            p = s["positions"][q[0]]
            if buy:
                s["cash"] += old_reserve - o["reserve"] - value - sum(fees.values())
                s["frozen"] += o["reserve"] - old_reserve
                p["total"] += qty
                p["cost"] += value + sum(fees.values())
                if not profile["t_plus_one"]:
                    p["sellable"] += qty
            else:
                s["cash"] += value - sum(fees.values())
                p["cost"] = p["cost"] * (p["total"] - qty) // p["total"]
                p["total"] -= qty
                p["frozen"] -= qty
            for k in fees:
                s["fees"][k] += fees[k]
                s["day_fees"][k] += fees[k]
            budget -= qty
            if not auction and book > 0:
                available[o["side"]] -= qty
            self.emit(
                "fill",
                quote_time=q[1],
                id=o["id"],
                symbol=q[0],
                side=o["side"],
                qty=qty,
                price=price,
                fees=fees,
                auction=auction,
                fallback=not auction and book < 0,
            )
            self.changed(q[0])
            if not o["remaining"]:
                self.finish(o, "filled")
