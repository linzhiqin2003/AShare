"""Broad random corpus: no fixed warm-up probes are counted as random coverage."""

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import paperbroker
from paperbroker import Broker, Costs, Instrument, Lots, MarketConfig, money
from paperbroker.reference import ReferenceBroker
from test_engine import assert_invariants

assert (
    Path(paperbroker.__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[3])
)


def run(seed):
    rng = np.random.default_rng(seed)
    cfg = MarketConfig(
        t_plus_one=bool(seed % 2),
        participation_ppm=int(rng.choice([50_000, 100_000, 1_000_000])),
        slippage_ticks=int(rng.integers(0, 3)),
        slippage_ppm=int(rng.choice([0, 0, 300, 2000])),
        costs=Costs(
            int(rng.integers(0, 400)),
            int(rng.choice([0, 50_000])),
            int(rng.choice([0, 500, 1000])),
        ),
        enforce_bands=bool(rng.integers(0, 4)),
    )
    names = ["A", "B", "C", "D"]
    cfg.instruments.update(
        {
            "A": Instrument(t_plus_one=cfg.t_plus_one, costs=cfg.costs),
            "B": Instrument(
                lots=Lots(1, 200),
                costs=cfg.costs,
                t_plus_one=cfg.t_plus_one,
                profile_id="star-v1",
            ),
            "C": Instrument(
                lots=Lots(1, 100),
                costs=cfg.costs,
                t_plus_one=cfg.t_plus_one,
                profile_id="bse-v1",
            ),
            "D": Instrument(
                tick=10,
                costs=Costs(cfg.costs.commission_ppm, cfg.costs.minimum, 0),
                t_plus_one=cfg.t_plus_one,
                profile_id="fund-v1",
            ),
        }
    )
    b, r = Broker(names, 200_000, cfg), ReferenceBroker(names, 200_000, cfg)
    cov = Counter()
    t = 0
    native_match = r.match

    def measured_match(q, volume, auction):
        # Count eligible resting orders at the actual matching call, not shapes
        # observed in input. Other time, quantity, price and rate conditions pass.
        st = r.state
        candidates = []
        output_start = len(r.output)
        budget = volume * cfg.participation_ppm // 1_000_000
        if not q[15] and budget > 0:
            for order in st["orders"]:
                if (
                    order["symbol"] != q[0]
                    or order["status"] != "open"
                    or not order["remaining"]
                ):
                    continue
                buy = order["side"] == "buy"
                profile = st["config"]["instruments"][q[0]]
                book, depth = (q[9], q[10]) if buy else (q[11], q[12])
                price = (
                    q[3]
                    if auction and q[17] == 1
                    else q[2]
                    if auction or book <= 0
                    else book
                )
                slip = (
                    cfg.slippage_ticks * profile["tick"]
                    + (price * cfg.slippage_ppm + 999999) // 1000000
                )
                raw = price + slip if buy else price - slip
                tick = profile["tick"]
                execution = ((raw + tick - 1) // tick if buy else raw // tick) * tick
                if (
                    execution <= 0
                    or (buy and execution > order["price"])
                    or (not buy and execution < order["price"])
                ):
                    continue
                if cfg.enforce_bands and not q[14] <= execution <= q[13]:
                    continue
                if q[1] <= order["time"]:
                    candidates.append(
                        (order["id"], "blocked:quote_time:" + order["side"])
                    )
                    continue
                if not auction and st["quote_received"][q[0]] < order["time"]:
                    candidates.append(
                        (order["id"], "blocked:preorder_volume:" + order["side"])
                    )
                    continue
                lock_price = q[3] if auction and q[17] == 1 else q[2]
                locked = lock_price >= q[13] if buy else lock_price <= q[14]
                if locked and (book <= 0 or depth == 0):
                    shape = (
                        "missing"
                        if book < 0
                        else "zero_price"
                        if book == 0
                        else "zero_size"
                    )
                    phase = (
                        ("open_auction" if q[17] == 1 else "close_auction")
                        if auction
                        else "continuous"
                    )
                    candidates.append(
                        (order["id"], f"blocked:lock:{phase}:{order['side']}:{shape}")
                    )
        native_match(q, volume, auction)
        executions = [e for e in r.output[output_start:] if e["type"] == "fill"]
        for order_id, key in candidates:
            # Earlier orders may have consumed the shared participation budget.
            if sum(e["qty"] for e in executions if e["id"] < order_id) < budget:
                assert not any(e["id"] == order_id for e in executions), key
                cov[key] += 1

    r.match = measured_match
    vol = [0] * 4
    stats = dict(min_cash=10**30)
    previous = json.loads(b.dump())

    def check(tag):
        nonlocal previous
        s = json.loads(b.dump())
        if s != r.state:
            raise AssertionError(f"seed {seed} {tag}: state differs")
        e1, e2 = b.events(), r.events()
        if e1 != e2:
            raise AssertionError(f"seed {seed} {tag}: events differ")
        assert_invariants(s, previous, e1, Counter(), last_interest[0])
        previous = s
        for e in e1:
            cov[e["type"]] += 1
            if e["type"] == "rejected":
                cov["rej:" + e["reason"]] += 1
            if e["type"] == "accepted":
                cov["odd_lot_accepted"] += (
                    s["orders"][e["id"] - s["order_offset"] - 1]["qty"] % 100 != 0
                )
            if e["type"] == "fill":
                cov[
                    "fill_profile:"
                    + s["config"]["instruments"][e["symbol"]]["profile_id"]
                ] += 1
                cov[
                    "fill:"
                    + e["side"]
                    + (
                        ":auction"
                        if e["auction"]
                        else ":fallback"
                        if e["fallback"]
                        else ":book"
                    )
                ] += 1
        stats["min_cash"] = min(stats["min_cash"], s["cash"])
        assert s["frozen"] >= 0, (seed, tag, "frozen<0")
        assert s["frozen"] == sum(
            o["reserve"] for o in s["orders"] if o["status"] == "open"
        ), (seed, tag, "frozen != reserves")
        for i, p in enumerate(s["positions"]):
            assert (
                0 <= p["sellable"]
                and 0 <= p["frozen"]
                and p["sellable"] + p["frozen"] <= p["total"]
            ), (seed, tag, p)
            assert p["frozen"] == sum(
                o["remaining"]
                for o in s["orders"]
                if o["status"] == "open" and o["side"] == "sell" and o["symbol"] == i
            )
        for o in s["orders"]:
            if o["status"] == "open":
                assert o["remaining"] > 0
        return s

    last_interest = [0]

    def ev(**k):
        last_interest[0] = k.get("interest", 0)
        nonlocal t
        t += int(rng.integers(0, 3))  # equal timestamps happen
        b.apply(dict(time=t, **k))
        r.apply(dict(time=t, **k))
        return check(k["type"])

    def px(lo=9.0, hi=11.0):
        return round(float(rng.integers(int(lo * 100), int(hi * 100) + 1)) / 100, 2)

    def quotes(kind=0, n=None):
        nonlocal t
        last_interest[0] = 0
        rows = []
        for selected in rng.choice(
            4, min(n or int(rng.integers(1, 4)), 4), replace=False
        ):
            s = int(selected)
            t += int(rng.integers(0, 3))
            step = int(rng.integers(0, 12))
            change = (
                0
                if step == 1
                else -int(rng.integers(1, 5000))
                if step == 0
                else int(rng.integers(1, 30000))
            )
            cov["input_volume_same"] += change == 0
            cov["input_volume_backwards"] += change < 0
            vol[s] += change
            vol[s] = max(vol[s], 0)
            last = float(rng.choice([9.0, 11.0, px(), px()]))
            mode = int(rng.choice([0, 0, 1, 1, 2, 2, 3, 4, 5]))
            ask = (
                None
                if mode == 0
                else 0
                if mode == 1
                else min(11.0, round(last + float(rng.integers(0, 4)) / 100, 2))
            )
            bid = (
                None
                if mode == 0
                else 0
                if mode == 2
                else max(9.0, round(last - float(rng.integers(0, 4)) / 100, 2))
            )
            rows.append(
                [
                    s,
                    t + int(rng.choice([-3, -1, 0, 0, 2, 4])),
                    money(last),
                    money(last if rng.integers(0, 2) else px()),
                    money(11),
                    money(9),
                    money(10),
                    vol[s],
                    money(10) * vol[s],
                    -1 if ask is None else money(ask),
                    int(rng.choice([0, 0, 0, int(rng.integers(1, 3000))])),
                    -1 if bid is None else money(bid),
                    int(rng.choice([0, 0, 0, int(rng.integers(1, 3000))])),
                    money(11),
                    money(9),
                    int(rng.integers(0, 15) == 0),
                    int(rng.integers(0, 20000)) if kind else 0,
                    kind,
                ]
            )
        a = np.array(rows, np.int64)
        for row in rows:
            cov["input_suspended"] += row[15] != 0
            cov["input_equal_time"] += (
                previous["quotes"][row[0]] is not None
                and row[1] == previous["quotes"][row[0]][1]
            )
            cov["input_upper_lock"] += row[2] == money(11) and row[9] <= 0
            cov["input_lower_lock"] += row[2] == money(9) and row[11] <= 0
            cov["input_auction"] += row[17] > 0
            for order in previous["orders"]:
                if order["symbol"] == row[0] and order["status"] == "open":
                    cov["nonmarketable_" + order["side"]] += (
                        order["price"] < row[9]
                        if order["side"] == "buy"
                        else order["price"] > row[11] > 0
                    )
        b.feed(a, received_at=t)
        r.feed(a, received_at=t)
        check("feed")

    for day_index in range(8):
        date = datetime(
            2026, 11, 11, 9, 0, tzinfo=timezone(timedelta(hours=8))
        ) + timedelta(days=day_index)
        day = int(date.strftime("%Y%m%d"))
        t = int(date.timestamp() * 1000)
        ev(type="session", day=day, phase="preopen")
        if day_index > 0 and rng.integers(0, 2):
            ev(
                type="corporate",
                actions=[
                    dict(
                        symbol=int(rng.integers(0, 4)),
                        cash_per_share=int(rng.choice([0, 1200, 5000])),
                        split_num=int(rng.choice([1, 13, 3, 11])),
                        split_den=int(rng.choice([1, 10, 2])),
                        reference_price=None,
                    )
                ],
            )
        ev(type="session", day=day, phase="open_auction")
        quotes(n=6)
        for phase, steps in (
            ("open_auction", 64),
            ("continuous", 90),
            ("break", 3),
            ("continuous", 60),
            ("close_auction", 64),
        ):
            ev(type="session", day=day, phase=phase)
            for _ in range(steps):
                c = int(rng.integers(0, 10))
                if c <= 4:
                    quotes(
                        kind={"open_auction": 1, "close_auction": 2}.get(phase, 0)
                        if rng.integers(0, 3)
                        else 0
                    )
                elif c <= 7:
                    s = int(rng.integers(0, 4))
                    side = str(rng.choice(["buy", "sell"]))
                    pos = json.loads(b.dump())["positions"][s]
                    qty = int(
                        rng.choice(
                            [
                                100,
                                200,
                                300,
                                1000,
                                pos["sellable"] or 100,
                                max(pos["sellable"] - 100, 100),
                                150,
                            ]
                        )
                    )
                    ev(
                        type="order",
                        symbol=s,
                        side=side,
                        kind=str(rng.choice(["limit", "market"])),
                        qty=qty,
                        price=money(
                            float(
                                rng.choice(
                                    [11 if side == "buy" else 9] * 3
                                    + [px(8.9, 11.1), px(8.9, 11.1)]
                                )
                            )
                        )
                        + int(rng.choice([0, 0, 0, 50])),
                    )
                elif c == 8:
                    ev(
                        type="cancel",
                        id=int(
                            rng.integers(
                                0,
                                json.loads(b.dump())["order_offset"]
                                + len(json.loads(b.dump())["orders"])
                                + 2,
                            )
                        ),
                    )
                else:
                    ev(
                        type="transfer",
                        amount=int(rng.choice([-50_000_000, 30_000_000, 1])),
                    )
            if phase in ("open_auction", "close_auction"):
                quotes(kind=1 if phase == "open_auction" else 2, n=4)
        ev(type="session", day=day, phase="closed")
        ev(type="settle", day=day, actions=[], interest=int(rng.integers(-300, 300)))
    return cov, stats["min_cash"]


def test_wide_random_differential():
    count = Counter()
    minimum = 10**30
    for seed in range(60):
        cov, cash = run(seed)
        count.update(cov)
        minimum = min(minimum, cash)
    required = [
        "fill:" + side + ":" + route
        for side in ("buy", "sell")
        for route in ("book", "fallback", "auction")
    ]
    required += [
        "odd_lot_accepted",
        "input_volume_same",
        "input_volume_backwards",
        "input_suspended",
        "input_equal_time",
        "input_upper_lock",
        "input_lower_lock",
        "input_auction",
        "nonmarketable_buy",
        "nonmarketable_sell",
        "corporate",
        "rej:quantity",
        "rej:tick",
        "rej:price_band",
        "rej:suspended",
        "rej:sellable",
        "rej:cash",
    ]
    required += [
        f"blocked:lock:{phase}:{side}:{shape}"
        for phase in ("continuous", "open_auction", "close_auction")
        for side in ("buy", "sell")
        for shape in ("missing", "zero_price", "zero_size")
    ]
    required += [
        "fill_profile:" + profile
        for profile in ("cash-equity-v1", "star-v1", "bse-v1", "fund-v1")
    ]
    assert all(count[key] > 0 for key in required), {
        key: count[key] for key in required if count[key] == 0
    }
    assert minimum >= 0
    print("PAPER_WIDE_COVERAGE=" + json.dumps(dict(sorted(count.items()))))
