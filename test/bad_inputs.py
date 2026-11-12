"""Bad-input matrix shared by the validation tests and the recorder.

Each case corrupts exactly one value (or one structural property) of an
otherwise valid input. Panel values are placed on the first row, the last
row and on both sides of the kernel's parallel scan chunk boundaries, so a
scanner that skips a chunk edge is caught. The expected exception type and
message were recorded with the pre-kernel NumPy validation
(`test/fixtures/bad_inputs.json`, written by `python -m test.bad_inputs`).
"""

import json
import sys
from pathlib import Path

import numpy as np
import panelbt as bt

# Must equal panelbt._core.VALIDATION_CHUNK (checked by the tests).
CHUNK = 1 << 14
T, N = 120, 300  # 36,000 cells: two full chunks plus a partial one
FIXTURE = Path(__file__).parent / "fixtures" / "bad_inputs.json"

PANEL_POSITIONS = {
    "first": 0,
    "last_row": (T - 1) * N + 7,
    "last": T * N - 1,
    "chunk_end": CHUNK - 1,
    "chunk_start": CHUNK,
    "chunk2_end": 2 * CHUNK - 1,
    "chunk2_start": 2 * CHUNK,
}
FUTURE_COLS = (N - 2, N - 1)


def base_market_arrays():
    rng = np.random.default_rng(7)
    p = rng.uniform(5, 50, (T, N))
    return dict(
        exec_price=p,
        close=p * 1.01,
        can_buy=np.ones((T, N), bool),
        can_sell=np.ones((T, N), bool),
        volume=np.full((T, N), 1e6),
        split=np.ones((T, N)),
        cash_div=np.zeros((T, N)),
        delist=np.zeros((T, N), bool),
        can_short=np.ones((T, N), bool),
    )


def base_futures(kind_cols=FUTURE_COLS):
    kind = np.zeros(N, bool)
    kind[list(kind_cols)] = True
    return dict(
        is_future=kind,
        multiplier=np.where(kind, 300.0, 1.0),
        margin_rate=np.full((T, N), 0.12),
        settlement=np.full((T, N), 4000.0),
        expiry=np.where(kind, T, T).astype(np.int64),
        fee_rate=np.full((T, N), 2e-5),
        fee_per_contract=0.0,
        delivery_fee_rate=0.0,
        delivery_fee_per_contract=0.0,
    )


def base_targets(nnz_rows=None):
    days = np.arange(0, T, 2, dtype=np.int64)
    per = 200 if nnz_rows is None else nnz_rows
    cols = np.tile(np.arange(per, dtype=np.int64), len(days))
    weights = np.full(len(days) * per, 0.9 / per)
    indptr = np.arange(len(days) + 1, dtype=np.int64) * per
    return dict(days=days, indptr=indptr, cols=cols, weights=weights)


def flat_set(a, position, value):
    a = np.array(a, copy=True)
    a.reshape(-1)[position] = value
    return a


def market_case(field, position, value, futures=False):
    def build():
        arrays = base_market_arrays()
        arrays[field] = flat_set(arrays[field], position, value)
        spec = bt.Futures(**base_futures()) if futures else None
        return bt.Market(**arrays, futures=spec)

    return build


def futures_case(field, position, value, market_field=None):
    def build():
        arrays = base_market_arrays()
        spec = base_futures()
        if market_field is not None:
            arrays[market_field] = flat_set(arrays[market_field], position, value)
        else:
            spec[field] = flat_set(spec[field], position, value)
        return bt.Market(**arrays, futures=bt.Futures(**spec))

    return build


def targets_case(field, position, value, length=None):
    def build():
        plan = base_targets(length)
        plan[field] = flat_set(plan[field], position, value)
        return bt.Targets(**plan)

    return build


def run_case(mutate, *, futures=False, config=None, kind="weights"):
    def build():
        arrays = base_market_arrays()
        spec = bt.Futures(**base_futures()) if futures else None
        market = bt.Market(**arrays, futures=spec)
        plan = base_targets()
        mutate(plan)
        return bt.run(market, bt.Targets(**plan, kind=kind), None, config)

    return build


def cases():
    out = {}
    values = {
        "exec_price": {"inf": np.inf, "ninf": -np.inf, "zero": 0.0, "neg": -1.0},
        "close": {"inf": np.inf, "zero": 0.0, "neg": -1.0},
        "volume": {"inf": np.inf, "neg": -1.0},
        "split": {"nan": np.nan, "inf": np.inf, "zero": 0.0, "neg": -2.0},
        "cash_div": {"nan": np.nan, "inf": np.inf, "neg": -0.1},
    }
    for field, kinds in values.items():
        for vname, value in kinds.items():
            for pname, position in PANEL_POSITIONS.items():
                out[f"market.{field}.{vname}.{pname}"] = market_case(
                    field, position, value
                )
    # Two bad fields: the first field in the documented order wins.
    out["market.order.exec_then_close"] = lambda: bt.Market(
        **{
            **base_market_arrays(),
            "exec_price": flat_set(base_market_arrays()["exec_price"], 5, -1.0),
            "close": flat_set(base_market_arrays()["close"], 0, -1.0),
        }
    )
    out["market.order.volume_then_split"] = lambda: bt.Market(
        **{
            **base_market_arrays(),
            "split": flat_set(base_market_arrays()["split"], 3, 0.0),
            "volume": flat_set(base_market_arrays()["volume"], 9, -1.0),
        }
    )
    # Futures columns: settlement and corporate actions.
    fut_positions = {
        "first": FUTURE_COLS[0],
        "last_row": (T - 1) * N + FUTURE_COLS[1],
        "chunk": (CHUNK // N) * N + FUTURE_COLS[0],
    }
    for vname, value in {
        "nan": np.nan,
        "inf": np.inf,
        "zero": 0.0,
        "neg": -5.0,
    }.items():
        for pname, position in fut_positions.items():
            out[f"futures.settlement.{vname}.{pname}"] = futures_case(
                "settlement", position, value
            )
    for field, value in {"split": 2.0, "cash_div": 0.5, "delist": True}.items():
        for pname, position in fut_positions.items():
            out[f"futures.{field}.{pname}"] = futures_case(
                field, position, value, market_field=field
            )
    for vname, value in {"nan": np.nan, "inf": np.inf, "neg": -0.01}.items():
        for field in ("margin_rate", "fee_rate"):
            for pname, position in PANEL_POSITIONS.items():
                out[f"futures.{field}.{vname}.{pname}"] = futures_case(
                    field, position, value
                )
    for vname, value in {
        "zero": 0.0,
        "neg": -1.0,
        "nan": np.nan,
        "inf": np.inf,
    }.items():
        out[f"futures.multiplier.{vname}"] = futures_case(
            "multiplier", FUTURE_COLS[1], value
        )
    for vname, value in {"neg": -1, "beyond": T + 1}.items():
        out[f"futures.expiry.{vname}"] = futures_case("expiry", FUTURE_COLS[0], value)
    # Targets: values and structure. The long plan spans several chunks.
    long_nnz = 400  # 60 rows × 400 = 24,000 entries
    t_positions = {
        "first": 0,
        "chunk_end": CHUNK - 1,
        "chunk_start": CHUNK,
        "last": 60 * long_nnz - 1,
    }
    for vname, value in {"nan": np.nan, "inf": np.inf, "ninf": -np.inf}.items():
        for pname, position in t_positions.items():
            out[f"targets.weights.{vname}.{pname}"] = targets_case(
                "weights", position, value, long_nnz
            )
    for pname, position in t_positions.items():
        out[f"targets.cols.neg.{pname}"] = targets_case("cols", position, -1, long_nnz)
    # Base days are 0, 2, ..., 118; a repeat equals its neighbour.
    for pname, (position, neighbour) in {
        "first": (0, 2),
        "middle": (30, 58),
        "last": (59, 116),
    }.items():
        out[f"targets.days.neg.{pname}"] = targets_case("days", position, -3)
        out[f"targets.days.repeat.{pname}"] = targets_case("days", position, neighbour)
    out["targets.indptr.start"] = targets_case("indptr", 0, 1)
    out["targets.indptr.end"] = targets_case("indptr", 60, 5)
    out["targets.indptr.decrease"] = targets_case("indptr", 30, 1)
    for pname, row in {"first": 0, "chunk": CHUNK // long_nnz, "last": 59}.items():
        start = row * long_nnz
        out[f"targets.cols.duplicate.{pname}"] = targets_case(
            "cols", start + long_nnz - 1, 0, long_nnz
        )
    # Context checks applied when a plan meets a market and configuration.
    out["run.days.beyond_t"] = run_case(
        lambda p: p.__setitem__("days", np.r_[p["days"][:-1], T])
    )
    out["run.cols.beyond_n"] = run_case(
        lambda p: p.__setitem__("cols", flat_set(p["cols"], 5, N))
    )
    for pname, position in {"first": 0, "middle": 6000, "last": 11999}.items():
        out[f"run.negative_without_short.{pname}"] = run_case(
            lambda p, q=position: p.__setitem__(
                "weights", flat_set(p["weights"], q, -0.001)
            )
        )
        out[f"run.row_sum.{pname}"] = run_case(
            lambda p, q=position: p.__setitem__(
                "weights", flat_set(p["weights"], q, 0.5)
            )
        )

    def fractional_contracts(p):
        p["cols"] = p["cols"].copy()
        p["cols"][199::200] = FUTURE_COLS[0]
        p["weights"] = np.where(p["cols"] == FUTURE_COLS[0], 2.0, 1.0)
        p["weights"][-1] = 2.5

    out["run.futures_fractional_shares"] = run_case(
        fractional_contracts, futures=True, kind="shares"
    )
    return out


def outcome(build):
    try:
        build()
    except Exception as exc:  # noqa: BLE001 - the recorded type is the result
        return [type(exc).__name__, str(exc)]
    return ["ok", ""]


def record():
    result = {name: outcome(build) for name, build in sorted(cases().items())}
    FIXTURE.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{len(result)} cases -> {FIXTURE}", file=sys.stderr)


if __name__ == "__main__":
    record()
