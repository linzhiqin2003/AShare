from typing import Any

from numpy.typing import NDArray

BUILD_MARKER: str
SOURCE_SHA256: str
VALIDATION_CHUNK: int

def run(
    floats: list[NDArray],
    masks: list[NDArray],
    plans: list[tuple[NDArray, NDArray, NDArray, NDArray, bool]],
    vectors: list[NDArray],
    scalars: list[float],
    flags: list[bool],
    first_delist: NDArray,
    actions: tuple[NDArray, NDArray],
    future: tuple[NDArray, NDArray, NDArray, list[NDArray]],
    validated: list[bool],
    context: tuple[bool, bool],
    threads: int,
) -> list[dict[str, Any]]: ...
def prepare_market(
    price: NDArray,
    close: NDArray,
    buy: NDArray,
    sell: NDArray,
    volume: NDArray,
    split: NDArray,
    dividend: NDArray,
    delist: NDArray,
    can_short: NDArray,
    validate: bool,
) -> tuple[NDArray, NDArray, NDArray]: ...
def validate_futures(
    kind: NDArray,
    multiplier: NDArray,
    expiry: NDArray,
    settlement: NDArray,
    split: NDArray,
    dividend: NDArray,
    delist: NDArray,
    rates: list[tuple[str, NDArray]],
) -> None: ...
def check_plan(
    days: NDArray, indptr: NDArray, cols: NDArray, weights: NDArray, validate: bool
) -> tuple[bool, bool, float]: ...
