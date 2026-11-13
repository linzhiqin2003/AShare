from typing import Any

from numpy.typing import NDArray

BUILD_MARKER: str
SOURCE_SHA256: str
VALIDATION_CHUNK: int

def _numpy_sums(probes: list[NDArray]) -> NDArray: ...
def run_weights_batch(
    returns: NDArray,
    buy: NDArray,
    sell: NDArray,
    targets: list[tuple[NDArray, NDArray, int, int, float, float, bool]],
) -> list[dict[str, Any]]: ...
def run_group_weights(
    factor: NDArray,
    logcap: NDArray,
    industry: NDArray,
    vectors: list[NDArray],
    universes: list[NDArray],
    price: NDArray,
    close: NDArray,
    adjustment: NDArray,
    buy: NDArray,
    sell: NDArray,
    options: tuple[int, int, int, int, int, bool, bool, bool],
) -> list[dict[str, Any]]: ...
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
