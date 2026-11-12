from typing import Any

from numpy.typing import NDArray

BUILD_MARKER: str
SOURCE_SHA256: str

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
    threads: int,
) -> list[dict[str, Any]]: ...
