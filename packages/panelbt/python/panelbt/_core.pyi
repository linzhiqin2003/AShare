from typing import Any

from numpy.typing import NDArray

BUILD_MARKER: str

def run(
    floats: list[NDArray],
    masks: list[NDArray],
    plans: list[tuple[NDArray, NDArray, NDArray, NDArray]],
    vectors: list[NDArray],
    scalars: list[float],
    flags: list[bool],
    threads: int,
) -> list[dict[str, Any]]: ...
