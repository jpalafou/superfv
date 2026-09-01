from typing import Dict, Literal, Tuple

XYZ_TUPLE: Tuple[Literal["x", "y", "z"], Literal["x", "y", "z"], Literal["x", "y", "z"]] = (
    "x",
    "y",
    "z",
)
AXIS_TO_DIM: Dict[int, Literal["x", "y", "z"]] = {1: "x", 2: "y", 3: "z"}
DIM_TO_AXIS: Dict[Literal["x", "y", "z"], int] = {"x": 1, "y": 2, "z": 3}


def get_transverse_dims(
    dim: Literal["x", "y", "z"], active_dims: Tuple[Literal["x", "y", "z"], ...] = XYZ_TUPLE
) -> Tuple[Literal["x", "y", "z"], ...]:
    return tuple(d for d in active_dims if d != dim)
