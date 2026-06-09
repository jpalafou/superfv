from types import ModuleType

from superfv.tools.device_management import ArrayLike
from superfv.tools.variable_index_map import VariableIndexMap


def trivial_source(idx: VariableIndexMap, u: ArrayLike, *, xp: ModuleType) -> ArrayLike:
    return xp.zeros_like(u)
