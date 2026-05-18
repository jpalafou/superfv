from types import ModuleType
from typing import Protocol

from .tools.device_management import ArrayLike
from .tools.variable_index_map import VariableIndexMap


class MultivarField(Protocol):
    def __call__(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: float,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...


class UnivarField(Protocol):
    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: float,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...
