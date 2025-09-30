from types import ModuleType
from typing import Optional, Protocol

from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap


class MultivarField(Protocol):
    def __call__(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float] = None,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...


class UnivarField(Protocol):
    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float] = None,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...
