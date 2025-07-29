from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class InterpolationScheme(ABC):
    name: str
    limiter: Optional[str] = None

    @abstractmethod
    def key(self) -> str:
        """Return a unique key for the interpolation scheme."""
        pass


@dataclass
class polyInterpolationScheme(InterpolationScheme):
    name: Literal["poly"] = "poly"
    limiter: Optional[Literal["zhang-shu"]] = None
    p: int = 0
    mode: Literal[1, 2, 3] = 1
    lazy_primitives: bool = False
    gauss_legendre: bool = False

    def __post_init__(self):
        if self.name != "poly":
            raise ValueError("polyInterpolationScheme must have name 'poly'")

    def key(self) -> str:
        return f"{self.name}{self.p}"


@dataclass
class musclInterpolationScheme(InterpolationScheme):
    name: Literal["muscl"] = "muscl"
    limiter: Optional[Literal["minmod", "moncen"]] = None
    p: int = 1

    def key(self) -> str:
        return f"{self.name}_{self.limiter}"

    def __post_init__(self):
        if self.name != "muscl":
            raise ValueError("musclInterpolationScheme must have name 'muscl'")
        if self.limiter is None:
            raise ValueError("musclInterpolationScheme requires a limiter.")
        if self.limiter not in ["minmod", "moncen"]:
            raise ValueError(
                "musclInterpolationScheme only supports 'minmod' or 'moncen' limiters."
            )
        if self.p != 1:
            raise ValueError("musclInterpolationScheme only supports p=1")
