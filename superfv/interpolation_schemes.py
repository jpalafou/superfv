from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class InterpolationScheme(ABC):
    name: str
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    limiter: Optional[str] = None

    @abstractmethod
    def key(self) -> str:
        """Return a unique key for the interpolation scheme."""
        pass

    def __post_init__(self):
        if self.flux_recipe is None:
            raise ValueError("InterpolationScheme requires a flux_recipe.")


@dataclass
class polyInterpolationScheme(InterpolationScheme):
    name: Literal["poly"] = "poly"
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    limiter: Optional[Literal["zhang-shu"]] = None
    p: int = 0
    lazy_primitives: bool = False
    gauss_legendre: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.name != "poly":
            raise ValueError("polyInterpolationScheme must have name 'poly'")

    def key(self) -> str:
        return f"{self.name}{self.p}"


@dataclass
class musclInterpolationScheme(InterpolationScheme):
    name: Literal["muscl"] = "muscl"
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    limiter: Optional[Literal["minmod", "moncen"]] = None
    p: int = 1

    def key(self) -> str:
        return f"{self.name}_{self.limiter}"

    def __post_init__(self):
        super().__post_init__()
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
