from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class InterpolationScheme(ABC):
    name: str
    p: int
    gauss_legendre: bool = False
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    lazy_primitives: bool = False
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
    p: int = 0
    gauss_legendre: bool = False
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    lazy_primitives: bool = False
    limiter: Optional[Literal["zhang-shu"]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name != "poly":
            raise ValueError("polyInterpolationScheme must have name 'poly'")

    def key(self) -> str:
        return f"{self.name}{self.p}"


@dataclass
class musclInterpolationScheme(InterpolationScheme):
    name: Literal["muscl"] = "muscl"
    p: int = 1
    gauss_legendre: bool = False
    flux_recipe: Optional[Literal[1, 2, 3]] = None
    lazy_primitives: bool = True
    limiter: Optional[Literal["minmod", "moncen"]] = None

    def key(self) -> str:
        return f"{self.name}_{self.limiter}"

    def __post_init__(self):
        super().__post_init__()
        if self.name != "muscl":
            raise ValueError("musclInterpolationScheme must have name 'muscl'")
        if self.p != 1:
            raise ValueError("musclInterpolationScheme only supports p=1")
