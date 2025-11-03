from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True, slots=True)
class LimiterConfig(ABC):
    """
    Base class for slope or flux limiter configurations.
    """

    @abstractmethod
    def key(self) -> str:
        """Return a unique key for the limiter configuration."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the limiter configuration to a dictionary."""
        pass


@dataclass(frozen=True, slots=True)
class InterpolationScheme(ABC):
    """
    Base class for interpolation schemes.

    Attributes:
        p: The polynomial degree.
        flux_recipe: The flux recipe to use:
            - 1: interpolate conservative nodes -> limit conservative nodes -> convert
            to primitive nodes -> compute fluxes
            - 2: interpolate conservative nodes -> convert to primitive nodes -> limit
            primitive nodes -> compute fluxes
            - 3: compute primitive cell averages -> interpolate primitive nodes ->
            limit primitive nodes -> compute fluxes
        limiter_config: The limiter configuration to use.
    """

    p: int
    flux_recipe: Literal[1, 2, 3]
    limiter_config: Optional[LimiterConfig] = None

    def __post_init__(self):
        if self.p < 0:
            raise ValueError("Polynomial degree p must be non-negative.")
        if self.flux_recipe not in (1, 2, 3):
            raise ValueError("Invalid flux recipe. Must be 1, 2, or 3.")
        if self.p == 0 and self.limiter_config is not None:
            raise ValueError("Limiter cannot be used with p=0 (first-order scheme).")

    @abstractmethod
    def key(self) -> str:
        """Return a unique key for the interpolation scheme."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the interpolation scheme to a dictionary."""
        pass


@dataclass(frozen=True, slots=True)
class polyInterpolationScheme(InterpolationScheme):
    """
    Configuration for arbitrary-degree polynomial interpolation schemes.

    Attributes:
        p: The polynomial degree.
        flux_recipe: The flux recipe to use:
            - 1: interpolate conservative nodes -> limit conservative nodes -> convert
            to primitive nodes -> compute fluxes
            - 2: interpolate conservative nodes -> convert to primitive nodes -> limit
            primitive nodes -> compute fluxes
            - 3: compute primitive cell averages -> interpolate primitive nodes ->
            limit primitive nodes -> compute fluxes
        limiter_config: The limiter configuration to use.
        gauss_legendre: Whether to use Gauss-Legendre quadrature.
        lazy_primitives: Whether to use the second-order evaluation for primitive cell
            averages (W_ave = W(U_ave)).
        adaptive_lazy: Write stuff here.
    """

    p: int
    flux_recipe: Literal[1, 2, 3]
    limiter_config: Optional[LimiterConfig] = None
    gauss_legendre: bool = False
    lazy_primitives: bool = False
    adaptive_lazy: bool = False

    def key(self) -> str:
        return f"poly{self.p}"

    def to_dict(self) -> dict:
        return dict(
            p=self.p,
            flux_recipe=self.flux_recipe,
            limiter_config=(
                None if self.limiter_config is None else self.limiter_config.to_dict()
            ),
            gauss_legendre=self.gauss_legendre,
            lazy_primitives=self.lazy_primitives,
        )
