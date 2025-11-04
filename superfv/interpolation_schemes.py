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
        lazy_primitives:
            - "none": Do not use second-order evaluation for primitive cell averages.
            - "full": Always use second-order evaluation for primitive cell averages.
            - "adaptive": Based on a shock-detection criterion, adaptively reduce the
                order of conservative cell centers, primitive cell centers, and
                primitive cell averages to second order.
        eta_max: Threshold for shock detection when `lazy_primitives` is "adaptive".
    """

    gauss_legendre: bool = False
    lazy_primitives: Literal["none", "full", "adaptive"] = "none"
    eta_max: Optional[float] = None

    def __post_init__(self):
        InterpolationScheme.__post_init__(self)
        if self.lazy_primitives not in ("none", "full", "adaptive"):
            raise ValueError(
                'Invalid lazy_primitives option. Must be "none", "full", or "adaptive".'
            )
        if self.lazy_primitives == "adaptive" and self.eta_max is None:
            raise ValueError(
                "eta_max must be provided when lazy_primitives is set to 'adaptive'."
            )

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
