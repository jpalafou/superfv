from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from .tools.device_management import ArrayLike


@dataclass(frozen=True, slots=True)
class LimiterConfig:
    """
    Base class for slope or flux limiter configurations.

    Attributes:
        shock_detection: Whether to enable shock detection.
        smooth_extrema_detection: Whether to enable smooth extrema detection.
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions if smooth
            extrema detection is enabled. Uniform regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        PAD_atol: Absolute tolerance for physical admissibility detection if
            physical_admissibility_detection is True.
        uniformity_tol: Tolerance for uniformity check when check_uniformity is True.
    """

    shock_detection: bool
    smooth_extrema_detection: bool
    check_uniformity: bool
    physical_admissibility_detection: bool
    eta_max: float = 0.0
    PAD_bounds: Optional[ArrayLike] = None
    PAD_atol: float = 0.0
    uniformity_tol: float = 1e-3

    def __post_init__(self):
        if self.shock_detection and self.eta_max is None:
            raise ValueError("eta_max must be provided when shock_detection is True.")
        if self.physical_admissibility_detection:
            if self.PAD_bounds is None:
                raise ValueError(
                    "PAD_bounds must be provided when physical_admissibility_detection"
                    " is True."
                )
            if self.PAD_atol is None:
                raise ValueError(
                    "PAD_atol must be provided when physical_admissibility_detection"
                    " is True."
                )

    def key(self) -> str:
        """Return a unique key for the limiter configuration."""
        return "generic_limiter"

    def to_dict(self) -> dict:
        """Convert the limiter configuration to a dictionary."""
        return dict(
            shock_detection=self.shock_detection,
            smooth_extrema_detection=self.smooth_extrema_detection,
            check_uniformity=self.check_uniformity,
            physical_admissibility_detection=self.physical_admissibility_detection,
            eta_max=self.eta_max,
            PAD_bounds=None if self.PAD_bounds is None else self.PAD_bounds.tolist(),
            PAD_atol=self.PAD_atol,
            uniformity_tol=self.uniformity_tol,
        )


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
    limiter_config: LimiterConfig = LimiterConfig(
        shock_detection=False,
        smooth_extrema_detection=False,
        check_uniformity=False,
        physical_admissibility_detection=False,
    )

    def __post_init__(self):
        if self.p < 0:
            raise ValueError("Polynomial degree p must be non-negative.")
        if self.flux_recipe not in (1, 2, 3):
            raise ValueError("Invalid flux recipe. Must be 1, 2, or 3.")

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
