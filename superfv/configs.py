from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal

import numpy as np

from .tools.device_management import ArrayLike


@dataclass(frozen=True, slots=True)
class SmoothExtremaDetectionParams:
    use_SED: bool
    clip_zero_tol: float = 1e-15


@dataclass(frozen=True, slots=True)
class MUSCL_Params:
    use_MUSCL: bool
    slope_limiter: Literal["minmod", "moncen", "PP2D", "none"]
    SED_params: SmoothExtremaDetectionParams

    def __post_init__(self):
        if not self.use_MUSCL:
            object.__setattr__(self, "slope_limiter", "none")
            object.__setattr__(self, "SED_params", SmoothExtremaDetectionParams(False, 0.0))


@dataclass(frozen=True, slots=True)
class PhysicalAdmissibilityParams:
    use_PAD: bool
    PAD_bounds: ArrayLike

    def __post_init__(self):
        if not self.use_PAD:
            object.__setattr__(self, "PAD_bounds", np.empty((0,)))


@dataclass(frozen=True, slots=True)
class ZhangShuParams:
    use_ZS: bool
    adaptive_dt: bool
    SED_params: SmoothExtremaDetectionParams
    PAD_params: PhysicalAdmissibilityParams
    adaptive_dt_tol: float = 1e-15
    theta_denom_tol: float = 1e-15
    include_corners: bool = True
    log_limiter_scalars: bool = True

    def __post_init__(self):
        if not self.use_ZS:
            object.__setattr__(self, "adaptive_dt", False)
            object.__setattr__(self, "SED_params", SmoothExtremaDetectionParams(False, 0.0))
            object.__setattr__(
                self, "PAD_params", PhysicalAdmissibilityParams(False, np.empty((0,)))
            )
            object.__setattr__(self, "log_limiter_scalars", False)
        if self.adaptive_dt and not self.PAD_params.use_PAD:
            raise ValueError(
                "Physical admissibility detection must be enabled when " "adaptive_dt is True."
            )


@dataclass(frozen=True, slots=True)
class NumericalAdmissibilityParams:
    use_NAD: bool
    rtol: float
    atol: float
    SED_params: SmoothExtremaDetectionParams
    delta: bool = False
    include_corners: bool = True

    def __post_init__(self):
        if not self.use_NAD:
            object.__setattr__(self, "SED_params", SmoothExtremaDetectionParams(False, 0.0))


@dataclass(frozen=True, slots=True)
class MOOD_Params:
    use_MOOD: bool
    NAD_params: NumericalAdmissibilityParams
    PAD_params: PhysicalAdmissibilityParams
    fallback_cascade: List[FiniteVolumeScheme]
    max_revs: int
    skip_trouble_counts: bool = False
    detect_closing_troubles: bool = True
    log_limiter_scalars: bool = True

    def __post_init__(self):
        if not self.use_MOOD:
            object.__setattr__(
                self,
                "NAD_params",
                NumericalAdmissibilityParams(
                    False, 0.0, 0.0, SmoothExtremaDetectionParams(False, 0.0)
                ),
            )
            object.__setattr__(
                self, "PAD_params", PhysicalAdmissibilityParams(False, np.empty((0,)))
            )
            object.__setattr__(self, "fallback_cascade", [])
            object.__setattr__(self, "max_revs", 0)
            object.__setattr__(self, "log_limiter_scalars", False)


@dataclass(frozen=True, slots=True)
class ShockDetectionParams:
    use_shock_detection: bool
    eta_max: float = 0.025


class LazyPrimitiveMode(Enum):
    FULL = 0
    NONE = 1
    ADAPTIVE = 2


class FluxRecipe(Enum):
    CONS_LIM_PRIM = 0
    CONS_PRIM_LIM = 1
    PRIM_PRIM_LIM = 2


class FluxQuadrature(Enum):
    TRANSVERSE = 0
    GAUSS_LEGENDRE = 1


@dataclass(frozen=True, slots=True)
class FiniteVolumeScheme:
    p: int
    flux_recipe: FluxRecipe
    flux_quadrature: FluxQuadrature
    lazy_primitive_mode: LazyPrimitiveMode
    MUSCL_params: MUSCL_Params
    zhang_shu_params: ZhangShuParams
    MOOD_params: MOOD_Params
    shock_detection_params: ShockDetectionParams

    def __post_init__(self):
        if self.p < 0:
            raise ValueError("Polynomial degree p must be non-negative.")
        if self.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE:
            if not self.shock_detection_params.use_shock_detection:
                raise ValueError(
                    "Shock detection must be enabled when lazy_primitive_mode is " "ADAPTIVE."
                )
        elif self.shock_detection_params.use_shock_detection:
            warnings.warn(
                "Shock detection is enabled but lazy_primitive_mode is not ADAPTIVE. "
                "Shock detection will be ignored.",
                UserWarning,
            )
            object.__setattr__(self, "shock_detection_params", ShockDetectionParams(False, 0.0))


@dataclass(frozen=True, slots=True)
class HydroParams:
    gamma: float
    riemann_solver: Literal["upwind", "llf", "hllc"]
    CFL: float
    dt_min: float = 1e-15
    rho_min: float = 1e-12
    P_min: float = 1e-12
    isothermal: bool = False
    iso_cs: float = 1.0


@dataclass(frozen=True, slots=True)
class MeshParams:
    nx: int
    ny: int
    nz: int
    nghost: int
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0


@dataclass(frozen=True, slots=True)
class SolverParams:
    hydro_params: HydroParams
    mesh_params: MeshParams
    finite_volume_scheme: FiniteVolumeScheme
    cupy: bool = False
    sync_timer: bool = False

    def __post_init__(self):
        if not self.cupy:
            object.__setattr__(self, "sync_timer", False)
