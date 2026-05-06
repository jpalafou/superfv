from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .boundary_conditions import CallableBC
from .field import MultivarField, UnivarField
from .tools.device_management import ArrayLike


@dataclass(frozen=True, slots=True)
class SmoothExtremaDetectionParameters:
    use_SED: bool
    clip_zero_tol: float = 1e-15


class MUSCL_SlopeLimiter(Enum):
    MINMOD = 0
    MONCEN = 1
    PP2D = 2
    NONE = 3


@dataclass(frozen=True, slots=True)
class MUSCL_Parameters:
    use_MUSCL: bool
    MUSCL_limiter: MUSCL_SlopeLimiter
    SED_params: SmoothExtremaDetectionParameters


@dataclass(frozen=True, slots=True)
class PhysicalAdmissibilityParameters:
    use_PAD: bool
    PAD_bounds: ArrayLike


@dataclass(frozen=True, slots=True)
class ZhangShuParameters:
    use_ZS: bool
    adaptive_dt: bool
    SED_params: SmoothExtremaDetectionParameters
    PAD_params: PhysicalAdmissibilityParameters
    adaptive_dt_tol: float = 1e-15
    theta_denom_tol: float = 1e-15
    include_corners: bool = True
    log_limiter_scalars: bool = True

    def __post_init__(self):
        if self.adaptive_dt and not self.PAD_params.use_PAD:
            raise ValueError(
                "Physical admissibility detection must be enabled when " "adaptive_dt is True."
            )


@dataclass(frozen=True, slots=True)
class ShockDetectionParameters:
    use_shock_detection: bool
    eta_max: float = 0.025


@dataclass(frozen=True, slots=True)
class NumericalAdmissibilityParameters:
    use_NAD: bool
    rtol: float
    atol: float
    SED_params: SmoothExtremaDetectionParameters
    delta: bool = False
    include_corners: bool = True


@dataclass(frozen=True, slots=True)
class MOOD_Parameters:
    use_MOOD: bool
    NAD_params: NumericalAdmissibilityParameters
    PAD_params: PhysicalAdmissibilityParameters
    fallback_cascade: List[FV_SchemeParameters]
    max_revs: int
    skip_trouble_counts: bool = False
    detect_closing_troubles: bool = True
    log_limiter_scalars: bool = True


class FallbackCascade(Enum):
    FULL = 0
    MUSCL = 1
    MUSCL0 = 2


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
class FV_SchemeParameters:
    p: int
    flux_recipe: FluxRecipe
    flux_quadrature: FluxQuadrature
    lazy_primitive_mode: LazyPrimitiveMode
    muscl_params: MUSCL_Parameters
    zhang_shu_params: ZhangShuParameters
    mood_params: MOOD_Parameters
    shock_detection_params: ShockDetectionParameters

    def __post_init__(self):
        if self.p < 0:
            raise ValueError("Polynomial degree p must be non-negative.")
        if self.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE:
            if not self.shock_detection_params.use_shock_detection:
                raise ValueError(
                    "Shock detection must be enabled when lazy_primitive_mode is ADAPTIVE."
                )
        if (
            sum(
                [
                    self.muscl_params.use_MUSCL,
                    self.zhang_shu_params.use_ZS,
                    self.mood_params.use_MOOD,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of MUSCL, Zhang-Shu, or MOOD limiting can be enabled at a time."
            )
        if self.muscl_params.use_MUSCL and self.p != 1:
            warnings.warn(
                f"Changing p from {self.p} to 1 since MUSCL limiting is enabled.", UserWarning
            )
            object.__setattr__(self, "p", 1)
        if self.p <= 1 and self.lazy_primitive_mode != LazyPrimitiveMode.FULL:
            warnings.warn(
                "Changing lazy_primitive_mode to FULL since FV scheme is second-order or lower.",
                UserWarning,
            )
            object.__setattr__(self, "lazy_primitive_mode", LazyPrimitiveMode.FULL)
        if (
            self.shock_detection_params.use_shock_detection
            and self.lazy_primitive_mode != LazyPrimitiveMode.ADAPTIVE
        ):
            warnings.warn(
                "Disabling shock detection since lazy_primitive_mode is not ADAPTIVE.",
                UserWarning,
            )
            object.__setattr__(self, "shock_detection_params", ShockDetectionParameters(False, 0.0))


class RiemannSolver(Enum):
    UPWIND = 0
    LLF = 1
    HLLC = 2


@dataclass(frozen=True, slots=True)
class HydroParameters:
    gamma: float
    riemann_solver: RiemannSolver
    CFL: float
    dt_min: float = 1e-15
    rho_min: float = 1e-12
    P_min: float = 1e-12
    isothermal: bool = False
    iso_cs: float = 1.0


@dataclass(frozen=True, slots=True)
class MeshParameters:
    nx: int
    ny: int
    nz: int
    nghost: int
    xlims: Tuple[float, float]
    ylims: Tuple[float, float]
    zlims: Tuple[float, float]


@dataclass(frozen=True, slots=True)
class InitialConditionParameters:
    ic: MultivarField
    passive_ics: List[UnivarField]


class BoundaryCondition(Enum):
    PERIODIC = 0
    DIRICHLET = 1
    FREE = 2
    SYMMETRIC = 3
    REFLECTIVE = 4
    ZEROS = 5
    ONES = 6
    PATCH = 7
    NONE = 8


@dataclass(frozen=True, slots=True)
class BoundaryConditionParameters:
    bcx: Tuple[BoundaryCondition, BoundaryCondition]
    bcy: Tuple[BoundaryCondition, BoundaryCondition]
    bcz: Tuple[BoundaryCondition, BoundaryCondition]
    bcx_callable_lower: Optional[CallableBC] = None
    bcx_callable_upper: Optional[CallableBC] = None
    bcy_callable_lower: Optional[CallableBC] = None
    bcy_callable_upper: Optional[CallableBC] = None
    bcz_callable_lower: Optional[CallableBC] = None
    bcz_callable_upper: Optional[CallableBC] = None


@dataclass(frozen=True, slots=True)
class SolverParams:
    hydro: HydroParameters
    ic: InitialConditionParameters
    mesh: MeshParameters
    bc: BoundaryConditionParameters
    fv_scheme: FV_SchemeParameters
    cupy: bool = False
    sync_timer: bool = True
