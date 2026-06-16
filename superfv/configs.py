from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from .boundary_conditions import BC, PatchBC
from .field import MultivarField, SourceTerm, UnivarField
from .riemann_solvers import RiemannSolver
from .slope_limiting.muscl import MUSCL_SlopeLimiter
from .tools.device_management import CUPY_AVAILABLE
from .tools.variable_index_map import VariableIndexMap


@dataclass(frozen=True, slots=True)
class SmoothExtremaDetectionParameters:
    use_SED: bool
    clip_zero_tol: float = 1e-15


@dataclass(frozen=True, slots=True)
class MUSCL_Parameters:
    use_MUSCL: bool
    MUSCL_limiter: MUSCL_SlopeLimiter
    SED_params: SmoothExtremaDetectionParameters


@dataclass(frozen=True, slots=True)
class PhysicalAdmissibilityParameters:
    use_PAD: bool
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]]

    def __post_init__(self):
        if self.use_PAD and not any(
            lb is not None or ub is not None for lb, ub in self.bounds.values()
        ):
            raise ValueError(
                "At least one variable must have a non-None bound when use_PAD is True."
            )


@dataclass(frozen=True, slots=True)
class ZhangShuParameters:
    use_ZS: bool
    adaptive_dt: bool
    SED_params: SmoothExtremaDetectionParameters
    PAD_params: PhysicalAdmissibilityParameters
    omit_vars: List[str]
    adaptive_dt_tol: float = 1e-15
    theta_denom_tol: float = 1e-15
    include_corners: bool = True

    def __post_init__(self):
        if not self.use_ZS:
            if self.adaptive_dt:
                raise ValueError(
                    "Adaptive time-stepping cannot be used if the Zhang-Shu limiter is not used."
                )
            if self.SED_params.use_SED:
                raise ValueError("SED cannot be used if the Zhang-Shu limiter is not used.")
            if self.PAD_params.use_PAD:
                raise ValueError("PAD cannot be used if the Zhang-Shu limiter is not used.")

        if self.adaptive_dt and not self.PAD_params.use_PAD:
            raise ValueError(
                "Physical admissibility detection must be enabled when adaptive_dt is True."
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
    omit_vars: List[str]
    delta: bool = False
    include_corners: bool = True

    def __post_init__(self):
        if not self.use_NAD and self.SED_params.use_SED:
            raise ValueError("Smooth extrema detection cannot be used if NAD is not used.")


@dataclass(frozen=True, slots=True)
class MOOD_Parameters:
    use_MOOD: bool
    NAD_params: NumericalAdmissibilityParameters
    PAD_params: PhysicalAdmissibilityParameters
    fallback_cascade: List[FV_SchemeParameters]
    max_revs: int
    blend_troubles: bool
    skip_trouble_counts: bool = False
    detect_closing_troubles: bool = True

    def __post_init__(self):
        if self.use_MOOD and self.blend_troubles and len(self.fallback_cascade) != 1:
            raise ValueError(
                "fallback_cascade must have exactly one scheme when blend_troubles is True."
            )

        if self.use_MOOD and self.max_revs < len(self.fallback_cascade):
            raise ValueError("max_revs must be at least the length of fallback_cascade.")

        if not self.use_MOOD:
            if self.NAD_params.use_NAD:
                raise ValueError("NAD cannot be used if MOOD is not used.")
            if self.PAD_params.use_PAD:
                raise ValueError("PAD cannot be used if MOOD is not used.")


class FallbackCascade(Enum):
    FULL = 0
    MUSCL = 1
    MUSCL0 = 2
    FIRST_ORDER = 3


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
    NONE = 2


@dataclass(frozen=True, slots=True)
class FV_SchemeParameters:
    name: str
    p: int
    flux_recipe: FluxRecipe
    flux_quadrature: FluxQuadrature
    lazy_primitive_mode: LazyPrimitiveMode
    muscl_params: MUSCL_Parameters
    zhang_shu_params: ZhangShuParameters
    mood_params: MOOD_Parameters
    shock_detection_params: ShockDetectionParameters

    def __post_init__(self):
        # Non-negative polynomial degree
        if self.p < 0:
            raise ValueError("Polynomial degree p must be non-negative.")

        # Couple shock detection with LazyPrimitiveMode.ADAPTIVE
        if self.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE:
            if not self.shock_detection_params.use_shock_detection:
                raise ValueError(
                    "Shock detection must be enabled when lazy_primitive_mode is ADAPTIVE."
                )
        elif self.shock_detection_params.use_shock_detection:
            warnings.warn(
                "Disabling shock detection since lazy_primitive_mode is not ADAPTIVE.",
                UserWarning,
            )
            object.__setattr__(self, "shock_detection_params", ShockDetectionParameters(False, 0.0))

        # Unique limiter choice
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

        # MUSCL p != 1 warning
        if self.muscl_params.use_MUSCL and self.p != 1:
            warnings.warn(
                f"Changing p from {self.p} to 1 since MUSCL limiting is enabled.", UserWarning
            )
            object.__setattr__(self, "p", 1)

        # p < 2 non lazy primitive warning
        if self.p < 2 and self.lazy_primitive_mode != LazyPrimitiveMode.FULL:
            warnings.warn(
                "Changing lazy_primitive_mode to FULL since FV scheme is second-order or lower.",
                UserWarning,
            )
            object.__setattr__(self, "lazy_primitive_mode", LazyPrimitiveMode.FULL)


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
    active_dims: Tuple[Literal["x", "y", "z"], ...]
    ndim: int

    def __post_init__(self):
        if self.ndim != len(self.active_dims):
            raise ValueError("ndim must be equal to the length of active_dims")


@dataclass
class InitialConditionParameters:
    ic: MultivarField
    passive_ics: Dict[str, UnivarField]
    sampling_p: int

    @property
    def npassives(self) -> int:
        return len(self.passive_ics)


@dataclass
class BoundaryConditionParameters:
    bcx: Tuple[BC, BC]
    bcy: Tuple[BC, BC]
    bcz: Tuple[BC, BC]
    bcx_callable_lower: Optional[Union[MultivarField, PatchBC]] = None
    bcx_callable_upper: Optional[Union[MultivarField, PatchBC]] = None
    bcy_callable_lower: Optional[Union[MultivarField, PatchBC]] = None
    bcy_callable_upper: Optional[Union[MultivarField, PatchBC]] = None
    bcz_callable_lower: Optional[Union[MultivarField, PatchBC]] = None
    bcz_callable_upper: Optional[Union[MultivarField, PatchBC]] = None
    sampling_p: Optional[int] = None

    def __post_init__(self):
        if bool(self.bcx[0] == BC.PERIODIC) != bool(self.bcx[1] == BC.PERIODIC):
            raise ValueError("Both lower and upper BCs in x must be PERIODIC or neither.")
        if bool(self.bcy[0] == BC.PERIODIC) != bool(self.bcy[1] == BC.PERIODIC):
            raise ValueError("Both lower and upper BCs in y must be PERIODIC or neither.")
        if bool(self.bcz[0] == BC.PERIODIC) != bool(self.bcz[1] == BC.PERIODIC):
            raise ValueError("Both lower and upper BCs in z must be PERIODIC or neither.")

        if self.bcx[0] == BC.DIRICHLET or self.bcx[0] == BC.PATCH:
            if self.bcx_callable_lower is None:
                raise ValueError(
                    "bcx_callable_lower must be provided for DIRICHLET or PATCH BC in x."
                )
        if self.bcx[1] == BC.DIRICHLET or self.bcx[1] == BC.PATCH:
            if self.bcx_callable_upper is None:
                raise ValueError(
                    "bcx_callable_upper must be provided for DIRICHLET or PATCH BC in x."
                )
        if self.bcy[0] == BC.DIRICHLET or self.bcy[0] == BC.PATCH:
            if self.bcy_callable_lower is None:
                raise ValueError(
                    "bcy_callable_lower must be provided for DIRICHLET or PATCH BC in y."
                )
        if self.bcy[1] == BC.DIRICHLET or self.bcy[1] == BC.PATCH:
            if self.bcy_callable_upper is None:
                raise ValueError(
                    "bcy_callable_upper must be provided for DIRICHLET or PATCH BC in y."
                )
        if self.bcz[0] == BC.DIRICHLET or self.bcz[0] == BC.PATCH:
            if self.bcz_callable_lower is None:
                raise ValueError(
                    "bcz_callable_lower must be provided for DIRICHLET or PATCH BC in z."
                )
        if self.bcz[1] == BC.DIRICHLET or self.bcz[1] == BC.PATCH:
            if self.bcz_callable_upper is None:
                raise ValueError(
                    "bcz_callable_upper must be provided for DIRICHLET or PATCH BC in z."
                )


@dataclass(frozen=True, slots=True)
class SolverParameters:
    hydro: HydroParameters
    ic: InitialConditionParameters
    source: SourceTerm
    mesh: MeshParameters
    bc: BoundaryConditionParameters
    fv_scheme: FV_SchemeParameters
    variable_index_map: VariableIndexMap
    cupy: bool = False
    cpp_backend: bool = False
    profile: bool = False
    output_path: Optional[Path] = None
    discard_after_writing: bool = True
    output_n_digits: int = 6

    def __post_init__(self):
        if self.cupy and not CUPY_AVAILABLE:
            raise ValueError("CuPy is not available but cupy is set to True.")

        # PAD bound dicts cannot contain variables not in the variable index map
        if (
            self.fv_scheme.zhang_shu_params.use_ZS
            and self.fv_scheme.zhang_shu_params.PAD_params.use_PAD
            and any(
                v not in self.variable_index_map.var_idx_map
                for v in self.fv_scheme.zhang_shu_params.PAD_params.bounds.keys()
            )
        ):
            raise ValueError(
                "All variables in zhang_shu_params.PAD_params.bounds must be in the variable index map."
            )
        if (
            self.fv_scheme.mood_params.use_MOOD
            and self.fv_scheme.mood_params.PAD_params.use_PAD
            and any(
                v not in self.variable_index_map.var_idx_map
                for v in self.fv_scheme.mood_params.PAD_params.bounds.keys()
            )
        ):
            raise ValueError(
                "All variables in mood_params.PAD_params.bounds must be in the variable index map."
            )

        # Omit vars lists cannot contain variables not in the variable index map
        if (
            self.fv_scheme.zhang_shu_params.use_ZS
            and self.fv_scheme.zhang_shu_params.omit_vars
            and any(
                v not in self.variable_index_map.var_idx_map
                for v in self.fv_scheme.zhang_shu_params.omit_vars
            )
        ):
            raise ValueError(
                "All variables in zhang_shu_params.omit_vars must be in the variable index map."
            )
        if (
            self.fv_scheme.mood_params.use_MOOD
            and self.fv_scheme.mood_params.NAD_params.omit_vars
            and any(
                v not in self.variable_index_map.var_idx_map
                for v in self.fv_scheme.mood_params.NAD_params.omit_vars
            )
        ):
            raise ValueError(
                "All variables in mood_params.NAD_params.omit_vars must be in the variable index map."
            )

        # If limiting conservatives, then all PAD bounds must be in primitives
        if (
            self.fv_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM
            and self.fv_scheme.zhang_shu_params.use_ZS
            and self.fv_scheme.zhang_shu_params.PAD_params.use_PAD
            and any(
                v not in self.variable_index_map.group_var_map["primitives"]
                for v in self.fv_scheme.zhang_shu_params.PAD_params.bounds.keys()
            )
        ):
            raise ValueError(
                "All variables with PAD bounds must be in primitives when using CONS_LIM_PRIM flux recipe."
            )

        # PP2D MUSCL slopes can only be used in 2D
        if (
            self.fv_scheme.muscl_params.use_MUSCL
            and self.fv_scheme.muscl_params.MUSCL_limiter == MUSCL_SlopeLimiter.PP2D
            and self.mesh.ndim != 2
        ):
            raise ValueError("PP2D MUSCL slopes can only be used in 2D.")

        # No flux quadarture in 1D
        if self.mesh.ndim == 1 and self.fv_scheme.flux_quadrature != FluxQuadrature.NONE:
            raise ValueError("Flux quadrature must be NONE for 1D simulations.")

        if self.cupy and self.cpp_backend:
            raise ValueError("CuPy and C++ backend cannot both be enabled at the same time.")


def generate_fv_scheme_parameters(
    name: str,
    p: int,
    flux_recipe: Optional[FluxRecipe] = None,
    flux_quadrature: Optional[FluxQuadrature] = None,
    lazy_primitive_mode: Optional[LazyPrimitiveMode] = None,
    muscl_params: Optional[MUSCL_Parameters] = None,
    zhang_shu_params: Optional[ZhangShuParameters] = None,
    mood_params: Optional[MOOD_Parameters] = None,
    shock_detection_params: Optional[ShockDetectionParameters] = None,
) -> FV_SchemeParameters:
    default_flux_recipe = FluxRecipe.CONS_PRIM_LIM
    default_flux_quadrature = FluxQuadrature.NONE
    default_lazy_primitive_mode = LazyPrimitiveMode.FULL
    _default_SED_params = SmoothExtremaDetectionParameters(False)
    _default_PAD_params = PhysicalAdmissibilityParameters(False, {})
    default_muscl_params = MUSCL_Parameters(
        use_MUSCL=False, MUSCL_limiter=MUSCL_SlopeLimiter.NONE, SED_params=_default_SED_params
    )
    default_zhang_shu_params = ZhangShuParameters(
        use_ZS=False,
        adaptive_dt=False,
        SED_params=_default_SED_params,
        PAD_params=_default_PAD_params,
        omit_vars=[],
    )
    default_mood_params = MOOD_Parameters(
        use_MOOD=False,
        NAD_params=NumericalAdmissibilityParameters(False, 0.0, 0.0, _default_SED_params, []),
        PAD_params=_default_PAD_params,
        fallback_cascade=[],
        max_revs=0,
        blend_troubles=False,
    )
    default_shock_detection_params = ShockDetectionParameters(False)

    return FV_SchemeParameters(
        name=name,
        p=p,
        flux_recipe=flux_recipe if flux_recipe is not None else default_flux_recipe,
        flux_quadrature=flux_quadrature if flux_quadrature is not None else default_flux_quadrature,
        lazy_primitive_mode=(
            lazy_primitive_mode if lazy_primitive_mode is not None else default_lazy_primitive_mode
        ),
        muscl_params=muscl_params if muscl_params is not None else default_muscl_params,
        zhang_shu_params=(
            zhang_shu_params if zhang_shu_params is not None else default_zhang_shu_params
        ),
        mood_params=mood_params if mood_params is not None else default_mood_params,
        shock_detection_params=(
            shock_detection_params
            if shock_detection_params is not None
            else default_shock_detection_params
        ),
    )
