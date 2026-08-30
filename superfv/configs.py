from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Tuple, Union, get_args

from .boundary_conditions import BC, PatchBC
from .field import MultivarField, SourceTerm, UnivarField
from .riemann_solvers import RiemannSolver
from .slope_limiting.muscl import MUSCL_SlopeLimiter
from .tools.device_management import CUPY_AVAILABLE
from .tools.variable_index_map import VariableIndexMap

LazyPrimitiveMode = Literal["full", "none", "adaptive"]
FluxRecipe = Literal["cons_lim_prim", "cons_prim_lim", "prim_prim_lim"]
FluxQuadrature = Literal["transverse", "gauss_legendre", "none"]

MAX_POLYNOMIAL_DEGREE = 7


def validate_literal_membership(value: Any, literal: Any, field_name: str) -> None:
    choices = get_args(literal)
    if value not in choices:
        raise ValueError(
            f"{field_name} must be one of {', '.join(repr(choice) for choice in choices)}; "
            f"got {value!r}."
        )


@dataclass(frozen=True, slots=True)
class SmoothExtremaDetectionParameters:
    use_SED: bool
    clip_zero_tol: float = 1e-15


@dataclass(frozen=True, slots=True)
class MUSCL_Parameters:
    use_MUSCL: bool
    MUSCL_limiter: MUSCL_SlopeLimiter
    SED_params: SmoothExtremaDetectionParameters

    def __post_init__(self):
        validate_literal_membership(self.MUSCL_limiter, MUSCL_SlopeLimiter, "MUSCL_limiter")
        if not self.use_MUSCL and self.SED_params.use_SED:
            raise ValueError("SED cannot be used if MUSCL limiting is not used.")


@dataclass(frozen=True, slots=True)
class PhysicalAdmissibilityDetectionParameters:
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
    PAD_params: PhysicalAdmissibilityDetectionParameters
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
    PAD_params: PhysicalAdmissibilityDetectionParameters
    eta_max: float = 0.025

    def __post_init__(self):
        if not self.use_shock_detection and self.PAD_params.use_PAD:
            raise ValueError(
                "Physical admissibility detection cannot be used if shock detection is not used."
            )


@dataclass(frozen=True, slots=True)
class NumericalAdmissibilityDetectionParameters:
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
    NAD_params: NumericalAdmissibilityDetectionParameters
    PAD_params: PhysicalAdmissibilityDetectionParameters
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


@dataclass(frozen=True, slots=True)
class FV_SchemeParameters:
    name: str
    p: int
    flux_recipe: FluxRecipe
    flux_quadrature: FluxQuadrature
    lazy_primitive_mode: LazyPrimitiveMode
    positivity_guard: bool
    riemann_solver: RiemannSolver
    muscl_params: MUSCL_Parameters
    zhang_shu_params: ZhangShuParameters
    mood_params: MOOD_Parameters
    shock_detection_params: ShockDetectionParameters

    def __post_init__(self):
        validate_literal_membership(self.flux_recipe, FluxRecipe, "flux_recipe")
        validate_literal_membership(self.flux_quadrature, FluxQuadrature, "flux_quadrature")
        validate_literal_membership(
            self.lazy_primitive_mode, LazyPrimitiveMode, "lazy_primitive_mode"
        )
        validate_literal_membership(self.riemann_solver, RiemannSolver, "riemann_solver")

        if self.p < 0 or self.p > MAX_POLYNOMIAL_DEGREE:
            raise ValueError(f"Polynomial degree p must be between 0 and {MAX_POLYNOMIAL_DEGREE}.")

        if self.lazy_primitive_mode == "adaptive":
            if not self.shock_detection_params.use_shock_detection:
                raise ValueError(
                    'Shock detection must be enabled when lazy_primitive_mode is "adaptive".'
                )
        elif self.shock_detection_params.use_shock_detection:
            raise ValueError(
                'Shock detection can only be enabled when lazy_primitive_mode is "adaptive".'
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
            raise ValueError("MUSCL limiting requires polynomial degree p == 1.")

        if self.zhang_shu_params.use_ZS and self.p == 0:
            raise ValueError("Zhang-Shu limiting requires polynomial degree p > 0.")

        if self.mood_params.use_MOOD and self.p == 0:
            raise ValueError("MOOD limiting requires polynomial degree p > 0.")

        if self.p < 2 and self.lazy_primitive_mode != "full":
            raise ValueError('lazy_primitive_mode must be "full" when polynomial degree p < 2.')


@dataclass(frozen=True, slots=True)
class HydroParameters:
    gamma: float
    CFL: float
    dissipation: bool = False
    nu: float = 0.0
    Chi: float = 0.0
    nu_dye: float = 0.0
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
        for i, dim in enumerate(self.active_dims):
            validate_literal_membership(dim, Literal["x", "y", "z"], f"active_dims[{i}]")
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
        for name, bcs in [("bcx", self.bcx), ("bcy", self.bcy), ("bcz", self.bcz)]:
            for i, bc in enumerate(bcs):
                validate_literal_membership(bc, BC, f"{name}[{i}]")

        if bool(self.bcx[0] == "periodic") != bool(self.bcx[1] == "periodic"):
            raise ValueError('Both lower and upper BCs in x must be "periodic" or neither.')
        if bool(self.bcy[0] == "periodic") != bool(self.bcy[1] == "periodic"):
            raise ValueError('Both lower and upper BCs in y must be "periodic" or neither.')
        if bool(self.bcz[0] == "periodic") != bool(self.bcz[1] == "periodic"):
            raise ValueError('Both lower and upper BCs in z must be "periodic" or neither.')

        if self.bcx[0] == "dirichlet" or self.bcx[0] == "patch":
            if self.bcx_callable_lower is None:
                raise ValueError(
                    'bcx_callable_lower must be provided for "dirichlet" or "patch" BC in x.'
                )
        if self.bcx[1] == "dirichlet" or self.bcx[1] == "patch":
            if self.bcx_callable_upper is None:
                raise ValueError(
                    'bcx_callable_upper must be provided for "dirichlet" or "patch" BC in x.'
                )
        if self.bcy[0] == "dirichlet" or self.bcy[0] == "patch":
            if self.bcy_callable_lower is None:
                raise ValueError(
                    'bcy_callable_lower must be provided for "dirichlet" or "patch" BC in y.'
                )
        if self.bcy[1] == "dirichlet" or self.bcy[1] == "patch":
            if self.bcy_callable_upper is None:
                raise ValueError(
                    'bcy_callable_upper must be provided for "dirichlet" or "patch" BC in y.'
                )
        if self.bcz[0] == "dirichlet" or self.bcz[0] == "patch":
            if self.bcz_callable_lower is None:
                raise ValueError(
                    'bcz_callable_lower must be provided for "dirichlet" or "patch" BC in z.'
                )
        if self.bcz[1] == "dirichlet" or self.bcz[1] == "patch":
            if self.bcz_callable_upper is None:
                raise ValueError(
                    'bcz_callable_upper must be provided for "dirichlet" or "patch" BC in z.'
                )


@dataclass(frozen=True, slots=True)
class SolverParameters:
    hydro: HydroParameters
    ic: InitialConditionParameters
    mesh: MeshParameters
    bc: BoundaryConditionParameters
    fv_scheme: FV_SchemeParameters
    variable_index_map: VariableIndexMap
    source: Optional[SourceTerm] = None
    cupy: bool = False
    profile: bool = False
    output_path: Optional[Path] = None
    discard_after_writing: bool = True
    output_n_digits: int = 6

    def __post_init__(self):
        if self.cupy and not CUPY_AVAILABLE:
            raise ValueError("CuPy is not available but cupy is set to True.")
        if self.hydro.nu_dye > 0.0 and "dye" not in self.variable_index_map.group_var_map.get(
            "passives", []
        ):
            raise ValueError('nu_dye > 0 requires a passive variable named "dye".')

        # PAD bound dicts must contain variables only in the "primitive" group
        def _check_PAD_bounds_in_primitives(PAD_params):
            valid_vars = set([])
            valid_vars.update(self.variable_index_map.group_var_map.get("primitives", []))
            valid_vars.update(self.variable_index_map.group_var_map.get("passives", []))
            valid_vars.update(["vx", "vy", "vz"])
            if PAD_params.use_PAD:
                for var in PAD_params.bounds.keys():
                    if var not in self.variable_index_map.var_idx_map:
                        raise ValueError(
                            f"PAD_bounds variable {var} is not in the variable index map."
                        )
                    if var not in valid_vars:
                        raise ValueError(f"PAD_bounds variable {var} is not primitive.")

        if (
            self.fv_scheme.zhang_shu_params.use_ZS
            and self.fv_scheme.zhang_shu_params.PAD_params.use_PAD
        ):
            _check_PAD_bounds_in_primitives(self.fv_scheme.zhang_shu_params.PAD_params)
        if self.fv_scheme.mood_params.use_MOOD and self.fv_scheme.mood_params.PAD_params.use_PAD:
            _check_PAD_bounds_in_primitives(self.fv_scheme.mood_params.PAD_params)
        if (
            self.fv_scheme.shock_detection_params.use_shock_detection
            and self.fv_scheme.shock_detection_params.PAD_params.use_PAD
        ):
            _check_PAD_bounds_in_primitives(self.fv_scheme.shock_detection_params.PAD_params)

        # Omit vars lists must contain variables in the "primitive" or "conservative" groups
        def _check_omit_vars_in_groups(omit_vars):
            if self.fv_scheme.flux_recipe == "cons_lim_prim":
                valid_vars = set([])
                valid_vars.update(self.variable_index_map.group_var_map.get("conservatives", []))
                valid_vars.update(self.variable_index_map.group_var_map.get("passives", []))
                valid_vars.update(["mx", "my", "mz"])
                valid_group_name = "conservatives"
            else:
                valid_vars = set([])
                valid_vars.update(self.variable_index_map.group_var_map.get("primitives", []))
                valid_vars.update(self.variable_index_map.group_var_map.get("passives", []))
                valid_vars.update(["vx", "vy", "vz"])
                valid_group_name = "primitives"
            for var in omit_vars:
                if var not in self.variable_index_map.var_idx_map:
                    raise ValueError(
                        f"`omit_vars` variable {var} is not in the variable index map."
                    )
                if var not in valid_vars:
                    raise ValueError(
                        f"`omit_vars` variable {var} is not in the {valid_group_name} group."
                    )

        if self.fv_scheme.zhang_shu_params.use_ZS:
            _check_omit_vars_in_groups(self.fv_scheme.zhang_shu_params.omit_vars)
        if self.fv_scheme.mood_params.use_MOOD and self.fv_scheme.mood_params.NAD_params.use_NAD:
            _check_omit_vars_in_groups(self.fv_scheme.mood_params.NAD_params.omit_vars)

        # PP2D MUSCL slopes can only be used in 2D
        if (
            self.fv_scheme.muscl_params.use_MUSCL
            and self.fv_scheme.muscl_params.MUSCL_limiter == "pp2d"
            and self.mesh.ndim != 2
        ):
            raise ValueError("PP2D MUSCL slopes can only be used in 2D.")

        # "none" flux quadrature in 1D and only 1D
        if self.mesh.ndim == 1 and self.fv_scheme.flux_quadrature != "none":
            raise ValueError('Flux quadrature must be "none" for 1D simulations.')
        elif self.mesh.ndim != 1 and self.fv_scheme.flux_quadrature == "none":
            raise ValueError('Flux quadrature cannot be "none" for 2D or 3D simulations.')

        if self.fv_scheme.riemann_solver == "hllc_teyssier":
            if self.mesh.ndim != 1:
                raise ValueError("The HLLC Teyssier Riemann solver only supports 1D simulations.")
            if "passives" in self.variable_index_map.group_var_map:
                raise ValueError(
                    "The HLLC Teyssier Riemann solver does not support passive scalars."
                )
            if self.cupy:
                raise ValueError("The HLLC Teyssier Riemann solver does not support CuPy.")

        if self.hydro.dissipation and self.fv_scheme.flux_quadrature == "gauss_legendre":
            raise ValueError(
                "Gauss-Legendre flux quadrature cannot be used with dissipative fluxes."
            )


def dummy_function(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "This function was replaced because the original function could not be pickled."
    )


def _pickle_or_dummy(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        pickle.dumps(obj)
    except (AttributeError, pickle.PicklingError, TypeError):
        return dummy_function
    return obj


def pickle_SolverParameters(params: SolverParameters, file: IO[bytes]) -> None:
    ic = replace(
        params.ic,
        ic=_pickle_or_dummy(params.ic.ic),
        passive_ics={
            name: _pickle_or_dummy(passive_ic) for name, passive_ic in params.ic.passive_ics.items()
        },
    )
    bc = replace(
        params.bc,
        bcx_callable_lower=_pickle_or_dummy(params.bc.bcx_callable_lower),
        bcx_callable_upper=_pickle_or_dummy(params.bc.bcx_callable_upper),
        bcy_callable_lower=_pickle_or_dummy(params.bc.bcy_callable_lower),
        bcy_callable_upper=_pickle_or_dummy(params.bc.bcy_callable_upper),
        bcz_callable_lower=_pickle_or_dummy(params.bc.bcz_callable_lower),
        bcz_callable_upper=_pickle_or_dummy(params.bc.bcz_callable_upper),
    )
    pickle.dump(
        replace(params, ic=ic, source=_pickle_or_dummy(params.source), bc=bc),
        file,
    )
