import pickle
import shutil
import time
import warnings
from dataclasses import asdict
from enum import Enum
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .axes import DIM_TO_AXIS, XYZ_TUPLE
from .boundary_conditions import BC, PatchBC, apply_bc
from .configs import (
    BoundaryConditionParameters,
    FallbackCascade,
    FluxQuadrature,
    FluxRecipe,
    FV_SchemeParameters,
    HydroParameters,
    InitialConditionParameters,
    LazyPrimitiveMode,
    MeshParameters,
    MOOD_Parameters,
    MUSCL_Parameters,
    MUSCL_SlopeLimiter,
    NumericalAdmissibilityParameters,
    PhysicalAdmissibilityParameters,
    RiemannSolver,
    ShockDetectionParameters,
    SmoothExtremaDetectionParameters,
    SolverParams,
    ZhangShuParameters,
)
from .field import MultivarField, UnivarField
from .finite_volume_driver import (
    apply_zhang_shu_limiter,
    integrate_cell_averages,
    integrate_gauss_legendre_face_nodes,
    integrate_transverse_nodes,
    interpolate_cell_centers,
    interpolate_face_nodes,
)
from .hydro import cons_to_prim, prim_to_cons, sound_speed
from .initial_conditions import square
from .mesh import UniformFVMesh
from .muscl_solver import update_fluxes_with_muscl_scheme
from .riemann_solvers import HLLC_RiemannSolver, LLF_RiemannSolver, UpwindRiemannSolver
from .slope_limiting.MOOD import mood_loop
from .slope_limiting.shock_detection import detect_shocks
from .stencils import conservative_interpolation
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager, xp
from .tools.slicing import crop, merge_slices
from .tools.snapshot import Snapshot, SnapshotData, SnapshotHistory
from .tools.step_history import MultiTimer, StepHistory, StepSummary, SubstepSummary
from .tools.variable_index_map import VariableIndexMap
from .tools.yaml_helper import yaml_dump

if CUPY_AVAILABLE:
    from .hydro import (
        make_cons_to_prim_elementwise_kernel,
        make_prim_to_cons_elementwise_kernel,
        sound_speed_cp,
    )


class TimeIntegrator(Enum):
    MUSCL_HANCOCK = 0
    FORWARD_EULER = 1
    SSPRK2 = 2
    SSPRK3 = 3
    RK4 = 4


class SnapshotMode(Enum):
    TARGET = 0
    EVERY = 1
    NONE = 2


class LogArrayAction(Enum):
    SUBSTEP_ADD = 0
    SUBSTEP_AVERAGE = 1
    RESET = 2


class HydroSolver:
    """
    SuperFV Hydrodynamics Solver for the compressible Euler equations in 1D, 2D, or 3D.

    Initialize the solver with the desired parameters:

        from superfv import HydroSolver, ic
        sim = HydroSolver(ic=ic.square, ...)

    Then call the `take_n_steps` or `run` methods to advance the simulation in time. For
    example, to take 10 steps:

        sim.take_n_steps(n=10, ...)

    or to run until t=1.0:

        sim.run(t=1.0, ...)

    Refer to the `step_history` to access the simulation history and the `snapshot_history`
    to access the snapshots. For example, to get the total energy across all steps:

        sim.step_history.get_history("E_total")

    or to get the density field from the 5th snapshot:

        idx = sim.params.variable_index_map
        sim.snapshot_history[5].u[idx("rho")]

    or to get the pressure field at time t=1.0:

        sim.snapshot_history(1.0).w[idx("P")]

    """

    # Initialization methods

    def __init__(
        self,
        # Hydro params
        gamma: float = 1.4,
        riemann_solver: RiemannSolver = RiemannSolver.HLLC,
        CFL: float = 0.8,
        dt_min: float = 1e-15,
        rho_min: float = 1e-12,
        P_min: float = 1e-12,
        isothermal: bool = False,
        iso_cs: float = 1.0,
        # IC params
        ic: MultivarField = square,
        passive_ics: Optional[Dict[str, UnivarField]] = None,
        # BC params
        bcx: Tuple[BC, BC] = (BC.PERIODIC, BC.PERIODIC),
        bcy: Tuple[BC, BC] = (BC.PERIODIC, BC.PERIODIC),
        bcz: Tuple[BC, BC] = (BC.PERIODIC, BC.PERIODIC),
        bcx_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
        bcx_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
        bcy_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
        bcy_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
        bcz_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
        bcz_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
        # Mesh params
        nx: int = 64,
        ny: int = 1,
        nz: int = 1,
        xlims: Tuple[float, float] = (0.0, 1.0),
        ylims: Tuple[float, float] = (0.0, 1.0),
        zlims: Tuple[float, float] = (0.0, 1.0),
        # Finite volume scheme params
        p: int = 0,
        flux_recipe: FluxRecipe = FluxRecipe.CONS_PRIM_LIM,
        flux_quadrature: FluxQuadrature = FluxQuadrature.TRANSVERSE,
        lazy_primitive_mode: LazyPrimitiveMode = LazyPrimitiveMode.FULL,
        # SED params
        use_SED: bool = True,
        clip_zero_tol: float = 1e-15,
        # MUSCL params
        use_MUSCL: bool = False,
        MUSCL_limiter: MUSCL_SlopeLimiter = MUSCL_SlopeLimiter.MONCEN,
        # PAD params
        PAD_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        # Zhang-Shu params
        use_ZS: bool = False,
        omit_vars_from_ZS: Optional[List[str]] = None,
        adaptive_dt: bool = False,
        adaptive_dt_tol: float = 1e-15,
        theta_denom_tol: float = 1e-15,
        # shock detection params
        eta_max: float = 0.025,
        # NAD params
        use_NAD: bool = True,
        rtol: float = 1e-5,
        atol: float = 0.0,
        omit_vars_from_NAD: Optional[List[str]] = None,
        delta: bool = False,
        # MOOD params
        use_MOOD: bool = False,
        fallback_cascade: FallbackCascade = FallbackCascade.MUSCL,
        max_revs: int = 1,
        blend_troubles: bool = False,
        skip_trouble_counts: bool = False,
        detect_closing_troubles: bool = True,
        # Shared Zhang-Shu / MOOD params
        include_corners: bool = True,
        log_limiter_scalars: bool = True,
        # Solver params
        cupy: bool = False,
        sync_timer: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        discard_after_writing: bool = True,
        overwrite: bool = False,
    ):
        """
        Initialize the HydroSolver with the specified parameters.

        Hydro parameters:
            gamma: Adiabatic index for the ideal gas equation of state.
            riemann_solver: Riemann solver specified by the `RiemannSolver` enum. Possible values
                include RiemannSolver.UPWIND, RiemannSolver.LLF, and RiemannSolver.HLLC.
            CFL: CFL number for time step calculation.
            dt_min: Minimum allowed time step size.
            rho_min: Minimum allowed density.
            P_min: Minimum allowed pressure.
            isothermal: If True, use an isothermal equation of state.
            iso_cs: Isothermal sound speed (only used if isothermal=True).

        Initial condition parameters:
            ic: Initial condition function that takes (idx, x, y, z, t, xp=xp) and returns an
                array of shape (nvars, nx, ny, nz).
            passive_ics: Optional dictionary with keys as variable names and values as functions
                that take (x, y, z, t, xp=xp) and return an array of shape (nx, ny, nz).

        Boundary condition parameters:
            bcx, bcy, bcz: Boundary conditions for x, y, z directions. Each is a tuple of two
                `BC` enums (lower, upper). Possible values include BC.PERIODIC, BC.DIRICHLET,
                BC.FREE, BC.SYMMETRIC, BC.REFLECTIVE, BC.PATCH, and BC.IC.
            bcx_callable_lower, bcx_callable_upper, bcy_callable_lower,
            bcy_callable_upper, bcz_callable_lower, bcz_callable_upper: Optional callable functions
                for Dirichlet boundary conditions. Each should take (idx, x, y, z, t, xp=xp) and
                return an array of shape (nvars, nx, ny, nz).

        Mesh parameters:
            nx, ny, nz: Number of cells in x, y, z directions for a uniform mesh.
            xlims, ylims, zlims: Tuples specifying the lower and upper bounds of the domain in
                x, y, z directions.

        Finite volume scheme parameters:
            p: Polynomial degree for the base finite volume scheme.
            flux_recipe: Flux recipe specified by the `FluxRecipe` enum. Possible values include
                FluxRecipe.CONS_LIM_PRIM, FluxRecipe.CONS_PRIM_LIM, and FluxRecipe.PRIM_PRIM_LIM.
            flux_quadrature: Flux quadrature specified by the `FluxQuadrature` enum. Possible
                values include FluxQuadrature.TRANSVERSE and FluxQuadrature.GAUSS_LEGENDRE.
            lazy_primitive_mode: Lazy primitive mode specified by the `LazyPrimitiveMode` enum.
                Possible values include LazyPrimitiveMode.FULL, LazyPrimitiveMode.NONE, and
                LazyPrimitiveMode.ADAPTIVE, which uses the shock detection threshold `eta_max`
                to determine when to use high-order primitive cell averages.

        Slope limiting parameters (smooth extrema detection):
            use_SED: If True, enable smooth extrema detection.
            clip_zero_tol: Tolerance for clipping near-zero values in SED.

        Slope limiting parameters (MUSCL):
            use_MUSCL: If True, enable MUSCL scheme.
            MUSCL_limiter: Slope limiter specified by the `MUSCL_SlopeLimiter` enum.
                Possible values include MUSCL_SlopeLimiter.MINMOD, MUSCL_SlopeLimiter.MONCEN,
                MUSCL_SlopeLimiter.PP2D (can only be used in 2D), and
                MUSCL_SlopeLimiter.NONE (no slope limiting).

        Slope limiting parameters (physical admissibility detection):
            PAD_bounds: Optional dictionary specifying lower and upper bounds for physical
                admissibility detection. Keys are variable names, and values are tuples of
                (lower_bound, upper_bound).

        Slope limiting parameters (Zhang-Shu):
            use_ZS: If True, enable Zhang-Shu limiter.
            omit_vars_from_ZS: Optional list of variable names to omit from Zhang-Shu limiter.
            adaptive_dt: If True, enable adaptive time stepping based on Zhang-Shu limiter.
            adaptive_dt_tol: Tolerance for adaptive time stepping.
            theta_denom_tol: Tolerance for denominator in theta calculation.

        Slope limiting parameters (shock detection):
            eta_max: Maximum allowed shock detection parameter.

        Slope limiting parameters (MOOD, numerical admissibility detection):
            use_NAD: If True, enable numerical admissibility detection.
            rtol: Relative tolerance for numerical admissibility detection.
            atol: Absolute tolerance for numerical admissibility detection.
            omit_vars_from_NAD: Optional list of variable names to omit from numerical
                admissibility detection.
            delta: If True, enable delta-based numerical admissibility detection.

        Slope limiting parameters (MOOD):
            use_MOOD: If True, enable MOOD scheme.
            fallback_cascade: Fallback cascade specified by the `FallbackCascade` enum. Possible
                values include FallbackCascade.FULL (which falls back to progressively lower-order
                schemes from `p` to 0 in increments of 1), FallbackCascade.MUSCL (which falls back
                to MUSCL), FallbackCascade.FIRST_ORDER (which falls back to first-order),
                and FallbackCascade.MUSCL0 (which falls back first to MUSCL and then first-order).
            max_revs: Maximum number of revisions allowed in MOOD scheme. Must be at least the
                number of fallback schemes in the cascade.
            blend_troubles: If True, blend troubled cells in MOOD scheme.
            skip_trouble_counts: If True, skip counting troubled cells in MOOD scheme.
            detect_closing_troubles: If True, detect closing troubled cells in MOOD scheme.

        Slope limiting parameters (shared between Zhang-Shu and MOOD):
            include_corners: If True, include corner cells when computing the discrete maximum
                principle.
            log_limiter_scalars: If True, log limiter scalars for analysis.

        Solver parameters:
            cupy: If True, use CuPy for GPU acceleration (requires CuPy to be installed).
            sync_timer: If True, synchronize timers when using CuPy.
            output_path: Optional path to save simulation outputs. If None, no outputs are saved.
            discard_after_writing: If True, discard arrays after writing to disk to save memory.
            overwrite: If True, overwrite existing output directory if it exists.
        """
        # Define the following attributes:
        self.arrays: ArrayManager
        self.mesh_arrays: ArrayManager
        self.params: SolverParams
        self.interior: Tuple[slice, slice, slice, slice]
        self.u0_func: MultivarField
        self.mesh: UniformFVMesh
        self.xp: ModuleType
        self.t: float
        self.t_wall_start: float
        self.substep_summary: SubstepSummary
        self.step_summary: StepSummary
        self.step_history: StepHistory
        self.snapshot_history: SnapshotHistory

        # These are straightforward
        self.arrays = ArrayManager()
        self.mesh_arrays = ArrayManager()

        # self.params requires many sub-parameters to be defined
        ic_params = InitialConditionParameters(
            ic=ic,
            passive_ics={} if passive_ics is None else passive_ics,
        )

        bc_params = BoundaryConditionParameters(
            bcx=bcx,
            bcy=bcy,
            bcz=bcz,
            bcx_callable_lower=bcx_callable_lower,
            bcx_callable_upper=bcx_callable_upper,
            bcy_callable_lower=bcy_callable_lower,
            bcy_callable_upper=bcy_callable_upper,
            bcz_callable_lower=bcz_callable_lower,
            bcz_callable_upper=bcz_callable_upper,
        )

        hydro_params = HydroParameters(
            gamma=gamma,
            riemann_solver=riemann_solver,
            CFL=CFL,
            dt_min=dt_min,
            rho_min=rho_min,
            P_min=P_min,
            isothermal=isothermal,
            iso_cs=iso_cs,
        )
        updated_PAD_bounds = self._configure_PAD_bounds(PAD_bounds, hydro_params)

        # Such as the fallback scheme cascade
        null_SED = SmoothExtremaDetectionParameters(False)
        null_MUSCL = MUSCL_Parameters(False, MUSCL_SlopeLimiter.NONE, null_SED)
        null_PAD = PhysicalAdmissibilityParameters(False, {})
        null_ZS = ZhangShuParameters(False, False, null_SED, null_PAD, [])
        null_NAD = NumericalAdmissibilityParameters(False, 0.0, 0.0, null_SED, [])
        null_MOOD = MOOD_Parameters(False, null_NAD, null_PAD, [], 0, False)
        null_shock = ShockDetectionParameters(False)

        fallback_cascade_list: List[FV_SchemeParameters] = []
        if use_MOOD:
            if fallback_cascade == FallbackCascade.FULL:
                for reduced_p in range(p - 1, -1, -1):
                    fallback_cascade_list.append(
                        FV_SchemeParameters(
                            name=f"fallback_p={reduced_p}",
                            p=reduced_p,
                            flux_recipe=flux_recipe,
                            flux_quadrature=flux_quadrature,
                            lazy_primitive_mode=lazy_primitive_mode,
                            muscl_params=null_MUSCL,
                            zhang_shu_params=null_ZS,
                            mood_params=null_MOOD,
                            shock_detection_params=null_shock,
                        )
                    )
            if fallback_cascade in (FallbackCascade.MUSCL, FallbackCascade.MUSCL0):
                fallback_cascade_list.append(
                    FV_SchemeParameters(
                        name="fallback_MUSCL",
                        p=1,
                        flux_recipe=flux_recipe,
                        flux_quadrature=flux_quadrature,
                        lazy_primitive_mode=lazy_primitive_mode,
                        muscl_params=MUSCL_Parameters(True, MUSCL_limiter, null_SED),
                        zhang_shu_params=null_ZS,
                        mood_params=null_MOOD,
                        shock_detection_params=null_shock,
                    )
                )
            if fallback_cascade in (FallbackCascade.MUSCL0, FallbackCascade.FIRST_ORDER):
                fallback_cascade_list.append(
                    FV_SchemeParameters(
                        name="fallback_p0",
                        p=0,
                        flux_recipe=flux_recipe,
                        flux_quadrature=flux_quadrature,
                        lazy_primitive_mode=lazy_primitive_mode,
                        muscl_params=null_MUSCL,
                        zhang_shu_params=null_ZS,
                        mood_params=null_MOOD,
                        shock_detection_params=null_shock,
                    )
                )

        fv_scheme_params = FV_SchemeParameters(
            name="base_scheme",
            p=p,
            flux_recipe=flux_recipe,
            flux_quadrature=flux_quadrature,
            lazy_primitive_mode=lazy_primitive_mode,
            muscl_params=MUSCL_Parameters(
                use_MUSCL=use_MUSCL and p > 0,
                MUSCL_limiter=MUSCL_limiter,
                SED_params=SmoothExtremaDetectionParameters(
                    use_SED and use_MUSCL and p > 0, clip_zero_tol
                ),
            ),
            zhang_shu_params=ZhangShuParameters(
                use_ZS=use_ZS and p > 0,
                adaptive_dt=adaptive_dt,
                SED_params=SmoothExtremaDetectionParameters(
                    use_SED and use_ZS and p > 0, clip_zero_tol
                ),
                PAD_params=PhysicalAdmissibilityParameters(
                    bool(updated_PAD_bounds) and use_ZS and p > 0, updated_PAD_bounds
                ),
                omit_vars=omit_vars_from_ZS or [],
                adaptive_dt_tol=adaptive_dt_tol,
                theta_denom_tol=theta_denom_tol,
                include_corners=include_corners,
                log_limiter_scalars=log_limiter_scalars,
            ),
            mood_params=MOOD_Parameters(
                use_MOOD=use_MOOD and p > 0,
                NAD_params=NumericalAdmissibilityParameters(
                    use_NAD=use_NAD and use_MOOD and p > 0,
                    rtol=rtol,
                    atol=atol,
                    SED_params=SmoothExtremaDetectionParameters(
                        use_SED and use_NAD and use_MOOD and p > 0, clip_zero_tol
                    ),
                    omit_vars=omit_vars_from_NAD or [],
                    delta=delta,
                    include_corners=include_corners,
                ),
                PAD_params=PhysicalAdmissibilityParameters(
                    bool(updated_PAD_bounds) and use_MOOD and p > 0, updated_PAD_bounds
                ),
                fallback_cascade=fallback_cascade_list,
                max_revs=max_revs,
                blend_troubles=blend_troubles,
                skip_trouble_counts=skip_trouble_counts,
                detect_closing_troubles=detect_closing_troubles,
                log_limiter_scalars=log_limiter_scalars,
            ),
            shock_detection_params=ShockDetectionParameters(
                lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE, eta_max
            ),
        )

        active_dims = self._compute_active_dims(nx, ny, nz)
        mesh_params = MeshParameters(
            nx=nx,
            ny=ny,
            nz=nz,
            nghost=self._compute_nghost(fv_scheme_params, len(active_dims)),
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            active_dims=active_dims,
            ndim=len(active_dims),
        )

        # At last, we can define self.params
        self.params = SolverParams(
            hydro=hydro_params,
            ic=ic_params,
            mesh=mesh_params,
            bc=bc_params,
            fv_scheme=fv_scheme_params,
            variable_index_map=self._define_complete_variable_index_map(
                ic_params, fv_scheme_params
            ),
            cupy=cupy and CUPY_AVAILABLE,
            sync_timer=sync_timer and cupy and CUPY_AVAILABLE,
            output_path=Path(output_path) if output_path is not None else None,
            discard_after_writing=discard_after_writing,
        )

        self._build_interior_slice()  # defines self.interior
        self._build_conservative_ic_u0_func()  # defines self.u0_func
        self._enable_ic_bc_or_none_if_inactive()
        self._build_mesh()  # defines self.mesh
        self._init_cupy(cupy, CUPY_AVAILABLE)  # defines self.xp
        self._allocate_arrays()
        self._init_time()  # defines self.t and self.t_wall_start
        self._init_step_and_snapshot_histories()  # self.step_history, self.snapshot_history
        self._reset_substep_summary()  # defines self.substep_summary
        self._reset_step_summary()  # defines self.step_summary
        self._prepare_output_directory(overwrite)
        self._summarize_step(take_snapshot=True)

    def _configure_PAD_bounds(
        self,
        PAD_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        hydro_params: HydroParameters,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        if PAD_bounds is None:
            PAD_bounds = {}

        update = {}
        if "rho" in PAD_bounds:
            if PAD_bounds["rho"][0] != hydro_params.rho_min:
                warnings.warn(
                    f"PAD lower bound for 'rho' is {PAD_bounds['rho'][0]}, "
                    f"which is different from the hydro parameter rho_min={hydro_params.rho_min}."
                )
        else:
            update["rho"] = (hydro_params.rho_min, None)
        if "P" in PAD_bounds:
            if PAD_bounds["P"][0] != hydro_params.P_min:
                warnings.warn(
                    f"PAD lower bound for 'P' is {PAD_bounds['P'][0]}, "
                    f"which is different from the hydro parameter P_min={hydro_params.P_min}."
                )
        else:
            update["P"] = (hydro_params.P_min, None)

        return {**PAD_bounds, **update}

    def _compute_nghost(self, fv_scheme: FV_SchemeParameters, ndim: int) -> int:
        dummy_stencil = conservative_interpolation.left_right(fv_scheme.p)

        # Ghost cell cost of interpolating cell center and face nodes
        nghost = dummy_stencil.shape[1] // 2

        if (
            fv_scheme.flux_recipe == FluxRecipe.PRIM_PRIM_LIM
            and fv_scheme.lazy_primitive_mode != LazyPrimitiveMode.FULL
        ):
            nghost *= 3

        if (
            fv_scheme.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE
            and fv_scheme.shock_detection_params.use_shock_detection
        ):
            nghost += 2

        # Ghost cell cost of integrating fluxes and Riemann Solver
        if fv_scheme.flux_quadrature == FluxQuadrature.TRANSVERSE and ndim >= 2:
            nghost += max(dummy_stencil.shape[1] // 2, 1)
        else:
            nghost += 1

        # Ghost cell cost of slope limiter
        nghost += max(
            1 if fv_scheme.zhang_shu_params.use_ZS else 0,
            (1 if fv_scheme.mood_params.NAD_params.use_NAD else 0)
            + (1 if fv_scheme.mood_params.blend_troubles else 0),
            3 if fv_scheme.muscl_params.SED_params.use_SED else 0,
            3 if fv_scheme.zhang_shu_params.SED_params.use_SED else 0,
            3 if fv_scheme.mood_params.NAD_params.SED_params.use_SED else 0,
        )

        return nghost

    def _compute_active_dims(self, nx: int, ny: int, nz: int) -> Tuple[Literal["x", "y", "z"], ...]:
        return tuple(dim for dim, n in zip(XYZ_TUPLE, [nx, ny, nz]) if n > 1)

    def _define_base_variable_index_map(self) -> VariableIndexMap:
        return VariableIndexMap(
            {"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4, "mx": 1, "my": 2, "mz": 3, "E": 4},
            group_var_map={
                "v": ["vx", "vy", "vz"],
                "m": ["mx", "my", "mz"],
                "primitives": ["rho", "v", "P"],
                "conservatives": ["rho", "m", "E"],
            },
        )

    def _define_complete_variable_index_map(
        self, ic_params: InitialConditionParameters, fv_scheme_params: FV_SchemeParameters
    ) -> VariableIndexMap:
        idx = self._define_base_variable_index_map()

        if ic_params.npassives > 0:
            for v in ic_params.passive_ics.keys():
                if v not in idx.var_idx_map:
                    idx.add_var(v, idx.nvars)
                idx.add_var_to_group(v, "passives")

        if fv_scheme_params.zhang_shu_params.omit_vars:
            for v in fv_scheme_params.zhang_shu_params.omit_vars:
                idx.add_var_to_group(v, "omit_ZS")

        if fv_scheme_params.mood_params.NAD_params.omit_vars:
            for v in fv_scheme_params.mood_params.NAD_params.omit_vars:
                idx.add_var_to_group(v, "omit_NAD")

        return idx

    def _build_interior_slice(self):
        active_dims = self.params.mesh.active_dims
        nghost = self.params.mesh.nghost
        interior = merge_slices(
            *[crop(DIM_TO_AXIS[dim], (nghost, -nghost), ndim=4) for dim in active_dims]
        )
        self.interior = (interior[0], interior[1], interior[2], interior[3])

    def _primitive_ic_w0_func(self) -> MultivarField:
        ic_params = self.params.ic

        def w0_func(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            out = ic_params.ic(idx, x, y, z, t, xp=xp)
            if ic_params.npassives > 0:
                for v, func in ic_params.passive_ics.items():
                    out[idx(v)] = func(x, y, z, t, xp=xp)
            return out

        return w0_func

    def _build_conservative_ic_u0_func(self):

        w0_func = self._primitive_ic_w0_func()

        def u0_func(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            w0_arr = w0_func(idx, x, y, z, t, xp=xp)
            u0_arr = xp.empty_like(w0_arr)
            self.primitives_to_conservatives(w0_arr, u0_arr)
            return u0_arr

        self.u0_func = u0_func

    def _enable_ic_bc_or_none_if_inactive(self):
        bc = self.params.bc

        # set BC to NONE if the corresponding dimension is inactive and disable those callable BCs
        if "x" not in self.params.mesh.active_dims:
            bc.bcx = (BC.NONE, BC.NONE)
            bc.bcx_callable_lower = None
            bc.bcx_callable_upper = None
        if "y" not in self.params.mesh.active_dims:
            bc.bcy = (BC.NONE, BC.NONE)
            bc.bcy_callable_lower = None
            bc.bcy_callable_upper = None
        if "z" not in self.params.mesh.active_dims:
            bc.bcz = (BC.NONE, BC.NONE)
            bc.bcz_callable_lower = None
            bc.bcz_callable_upper = None

        # set BC to DIRICHLET with u0_func if BC is IC
        for dim in ["x", "y", "z"]:
            bcdim0 = getattr(bc, f"bc{dim}")[0]
            bcdim1 = getattr(bc, f"bc{dim}")[1]

            if bcdim0 != BC.IC and bcdim1 != BC.IC:
                continue

            new_bcdim = [bcdim0, bcdim1]
            if bcdim0 == BC.IC:
                new_bcdim[0] = BC.DIRICHLET
                setattr(bc, f"bc{dim}_callable_lower", self.u0_func)
            if bcdim1 == BC.IC:
                new_bcdim[1] = BC.DIRICHLET
                setattr(bc, f"bc{dim}_callable_upper", self.u0_func)
            setattr(bc, f"bc{dim}", tuple(new_bcdim))

    def _build_mesh(self):
        mesh_params = self.params.mesh

        self.mesh = UniformFVMesh(
            nx=mesh_params.nx,
            ny=mesh_params.ny,
            nz=mesh_params.nz,
            nghost=mesh_params.nghost,
            xlims=mesh_params.xlims,
            ylims=mesh_params.ylims,
            zlims=mesh_params.zlims,
            active_dims=mesh_params.active_dims,
            array_manager=self.mesh_arrays,
        )

    def _init_cupy(self, tried_cupy: bool, cupy_available: bool):
        if tried_cupy and not cupy_available:
            warnings.warn("CuPy is not available. Using NumPy instead.")
        if not tried_cupy and cupy_available:
            warnings.warn("CuPy not requested, but CuPy is available. Using NumPy instead.")

        if self.params.cupy:
            self.arrays.transfer_to("gpu")
            self.mesh_arrays.transfer_to("gpu")

            self.xp = xp
        else:
            self.xp = np

    def _compute_ic_array(self) -> ArrayLike:
        params = self.params
        idx = params.variable_index_map
        nvars = idx.nvars
        mesh = self.mesh
        nx, ny, nz = mesh.shape

        u0 = self.xp.empty((nvars, nx, ny, nz))
        if params.fv_scheme.p > 1:
            mesh.perform_GaussLegendre_quadrature(
                lambda x, y, z: self.u0_func(idx, x, y, z, 0.0, xp=self.xp),
                u0,
                mesh_region="core",
                cell_region="interior",
                p=params.fv_scheme.p,
            )
        else:
            u0[...] = self.u0_func(idx, *mesh.get_cell_centers(), 0.0, xp=self.xp)

        return u0

    def _allocate_arrays(self):
        nvars = self.params.variable_index_map.nvars
        fv_scheme = self.params.fv_scheme
        mesh = self.mesh
        nx, ny, nz = mesh.shape
        _nx_, _ny_, _nz_ = mesh._shape_
        active_dims = mesh.active_dims
        arrays = self.arrays

        # define cell-centered/cell-averaged arrays
        arrays.add("u", self._compute_ic_array())
        arrays.add("unew", np.empty((nvars, nx, ny, nz)))
        arrays.add("_u_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_ucc_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_wcc_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_w_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_w1_", np.empty((nvars, _nx_, _ny_, _nz_)))

        # define arrays associated faces along the x-direction
        if "x" in active_dims:
            arrays.add("_F_", np.empty((nvars, nx + 1, _ny_, _nz_)))
            arrays.add("F", np.empty((nvars, nx + 1, ny, nz)))

        # define arrays associated with faces along the y-direction
        if "y" in active_dims:
            arrays.add("_G_", np.empty((nvars, _nx_, ny + 1, _nz_)))
            arrays.add("G", np.empty((nvars, nx, ny + 1, nz)))

        # define arrays associated with faces along the z-direction
        if "z" in active_dims:
            arrays.add("_H_", np.empty((nvars, _nx_, _ny_, nz + 1)))
            arrays.add("H", np.empty((nvars, nx, ny, nz + 1)))

        # define slope-limiting arrays
        arrays.add("_alpha_", np.ones((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_has_shock_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))

        # define Zhang-Shu limiter arrays
        arrays.add("_theta_", np.ones((nvars, _nx_, _ny_, _nz_)))

        # define MOOD arrays
        for scheme in [fv_scheme] + fv_scheme.mood_params.fallback_cascade:
            if "x" in active_dims:
                arrays.add("F_" + scheme.name, np.empty((nvars, nx + 1, ny, nz)))
            if "y" in active_dims:
                arrays.add("G_" + scheme.name, np.empty((nvars, nx, ny + 1, nz)))
            if "z" in active_dims:
                arrays.add("H_" + scheme.name, np.empty((nvars, nx, ny, nz + 1)))

        arrays.add("_qold_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_qnew_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_troubles_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))
        arrays.add("revisable_troubles", np.zeros((1, nx, ny, nz), dtype=np.int32))
        arrays.add("_cascade_idx_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))

        # define snapshot arrays
        arrays.add("has_shock_log", np.zeros((1, nx, ny, nz)))
        arrays.add("theta_log", np.ones((nvars, nx, ny, nz)))
        arrays.add("troubles_log", np.zeros((1, nx, ny, nz)))
        arrays.add("cascade_idx_log", np.zeros((1, nx, ny, nz)))

    def _init_time(self):
        self.t = 0.0
        self.t_wall_start = np.nan

    def _init_step_and_snapshot_histories(self):
        self.step_history = StepHistory([])
        self.snapshot_history = SnapshotHistory([])

    def _reset_substep_summary(self):
        self.substep_summary = SubstepSummary(
            substep=-1, t_wall=np.nan, n_MOOD_revisions=0, n_troubles_hist=[]
        )

    def _reset_step_summary(self, n: int = 0):
        self.step_summary = StepSummary(
            step=n,
            t_sim=np.nan,
            t_wall=np.nan,
            n_dt_revisions=0,
            rho_min=np.nan,
            E_total=np.nan,
            substeps=[],
            timer=MultiTimer(["take_snapshot", "take_step", "compute_dt"]),
        )

    def _take_snapshot(self):
        self.step_summary.timer.start("take_snapshot", self.params.sync_timer)  # TIMER START

        params = self.params
        output_path = params.output_path

        u = self.arrays["u"]
        w = self.xp.empty_like(u)

        self.conservatives_to_primitives(u, w)

        if output_path is None:
            path = None
        else:
            n = len(self.snapshot_history)
            ndigits = params.output_n_digits
            path = output_path / f"output_{n:0{ndigits}d}.pkl"

        snapshot = Snapshot(
            t=self.t,
            data=SnapshotData(
                u=self.arrays.get_numpy_copy("u"),
                w=self.xp.asnumpy(w) if self.params.cupy else w,
                has_shock=(
                    self.arrays.get_numpy_copy("has_shock_log")
                    if params.fv_scheme.shock_detection_params.use_shock_detection
                    else np.empty((0,))
                ),
                theta=(
                    self.arrays.get_numpy_copy("theta_log")
                    if params.fv_scheme.zhang_shu_params.use_ZS
                    else np.empty((0,))
                ),
                troubles=(
                    self.arrays.get_numpy_copy("troubles_log")
                    if params.fv_scheme.mood_params.use_MOOD
                    else np.empty((0,))
                ),
                cascade_idx=(
                    self.arrays.get_numpy_copy("cascade_idx_log")
                    if params.fv_scheme.mood_params.use_MOOD
                    else np.empty((0,))
                ),
            ),
            path=path,
        )

        if snapshot.path is not None:
            if len(self.step_history) > 0:
                print()  # for better separation of snapshot logs in the terminal

            with open(snapshot.path.parent / "output_times.txt", "a") as f:
                f.write(f"{snapshot.path.name},{snapshot.t}\n")
            snapshot.dump(params.discard_after_writing)
        self.snapshot_history.append(snapshot)

        self.step_summary.timer.stop("take_snapshot", self.params.sync_timer)  # TIMER STOP

    def _write_params_files(self):
        output_path = self.params.output_path
        if output_path is not None:
            with open(output_path / "params.yaml", "w") as f:
                f.write(yaml_dump(asdict(self.params)))
            with open(output_path / "params.pkl", "wb") as f:
                pickle.dump(self.params, f)

    def _write_mesh(self):
        output_path = self.params.output_path
        mesh = self.mesh

        if output_path is not None:
            with open(output_path / "mesh.pkl", "wb") as f:
                pickle.dump(mesh, f)

    def _prepare_output_directory(self, overwrite: bool):
        """
        Prepare the output directory and write the configuration file.
        """
        output_path = self.params.output_path

        if output_path is None:
            return
        if output_path.exists() and overwrite:
            warnings.warn(f"Output path '{output_path}' already exists. Overwriting.")
            shutil.rmtree(output_path)
        elif output_path.exists():
            raise FileExistsError(f"Output path '{output_path}' already exists.")

        output_path.mkdir(parents=True, exist_ok=False)

        self._write_params_files()
        self._write_mesh()

    def _summarize_step(self, take_snapshot: bool = False):
        idx = self.params.variable_index_map
        step_summary = self.step_summary

        u = self.arrays["u"]

        step_summary.t_sim = self.t
        step_summary.t_wall = time.time() - self.t_wall_start
        step_summary.rho_min = self.xp.min(u[idx("rho")]).item()
        step_summary.E_total = self.xp.sum(u[idx("E")]).item()

        if take_snapshot:
            self._take_snapshot()  # updates step_summary.timer

        self.step_history.append(step_summary)
        self._reset_step_summary(step_summary.step + 1)

    # Helper functions

    @cached_property
    def _prim_to_cons_cp(self):
        return make_prim_to_cons_elementwise_kernel(self.params.ic.npassives)

    @cached_property
    def _cons_to_prim_cp(self):
        return make_cons_to_prim_elementwise_kernel(self.params.ic.npassives)

    @cached_property
    def riemann_solver(self):
        params = self.params
        rs = params.hydro.riemann_solver
        npassives = params.ic.npassives

        match rs:
            case RiemannSolver.UPWIND:
                return UpwindRiemannSolver(npassives)
            case RiemannSolver.LLF:
                return LLF_RiemannSolver(npassives)
            case RiemannSolver.HLLC:
                return HLLC_RiemannSolver(npassives)
            case _:
                raise ValueError(f"Unknown Riemann solver: {rs}")

    # Hydro solver functions

    def primitives_to_conservatives(self, w: ArrayLike, u: ArrayLike):
        """
        Write conservative variables into `u` using primitive variables from `w`.
        """
        params = self.params
        idx = params.variable_index_map

        if params.cupy:
            self._prim_to_cons_cp(
                w[idx("rho")],
                w[idx("vx")],
                w[idx("vy")],
                w[idx("vz")],
                w[idx("P")],
                params.hydro.gamma,
                *(w[idx(v)] for v in idx.group_var_map.get("passives", [])),
                u[idx("rho")],
                u[idx("mx")],
                u[idx("my")],
                u[idx("mz")],
                u[idx("E")],
                *(u[idx(v)] for v in idx.group_var_map.get("passives", [])),
            )
        else:
            u[...] = prim_to_cons(idx, w, params.hydro.gamma)

    def conservatives_to_primitives(self, u: ArrayLike, w: ArrayLike):
        """
        Write primitive variables into `w` from conservative variables `u`.
        """
        params = self.params
        idx = params.variable_index_map

        if params.cupy:
            self._cons_to_prim_cp(
                u[idx("rho")],
                u[idx("mx")],
                u[idx("my")],
                u[idx("mz")],
                u[idx("E")],
                params.hydro.gamma,
                params.hydro.isothermal,
                params.hydro.iso_cs,
                *(u[idx(v)] for v in idx.group_var_map.get("passives", [])),
                w[idx("rho")],
                w[idx("vx")],
                w[idx("vy")],
                w[idx("vz")],
                w[idx("P")],
                *(w[idx(v)] for v in idx.group_var_map.get("passives", [])),
            )
        else:
            w[...] = cons_to_prim(
                idx,
                u,
                params.hydro.gamma,
                params.hydro.isothermal,
                params.hydro.iso_cs,
            )

    def compute_sound_speed(self, w: ArrayLike, c: ArrayLike):
        """
        Compute the sound speed from primitives `w` and write the result to `c`.
        """
        params = self.params
        idx = params.variable_index_map

        if params.hydro.isothermal:
            c[...] = params.hydro.iso_cs
        elif params.cupy:
            c[...] = sound_speed_cp(w[idx("rho")], w[idx("P")], params.hydro.gamma)
        else:
            c[...] = sound_speed(idx, w, params.hydro.gamma)

    def compute_dt(self, t: float, u: ArrayLike) -> float:
        params = self.params
        idx = params.variable_index_map
        mesh = self.mesh

        w = self.xp.empty_like(u)
        c = self.xp.empty_like(u[0, ...])
        sum_of_s_over_h = self.xp.zeros_like(u[0, ...])

        self.conservatives_to_primitives(u, w)
        self.compute_sound_speed(w, c)

        for dim, h in zip(["x", "y", "z"], [mesh.hx, mesh.hy, mesh.hz]):
            if dim not in params.mesh.active_dims:
                continue
            s = self.xp.abs(w[idx("v" + dim)]) + c
            sum_of_s_over_h += s / h

        max_speed = self.xp.max(sum_of_s_over_h).item()
        dt = params.hydro.CFL / max_speed

        return dt

    def apply_bc(self, _u_: ArrayLike, t: float, p: int):
        bc_params = self.params.bc
        mesh_params = self.params.mesh
        idx = self.params.variable_index_map
        mesh = self.mesh

        apply_bc(
            self.xp,
            _u_,
            mesh_params.nghost,
            bc_params.bcx,
            bc_params.bcy,
            bc_params.bcz,
            bc_params.bcx_callable_lower,
            bc_params.bcx_callable_upper,
            bc_params.bcy_callable_lower,
            bc_params.bcy_callable_upper,
            bc_params.bcz_callable_lower,
            bc_params.bcz_callable_upper,
            idx,
            mesh,
            t,
            p,
        )

    def detect_shocks(self, fv_scheme: FV_SchemeParameters):
        """
        Detect shocks in "_u_" or "_w_" and write the result to "_has_shock_".
        """
        if not fv_scheme.shock_detection_params.use_shock_detection:
            raise ValueError("Shock detection is not enabled in the provided FV scheme.")

        params = self.params
        idx = params.variable_index_map
        active_dims = params.mesh.active_dims
        eta_max = fv_scheme.shock_detection_params.eta_max
        arrays = self.arrays

        _u_ = arrays["_u_"]
        _w_ = arrays["_w_"]
        _has_shock_ = arrays["_has_shock_"]
        _q_ref_ = self.xp.empty_like(_u_)
        _c_ = self.xp.empty_like(_u_[:1, ...])
        _eta_ = self.xp.empty_like(_u_)

        self.compute_sound_speed(_w_, _c_)

        if fv_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            # Detect shocks using conservatives
            _q_ref_[...] = _u_
            _c_ *= _u_[idx("rho", keepdims=True)]
            for dim in active_dims:
                _q_ref_[idx("v" + dim)] = _c_
            detect_shocks(_u_, _q_ref_, _eta_, _has_shock_, active_dims, eta_max)
        else:
            # Detect shocks using primitives
            _q_ref_[...] = _w_
            for dim in active_dims:
                _q_ref_[idx("v" + dim)] = _c_
            detect_shocks(_w_, _q_ref_, _eta_, _has_shock_, active_dims, eta_max)

    def update_cell_centers_and_primitive_cell_averages(
        self, t: float, u: ArrayLike, fv_scheme: FV_SchemeParameters
    ):
        """
        Update the ghost-cell-padded cell-averaged and cell-centered arrays
        (_u_, _ucc_, _wcc_, and _w_) based on time `t`, the conservative variables `u`, and
        the provided `fv_scheme`.
        """
        idx = self.params.variable_index_map
        active_dims = self.mesh.active_dims
        arrays = self.arrays

        # allocate arrays
        _u_ = arrays["_u_"]
        _ucc_ = arrays["_ucc_"]
        _wcc_ = arrays["_wcc_"]
        _w_ = arrays["_w_"]
        _w1_ = arrays["_w1_"]
        _has_shock_ = arrays["_has_shock_"]

        # 0) conservatives FV averages + BC
        _u_[self.interior] = u
        self.apply_bc(_u_, t, fv_scheme.p)

        # 1) conservative and primitive centroids
        interpolate_cell_centers(_u_, _ucc_, active_dims, fv_scheme.p)
        self.conservatives_to_primitives(_ucc_, _wcc_)

        # 2) primitive FV averages
        self.conservatives_to_primitives(_u_, _w1_)

        match fv_scheme.lazy_primitive_mode:
            case LazyPrimitiveMode.NONE:
                integrate_cell_averages(_wcc_, _w_, active_dims, fv_scheme.p)
            case LazyPrimitiveMode.FULL:
                _w_[...] = _w1_
            case LazyPrimitiveMode.ADAPTIVE:
                integrate_cell_averages(_wcc_, _w_, active_dims, fv_scheme.p)
                self.detect_shocks(fv_scheme)  # updates "_has_shock_"

                # Flag PAD violations as shocks
                if fv_scheme.mood_params.PAD_params.use_PAD:
                    raise NotImplementedError(
                        "Adaptive primitive mode with MOOD PAD is not implemented yet."
                    )
                if fv_scheme.zhang_shu_params.PAD_params.use_PAD:
                    for v, (lb, ub) in fv_scheme.zhang_shu_params.PAD_params.bounds.items():
                        if lb is not None:
                            self.xp.maximum(_has_shock_, _w_[idx(v)] < lb, out=_has_shock_)
                        if ub is not None:
                            self.xp.maximum(_has_shock_, _w_[idx(v)] > ub, out=_has_shock_)

                _w_[...] = self.xp.where(_has_shock_, _w_, _w1_)

        # ensure density is always transformed in the lazy way
        if fv_scheme.lazy_primitive_mode != LazyPrimitiveMode.FULL:
            _w_[idx("rho"), ...] = _w1_[idx("rho"), ...]

    def _compute_nnodes_per_face(self, fv_scheme: FV_SchemeParameters) -> int:
        ndim = self.mesh.ndim
        n_gauss_legendre = conservative_interpolation.n_gauss_legendre_nodes(fv_scheme.p)

        if fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
            return n_gauss_legendre ** (ndim - 1)
        else:
            return 1

    def _prepare_interpolation_arrays(
        self, dim: Literal["x", "y", "z"], fv_scheme: FV_SchemeParameters
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        params = self.params
        active_dims = params.mesh.active_dims
        nvars = params.variable_index_map.nvars
        nx, ny, nz = self.mesh.shape
        _nx_, _ny_, _nz_ = self.mesh._shape_
        nnodes = self._compute_nnodes_per_face(fv_scheme)

        if dim not in active_dims:
            raise ValueError(f"Dimension '{dim}' is not active.")

        if dim == "x":
            _x_nodes_ = self.xp.empty((nvars, _nx_, _ny_, _nz_, 2 * nnodes))
            _f_nodes_ = self.xp.empty((nvars, nx + 1, _ny_, _nz_, nnodes))
            _F_integral_ = self.xp.empty((nvars, nx + 1, _ny_, _nz_))

            return _x_nodes_, _f_nodes_, _F_integral_

        if dim == "y":
            _y_nodes_ = self.xp.empty((nvars, _nx_, _ny_, _nz_, 2 * nnodes))
            _g_nodes_ = self.xp.empty((nvars, _nx_, ny + 1, _nz_, nnodes))
            _G_integral_ = self.xp.empty((nvars, _nx_, ny + 1, _nz_))

            return _y_nodes_, _g_nodes_, _G_integral_

        if dim == "z":
            _z_nodes_ = self.xp.empty((nvars, _nx_, _ny_, _nz_, 2 * nnodes))
            _h_nodes_ = self.xp.empty((nvars, _nx_, _ny_, nz + 1, nnodes))
            _H_integral_ = self.xp.empty((nvars, _nx_, _ny_, nz + 1))

            return _z_nodes_, _h_nodes_, _H_integral_

    def _convert_nodes_to_primitives(
        self,
        _x_nodes_: Optional[ArrayLike],
        _y_nodes_: Optional[ArrayLike],
        _z_nodes_: Optional[ArrayLike],
    ):
        for _nodes_ in [_x_nodes_, _y_nodes_, _z_nodes_]:
            if _nodes_ is not None:
                self.conservatives_to_primitives(_nodes_, _nodes_)

    def zhang_shu_limiter(
        self,
        fv_scheme: FV_SchemeParameters,
        _x_nodes_: Optional[ArrayLike],
        _y_nodes_: Optional[ArrayLike],
        _z_nodes_: Optional[ArrayLike],
    ):
        flux_recipe = fv_scheme.flux_recipe
        arrays = self.arrays

        _q_ = arrays["_u_"] if flux_recipe == FluxRecipe.CONS_LIM_PRIM else arrays["_w_"]
        _qcc_ = arrays["_ucc_"] if flux_recipe == FluxRecipe.CONS_LIM_PRIM else arrays["_wcc_"]
        _theta_ = arrays["_theta_"]
        _alpha_ = arrays["_alpha_"]

        apply_zhang_shu_limiter(
            _q_,
            _qcc_,
            _theta_,
            _alpha_,
            self.params.variable_index_map,
            fv_scheme.zhang_shu_params,
            _x_nodes_,
            _y_nodes_,
            _z_nodes_,
        )

    def _ensure_positive_nodes(
        self,
        _x_nodes_: Optional[ArrayLike],
        _y_nodes_: Optional[ArrayLike],
        _z_nodes_: Optional[ArrayLike],
    ):
        params = self.params
        idx = params.variable_index_map

        for _nodes_ in [_x_nodes_, _y_nodes_, _z_nodes_]:
            if _nodes_ is None:
                continue
            self.xp.maximum(_nodes_[idx("rho")], params.hydro.rho_min, out=_nodes_[idx("rho")])
            self.xp.maximum(_nodes_[idx("P")], params.hydro.P_min, out=_nodes_[idx("P")])

    def _update_nodal_fluxes(
        self,
        _x_nodes_: ArrayLike,
        _f_nodes_: ArrayLike,
        fv_scheme: FV_SchemeParameters,
        dim: Literal["x", "y", "z"],
    ):
        params = self.params
        nghost = params.mesh.nghost
        nnodes = _f_nodes_.shape[4]
        axis = DIM_TO_AXIS[dim]

        left_of_interface = merge_slices(
            crop(4, (nnodes, 2 * nnodes), ndim=5), crop(axis, (nghost - 1, -nghost), ndim=5)
        )
        right_of_interface = merge_slices(
            crop(4, (None, nnodes), ndim=5), crop(axis, (nghost, -nghost + 1), ndim=5)
        )

        _f_nodes_[...] = self.riemann_solver(
            params.variable_index_map,
            _x_nodes_[left_of_interface],
            _x_nodes_[right_of_interface],
            dim,
            params.hydro.gamma,
            params.hydro.isothermal,
            params.hydro.iso_cs,
        )

    def _update_flux_arrays(
        self,
        _F_integral_: Optional[ArrayLike],
        _G_integral_: Optional[ArrayLike],
        _H_integral_: Optional[ArrayLike],
    ):
        params = self.params
        nghost = self.params.mesh.nghost
        arrays = self.arrays

        for dim, _Fintegral_ in zip(["x", "y", "z"], [_F_integral_, _G_integral_, _H_integral_]):
            if _Fintegral_ is None:
                continue  # this dim is not active
            Fluxes = arrays[{"x": "F", "y": "G", "z": "H"}[dim]]

            if params.mesh.ndim == 1:
                Fluxes[...] = _Fintegral_  # no need for flux integral in 1D
                continue

            interior = merge_slices(
                *[
                    crop(axis, (nghost, -nghost), ndim=4)
                    for d, axis in zip(["x", "y", "z"], [1, 2, 3])
                    if d in params.mesh.active_dims and d != dim
                ],
            )
            Fluxes[...] = _Fintegral_[interior]

    def update_fluxes(self, fv_scheme: FV_SchemeParameters):
        """
        Update the flux arrays "F", "G", and/or "H" with the specified fv_scheme based on the
        current cell averaged / cell centered values "_u_", "_ucc_", "_wcc_", and/or "_w_".
        """
        if fv_scheme.muscl_params.use_MUSCL:
            update_fluxes_with_muscl_scheme(self, fv_scheme, 0.0)
            return

        params = self.params
        active_dims = params.mesh.active_dims
        arrays = self.arrays

        _q_ = arrays["_w_"] if fv_scheme.flux_recipe == FluxRecipe.PRIM_PRIM_LIM else arrays["_u_"]

        # Allocate temporary arrays
        tmp = {dim: self._prepare_interpolation_arrays(dim, fv_scheme) for dim in active_dims}
        _x_nodes_ = tmp["x"][0] if "x" in tmp else None
        _y_nodes_ = tmp["y"][0] if "y" in tmp else None
        _z_nodes_ = tmp["z"][0] if "z" in tmp else None

        # Interpolate nodes at cell faces
        for dim in active_dims:
            _nodes_, _, _ = tmp[dim]
            interpolate_face_nodes(
                _q_,
                _nodes_,
                dim,
                active_dims,
                fv_scheme.p,
                fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE,
            )

        # Convert nodes to primitives if needed, apply a priori limiting, and ensure positivity
        if fv_scheme.flux_recipe == FluxRecipe.CONS_PRIM_LIM:
            self._convert_nodes_to_primitives(_x_nodes_, _y_nodes_, _z_nodes_)

        if fv_scheme.zhang_shu_params.use_ZS:
            self.zhang_shu_limiter(fv_scheme, _x_nodes_, _y_nodes_, _z_nodes_)

        if fv_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            self._convert_nodes_to_primitives(_x_nodes_, _y_nodes_, _z_nodes_)
        self._ensure_positive_nodes(_x_nodes_, _y_nodes_, _z_nodes_)

        # Compute flux integrals
        for dim in active_dims:
            _nodes_, _fnodes_, _Fintegral_ = tmp[dim]

            self._update_nodal_fluxes(_nodes_, _fnodes_, fv_scheme, dim)

            if params.mesh.ndim == 1:
                _Fintegral_[...] = _fnodes_[..., 0]  # no need for flux integral in 1D
            elif fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                integrate_gauss_legendre_face_nodes(
                    _fnodes_, _Fintegral_, dim, active_dims, fv_scheme.p
                )
            else:
                integrate_transverse_nodes(_fnodes_, _Fintegral_, dim, active_dims, fv_scheme.p)
        self._update_flux_arrays(
            tmp["x"][2] if "x" in tmp else None,
            tmp["y"][2] if "y" in tmp else None,
            tmp["z"][2] if "z" in tmp else None,
        )

    def compute_flux_jvp(
        self,
        q: ArrayLike,
        vec: ArrayLike,
        dim: Literal["x", "y", "z"],
        primitives: bool = True,
    ) -> ArrayLike:
        """
        Compute the Jacobian-vector product for the flux function.
        """
        if not primitives:
            raise NotImplementedError("JVP for conservative variable fluxes not implemented yet.")

        idx = self.params.variable_index_map
        hydro_params = self.params.hydro

        jvp = self.xp.zeros_like(q)

        iv = idx("v" + dim)

        jvp[idx("rho")] = q[iv] * vec[idx("rho")] + q[idx("rho")] * vec[iv]
        jvp[idx("vx")] = q[iv] * vec[idx("vx")]
        jvp[idx("vy")] = q[iv] * vec[idx("vy")]
        jvp[idx("vz")] = q[iv] * vec[idx("vz")]
        jvp[iv] += (1 / q[idx("rho")]) * vec[idx("P")]
        jvp[idx("P")] = (
            hydro_params.iso_cs**2 * jvp[idx("rho")]
            if hydro_params.isothermal
            else hydro_params.gamma * q[idx("P")] * vec[iv] + q[iv] * vec[idx("P")]
        )
        if "passives" in idx.group_var_map:
            jvp[idx("passives")] = q[iv] * vec[idx("passives")] + q[idx("passives")] * vec[iv]

        return jvp

    # ODE solver functions

    def compute_time_derivative(self) -> ArrayLike:
        active_dims = self.params.mesh.active_dims
        hx, hy, hz = self.mesh.hx, self.mesh.hy, self.mesh.hz

        dudt = self.xp.zeros_like(self.arrays["u"])

        if "x" in active_dims:
            F_fluxes = self.arrays["F"]
            dudt -= (F_fluxes[:, 1:, :, :] - F_fluxes[:, :-1, :, :]) / hx

        if "y" in active_dims:
            G_fluxes = self.arrays["G"]
            dudt -= (G_fluxes[:, :, 1:, :] - G_fluxes[:, :, :-1, :]) / hy

        if "z" in active_dims:
            H_fluxes = self.arrays["H"]
            dudt -= (H_fluxes[:, :, :, 1:] - H_fluxes[:, :, :, :-1]) / hz

        return dudt

    def f(self, t: float, u: ArrayLike, dt: float) -> ArrayLike:
        base_scheme = self.params.fv_scheme

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.update_cell_centers_and_primitive_cell_averages(t, u, base_scheme)
            self.update_fluxes(base_scheme)

            if base_scheme.mood_params.use_MOOD:
                mood_loop(self, t, dt)

        return self.compute_time_derivative()

    def _summarize_substep(self):
        Step_summary = self.step_summary
        substep_summary = self.substep_summary

        substep_summary.substep = (
            1 if not Step_summary.substeps else Step_summary.substeps[-1].substep + 1
        )
        substep_summary.t_wall = time.time() - self.t_wall_start

        Step_summary.substeps.append(substep_summary)
        self._reset_substep_summary()

    def _update_log_arrays(self, action: LogArrayAction):
        interior = self.interior
        arrays = self.arrays

        match action:
            case LogArrayAction.SUBSTEP_ADD:
                arrays["has_shock_log"] += arrays["_has_shock_"][interior]
                arrays["theta_log"] += arrays["_theta_"][interior]
                arrays["troubles_log"] += arrays["_troubles_"][interior]
                arrays["cascade_idx_log"] += arrays["_cascade_idx_"][interior]
            case LogArrayAction.SUBSTEP_AVERAGE:
                n_substeps = max(1, len(self.step_summary.substeps))

                arrays["has_shock_log"] /= n_substeps
                arrays["theta_log"] /= n_substeps
                arrays["troubles_log"] /= n_substeps
                arrays["cascade_idx_log"] /= n_substeps
            case LogArrayAction.RESET:
                arrays["has_shock_log"][...] = 0.0
                arrays["theta_log"][...] = 0.0
                arrays["troubles_log"][...] = 0.0
                arrays["cascade_idx_log"][...] = 0.0

    def _close_substep(self):
        self._summarize_substep()
        self._update_log_arrays(LogArrayAction.SUBSTEP_ADD)

    def _update_unew(self, t: float, u: ArrayLike, dt: float, time_integrator: TimeIntegrator):
        params = self.params
        arrays = self.arrays

        unew = arrays["unew"]

        match time_integrator:
            case TimeIntegrator.MUSCL_HANCOCK:
                if not params.fv_scheme.muscl_params.use_MUSCL:
                    raise ValueError("MUSCL-Hancock time integrator requires a MUSCL FV scheme.")

                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    self.update_cell_centers_and_primitive_cell_averages(t, u, params.fv_scheme)
                    update_fluxes_with_muscl_scheme(self, params.fv_scheme, dt)
                k0 = self.compute_time_derivative()

                unew[...] = u + dt * k0
                return
            case TimeIntegrator.FORWARD_EULER:
                unew[...] = u + self.f(t, u, dt) * dt
                self._close_substep()
                return
            case TimeIntegrator.SSPRK2:
                k0 = self.xp.empty_like(u)
                k1 = self.xp.empty_like(u)

                k0[...] = self.f(t, u, dt)
                unew[...] = u + dt * k0
                self._close_substep()

                k1[...] = self.f(t + dt, unew, dt)
                unew[...] = 0.5 * u + 0.5 * (unew + dt * k1)
                self._close_substep()
                return
            case TimeIntegrator.SSPRK3:
                k0 = self.xp.empty_like(u)
                k1 = self.xp.empty_like(u)
                k2 = self.xp.empty_like(u)

                k0[...] = self.f(t, u, dt)
                unew[...] = u + dt * k0
                self._close_substep()

                k1[...] = self.f(t + dt, unew, dt)
                self._close_substep()

                k2[...] = self.f(t + 0.5 * dt, u + 0.25 * dt * k0 + 0.25 * dt * k1, dt)
                unew[...] = u + (1 / 6) * dt * (k0 + k1 + 4 * k2)
                self._close_substep()
                return
            case TimeIntegrator.RK4:
                k0 = self.xp.empty_like(u)
                k1 = self.xp.empty_like(u)
                k2 = self.xp.empty_like(u)
                k3 = self.xp.empty_like(u)

                k0[...] = self.f(t, u, dt)
                self._close_substep()

                k1[...] = self.f(t + 0.5 * dt, u + 0.5 * dt * k0, dt)
                self._close_substep()

                k2[...] = self.f(t + 0.5 * dt, u + 0.5 * dt * k1, dt)
                self._close_substep()

                k3[...] = self.f(t + dt, u + dt * k2, dt)
                unew[...] = u + (1 / 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
                self._close_substep()
                return
            case _:
                raise ValueError(f"Unsupported time integrator: {time_integrator}")

    def _open_step(self):
        self._update_log_arrays(LogArrayAction.RESET)
        self.step_summary.timer.start("take_step", self.params.sync_timer)  # TIMER START

    def _close_step(self, take_snapshot: bool):
        self.step_summary.timer.stop("take_step", self.params.sync_timer)  # TIMER STOP
        self._update_log_arrays(LogArrayAction.SUBSTEP_AVERAGE)
        self._summarize_step(take_snapshot)

    def _start_wall_timer(self):
        self.t_wall_start = time.time()
        self._progress_line_len = 0

    def _build_message(
        self,
        current_time: Optional[float] = None,
        stopping_time: Optional[float] = None,
        current_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> str:
        step_summary = self.step_history[-1]
        energy_conservation = abs(step_summary.E_total - self.step_history[0].E_total)

        parts = ["SuperFV: "]
        if current_steps is not None and total_steps is not None:
            parts.append(f"step {current_steps}/{total_steps}")
        elif current_steps is not None:
            parts.append("1 step" if current_steps == 1 else f"{current_steps} steps")
        if current_time is not None and stopping_time is not None:
            parts.append(f"[t={current_time:.2e}/{stopping_time:.2e}]")
        parts.append(f"rho_min={step_summary.rho_min:.2e}")
        parts.append(f"E_cons={energy_conservation:.2e}")
        parts.append(f"wall={step_summary.t_wall:.2e}s")

        message = parts[0] + " | ".join(parts[1:])
        return message

    def _print_message(self, message: str):
        print(f"\r{message}", end="", flush=True)

    def _check_dt(self, dt: float):
        if not np.isfinite(dt):
            raise RuntimeError(f"Computed dt={dt} is not finite.")
        if dt < self.params.hydro.dt_min:
            raise RuntimeError(
                f"Computed dt={dt} is smaller than the minimum allowed={self.params.hydro.dt_min}."
            )

    def _check_unew_for_dt_revision(self) -> bool:
        params = self.params
        tol = params.fv_scheme.zhang_shu_params.adaptive_dt_tol
        idx = params.variable_index_map

        if not params.fv_scheme.zhang_shu_params.adaptive_dt:
            raise RuntimeError(
                "DT revision check called but adaptive_dt is not enabled in the parameters."
            )

        unew = self.arrays["unew"]
        wnew = self.xp.empty_like(unew)

        self.conservatives_to_primitives(unew, wnew)

        for v, (lb, ub) in params.fv_scheme.zhang_shu_params.PAD_params.bounds.items():
            if lb is not None and self.xp.any(wnew[idx(v)] < lb - tol):
                return True
            if ub is not None and self.xp.any(wnew[idx(v)] > ub + tol):
                return True

        return False

    def _take_step(self, time_integrator: TimeIntegrator, dt_min: Optional[float] = None):
        """
        Update "u" and self.t by taking a single step in time with the specified time integrator
        and optional minimum dt constraint.
        """
        arrays = self.arrays
        t = self.t

        u = arrays["u"]
        unew = arrays["unew"]

        # Compute dt which may need to be clipped
        self.step_summary.timer.start("compute_dt", self.params.sync_timer)  # TIMER START
        dt = self.compute_dt(t, u)
        self._check_dt(dt)
        if dt_min is not None:
            dt = min(dt, dt_min)
        self.step_summary.timer.stop("compute_dt", self.params.sync_timer)  # TIMER STOP

        # Compute new state
        self._update_unew(t, u, dt, time_integrator)

        # Revise time-step size if needed
        if self.params.fv_scheme.zhang_shu_params.adaptive_dt:
            while self._check_unew_for_dt_revision():
                dt /= 2
                self._check_dt(dt)
                self._update_unew(t, u, dt, time_integrator)
                self.step_summary.n_dt_revisions += 1

        # Update the state
        u[...] = unew
        self.t += dt

    def _finish_run(self):
        output_path = self.params.output_path

        if output_path is None:
            return

        # Write step history to output path
        with open(output_path / "step_history.pkl", "wb") as f:
            pickle.dump(self.step_history, f)

        # Dump all snapshots and write snapshot history to output path
        for snapshot in self.snapshot_history:
            if snapshot.data is not None:
                snapshot.clear()
        with open(output_path / "snapshot_history.pkl", "wb") as f:
            pickle.dump(self.snapshot_history, f)

    def take_n_steps(
        self,
        n: int,
        time_integrator: TimeIntegrator = TimeIntegrator.SSPRK3,
        snapshot_mode: SnapshotMode = SnapshotMode.TARGET,
        print_update: bool = True,
        print_frequency: int = 100,
    ):
        self._start_wall_timer()

        print_frequency = max(1, print_frequency)
        for i in range(1, n + 1):
            take_snapshot_this_step = (
                snapshot_mode == SnapshotMode.TARGET and i == n
            ) or snapshot_mode == SnapshotMode.EVERY

            self._open_step()
            self._take_step(time_integrator)
            self._close_step(take_snapshot_this_step)

            if print_update and (i % print_frequency == 0 or i == n):
                self._print_message(self._build_message(current_steps=i, total_steps=n))

        if print_update:
            self._print_message(self._build_message(current_steps=n, total_steps=n) + " (done)\n")

        self._finish_run()

    def run(
        self,
        t: Union[float, List[float]],
        time_integrator: TimeIntegrator = TimeIntegrator.SSPRK3,
        snapshot_mode: SnapshotMode = SnapshotMode.TARGET,
        allow_overshoot: bool = False,
        print_update: bool = True,
        print_frequency: int = 100,
    ):
        self._start_wall_timer()
        print_frequency = max(1, print_frequency)

        if not isinstance(t, list):
            target_times = [t]
        else:
            if t != sorted(set(t)):
                raise ValueError("Target times must be given in sorted order without duplicates.")
            target_times = t
        tstop = target_times[-1]

        if any(targets < 0 for targets in target_times):
            raise ValueError("Target times must be non-negative.")

        while target_times:
            self._open_step()
            self._take_step(
                time_integrator, dt_min=None if allow_overshoot else target_times[0] - self.t
            )

            take_snapshot_this_step = snapshot_mode == SnapshotMode.EVERY
            if self.t >= target_times[0]:
                if snapshot_mode == SnapshotMode.TARGET:
                    take_snapshot_this_step = True
                target_times.pop(0)
            self._close_step(take_snapshot_this_step)

            n = len(self.step_history)
            if print_update and n % print_frequency == 0:
                self._print_message(
                    self._build_message(current_steps=n, current_time=self.t, stopping_time=tstop)
                )

        if print_update:
            self._print_message(
                self._build_message(current_steps=n, current_time=self.t, stopping_time=tstop)
                + " (done)\n"
            )

        self._finish_run()
