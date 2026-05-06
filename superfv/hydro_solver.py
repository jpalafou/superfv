import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .boundary_conditions import CallableBC
from .configs import (
    BoundaryCondition,
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
from .initial_conditions import square
from .tools.device_management import ArrayManager
from .tools.yaml_helper import yaml_dump


class HydroSolver:
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
        passive_ics: Optional[List[UnivarField]] = None,
        # BC params
        bcx: Tuple[BoundaryCondition, BoundaryCondition] = BoundaryCondition.PERIODIC,
        bcy: Tuple[BoundaryCondition, BoundaryCondition] = BoundaryCondition.NONE,
        bcz: Tuple[BoundaryCondition, BoundaryCondition] = BoundaryCondition.NONE,
        bcx_callable_lower: Optional[CallableBC] = None,
        bcx_callable_upper: Optional[CallableBC] = None,
        bcy_callable_lower: Optional[CallableBC] = None,
        bcy_callable_upper: Optional[CallableBC] = None,
        bcz_callable_lower: Optional[CallableBC] = None,
        bcz_callable_upper: Optional[CallableBC] = None,
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
        PAD_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        # Zhang-Shu params
        use_ZS: bool = False,
        adaptive_dt: bool = False,
        adaptive_dt_tol: float = 1e-15,
        theta_denom_tol: float = 1e-15,
        # shock detection params
        eta_max: float = 0.025,
        # NAD params
        use_NAD: bool = True,
        rtol: float = 1e-5,
        atol: float = 0.0,
        delta: bool = False,
        # MOOD params
        use_MOOD: bool = False,
        fallback_cascade: FallbackCascade = FallbackCascade.MUSCL0,
        max_revs: int = 1,
        skip_trouble_counts: bool = False,
        detect_closing_troubles: bool = True,
        # Shared Zhang-Shu / MOOD params
        include_corners: bool = True,
        log_limiter_scalars: bool = True,
        # Other params
        cupy: bool = False,
        sync_timer: bool = True,
    ):
        # Define the following attributes:
        self.arrays: ArrayManager
        self.params: SolverParams

        self.arrays = ArrayManager()

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

        # parse fallback scheme cascade since FV_SchemeParameters requires a fallback_cascade list
        null_SED = SmoothExtremaDetectionParameters(False)
        null_MUSCL = MUSCL_Parameters(False, MUSCL_SlopeLimiter.NONE, null_SED)
        null_PAD = PhysicalAdmissibilityParameters(False, np.empty((0,)))
        null_ZS = ZhangShuParameters(False, False, null_SED, null_PAD)
        null_NAD = NumericalAdmissibilityParameters(False, 0.0, 0.0, null_SED)
        null_MOOD = MOOD_Parameters(False, null_NAD, null_PAD, [], 0)
        null_shock = ShockDetectionParameters(False)

        fallback_cascade_list: List[FV_SchemeParameters] = []
        if fallback_cascade == FallbackCascade.FULL:
            for reduced_p in range(p, -1, -1):
                fallback_cascade_list.append(
                    FV_SchemeParameters(
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
        elif fallback_cascade in (FallbackCascade.MUSCL, FallbackCascade.MUSCL0):
            fallback_cascade_list.append(
                FV_SchemeParameters(
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
        if fallback_cascade == FallbackCascade.MUSCL0:
            fallback_cascade_list.append(
                FV_SchemeParameters(
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

        # FV_SchemeParameters requires a PAD bounds array
        self.define_PAD_array(PAD_bounds)

        SED_params = SmoothExtremaDetectionParameters(use_SED=use_SED, clip_zero_tol=clip_zero_tol)
        PAD_params = PhysicalAdmissibilityParameters(
            use_PAD=PAD_bounds is not None, PAD_bounds=self.arrays["PAD_bounds"]
        )
        shock_detection_params = ShockDetectionParameters(
            use_shock_detection=lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE, eta_max=eta_max
        )

        fv_scheme_params = FV_SchemeParameters(
            p=p,
            flux_recipe=flux_recipe,
            flux_quadrature=flux_quadrature,
            lazy_primitive_mode=lazy_primitive_mode,
            muscl_params=MUSCL_Parameters(
                use_MUSCL=use_MUSCL, MUSCL_limiter=MUSCL_limiter, SED_params=SED_params
            ),
            zhang_shu_params=ZhangShuParameters(
                use_ZS=use_ZS,
                adaptive_dt=adaptive_dt,
                SED_params=SED_params,
                PAD_params=PAD_params,
                adaptive_dt_tol=adaptive_dt_tol,
                theta_denom_tol=theta_denom_tol,
                include_corners=include_corners,
                log_limiter_scalars=log_limiter_scalars,
            ),
            mood_params=MOOD_Parameters(
                use_MOOD=use_MOOD,
                NAD_params=NumericalAdmissibilityParameters(
                    use_NAD=use_NAD,
                    rtol=rtol,
                    atol=atol,
                    SED_params=SED_params,
                    delta=delta,
                    include_corners=include_corners,
                ),
                PAD_params=PAD_params,
                fallback_cascade=fallback_cascade_list,
                max_revs=max_revs,
                skip_trouble_counts=skip_trouble_counts,
                detect_closing_troubles=detect_closing_troubles,
                log_limiter_scalars=log_limiter_scalars,
            ),
            shock_detection_params=shock_detection_params,
        )

        # MeshParameters requires nghost, which depends on the FV scheme
        nghost = self.compute_nghost(fv_scheme_params)
        mesh_params = MeshParameters(
            nx=nx, ny=ny, nz=nz, nghost=nghost, xlims=xlims, ylims=ylims, zlims=zlims
        )

        self.params = SolverParams(
            hydro=hydro_params,
            ic=ic_params,
            mesh=mesh_params,
            bc=bc_params,
            fv_scheme=fv_scheme_params,
            cupy=cupy,
            sync_timer=sync_timer,
        )

    def define_PAD_array(self, PAD_bounds: Optional[Dict[str, Tuple[float, float]]]):
        warnings.warn("Using dummy PAD bounds array for now.")
        PAD_array = np.empty((0,))
        self.arrays.add("PAD_bounds", PAD_array)

    def compute_nghost(self, fv_scheme_params: FV_SchemeParameters) -> int:
        warnings.warn("Using dummy nghost value for now.")
        return 0

    def write_config_file(self, path: str):
        with open(Path(path) / "config.yaml", "w") as f:
            f.write(yaml_dump(asdict(self.params)))
