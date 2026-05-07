import warnings
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .boundary_conditions import BC, PatchBC
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
from .hydro import prim_to_cons
from .initial_conditions import square
from .mesh import UniformFVMesh
from .stencils import conservative_interpolation
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager
from .tools.variable_index_map import VariableIndexMap
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
        self.mesh_arrays: ArrayManager
        self.params: SolverParams
        self.u0_func: MultivarField
        self.mesh: UniformFVMesh

        self.arrays = ArrayManager()
        self.mesh_arrays = ArrayManager()

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
        self._define_PAD_array(PAD_bounds)

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

        mesh_params = MeshParameters(
            nx=nx,
            ny=ny,
            nz=nz,
            nghost=self._compute_nghost(fv_scheme_params),
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            active_dims=self._compute_active_dims(nx, ny, nz),
        )

        self.params = SolverParams(
            hydro=hydro_params,
            ic=ic_params,
            mesh=mesh_params,
            bc=bc_params,
            fv_scheme=fv_scheme_params,
            variable_index_map=self._define_complete_variable_index_map(ic_params),
            cupy=cupy and CUPY_AVAILABLE,
            sync_timer=sync_timer,
        )

        self._build_conservative_ic_u0_func()
        self._enable_ic_bc_or_none_if_inactive()
        self._build_mesh()
        self._allocate_arrays()
        self._init_cupy()

    def _define_PAD_array(self, PAD_bounds: Optional[Dict[str, Tuple[float, float]]]):
        warnings.warn("Using dummy PAD bounds array for now.")
        PAD_array = np.empty((0,))
        self.arrays.add("PAD_bounds", PAD_array)

    def _compute_nghost(self, fv_scheme_params: FV_SchemeParameters) -> int:
        warnings.warn("Using dummy nghost value for now.")
        return 0

    def _compute_active_dims(self, nx: int, ny: int, nz: int) -> Tuple[Literal["x", "y", "z"], ...]:
        dims = ["x", "y", "z"]
        return tuple(dim for dim, n in zip(dims, [nx, ny, nz]) if n > 1)

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
        self, ic_params: InitialConditionParameters
    ) -> VariableIndexMap:
        idx = self._define_base_variable_index_map()

        if ic_params.npassives > 0:
            for v in ic_params.passive_ics.keys():
                if v not in idx.var_idx_map:
                    idx.add_var(v, idx.nvars)
            idx.add_var_to_group("passives", ic_params.passive_ics.keys())

        return idx

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
            out = self.ic(idx, x, y, z, t, xp=xp)
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
        params = self.params.mesh

        self.mesh = UniformFVMesh(
            nx=params.nx,
            ny=params.ny,
            nz=params.nz,
            nghost=params.nghost,
            xlims=params.xlims,
            ylims=params.ylims,
            zlims=params.zlims,
            active_dims=params.active_dims,
            array_manager=self.mesh_arrays,
        )

    def _compute_ninterps_per_face(self, quantity: Literal["nodes", "lines"]) -> int:
        fv_scheme = self.params.fv_scheme
        ndim = self.mesh.ndim
        n_gauss_legendre = conservative_interpolation.n_gauss_legendre_nodes(fv_scheme.p)

        if quantity == "nodes":
            if fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                return n_gauss_legendre ** (ndim - 1)
            else:
                return 1
        elif quantity == "lines":
            if ndim == 1:
                return 0
            if fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                return n_gauss_legendre ** (ndim - 2)
            else:
                return 1
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

    def _allocate_arrays(self):
        nvars = self.params.variable_index_map.nvars
        nx, ny, nz = self.mesh.shape
        _nx_, _ny_, _nz_ = self.mesh._shape_

        arrays = self.arrays

        arrays.add("u", np.empty((nvars, nx, ny, nz)))

    def _init_cupy(self):
        if self.params.cupy:
            self.arrays.transfer_to("gpu")
            self.mesh_arrays.transfer_to("gpu")

    def write_config_file(self, path: str):
        with open(Path(path) / "config.yaml", "w") as f:
            f.write(yaml_dump(asdict(self.params)))

    def primitives_to_conservatives(self, w: ArrayLike, u: ArrayLike):
        idx = self.variable_index_map
        params = self.params

        if params.cupy:
            self.prim_to_cons_cp(
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
