import warnings
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

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
from .explicit_ODE_solver import ExplicitODESolver
from .field import MultivarField, UnivarField
from .finite_volume_driver import (
    integrate_cell_averages,
    integrate_gauss_legendre_face_nodes,
    integrate_transverse_nodes,
    interpolate_cell_centers,
    interpolate_face_nodes,
)
from .hydro import cons_to_prim, prim_to_cons, sound_speed
from .initial_conditions import square
from .mesh import UniformFVMesh
from .riemann_solvers import hllc
from .stencils import conservative_interpolation
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager, xp
from .tools.slicing import crop, merge_slices
from .tools.variable_index_map import VariableIndexMap
from .tools.yaml_helper import yaml_dump

if CUPY_AVAILABLE:
    from .hydro import (
        make_cons_to_prim_elementwise_kernel,
        make_prim_to_cons_elementwise_kernel,
        sound_speed_cp,
    )
    from .riemann_solvers import make_hllc_elementwise_kernel


class HydroSolver(ExplicitODESolver):

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
        self.interior: Tuple[slice, slice, slice, slice]
        self.u0_func: MultivarField
        self.mesh: UniformFVMesh
        self.xp: ModuleType

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
        elif fallback_cascade in (FallbackCascade.MUSCL, FallbackCascade.MUSCL0):
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
        if fallback_cascade == FallbackCascade.MUSCL0:
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
            name="base_scheme",
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
            ndim=len(self._compute_active_dims(nx, ny, nz)),
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

        self._build_interior_slice()
        self._build_conservative_ic_u0_func()
        self._enable_ic_bc_or_none_if_inactive()
        self._build_mesh()
        self._init_cupy(cupy, CUPY_AVAILABLE)
        self._compute_ic_array_and_initialize_ODE_solver()
        self._allocate_arrays()

    def _define_PAD_array(self, PAD_bounds: Optional[Dict[str, Tuple[float, float]]]):
        warnings.warn("Using dummy PAD bounds array for now.")
        PAD_array = np.empty((0,))
        self.arrays.add("PAD_bounds", PAD_array)

    def _compute_nghost(self, fv_scheme_params: FV_SchemeParameters) -> int:
        warnings.warn("Using dummy nghost value for now.")
        return 2

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
            out = ic_params.ic(idx, x, y, z, t, xp=xp)
            if ic_params.npassives > 0:
                for v, func in ic_params.passive_ics.items():
                    out[idx(v)] = func(x, y, z, t, xp=xp)
            return out

        return w0_func

    def _build_interior_slice(self):
        active_dims = self.params.mesh.active_dims
        nghost = self.params.mesh.nghost
        self.interior = merge_slices(
            *[crop({"x": 1, "y": 2, "z": 3}[dim], (nghost, -nghost), ndim=4) for dim in active_dims]
        )

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

    def _compute_ic_array_and_initialize_ODE_solver(self):
        params = self.params
        idx = params.variable_index_map
        nvars = idx.nvars
        mesh = self.mesh
        nx, ny, nz = mesh.shape
        arrays = self.arrays

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

        ExplicitODESolver.__init__(self, u0, arrays, params.hydro.dt_min)  # adds "u0" to arrays

    def _compute_ninterps_per_face(
        self, fv_scheme: FV_SchemeParameters, quantity: Literal["nodes", "lines"]
    ) -> int:
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
        fv_scheme = self.params.fv_scheme
        mesh = self.mesh
        nx, ny, nz = mesh.shape
        _nx_, _ny_, _nz_ = mesh._shape_
        active_dims = mesh.active_dims
        ndim = mesh.ndim
        arrays = self.arrays

        n_lines = self._compute_ninterps_per_face(fv_scheme, "lines")
        n_nodes = self._compute_ninterps_per_face(fv_scheme, "nodes")

        # define cell-centered/cell-averaged arrays
        arrays.add("dudt", np.empty((nvars, nx, ny, nz)))
        arrays.add("_u_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_ucc_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_wcc_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_wp_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_w1_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_w_", np.empty((nvars, _nx_, _ny_, _nz_)))

        # define arrays associated faces along the x-direction
        if "x" in active_dims:
            arrays.add("_x_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, 2 * n_nodes)))
            arrays.add("_f_nodes_", np.empty((nvars, nx + 1, _ny_, _nz_, n_nodes)))
            arrays.add("_F_", np.empty((nvars, nx + 1, _ny_, _nz_, 1)))
            arrays.add("F", np.empty((nvars, nx + 1, ny, nz)))
            if ndim == 3:
                arrays.add("_f_lines_", np.empty((nvars, nx + 1, _ny_, _nz_, n_lines)))

        # define arrays associated with faces along the y-direction
        if "y" in active_dims:
            arrays.add("_y_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, 2 * n_nodes)))
            arrays.add("_g_nodes_", np.empty((nvars, _nx_, ny + 1, _nz_, n_nodes)))
            arrays.add("_G_", np.empty((nvars, _nx_, ny + 1, _nz_, 1)))
            arrays.add("G", np.empty((nvars, nx, ny + 1, nz)))
            if ndim == 3:
                arrays.add("_g_lines_", np.empty((nvars, _nx_, ny + 1, _nz_, n_lines)))

        # define arrays associated with faces along the z-direction
        if "z" in active_dims:
            arrays.add("_z_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, 2 * n_nodes)))
            arrays.add("_h_nodes_", np.empty((nvars, _nx_, _ny_, nz + 1, n_nodes)))
            arrays.add("_H_", np.empty((nvars, _nx_, _ny_, nz + 1, 1)))
            arrays.add("H", np.empty((nvars, nx, ny, nz + 1)))
            if ndim == 3:
                arrays.add("_h_lines_", np.empty((nvars, _nx_, _ny_, nz + 1, n_lines)))

        # define buffer and interpolation arrays
        arrays.add("_buffer1_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        if ndim >= 2:
            if fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                arrays.add("_faces_", np.empty((nvars, _nx_, _ny_, _nz_, 2)))
            arrays.add("_midline_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        if ndim == 3:
            if fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                ngl = conservative_interpolation.n_gauss_legendre_nodes(fv_scheme.p)
                arrays.add("_lines_", np.empty((nvars, _nx_, _ny_, _nz_, 2 * ngl)))
            arrays.add("_midface_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))

        # define slope-limiting arrays
        arrays.add("_flux_jvp_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_M_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_m_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_alpha_", np.ones((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_eta_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_has_shock_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))
        if "x" in active_dims:
            arrays.add("_xslopes_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        if "y" in active_dims:
            arrays.add("_yslopes_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        if "z" in active_dims:
            arrays.add("_zslopes_", np.zeros((nvars, _nx_, _ny_, _nz_)))

        # define Zhang-Shu limiter arrays
        arrays.add("_theta_", np.ones((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("theta_log", np.ones((nvars, nx, ny, nz)))
        arrays.add("_Mj_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_mj_", np.empty((nvars, _nx_, _ny_, _nz_)))

        total_nodes = 1 + self.mesh.ndim * 2 * n_nodes
        arrays.add("_wj_all_", np.empty((nvars, _nx_, _ny_, _nz_, total_nodes)))

        # define MOOD arrays
        for scheme in fv_scheme.mood_params.fallback_cascade:
            if "x" in active_dims:
                arrays.add("F_" + scheme.name, np.empty((nvars, nx + 1, ny, nz)))
            if "y" in active_dims:
                arrays.add("G_" + scheme.name, np.empty((nvars, nx, ny + 1, nz)))
            if "z" in active_dims:
                arrays.add("H_" + scheme.name, np.empty((nvars, nx, ny, nz + 1)))

        arrays.add("_unew_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_wnew_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_NAD_violations_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_PAD_violations_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_troubles_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))
        arrays.add("_troubles2_", np.zeros((1, _nx_, _ny_, _nz_), dtype=np.int32))
        arrays.add("_cascade_idx_", np.zeros((1, _nx_, _ny_, _nz_), dtype=int))
        arrays.add("_blended_cascade_idx_", np.zeros((1, _nx_, _ny_, _nz_)))
        arrays.add("_mask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1), dtype=int))
        arrays.add("_fmask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1)))
        arrays.add("troubles_log", np.zeros((1, nx, ny, nz)))
        arrays.add("cascade_idx_log", np.zeros((1, nx, ny, nz)))

    # Helper functions

    @cached_property
    def _prim_to_cons_cp(self):
        return make_prim_to_cons_elementwise_kernel(self.params.ic.npassives)

    @cached_property
    def _cons_to_prim_cp(self):
        return make_cons_to_prim_elementwise_kernel(self.params.ic.npassives)

    @cached_property
    def _riemann_solver_func(self):
        rs = self.params.hydro.riemann_solver

        if self.params.cupy:
            if rs == RiemannSolver.HLLC:
                return make_hllc_elementwise_kernel(self.params.ic.npassives)
            else:
                raise NotImplementedError(
                    f"CuPy kernel for Riemann solver '{rs}' is not implemented."
                )
        else:
            if rs == RiemannSolver.HLLC:
                return hllc
            else:
                raise NotImplementedError(f"Riemann solver '{rs}' is not implemented.")

    # Hydro solver functions

    def primitives_to_conservatives(self, w: ArrayLike, u: ArrayLike):
        """
        Write conservatives to `u` using primitives from `u`.
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
        Write to primitives `w` from conservatives `u`.
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
        Write sound speed to `c` from primitives `w`.
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

        w = self.xp.empty_like(u)
        c = self.xp.empty_like(u[0, ...])
        sum_of_s_over_h = self.xp.zeros_like(u[0, ...])

        self.conservatives_to_primitives(u, w)
        self.compute_sound_speed(w, c)

        for dim, h in zip(["x", "y", "z"], [params.mesh.hx, params.mesh.hy, params.mesh.hz]):
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

    def update_cell_centers_and_primitive_cell_averages(
        self, t: float, u: ArrayLike, fv_scheme: FV_SchemeParameters
    ):
        idx = self.params.variable_index_map
        active_dims = self.mesh.active_dims
        ndim = self.mesh.ndim
        na = self.xp.newaxis
        arrays = self.arrays

        # allocate arrays
        _u_ = arrays["_u_"]
        _ucc_ = arrays["_ucc_"]  # shape (..., 1)
        _wcc_ = arrays["_wcc_"]  # shape (..., 1)
        _wp_ = arrays["_wp_"]  # shape (..., 1)
        _w1_ = arrays["_w1_"]
        _w_ = arrays["_w_"]
        _temp1_ = arrays["_midline_"] if ndim >= 2 else None  # shape (..., 1) or None
        _temp2_ = arrays["_midface_"] if ndim == 3 else None  # shape (..., 1) or None

        # 0) conservatives FV averages + BC
        _u_[self.interior] = u
        self.apply_bc(_u_, t, fv_scheme.p)

        # 1) conservative and primitive centroids
        interpolate_cell_centers(_u_[..., na], _ucc_, active_dims, fv_scheme.p, _temp1_, _temp2_)
        self.conservatives_to_primitives(_ucc_, _wcc_)

        # 2) primitive FV averages
        self.conservatives_to_primitives(_u_, _w1_)

        match fv_scheme.lazy_primitive_mode:
            case LazyPrimitiveMode.NONE:
                integrate_cell_averages(_wcc_, _wp_, active_dims, fv_scheme.p, _temp1_, _temp2_)
                _w_[...] = _wp_[..., 0]
            case LazyPrimitiveMode.FULL:
                _w_[...] = _w1_
            case LazyPrimitiveMode.ADAPTIVE:
                raise NotImplementedError("Adaptive lazy primitives not implemented yet.")

        # ensure density is always transformed in the lazy way
        if fv_scheme.lazy_primitive_mode != LazyPrimitiveMode.FULL:
            _w_[idx("rho"), ...] = _w1_[idx("rho"), ...]

    def _convert_nodes_to_primitives(self):
        params = self.params
        arrays = self.arrays

        for dim in params.mesh.active_dims:
            _nodes_ = arrays[f"_{dim}_nodes_"]
            self.conservatives_to_primitives(_nodes_, _nodes_)

    def _ensure_positive_nodes(self):
        params = self.params
        idx = params.variable_index_map
        arrays = self.arrays

        for dim in params.mesh.active_dims:
            _nodes_ = arrays[f"_{dim}_nodes_"]
            self.xp.maximum(_nodes_[idx("rho")], params.hydro.rho_min, out=_nodes_[idx("rho")])
            self.xp.maximum(_nodes_[idx("P")], params.hydro.P_min, out=_nodes_[idx("P")])

    def riemann_solver(
        self,
        left_of_interface: ArrayLike,
        right_of_interface: ArrayLike,
        fluxes: ArrayLike,
        dim: str,
    ):
        params = self.params
        idx = params.variable_index_map

        if params.cupy:
            cl = self.xp.empty_like(left_of_interface[0, ...])
            cr = self.xp.empty_like(right_of_interface[0, ...])

            self.compute_sound_speed(left_of_interface, cl)
            self.compute_sound_speed(right_of_interface, cr)

            self._riemann_solver_func(
                left_of_interface[idx("rho")],
                right_of_interface[idx("rho")],
                left_of_interface[idx("vx")],
                right_of_interface[idx("vx")],
                left_of_interface[idx("vy")],
                right_of_interface[idx("vy")],
                left_of_interface[idx("vz")],
                right_of_interface[idx("vz")],
                left_of_interface[idx("P")],
                right_of_interface[idx("P")],
                cl,
                cr,
                params.hydro.gamma,
                {"x": 1, "y": 2, "z": 3}[dim],
                *[
                    x
                    for v in idx.group_var_map.get("passives", [])
                    for x in (left_of_interface[idx(v)], right_of_interface[idx(v)])
                ],
                fluxes[idx("rho")],
                fluxes[idx("mx")],
                fluxes[idx("my")],
                fluxes[idx("mz")],
                fluxes[idx("E")],
                *[fluxes[idx(v)] for v in idx.group_var_map.get("passives", [])],
            )
        else:
            fluxes[...] = self._riemann_solver_func(
                idx,
                left_of_interface,
                right_of_interface,
                dim,
                params.hydro.gamma,
                params.hydro.isothermal,
                params.hydro.iso_cs,
            )

    def _update_nodal_fluxes(self, fv_scheme: FV_SchemeParameters, dim: Literal["x", "y", "z"]):
        params = self.params
        nghost = params.mesh.nghost
        nnodes = self._compute_ninterps_per_face(fv_scheme, "nodes")
        axis = {"x": 1, "y": 2, "z": 3}[dim]
        arrays = self.arrays

        _nodes_ = arrays[f"_{dim}_nodes_"]
        _fnodes_ = arrays[{"x": "_f_nodes_", "y": "_g_nodes_", "z": "_h_nodes_"}[dim]]

        left_of_interface = merge_slices(
            crop(4, (nnodes, 2 * nnodes), ndim=5), crop(axis, (nghost - 1, -nghost), ndim=5)
        )
        right_of_interface = merge_slices(
            crop(4, (None, nnodes), ndim=5), crop(axis, (nghost, -nghost + 1), ndim=5)
        )

        self.riemann_solver(_nodes_[left_of_interface], _nodes_[right_of_interface], _fnodes_, dim)

    def _update_flux_arrays(self):
        params = self.params
        nghost = self.params.mesh.nghost
        arrays = self.arrays

        for dim in params.mesh.active_dims:
            _fnodes_ = arrays[{"x": "_f_nodes_", "y": "_g_nodes_", "z": "_h_nodes_"}[dim]]
            _Fluxes_ = arrays[{"x": "_F_", "y": "_G_", "z": "_H_"}[dim]]
            Fluxes = arrays[{"x": "F", "y": "G", "z": "H"}[dim]]

            if params.mesh.ndim == 1:
                Fluxes[...] = _fnodes_[..., 0]  # no need for flux integral in 1D
                continue

            interior = merge_slices(
                *[
                    crop(axis, (nghost, -nghost), ndim=5)
                    for d, axis in zip(["x", "y", "z"], [1, 2, 3])
                    if d in params.mesh.active_dims and d != dim
                ],
                (slice(None), slice(None), slice(None), slice(None), 0),
            )
            Fluxes[...] = _Fluxes_[interior]

    def update_fluxes(self, fv_scheme: FV_SchemeParameters):
        params = self.params
        active_dims = params.mesh.active_dims
        na = self.xp.newaxis
        arrays = self.arrays

        _q_ = arrays["_w_"] if fv_scheme.flux_recipe == FluxRecipe.PRIM_PRIM_LIM else arrays["_u_"]
        _qtemp1_ = arrays["_faces_"] if params.mesh.ndim >= 2 else None
        _qtemp2_ = arrays["_lines_"] if params.mesh.ndim == 3 else None

        for dim in params.mesh.active_dims:
            _nodes_ = arrays[f"_{dim}_nodes_"]
            interpolate_face_nodes(
                _q_[..., na],
                _nodes_,
                dim,
                active_dims,
                fv_scheme.p,
                fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE,
                _qtemp1_,
                _qtemp2_,
            )

        if fv_scheme.flux_recipe == FluxRecipe.CONS_PRIM_LIM:
            self._convert_nodes_to_primitives()

        # slope limiting goes here
        if fv_scheme.zhang_shu_params.use_ZS:
            raise NotImplementedError("Zhang-Shu limiter not implemented yet.")

        if fv_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            self._convert_nodes_to_primitives()

        self._ensure_positive_nodes()
        for dim in params.mesh.active_dims:
            _fnodes_ = arrays[{"x": "_f_nodes_", "y": "_g_nodes_", "z": "_h_nodes_"}[dim]]
            _Fluxes_ = arrays[{"x": "_F_", "y": "_G_", "z": "_H_"}[dim]]

            self._update_nodal_fluxes(fv_scheme, dim)

            if params.mesh.ndim == 1:
                continue
            elif fv_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
                integrate_gauss_legendre_face_nodes(
                    _fnodes_, _Fluxes_, dim, active_dims, fv_scheme.p
                )
            else:
                _Ftemp_ = (
                    arrays[{"x": "_f_lines_", "y": "_g_lines_", "z": "_h_lines_"}[dim]]
                    if params.mesh.ndim == 3
                    else None
                )
                integrate_transverse_nodes(
                    _fnodes_, _Fluxes_, dim, active_dims, fv_scheme.p, _Ftemp_
                )

        self._update_flux_arrays()

    # Useful user functions

    def write_config_file(self, path: str):
        with open(Path(path) / "config.yaml", "w") as f:
            f.write(yaml_dump(asdict(self.params)))

    # ODE solver functions

    def f(self, t: float, u: ArrayLike) -> ArrayLike:
        warnings.warn("Using dummy f")
        return 0.0 * u

    def build_opening_message(self) -> str:
        return "dummy opening message"

    def build_update_message(self) -> str:
        return "dummy update message"

    def build_closing_message(self) -> str:
        return "dummy closing message"

    def prepare_snapshot_data(self) -> Any:
        warnings.warn("Using dummy snapshot data")
        return None
