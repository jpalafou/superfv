from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from superfv.tools.variable_index_map import VariableIndexMap

from . import initial_conditions as ics
from .advection_driver import (
    compute_advection_dt,
    update_advection_fluxes,
    update_advection_workspace,
)
from .boundary_conditions import BC, PatchBC
from .configs import FluxQuadrature, HydroParameters, MUSCL_SlopeLimiter
from .field import MultivarField, SourceTerm, UnivarField
from .finite_volume_driver import compute_fv_dudt
from .hydro_solver import HydroSolver
from .mesh import UniformFiniteVolumeMesh
from .slope_limiting.mood import mood_loop
from .tools.device_management import ArrayLike
from .tools.snapshot import Snapshot


class AdvectionSolver(HydroSolver):
    """
    SuperFV solver for the advection equation in 1D, 2D, or 3D.

    Initialize the solver with the desired parameters:

        from superfv import AdvectionSolver, ics
        sim = AdvectionSolver(ic=ics.square, ...)

    Then call the `take_n_steps` or `run` methods to advance the simulation in time. For
    example, to take 10 steps:

        sim.take_n_steps(n=10, ...)

    or to run until t=1.0:

        sim.run(t=1.0, ...)

    Refer to the `step_history` to access the simulation history and the `snapshot_history`
    to access the snapshots. For example, to get the minimum density across all steps:

        sim.step_history.get_history("rho_min")

    or to get the density field from the 5th snapshot:

        idx = sim.params.variable_index_map
        sim.snapshot_history[5].u[idx("rho")]

    or to get a passive field named "dye" at time t=1.0:

        sim.snapshot_history(1.0).u[idx("dye")]

    """

    def __init__(
        self,
        # Hydro params
        CFL: float = 0.8,
        dt_min: float = 1e-15,
        # IC params
        ic: MultivarField = partial(ics.square, vx=1),
        passive_ics: Optional[Dict[str, UnivarField]] = None,
        sampling_p: Optional[int] = None,
        # Source term params
        source: Optional[SourceTerm] = None,
        # BC params
        bcx: Tuple[BC, BC] = ("periodic", "periodic"),
        bcy: Tuple[BC, BC] = ("periodic", "periodic"),
        bcz: Tuple[BC, BC] = ("periodic", "periodic"),
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
        flux_quadrature: FluxQuadrature = "transverse",
        # SED params
        use_SED: bool = True,
        clip_zero_tol: float = 1e-15,
        # MUSCL params
        use_MUSCL: bool = False,
        MUSCL_limiter: MUSCL_SlopeLimiter = "moncen",
        # PAD params
        PAD_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        # Zhang-Shu params
        use_ZS: bool = False,
        adaptive_dt: bool = False,
        adaptive_dt_tol: float = 1e-15,
        theta_denom_tol: float = 1e-15,
        # NAD params
        use_NAD: bool = True,
        rtol: float = 1e-5,
        atol: float = 0.0,
        delta: bool = False,
        # MOOD params
        use_MOOD: bool = False,
        fallback_cascade: Literal["full", "muscl", "muscl0", "first_order"] = "muscl",
        max_revs: int = 1,
        blend_troubles: bool = False,
        skip_trouble_counts: bool = False,
        detect_closing_troubles: bool = True,
        # Shared Zhang-Shu / MOOD params
        omit_vars: Optional[List[str]] = None,
        include_corners: bool = True,
        use_global_bounds: bool = False,
        # Solver params
        cupy: bool = False,
        profile: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        discard_after_writing: bool = True,
        overwrite: bool = False,
        bonus_snapshot_routine: Optional[Callable[[Snapshot], None]] = None,
    ):
        """
        Initialize the AdvectionSolver with the specified parameters.

        Advection parameters:
            CFL: CFL number for time step calculation.
            dt_min: Minimum allowed time step size.

        Initial condition parameters:
            ic: Initial condition function that takes (idx, x, y, z, t, xp=xp) and returns an
                array of shape (nvars, nx, ny, nz).
            passive_ics: Optional dictionary with keys as variable names and values as functions
                that take (x, y, z, t, xp=xp) and return an array of shape (nx, ny, nz).
            sampling_p: If specified, use this polynomial degree for Gauss-Legendre quadrature when
                computing cell averages from the initial condition or Dirichlet boundary
                conditions. If None, defaults to the polynomial degree `p`.

        Source term parameters:
            source: Optional source term function that takes (idx, u, xp=xp) and returns an array
                of shape (nvars, nx, ny, nz).

        Boundary condition parameters:
            bcx, bcy, bcz: Boundary conditions for x, y, z directions. Each is a tuple of two
                boundary condition literals (lower, upper).
                "periodic": Wrap ghost cells from the opposite side; both sides must be periodic.
                "dirichlet": Fill ghost cells from the corresponding boundary callable.
                "free": Copy the adjacent interior value; useful for outflow boundaries.
                "symmetric": Mirror interior values into ghost cells.
                "reflective": Mirror interior values and flip the normal velocity.
                "zeros": Fill ghost cells with 0.
                "ones": Fill ghost cells with 1.
                "patch": Let a `PatchBC` callable mutate the boundary region directly.
                "none": Leave the boundary untouched; used internally for inactive dimensions.
                "ic": Use the initial condition as a Dirichlet boundary.
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
            flux_quadrature: Flux quadrature literal.
                "transverse": Use transverse face-node integration in 2D and 3D runs.
                "gauss_legendre": Use Gauss-Legendre face quadrature in 2D and 3D runs.
                "none": Use no face quadrature; required and selected automatically in 1D.

        Slope limiting parameters (smooth extrema detection):
            use_SED: If True, enable smooth extrema detection.
            clip_zero_tol: Tolerance for clipping near-zero values in SED.

        Slope limiting parameters (MUSCL):
            use_MUSCL: If True, enable MUSCL scheme.
            MUSCL_limiter: Slope limiter literal.
                "minmod": Most diffusive TVD limiter.
                "moncen": Monotonized central limiter; less diffusive default.
                "pp2d": Positivity-preserving 2D limiter; requires exactly two active dimensions.
                "none": Use unlimited centered slopes.

        Slope limiting parameters (physical admissibility detection):
            PAD_bounds: Optional dictionary specifying lower and upper bounds for physical
                admissibility detection. Keys are variable names, and values are tuples of
                (lower_bound, upper_bound).

        Slope limiting parameters (Zhang-Shu):
            use_ZS: If True, enable Zhang-Shu limiter.
            adaptive_dt: If True, enable adaptive time stepping based on Zhang-Shu limiter.
            adaptive_dt_tol: Tolerance for adaptive time stepping.
            theta_denom_tol: Tolerance for denominator in theta calculation.

        Slope limiting parameters (MOOD, numerical admissibility detection):
            use_NAD: If True, enable numerical admissibility detection.
            rtol: Relative tolerance for numerical admissibility detection.
            atol: Absolute tolerance for numerical admissibility detection.
            delta: If True, enable delta-based numerical admissibility detection.

        Slope limiting parameters (MOOD):
            use_MOOD: If True, enable MOOD scheme.
            fallback_cascade: Fallback cascade literal.
                "full": Fall back through progressively lower polynomial degrees down to p=0.
                "muscl": Fall back to a MUSCL scheme.
                "muscl0": Fall back to MUSCL, then to first-order.
                "first_order": Fall back directly to p=0.
            max_revs: Maximum number of revisions allowed in MOOD scheme. Must be at least the
                number of fallback schemes in the cascade.
            blend_troubles: If True, blend troubled cells in MOOD scheme.
            skip_trouble_counts: If True, skip counting troubled cells in MOOD scheme.
            detect_closing_troubles: If True, detect closing troubled cells in MOOD scheme.

        Slope limiting parameters (shared between Zhang-Shu and MOOD):
            omit_vars: Optional list of variable names to omit from Zhang-Shu limiter or NAD. Note
                that PAD may still be active in other parts of the code, such as the Zhang-Shu
                adaptive time step check or MOOD troubled cell detection.
            include_corners: If True, include corner cells when computing the discrete maximum
                principle.
            use_global_bounds: If True, set M and m to global bounds determined by the provided
                `PAD_bounds` in Zhang-Shu or MOOD schemes. Cannot be used with SED and turns off
                PAD in MOOD schemes. Expects all simulation variables to be included in either
                `PAD_bounds` or `omit_vars`.

        Solver parameters:
            cupy: If True, use CuPy for GPU acceleration (requires CuPy to be installed).
            profile: If True, many timers are added to the solver beyond "take_step".
            output_path: Optional path to save simulation outputs. If None, no outputs are saved.
            discard_after_writing: If True, discard arrays after writing to disk to save memory.
            overwrite: If True, overwrite existing output directory if it exists.
            bonus_snapshot_routine: Optional callable function that takes a `Snapshot` object and
                performs additional operations on it after a snapshot is taken. If None, no
                additional operations are performed.
        """
        super().__init__(
            # Hydro params
            gamma=np.nan,
            riemann_solver="upwind",
            CFL=CFL,
            nu=0.0,
            Chi=0.0,
            nu_dye=0.0,
            dt_min=dt_min,
            positivity_guard=False,
            rho_min=np.nan,
            P_min=np.nan,
            isothermal=False,
            iso_cs=np.nan,
            # IC params
            ic=ic,
            passive_ics=passive_ics,
            sampling_p=sampling_p,
            # Source term params
            source=source,
            # BC params
            bcx=bcx,
            bcy=bcy,
            bcz=bcz,
            bcx_callable_lower=bcx_callable_lower,
            bcx_callable_upper=bcx_callable_upper,
            bcy_callable_lower=bcy_callable_lower,
            bcy_callable_upper=bcy_callable_upper,
            bcz_callable_lower=bcz_callable_lower,
            bcz_callable_upper=bcz_callable_upper,
            # Mesh params
            nx=nx,
            ny=ny,
            nz=nz,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            # Finite volume scheme params
            p=p,
            flux_recipe="cons_prim_lim",
            flux_quadrature=flux_quadrature,
            lazy_primitive_mode="full",
            # SED params
            use_SED=use_SED,
            clip_zero_tol=clip_zero_tol,
            # MUSCL params
            use_MUSCL=use_MUSCL,
            MUSCL_limiter=MUSCL_limiter,
            # PAD params
            PAD_bounds=PAD_bounds,
            # Zhang-Shu params
            use_ZS=use_ZS,
            adaptive_dt=adaptive_dt,
            adaptive_dt_tol=adaptive_dt_tol,
            theta_denom_tol=theta_denom_tol,
            # shock detection params
            eta_max=np.nan,
            # NAD params
            use_NAD=use_NAD,
            rtol=rtol,
            atol=atol,
            delta=delta,
            # MOOD params
            use_MOOD=use_MOOD,
            fallback_cascade=fallback_cascade,
            fallback_riemann_solver=None,
            max_revs=max_revs,
            blend_troubles=blend_troubles,
            skip_trouble_counts=skip_trouble_counts,
            detect_closing_troubles=detect_closing_troubles,
            # Shared Zhang-Shu / MOOD params
            omit_vars=["vx", "vy", "vz", "P"] + (omit_vars if omit_vars is not None else []),
            include_corners=include_corners,
            use_global_bounds=use_global_bounds,
            # Solver params
            cupy=cupy,
            profile=profile,
            output_path=output_path,
            discard_after_writing=discard_after_writing,
            overwrite=overwrite,
            bonus_snapshot_routine=bonus_snapshot_routine,
        )

    def _configure_PAD_bounds(
        self,
        PAD_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        hydro_params: HydroParameters,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        if PAD_bounds is None:
            PAD_bounds = {}
        return {v: b for v, b in PAD_bounds.items() if v not in {"v", "vx", "vy", "vz", "P", "E"}}

    def _build_conservative_ic_u0_func(self):
        w0_func = self._primitive_ic_w0_func()

        def u0_func(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: float,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            out = w0_func(idx, x, y, z, t, xp=xp)
            out[idx("P")] = np.nan  # P and E share an unused slot for the Riemann solver.
            return out

        self.w0_func = u0_func
        self.u0_func = u0_func

    def _compute_dt(
        self,
        u: ArrayLike,
        idx: VariableIndexMap,
        mesh: UniformFiniteVolumeMesh,
        hydro_params: HydroParameters,
    ) -> float:
        return compute_advection_dt(u, idx, mesh, hydro_params.CFL)

    def _check_unew_for_dt_revision(self) -> bool:
        params = self.params
        tol = params.fv_scheme.zhang_shu_params.adaptive_dt_tol
        idx = params.variable_index_map

        if not params.fv_scheme.zhang_shu_params.adaptive_dt:
            raise RuntimeError(
                "DT revision check called but adaptive_dt is not enabled in the parameters."
            )

        unew = self.arrays["unew"]

        for v, (lb, ub) in params.fv_scheme.zhang_shu_params.PAD_params.bounds.items():
            if lb is not None and self.xp.any(unew[idx(v)] < lb - tol):
                return True
            if ub is not None and self.xp.any(unew[idx(v)] > ub + tol):
                return True

        return False

    def f(self, t: float, u: ArrayLike, dt: float, hancock: bool = False) -> ArrayLike:
        with self._time("f"):
            params = self.params
            idx = params.variable_index_map
            base_scheme = params.fv_scheme
            bc_params = params.bc
            mesh = self.mesh
            active_dims = params.mesh.active_dims
            arrays = self.arrays

            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                update_advection_workspace(
                    u,
                    arrays["_u_"],
                    t,
                    idx,
                    mesh,
                    bc_params,
                    self.step_summary.timer if self.params.profile else None,
                )

                update_advection_fluxes(
                    arrays["_u_"],
                    arrays["_F_"] if "x" in active_dims else np.array([]),
                    arrays["_G_"] if "y" in active_dims else np.array([]),
                    arrays["_H_"] if "z" in active_dims else np.array([]),
                    arrays["_theta_"] if base_scheme.zhang_shu_params.use_ZS else np.array([]),
                    arrays["_qcc_"] if base_scheme.zhang_shu_params.use_ZS else np.array([]),
                    arrays["_alpha_"],
                    idx,
                    active_dims,
                    mesh,
                    base_scheme,
                    dt if hancock else 0.0,
                    self.step_summary.timer if self.params.profile else None,
                )

                # Compute source term
                if self.params.source is not None:
                    arrays["source"][...] = self.params.source(idx, u, xp=self.xp)

                if base_scheme.mood_params.use_MOOD:
                    with self._time("mood_loop"):
                        # Primitive and conservative states are identical for advection.
                        mood_loop(
                            arrays["_u_"],
                            arrays["_u_"],
                            arrays["_u_"],
                            arrays["_unew_"],
                            arrays["_unew_"],
                            arrays["_alpha_"],
                            np.array([]),  # fallback schemes could in principle use ZS
                            np.array([]),  # fallback schemes could in principle use ZS
                            arrays["_troubles_"],
                            arrays["revisable_troubles"],
                            arrays["_cascade_idx_"],
                            arrays["_F_"] if "x" in active_dims else np.array([]),
                            arrays["_G_"] if "y" in active_dims else np.array([]),
                            arrays["_H_"] if "z" in active_dims else np.array([]),
                            arrays["_F_fallback_"] if "x" in active_dims else np.array([]),
                            arrays["_G_fallback_"] if "y" in active_dims else np.array([]),
                            arrays["_H_fallback_"] if "z" in active_dims else np.array([]),
                            arrays["source"],
                            idx,
                            t,
                            dt,
                            mesh,
                            base_scheme,
                            bc_params,
                            self.substep_summary,
                            None,
                            self.step_summary.timer if self.params.profile else None,
                        )

            with self._time("compute_dudt"):
                return compute_fv_dudt(
                    arrays["F"] if "x" in active_dims else np.array([]),
                    arrays["G"] if "y" in active_dims else np.array([]),
                    arrays["H"] if "z" in active_dims else np.array([]),
                    arrays["source"],
                    mesh,
                )
