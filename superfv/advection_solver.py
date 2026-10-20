import time
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
from .configs import HydroParameters, MUSCL_SlopeLimiter
from .field import MultivarField, SourceTerm, UnivarField
from .finite_volume_driver import compute_fv_dudt
from .hydro_solver import HydroSolver
from .mesh import UniformFiniteVolumeMesh
from .slope_limiting.mood import mood_loop
from .tools.device_management import ArrayLike
from .tools.snapshot import Snapshot


class AdvectionSolver(HydroSolver):
    def __init__(
        self,
        # Hydro params
        CFL: float = 0.8,
        nu: float = 0.0,
        Chi: float = 0.0,
        nu_dye: float = 0.0,
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
        super().__init__(
            # Hydro params
            gamma=np.nan,
            riemann_solver="upwind",
            CFL=CFL,
            nu=nu,
            Chi=Chi,
            nu_dye=nu_dye,
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
            flux_quadrature="transverse",
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
            omit_vars=omit_vars,
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

    def _summarize_step(self, take_snapshot: bool = False):
        step_summary = self.step_summary

        step_summary.t_wall = time.time() - self.t_wall_start

        if take_snapshot:
            self._take_snapshot()  # updates step_summary.timer

        self.step_history.append(step_summary)
        self._reset_step_summary(step_summary.step + 1)

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
