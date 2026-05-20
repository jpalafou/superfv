from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from superfv.axes import DIM_TO_AXIS

from .configs import FluxRecipe, FV_SchemeParameters, HydroParameters
from .slope_limiting.muscl import (
    MUSCL_SlopeLimiter,
    compute_MUSCL_slopes,
    compute_PP2D_slopes,
)
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import crop, replace_slice
from .tools.variable_index_map import VariableIndexMap

if TYPE_CHECKING:
    from .hydro_solver import HydroSolver

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def _allocate_MUSCL_arrays(sim: HydroSolver):
    active_dims = sim.params.mesh.active_dims
    nvars = sim.params.variable_index_map.nvars
    _nx_, _ny_, _nz_ = sim.mesh._shape_
    arrays = sim.arrays

    if "_flux_jvp_" not in sim.arrays:
        arrays.add("_predictor_q_", sim.xp.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_flux_jvp_", sim.xp.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_nodes_", sim.xp.empty((nvars, _nx_, _ny_, _nz_, 2)))
        if "x" in active_dims:
            arrays.add("_x_slopes_", sim.xp.empty((nvars, _nx_, _ny_, _nz_)))
        if "y" in active_dims:
            arrays.add("_y_slopes_", sim.xp.empty((nvars, _nx_, _ny_, _nz_)))
        if "z" in active_dims:
            arrays.add("_z_slopes_", sim.xp.empty((nvars, _nx_, _ny_, _nz_)))


def compute_flux_jvp(
    q: ArrayLike,
    vec: ArrayLike,
    dim: Literal["x", "y", "z"],
    idx: VariableIndexMap,
    hydro_params: HydroParameters,
    primitives: bool = True,
) -> ArrayLike:
    """
    Compute the Jacobian-vector product for the flux function.
    """
    if not primitives:
        raise NotImplementedError("JVP for conservative variable fluxes not implemented yet.")

    jvp = cp.zeros_like(q) if CUPY_AVAILABLE and isinstance(q, cp.ndarray) else np.zeros_like(q)

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


def update_fluxes_with_muscl_scheme(
    sim: HydroSolver, muscl_scheme: FV_SchemeParameters, hancock_dt: float = 0.0
):
    sim.step_summary.timer.start("update_fluxes", sim.params.sync_timer)  # TIMER START

    if not muscl_scheme.muscl_params.use_MUSCL:
        raise ValueError("update_fluxes_with_muscl_scheme should only be called for MUSCL schemes.")

    muscl_params = muscl_scheme.muscl_params
    hydro_params = sim.params.hydro
    idx = sim.params.variable_index_map
    active_dims = sim.params.mesh.active_dims
    hx, hy, hz = sim.mesh.hx, sim.mesh.hy, sim.mesh.hz
    nghost = sim.params.mesh.nghost
    arrays = sim.arrays

    _allocate_MUSCL_arrays(sim)

    _q_ = arrays["_u_"] if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM else arrays["_w_"]
    _alpha_ = arrays["_alpha_"]
    _xyz_slopes_ = {dim: arrays[f"_{dim}_slopes_"] for dim in active_dims}
    _predictor_q_ = arrays["_predictor_q_"]
    _jvp_ = arrays["_flux_jvp_"]
    _nodes_ = arrays["_nodes_"]
    _FGH_nodes_ = {dim: arrays[{"x": "_F_", "y": "_G_", "z": "_H_"}[dim]] for dim in active_dims}

    # compute smooth extrema
    if muscl_params.SED_params.use_SED:
        compute_alpha(_q_, _alpha_, active_dims, muscl_params.SED_params.clip_zero_tol)

    # Compute MUSCL slopes
    if muscl_params.MUSCL_limiter == MUSCL_SlopeLimiter.PP2D:
        if len(active_dims) != 2:
            raise ValueError("PP2D MUSCL slopes can only be used in 2D.")
        _slopes1_ = _xyz_slopes_[active_dims[0]]
        _slopes2_ = _xyz_slopes_[active_dims[1]]

        compute_PP2D_slopes(
            _q_, _alpha_, _slopes1_, _slopes2_, active_dims, 1e-20, muscl_params.SED_params.use_SED
        )
    else:
        for dim in active_dims:
            _slopes_ = _xyz_slopes_[dim]

            compute_MUSCL_slopes(
                _q_,
                _alpha_,
                _slopes_,
                dim,
                muscl_params.MUSCL_limiter,
                muscl_params.SED_params.use_SED,
            )

    # Compute predictor step
    _predictor_q_[...] = _q_

    if hancock_dt > 0:
        for dim in active_dims:
            h = {"x": hx, "y": hy, "z": hz}[dim]
            _slopes_ = _xyz_slopes_[dim]

            _jvp_[...] = compute_flux_jvp(
                _q_,
                _slopes_,
                dim,
                idx,
                hydro_params,
                primitives=muscl_scheme.flux_recipe != FluxRecipe.CONS_LIM_PRIM,
            )
            _predictor_q_[...] -= 0.5 * _jvp_ * hancock_dt / h

    # Corrector step
    for dim in active_dims:
        _slopes_ = _xyz_slopes_[dim]
        _fluxes_ = _FGH_nodes_[dim]

        _nodes_[..., 0] = _predictor_q_ - 0.5 * _slopes_
        _nodes_[..., 1] = _predictor_q_ + 0.5 * _slopes_

        if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            sim.conservatives_to_primitives(_nodes_, _nodes_)

        sim.step_summary.timer.start("riemann_solver", sim.params.sync_timer)  # TIMER START
        axis = DIM_TO_AXIS[dim]
        left_of_interface = replace_slice(crop(axis, (nghost - 1, -nghost), ndim=5), 4, 1)
        right_of_interface = replace_slice(crop(axis, (nghost, -nghost + 1), ndim=5), 4, 0)

        sim.riemann_solver(
            _nodes_[left_of_interface],
            _nodes_[right_of_interface],
            _fluxes_,
            dim,
            sim.params.variable_index_map,
            sim.params.hydro.gamma,
            sim.params.hydro.isothermal,
            sim.params.hydro.iso_cs,
        )
        sim.step_summary.timer.stop("riemann_solver", sim.params.sync_timer)  # TIMER STOP

    sim._update_flux_arrays(
        _FGH_nodes_["x"] if "x" in active_dims else None,
        _FGH_nodes_["y"] if "y" in active_dims else None,
        _FGH_nodes_["z"] if "z" in active_dims else None,
    )

    sim.step_summary.timer.stop("update_fluxes", sim.params.sync_timer)  # TIMER STOP
