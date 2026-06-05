from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Literal

import numpy as np

from superfv.axes import DIM_TO_AXIS

from .configs import FluxRecipe, FV_SchemeParameters
from .slope_limiting.muscl import (
    MUSCL_SlopeLimiter,
    compute_MUSCL_slopes,
    compute_PP2D_slopes,
)
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import crop
from .tools.variable_index_map import VariableIndexMap

if TYPE_CHECKING:
    from .hydro_solver import HydroSolver

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def _compute_flux_jvp_np(
    w: np.ndarray,
    vec: np.ndarray,
    jvp: np.ndarray,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
):
    iv = idx("v" + dim)

    jvp[idx("rho")] = w[iv] * vec[idx("rho")] + w[idx("rho")] * vec[iv]
    jvp[idx("vx")] = w[iv] * vec[idx("vx")]
    jvp[idx("vy")] = w[iv] * vec[idx("vy")]
    jvp[idx("vz")] = w[iv] * vec[idx("vz")]
    jvp[iv] += (1 / w[idx("rho")]) * vec[idx("P")]
    jvp[idx("P")] = (
        iso_cs**2 * jvp[idx("rho")]
        if isothermal
        else gamma * w[idx("P")] * vec[iv] + w[iv] * vec[idx("P")]
    )
    if "passives" in idx.group_var_map:
        jvp[idx("passives")] = w[iv] * vec[idx("passives")] + w[idx("passives")] * vec[iv]


@lru_cache(maxsize=None)
def _make_compute_flux_jvp_kernel(npassives: int):
    in_params = (
        "float64 rho, float64 v1, float64 v2, float64 v3, float P, "
        "float64 vec_rho, float64 vec_v1, float64 vec_v2, float64 vec_v3, float vec_P, "
        "float64 gamma, bool isothermal, float64 iso_cs, int32 dim"
    )
    for i in range(npassives):
        in_params += f", float64 passives_{i}"
    for i in range(npassives):
        in_params += f", float64 vec_passives_{i}"

    out_params = "float64 jvp_rho, float64 jvp_v1, float64 jvp_v2, float64 jvp_v3, float64 jvp_P"
    for i in range(npassives):
        out_params += f", float64 jvp_passives_{i}"

    body = """
    if (dim == 1) {
        jvp_rho = v1 * vec_rho + rho * vec_v1;
        jvp_v1 = v1 * vec_rho + rho * vec_v1 + (1 / rho) * vec_P;
        jvp_v2 = v2 * vec_rho + rho * vec_v2;
        jvp_v3 = v3 * vec_rho + rho * vec_v3;
        jvp_P = isothermal ?
            iso_cs**2 * jvp_rho : gamma * w[idx("P")] * vec[iv] + w[iv] * vec[idx("P")];

    """
    for i in range(npassives):
        body += f"    jvp_passives_{i} = v1 * vec_passives_{i} + rho * vec_v1;"
    body += """
    } else if (dim == 2) {
        jvp_rho = v1 * vec_rho + rho * vec_v1;
        jvp_v1 = v1 * vec_rho + rho * vec_v1 + (1 / rho) * vec_P;
        jvp_v2 = v2 * vec_rho + rho * vec_v2;
        jvp_v3 = v3 * vec_rho + rho * vec_v3;
        jvp_P = isothermal ?
            iso_cs**2 * jvp_rho : gamma * w[idx("P")] * vec[iv] + w[iv] * vec[idx("P")];
    """
    for i in range(npassives):
        body += f"    jvp_passives_{i} = v2 * vec_passives_{i} + rho * vec_v2;"
    body += """
    } else if(dim == 3) {
        jvp_rho = v1 * vec_rho + rho * vec_v1;
        jvp_v1 = v1 * vec_rho + rho * vec_v1 + (1 / rho) *	vec_P;
        jvp_v2 = v2 * vec_rho + rho * vec_v2;
        jvp_v3 = v3 * vec_rho + rho * vec_v3;
        jvp_P = isothermal ?
            iso_cs**2*jvp_rho : gamma*w[idx("P")]*vec[iv] + w[iv]*vec[idx("P")];
    }
    """
    for i in range(npassives):
        body += f"    jvp_passives_{i} = v3 * vec_passives_{i} + rho * vec_v3;"

    return cp.ElementwiseKernel(in_params, out_params, body, name="compute_flux_jvp")


def compute_flux_jvp(
    w: ArrayLike,
    vec: ArrayLike,
    jvp: ArrayLike,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
    primitives: bool = True,
):
    if not primitives:
        raise NotImplementedError("JVP for conservative variable fluxes not implemented yet.")

    iv = idx("v" + dim)

    jvp[idx("rho")] = w[iv] * vec[idx("rho")] + w[idx("rho")] * vec[iv]
    jvp[idx("vx")] = w[iv] * vec[idx("vx")]
    jvp[idx("vy")] = w[iv] * vec[idx("vy")]
    jvp[idx("vz")] = w[iv] * vec[idx("vz")]
    jvp[iv] += (1 / w[idx("rho")]) * vec[idx("P")]
    jvp[idx("P")] = (
        iso_cs**2 * jvp[idx("rho")]
        if isothermal
        else gamma * w[idx("P")] * vec[iv] + w[iv] * vec[idx("P")]
    )
    if "passives" in idx.group_var_map:
        jvp[idx("passives")] = w[iv] * vec[idx("passives")] + w[idx("passives")] * vec[iv]

    return

    if CUPY_AVAILABLE and isinstance(w, cp.ndarray):
        npassives = len(idx.group_var_map.get("passives", []))
        kernel = _make_compute_flux_jvp_kernel(npassives)
        kernel(
            w[idx("rho")],
            w[idx("vx")],
            w[idx("vy")],
            w[idx("vz")],
            w[idx("P")],
            vec[idx("rho")],
            vec[idx("vx")],
            vec[idx("vy")],
            vec[idx("vz")],
            vec[idx("P")],
            gamma,
            isothermal,
            iso_cs,
            {"x": 1, "y": 2, "z": 3}[dim],
            *[w[idx(v)] for v in idx.group_var_map.get("passives", [])],
            *[vec[idx(v)] for v in idx.group_var_map.get("passives", [])],
        )
    else:
        return _compute_flux_jvp_np(w, vec, jvp, idx, dim, gamma, isothermal, iso_cs)


def update_fluxes_with_muscl_scheme(
    sim: HydroSolver, muscl_scheme: FV_SchemeParameters, hancock_dt: float = 0.0
):
    sim._start_timer("update_fluxes")  # TIMER START

    if not muscl_scheme.muscl_params.use_MUSCL:
        raise ValueError("update_fluxes_with_muscl_scheme should only be called for MUSCL schemes.")

    muscl_params = muscl_scheme.muscl_params
    hydro_params = sim.params.hydro
    idx = sim.params.variable_index_map
    active_dims = sim.params.mesh.active_dims
    hx, hy, hz = sim.mesh.hx, sim.mesh.hy, sim.mesh.hz
    nghost = sim.params.mesh.nghost
    na = sim.xp.newaxis
    arrays = sim.arrays

    _u_ = arrays["_u_"]
    _w_ = arrays["_w_"]
    _q_ = _u_ if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _w_
    _alpha_ = arrays["_alpha_"]
    xyz_slope_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}

    # Detect smooth extrema
    if muscl_params.SED_params.use_SED:
        compute_alpha(_q_, _alpha_, active_dims, muscl_params.SED_params.clip_zero_tol)

    # Compute MUSCL slopes
    if muscl_params.MUSCL_limiter == MUSCL_SlopeLimiter.PP2D:
        if len(active_dims) != 2:
            raise ValueError("PP2D MUSCL slopes can only be used in 2D.")

        _slopes1_ = sim.xp.empty_like(_q_)  # TEMP ARRAY
        _slopes2_ = sim.xp.empty_like(_q_)  # TEMP ARRAY
        compute_PP2D_slopes(
            _q_, _alpha_, _slopes1_, _slopes2_, active_dims, 1e-20, muscl_params.SED_params.use_SED
        )
        xyz_slope_dict[active_dims[0]] = _slopes1_
        xyz_slope_dict[active_dims[1]] = _slopes2_
    else:
        for dim in active_dims:
            _slopes_ = sim.xp.empty_like(_q_)  # TEMP ARRAY
            compute_MUSCL_slopes(
                _q_,
                _alpha_,
                _slopes_,
                dim,
                muscl_params.MUSCL_limiter,
                muscl_params.SED_params.use_SED,
            )
            xyz_slope_dict[dim] = _slopes_

    # Compute predictor step
    _predictor_q_ = _q_.copy()

    if hancock_dt > 0:
        _jvp_ = sim.xp.empty_like(_q_)  # TEMP ARRAY

        for dim in active_dims:
            h = {"x": hx, "y": hy, "z": hz}[dim]

            compute_flux_jvp(
                _q_,
                xyz_slope_dict[dim],
                _jvp_,
                idx,
                dim,
                gamma=hydro_params.gamma,
                isothermal=hydro_params.isothermal,
                iso_cs=hydro_params.iso_cs,
                primitives=muscl_scheme.flux_recipe != FluxRecipe.CONS_LIM_PRIM,
            )
            _predictor_q_ -= 0.5 * _jvp_ * hancock_dt / h

    # Corrector step
    for dim in active_dims:
        _Fluxes_ = arrays[{"x": "_F_", "y": "_G_", "z": "_H_"}[dim]]

        # Compute cell faces
        _slopes_ = xyz_slope_dict[dim]
        _left_face_ = _predictor_q_ - 0.5 * _slopes_  # TEMP ARRAY
        _right_face_ = _predictor_q_ + 0.5 * _slopes_  # TEMP ARRAY

        # Ensure faces are positive and primitive
        if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            sim.conservatives_to_primitives(_left_face_, _left_face_)
            sim.conservatives_to_primitives(_right_face_, _right_face_)

        sim._ensure_positive_nodes(_left_face_[..., na], _w_)
        sim._ensure_positive_nodes(_right_face_[..., na], _w_)

        # Gather slices
        axis = DIM_TO_AXIS[dim]
        left_of_interface = crop(axis, (nghost - 1, -nghost), ndim=4)
        right_of_interface = crop(axis, (nghost, -nghost + 1), ndim=4)

        # Perform Riemann solve
        sim._start_timer("riemann_solver")  # TIMER START
        sim.riemann_solver(
            _right_face_[left_of_interface],
            _left_face_[right_of_interface],
            _Fluxes_,
            dim,
            sim.params.variable_index_map,
            sim.params.hydro.gamma,
            sim.params.hydro.isothermal,
            sim.params.hydro.iso_cs,
        )
        sim._stop_timer("riemann_solver")  # TIMER STOP

    sim._stop_timer("update_fluxes")  # TIMER STOP
