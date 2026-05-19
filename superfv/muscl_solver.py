from __future__ import annotations

from typing import TYPE_CHECKING

from superfv.axes import DIM_TO_AXIS

from .configs import FluxRecipe, FV_SchemeParameters
from .slope_limiting.muscl import (
    MUSCL_SlopeLimiter,
    compute_MUSCL_slopes,
    compute_PP2D_slopes,
)
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .tools.slicing import crop, replace_slice

if TYPE_CHECKING:
    from .hydro_solver import HydroSolver


def update_fluxes_with_muscl_scheme(
    sim: HydroSolver, muscl_scheme: FV_SchemeParameters, hancock_dt: float = 0.0
):
    if not muscl_scheme.muscl_params.use_MUSCL:
        raise ValueError("update_fluxes_with_muscl_scheme should only be called for MUSCL schemes.")

    muscl_params = muscl_scheme.muscl_params
    active_dims = sim.params.mesh.active_dims
    nvars = sim.params.variable_index_map.nvars
    nx, ny, nz = sim.mesh.shape
    _nx_, _ny_, _nz_ = sim.mesh._shape_
    hx, hy, hz = sim.mesh.hx, sim.mesh.hy, sim.mesh.hz
    nghost = sim.params.mesh.nghost
    arrays = sim.arrays

    _alpha_ = arrays["_alpha_"]
    _q_ = arrays["_u_"] if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM else arrays["_w_"]
    _predictor_q_ = sim.xp.empty_like(_q_)
    _jvp_ = sim.xp.empty_like(_q_)
    _xyz_nodes_ = {dim: sim.xp.empty(_q_.shape + (2,)) for dim in active_dims}
    _xyz_slopes_ = {dim: sim.xp.empty_like(_q_) for dim in active_dims}
    _FGH_nodes_ = {
        dim: sim.xp.empty(
            (
                nvars,
                nx + 1 if dim == "x" else _nx_,
                ny + 1 if dim == "y" else _ny_,
                nz + 1 if dim == "z" else _nz_,
            )
        )
        for dim in active_dims
    }

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

            _jvp_[...] = sim.compute_flux_jvp(
                _q_,
                _slopes_,
                dim,
                primitives=muscl_scheme.flux_recipe != FluxRecipe.CONS_LIM_PRIM,
            )
            _predictor_q_[...] -= 0.5 * _jvp_ * hancock_dt / h

    # Corrector step
    for dim in active_dims:
        _slopes_ = _xyz_slopes_[dim]
        _nodes_ = _xyz_nodes_[dim]
        _fluxes_ = _FGH_nodes_[dim]

        _nodes_[..., 0] = _predictor_q_ - 0.5 * _slopes_
        _nodes_[..., 1] = _predictor_q_ + 0.5 * _slopes_

        if muscl_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            sim.conservatives_to_primitives(_nodes_, _nodes_)

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
    sim._update_flux_arrays(
        _FGH_nodes_["x"] if "x" in active_dims else None,
        _FGH_nodes_["y"] if "y" in active_dims else None,
        _FGH_nodes_["z"] if "z" in active_dims else None,
    )
