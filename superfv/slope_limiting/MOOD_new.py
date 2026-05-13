from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

from superfv.configs import (
    FluxRecipe,
    FV_SchemeParameters,
    NumericalAdmissibilityParameters,
)
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import compute_alpha
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices
from superfv.tools.variable_index_map import VariableIndexMap

if TYPE_CHECKING:
    from superfv.hydro_solver import HydroSolver


def numerical_admissibility_detection(
    _qold_: np.ndarray,
    _qnew_: np.ndarray,
    _troubles_: np.ndarray,
    _alpha_: np.ndarray,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    params: NumericalAdmissibilityParameters,
):
    """
    Update _troubles_ based on NAD criteria.
    """
    _NAD_troubles_ = np.zeros_like(_qold_, dtype=bool)
    _dmp_M_ = np.empty_like(_qold_)
    _dmp_m_ = np.empty_like(_qold_)
    _lower_ = np.empty_like(_qold_)
    _upper_ = np.empty_like(_qold_)

    # Update DMP
    compute_dmp(_qold_, _dmp_M_, _dmp_m_, active_dims, params.include_corners)

    # compute lower and upper bounds for NAD
    if params.delta:
        _delta_ = np.empty_like(_qold_)

        _delta_[...] = _dmp_M_ - _dmp_m_
        _lower_[...] = _dmp_m_ - params.rtol * _delta_ - params.atol
        _upper_[...] = _dmp_M_ + params.rtol * _delta_ + params.atol
    else:
        _lower_[...] = _dmp_m_ - params.rtol * np.abs(_dmp_m_) - params.atol
        _upper_[...] = _dmp_M_ + params.rtol * np.abs(_dmp_M_) + params.atol

    # Detect NAD violations
    _NAD_troubles_ |= _qnew_ < _lower_
    _NAD_troubles_ |= _qnew_ > _upper_

    # SED relaxation
    if params.SED_params.use_SED:
        _NAD_troubles_ &= _alpha_ < 1.0

    # Omit variables from detection
    if params.omit_vars:
        omit_indices = [idx(v) for v in params.omit_vars]
        _NAD_troubles_[omit_indices] = False

    # Update troubled cells
    np.maximum(_troubles_, np.any(_NAD_troubles_, axis=0), out=_troubles_)


def detect_troubled_cells(sim: HydroSolver, t: float, dt: float) -> int:
    """
    Update "_troubles_" and return the number of revisable troubled cells.
    """
    interior = sim.interior
    params = sim.params
    mood_params = params.fv_scheme.mood_params
    n_cascade = len(mood_params.fallback_cascade)
    active_dims = params.mesh.active_dims
    idx = params.variable_index_map
    arrays = sim.arrays

    _uold_ = arrays["_u_"]
    _troubles_ = arrays["_troubles_"]
    rev_troubles = arrays["revisable_troubles"]
    _alpha_ = arrays["_alpha_"]
    _cascade_idx_ = arrays["_cascade_idx_"]
    _qold_ = sim.xp.empty_like(_uold_)
    _qnew_ = sim.xp.empty_like(_uold_)

    # Reset troubled cells
    _troubles_[...] = 0.0

    # Compute candidate solution qnew
    _qnew_[interior] = _uold_[interior] + dt * sim.compute_time_derivative()
    sim.apply_bc(_qnew_, t, params.fv_scheme.p)

    if params.fv_scheme.flux_recipe != FluxRecipe.CONS_LIM_PRIM:
        sim.conservatives_to_primitives(_uold_, _qold_)
        sim.conservatives_to_primitives(_qnew_, _qnew_)
    else:
        _qold_[...] = _uold_[...]

    # Update smooth extrema
    if mood_params.NAD_params.use_NAD and mood_params.NAD_params.SED_params.use_SED:
        compute_alpha(_qold_, _alpha_, active_dims)

    # Detect troubled cells
    if params.cupy:
        raise NotImplementedError("MOOD is not yet implemented for GPU arrays.")

    if mood_params.NAD_params.use_NAD:
        numerical_admissibility_detection(
            _qold_, _qnew_, _troubles_, _alpha_, idx, active_dims, mood_params.NAD_params
        )

    if mood_params.PAD_params.use_PAD:
        for v, (lb, ub) in mood_params.PAD_params.physical_bounds.items():
            if lb is not None:
                sim.xp.maximum(_troubles_, _qnew_[idx[v]] < lb, out=_troubles_)
            if ub is not None:
                sim.xp.maximum(_troubles_, _qnew_[idx[v]] > ub, out=_troubles_)

    # Return number of revisable troubled cells
    sim.xp.minimum(_troubles_[interior], _cascade_idx_[interior] < n_cascade, out=rev_troubles)
    return sim.xp.sum(rev_troubles).item()


def init_mood(sim: HydroSolver):
    """
    Reset _cascade_idx_ and assign the base fluxes "F_[sim.params.fv_scheme.name]", ...
    """
    params = sim.params
    active_dims = params.mesh.active_dims
    arrays = sim.arrays

    arrays["_cascade_idx_"][...] = 0

    # Copy initial fluxes
    if "x" in active_dims:
        arrays["F_" + params.fv_scheme.name] = arrays["F"].copy()
    if "y" in active_dims:
        arrays["G_" + params.fv_scheme.name] = arrays["G"].copy()
    if "z" in active_dims:
        arrays["H_" + params.fv_scheme.name] = arrays["H"].copy()


def compute_fallback_fluxes(sim: HydroSolver, fallback_scheme: FV_SchemeParameters):
    """
    Update the flux arrays "F_[fallback_scheme.name]", ... with a specified fallback scheme.
    """
    active_dims = sim.params.mesh.active_dims
    arrays = sim.arrays

    sim.update_fluxes(fallback_scheme)

    if "x" in active_dims:
        arrays["F_" + fallback_scheme.name][...] = arrays["F"].copy()
    if "y" in active_dims:
        arrays["G_" + fallback_scheme.name][...] = arrays["G"].copy()
    if "z" in active_dims:
        arrays["H_" + fallback_scheme.name][...] = arrays["H"].copy()


def map_cells_values_to_face_values(
    xp: ModuleType, _cell_values_: ArrayLike, axis: int
) -> ArrayLike:
    """
    Return an array of face values obtained by taking the maximum of the adjacent
    cell values along the specified axis.
    """
    fv_shape = list(_cell_values_.shape)
    fv_shape[axis] -= 1
    _face_values_ = xp.empty(fv_shape, dtype=_cell_values_.dtype)

    lft_slc = crop(axis, (None, -1), ndim=4)
    rgt_slc = crop(axis, (1, None), ndim=4)

    _face_values_[...] = _cell_values_[lft_slc]
    xp.maximum(_face_values_, _cell_values_[rgt_slc], out=_face_values_)
    return _face_values_


def blend_troubled_cells(
    xp: ModuleType, _troubles_: ArrayLike, active_dims: Tuple[Literal["x", "y", "z"]]
) -> ArrayLike:
    """
    Return an floating-point array of troubled cells blended with their neighbors.
    """
    ndim = len(active_dims)

    _blended_ = xp.empty_like(_troubles_, dtype=xp.float64)
    _blended_[...] = _troubles_

    if ndim == 1:
        axis = {"x": 1, "y": 2, "z": 3}[active_dims[0]]
        lft_slc = crop(axis, (None, -1), ndim=4)
        rgt_slc = crop(axis, (1, None), ndim=4)

        # First neighbors
        _blended_[lft_slc] = xp.maximum(0.75 * _troubles_[rgt_slc], _blended_[lft_slc])
        _blended_[rgt_slc] = xp.maximum(0.75 * _troubles_[lft_slc], _blended_[rgt_slc])

        # Second neighbors
        _blended_[lft_slc] = xp.maximum(0.25 * (_blended_[rgt_slc] > 0), _blended_[lft_slc])
        _blended_[rgt_slc] = xp.maximum(0.25 * (_blended_[lft_slc] > 0), _blended_[rgt_slc])
    elif ndim == 2:
        axis1 = {"x": 1, "y": 2, "z": 3}[active_dims[0]]
        axis2 = {"x": 1, "y": 2, "z": 3}[active_dims[1]]

        lft_slc1 = crop(axis1, (None, -1), ndim=4)
        rgt_slc1 = crop(axis1, (1, None), ndim=4)
        lft_slc2 = crop(axis2, (None, -1), ndim=4)
        rgt_slc2 = crop(axis2, (1, None), ndim=4)
        lft_lft = merge_slices(lft_slc1, lft_slc2)
        lft_rgt = merge_slices(lft_slc1, rgt_slc2)
        rgt_lft = merge_slices(rgt_slc1, lft_slc2)
        rgt_rgt = merge_slices(rgt_slc1, rgt_slc2)

        # First neighbors
        _blended_[lft_slc1] = xp.maximum(0.75 * _troubles_[rgt_slc1], _blended_[lft_slc1])
        _blended_[rgt_slc1] = xp.maximum(0.75 * _troubles_[lft_slc1], _blended_[rgt_slc1])
        _blended_[lft_slc2] = xp.maximum(0.75 * _troubles_[rgt_slc2], _blended_[lft_slc2])
        _blended_[rgt_slc2] = xp.maximum(0.75 * _troubles_[lft_slc2], _blended_[rgt_slc2])

        # Second neighbors
        _blended_[lft_lft] = xp.maximum(0.5 * _troubles_[rgt_rgt], _blended_[lft_lft])
        _blended_[lft_rgt] = xp.maximum(0.5 * _troubles_[rgt_lft], _blended_[lft_rgt])
        _blended_[rgt_lft] = xp.maximum(0.5 * _troubles_[lft_rgt], _blended_[rgt_lft])
        _blended_[rgt_rgt] = xp.maximum(0.5 * _troubles_[lft_lft], _blended_[rgt_rgt])

        # Third neighbors
        _blended_[lft_slc2] = xp.maximum(0.25 * (_blended_[rgt_slc2] > 0), _blended_[lft_slc2])
        _blended_[rgt_slc2] = xp.maximum(0.25 * (_blended_[lft_slc2] > 0), _blended_[rgt_slc2])
        _blended_[lft_slc1] = xp.maximum(0.25 * (_blended_[rgt_slc1] > 0), _blended_[lft_slc1])
        _blended_[rgt_slc1] = xp.maximum(0.25 * (_blended_[lft_slc1] > 0), _blended_[rgt_slc1])

    elif ndim == 3:
        raise NotImplementedError("3D blending is not implemented yet.")

    return _blended_


def get_face_mask(
    xp: ModuleType,
    _cv_mask_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"]],
    nghost: int,
) -> ArrayLike:
    """
    Slice a cell-centered mask with ghost cells to obtain a face-centered mask with
    no ghost cells.
    """
    _fv_ = map_cells_values_to_face_values(xp, _cv_mask_, {"x": 1, "y": 2, "z": 3}[dim])
    interior = merge_slices(
        *[
            crop(
                {"x": 1, "y": 2, "z": 3}[d],
                (nghost - 1, -nghost + 1) if d == dim else (nghost, -nghost),
                ndim=4,
            )
            for d in active_dims
        ]
    )
    return _fv_[interior]


def assign_blended_fluxes(sim: HydroSolver):
    """
    Update the flux arrays "F", ... by blending the high-order and fallback fluxes based on
    a blended troubled cell mask.
    """
    params = sim.params
    fv_scheme = params.fv_scheme
    fallback_scheme = fv_scheme.mood_params.fallback_cascade[0]
    active_dims = params.mesh.active_dims
    nghost = params.mesh.nghost
    arrays = sim.arrays

    _troubles_ = arrays["_cascade_idx_"]
    _blended_troubles_ = blend_troubled_cells(sim.xp, _troubles_, active_dims)

    for dim in active_dims:
        name = {"x": "F", "y": "G", "z": "H"}[dim]
        theta_fv = get_face_mask(sim.xp, _blended_troubles_, dim, active_dims, nghost)

        F_high_order = arrays[name + "_" + fv_scheme.name]
        F_fallback = arrays[name + "_" + fallback_scheme.name]

        arrays[name] = (1 - theta_fv) * F_high_order + theta_fv * F_fallback


def assign_fluxes(sim: HydroSolver):
    """
    Update the flux arrays "F", ... based on the current "_cascade_idx_"
    """
    params = sim.params
    fv_scheme = params.fv_scheme
    cascade = fv_scheme.mood_params.fallback_cascade
    active_dims = params.mesh.active_dims
    nghost = params.mesh.nghost
    arrays = sim.arrays

    if fv_scheme.mood_params.blend_troubles:
        assign_blended_fluxes(sim)
        return

    _cascade_idx_ = arrays["_cascade_idx_"]

    # Assign fluxes based on cascade index
    for dim in active_dims:
        name = {"x": "F", "y": "G", "z": "H"}[dim]
        arrays[name][...] = 0.0
        cascade_face_mask = get_face_mask(sim.xp, _cascade_idx_, dim, active_dims, nghost)
        for i, scheme in enumerate([fv_scheme] + cascade):
            in_mask = cascade_face_mask == i
            arrays[name] += arrays[name + "_" + scheme.name] * in_mask


def mood_loop(sim: HydroSolver, t: float, dt: float):
    """
    Revise flux arrays "F", .. until no more revisable troubled cells are detected or the maximum
    number of revisions is reached. Update substep summary.
    """
    params = sim.params
    mood_params = params.fv_scheme.mood_params
    n_cascade = len(mood_params.fallback_cascade)
    substep_summary = sim.substep_summary
    arrays = sim.arrays

    _cascade_idx_ = arrays["_cascade_idx_"]
    _troubles_ = arrays["_troubles_"]

    # Initialize MOOD arrays
    init_mood(sim)
    max_fallback_idx = 0

    for _ in range(mood_params.max_revs):
        n_troubles = detect_troubled_cells(sim, t, dt)
        substep_summary.n_troubles_hist.append(n_troubles)

        # Do not revise if no revisable troubled cells
        if n_troubles == 0:
            break

        # Update cascade index and compute new fluxes
        sim.xp.minimum(_cascade_idx_ + _troubles_, n_cascade, out=_cascade_idx_)

        i_max = sim.xp.max(_cascade_idx_).item()
        if i_max == max_fallback_idx + 1:
            compute_fallback_fluxes(sim, mood_params.fallback_cascade[i_max - 1])
            max_fallback_idx += 1
        elif i_max != max_fallback_idx:
            raise RuntimeError(
                f"Unexpected cascade index: {i_max} (previous max was {max_fallback_idx})"
            )

        # Update state
        substep_summary.n_MOOD_revisions += 1
        assign_fluxes(sim)
