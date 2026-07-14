from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.boundary_conditions import BC, apply_bc
from superfv.configs import (
    BoundaryConditionParameters,
    FluxRecipe,
    FV_SchemeParameters,
    HydroParameters,
    MOOD_Parameters,
    NumericalAdmissibilityParameters,
)
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.finite_volume_driver import (
    apply_fv_bc,
    compute_fv_dudt,
    get_interior_view,
    update_fv_fluxes,
)
from superfv.hydro import cons_to_prim
from superfv.mesh import UniformFiniteVolumeMesh
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import compute_alpha
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices, replace_slice
from superfv.tools.step_history import MultiTimer, SubstepSummary
from superfv.tools.variable_index_map import VariableIndexMap


def compute_candidate_solution(
    _uold_: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
    H: ArrayLike,
    S: ArrayLike,
    _unew_: ArrayLike,
    _wnew_: ArrayLike,
    idx: VariableIndexMap,
    t: float,
    dt: float,
    mesh: UniformFiniteVolumeMesh,
    bc_params: BoundaryConditionParameters,
    hydro_params: HydroParameters,
):
    interior = get_interior_view(mesh.active_dims, mesh.nghost)
    dudt = compute_fv_dudt(F, G, H, S, mesh)
    _unew_[interior] = _uold_[interior] + dt * dudt

    apply_fv_bc(_unew_, idx, mesh, t, bc_params, hydro_params)
    cons_to_prim(
        _unew_,
        _wnew_,
        idx,
        hydro_params.gamma,
        hydro_params.isothermal,
        hydro_params.iso_cs,
    )


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
    Update `_troubles_` based on NAD criteria compute from `_qold_` and `_qnew_` with
    optional relaxation from the smooth extrema detector `_alpha_`. Renders a single
    ghost cell layer along each active dimension of the output array invalid.
    """
    _NAD_troubles_ = np.zeros_like(_qold_, dtype=bool)
    _dmp_M_ = np.empty_like(_qold_)
    _dmp_m_ = np.empty_like(_qold_)

    # Update DMP
    compute_dmp(_qold_, _dmp_M_, _dmp_m_, active_dims, params.include_corners)

    # compute lower and upper bounds for NAD
    if params.delta:
        _delta_ = _dmp_M_ - _dmp_m_  # TEMP ARRAY
        _lower_ = _dmp_m_ - params.rtol * _delta_ - params.atol  # TEMP ARRAY
        _upper_ = _dmp_M_ + params.rtol * _delta_ + params.atol  # TEMP ARRAY
    else:
        _lower_ = _dmp_m_ - params.rtol * np.abs(_dmp_m_) - params.atol  # TEMP ARRAY
        _upper_ = _dmp_M_ + params.rtol * np.abs(_dmp_M_) + params.atol  # TEMP ARRAY

    # Detect NAD violations
    _NAD_troubles_ |= _qnew_ < _lower_
    _NAD_troubles_ |= _qnew_ > _upper_

    # SED relaxation
    if params.SED_params.use_SED:
        _NAD_troubles_ &= _alpha_ < 1.0

    # Omit variables from detection
    if "omit_NAD" in idx.group_var_map:
        _NAD_troubles_[idx("omit_NAD")] = False

    # Update troubled cells
    np.maximum(_troubles_, np.any(_NAD_troubles_, axis=0), out=_troubles_)


def apply_troubles_bc(
    _troubles_: ArrayLike,
    nghost: int,
    bc_params: BoundaryConditionParameters,
):
    none = (BC.NONE, BC.NONE)
    periodic = (BC.PERIODIC, BC.PERIODIC)
    zeros = (BC.ZEROS, BC.ZEROS)
    apply_bc(
        _troubles_,
        nghost,
        {none: none, periodic: periodic}.get(bc_params.bcx, zeros),
        {none: none, periodic: periodic}.get(bc_params.bcy, zeros),
        {none: none, periodic: periodic}.get(bc_params.bcz, zeros),
    )


def detect_troubled_cells(
    _uold_: ArrayLike,
    _wold_: ArrayLike,
    _unew_: ArrayLike,
    _wnew_: ArrayLike,
    _troubles_: ArrayLike,
    _cascade_idx_: ArrayLike,
    revisable_troubles: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    base_scheme: FV_SchemeParameters,
    bc_params: BoundaryConditionParameters,
) -> int:
    """
    Update "_troubles_" and return the number of revisable troubled cells.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_uold_, cp.ndarray) else np

    mood_params = base_scheme.mood_params
    n_cascade = len(mood_params.fallback_cascade)
    active_dims = mesh.active_dims
    interior = get_interior_view(active_dims, mesh.nghost)

    # Reset troubled cells
    _troubles_[...] = 0.0

    # Assign _qold_ and _qnew_, the NAD arrays
    _qold_ = _uold_ if base_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _wold_
    _qnew_ = _unew_ if base_scheme.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _wnew_

    # Update smooth extrema
    if mood_params.NAD_params.use_NAD and mood_params.NAD_params.SED_params.use_SED:
        compute_alpha(_qold_, _alpha_, active_dims, mood_params.NAD_params.SED_params.clip_zero_tol)

    if CUPY_AVAILABLE and isinstance(_uold_, cp.ndarray):
        # Detect troubled cells with CuPy using a custom kernel
        _M_ = cp.empty_like(_qold_)
        _m_ = cp.empty_like(_qold_)

        # NAD mask
        omit_vars_idxs = [idx(v) for v in mood_params.NAD_params.omit_vars]
        NAD_mask = cp.array(
            [int(i not in omit_vars_idxs) for i in range(idx.nvars)], dtype=np.int32
        )

        # PAD bounds
        physical_bounds = cp.empty((idx.nvars, 2), dtype=np.float64)
        idx_bound_map = {idx(v): (lb, ub) for v, (lb, ub) in mood_params.PAD_params.bounds.items()}
        for i in range(idx.nvars):
            lb = idx_bound_map[i][0] if i in idx_bound_map else None
            ub = idx_bound_map[i][1] if i in idx_bound_map else None

            physical_bounds[i, 0] = lb if lb is not None else -cp.inf
            physical_bounds[i, 1] = ub if ub is not None else cp.inf

        compute_dmp(_qold_, _M_, _m_, active_dims, mood_params.NAD_params.include_corners)

        detect_troubles_kernel_helper(
            _qnew_,
            _M_,
            _m_,
            NAD_mask,
            _alpha_,
            _wnew_,
            physical_bounds,
            _troubles_,
            mood_params.NAD_params.use_NAD,
            mood_params.NAD_params.SED_params.use_SED,
            mood_params.PAD_params.use_PAD,
            mood_params.NAD_params.delta,
            mood_params.NAD_params.rtol,
            mood_params.NAD_params.atol,
        )
    else:
        # Detect troubled cells with NumPy
        if mood_params.NAD_params.use_NAD:
            numerical_admissibility_detection(
                _qold_, _qnew_, _troubles_, _alpha_, idx, active_dims, mood_params.NAD_params
            )

        if mood_params.PAD_params.use_PAD:
            for v, (lb, ub) in mood_params.PAD_params.bounds.items():
                if lb is not None:
                    np.maximum(_troubles_, _wnew_[idx(v)] < lb, out=_troubles_)
                if ub is not None:
                    np.maximum(_troubles_, _wnew_[idx(v)] > ub, out=_troubles_)

    # Update _troubles_ ghost cells
    apply_troubles_bc(_troubles_, mesh.nghost, bc_params)

    # Return number of revisable troubled cells
    xp.minimum(_troubles_[interior], _cascade_idx_[interior] < n_cascade, out=revisable_troubles)
    return xp.sum(revisable_troubles).item()


def init_mood(
    _F_base_: ArrayLike,
    _G_base_: ArrayLike,
    _H_base_: ArrayLike,
    _F_fallback_: ArrayLike,
    _G_fallback_: ArrayLike,
    _H_fallback_: ArrayLike,
    _cascade_idx_: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
):
    """
    Reset _cascade_idx_ and assign the base fluxes, ...
    """
    _cascade_idx_[...] = 0

    # Copy initial fluxes
    if "x" in active_dims:
        _F_fallback_[0, ...] = _F_base_
    if "y" in active_dims:
        _G_fallback_[0, ...] = _G_base_
    if "z" in active_dims:
        _H_fallback_[0, ...] = _H_base_


def map_cells_values_to_face_values(_cell_values_: ArrayLike, axis: int) -> ArrayLike:
    """
    Return an array of face values obtained by taking the maximum of the adjacent
    cell values along the specified axis.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_cell_values_, cp.ndarray) else np

    fv_shape = list(_cell_values_.shape)
    fv_shape[axis] -= 1

    lft_slc = crop(axis, (None, -1), ndim=4)
    rgt_slc = crop(axis, (1, None), ndim=4)

    _face_values_ = _cell_values_[lft_slc].copy()
    xp.maximum(_face_values_, _cell_values_[rgt_slc], out=_face_values_)
    return _face_values_


def blend_troubled_cells(
    _troubles_: ArrayLike, active_dims: Tuple[Literal["x", "y", "z"], ...]
) -> ArrayLike:
    """
    Return an floating-point array of troubled cells blended with their neighbors.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_troubles_, cp.ndarray) else np

    ndim = len(active_dims)

    _blended_ = xp.empty_like(_troubles_, dtype=xp.float64)
    _blended_[...] = _troubles_

    if ndim == 1:
        axis = DIM_TO_AXIS[active_dims[0]]
        lft_slc = crop(axis, (None, -1), ndim=4)
        rgt_slc = crop(axis, (1, None), ndim=4)

        # First neighbors
        _blended_[lft_slc] = xp.maximum(0.75 * _troubles_[rgt_slc], _blended_[lft_slc])
        _blended_[rgt_slc] = xp.maximum(0.75 * _troubles_[lft_slc], _blended_[rgt_slc])

        # Second neighbors
        _blended_[lft_slc] = xp.maximum(0.25 * (_blended_[rgt_slc] > 0), _blended_[lft_slc])
        _blended_[rgt_slc] = xp.maximum(0.25 * (_blended_[lft_slc] > 0), _blended_[rgt_slc])
    elif ndim == 2:
        axis1 = DIM_TO_AXIS[active_dims[0]]
        axis2 = DIM_TO_AXIS[active_dims[1]]

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
    _cv_mask_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    nghost: int,
) -> ArrayLike:
    """
    Slice a cell-centered mask with ghost cells to obtain a face-centered mask with
    no ghost cells.
    """
    _fv_ = map_cells_values_to_face_values(_cv_mask_, DIM_TO_AXIS[dim])
    interior = merge_slices(
        *[
            crop(
                DIM_TO_AXIS[d],
                (nghost - 1, -nghost + 1) if d == dim else (nghost, -nghost),
                ndim=4,
            )
            for d in active_dims
        ]
    )
    return _fv_[interior]


def blend_fluxes(
    F_high_order: ArrayLike, F_fallback: ArrayLike, theta: ArrayLike, F_blended: ArrayLike
):
    """
    Return a blended flux array based on the high-order fluxes, fallback fluxes, and a
    face-centered mask of troubled cells.
    """
    if CUPY_AVAILABLE and isinstance(F_high_order, cp.ndarray):
        blend_fluxes_elementwise_kernel(F_high_order, F_fallback, theta, F_blended)
        return
    F_blended[...] = (1 - theta) * F_high_order + theta * F_fallback


def assign_blended_fluxes(
    _cascade_idx_: ArrayLike,
    F_fallback: ArrayLike,
    G_fallback: ArrayLike,
    H_fallback: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
    H: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    nghost: int,
    blend_troubles: bool,
    i_max_computed: int,
):
    """
    Update the flux arrays "F", ... by blending the high-order and fallback fluxes based on
    a blended troubled cell mask.
    """
    if i_max_computed != 1:
        raise NotImplementedError(
            "Blending is only implemented for a single pre-computed fallback scheme "
            "(i_max_computed=1)."
        )

    if blend_troubles:
        _blended_troubles_ = blend_troubled_cells(_cascade_idx_, active_dims)
    else:
        _blended_troubles_ = _cascade_idx_.astype(np.float64)

    for dim in active_dims:
        _theta_ = get_face_mask(_blended_troubles_, dim, active_dims, nghost)
        F_in = {"x": F_fallback, "y": G_fallback, "z": H_fallback}[dim]
        F_out = {"x": F, "y": G, "z": H}[dim]
        blend_fluxes(F_in[0, ...], F_in[1, ...], _theta_, F_out)


def add_specified_fluxes(cascade_idx_arr: ArrayLike, F_i: ArrayLike, F_total: ArrayLike, i: int):
    """
    Add the specified fluxes F_i to F_total based on the cascade index array.
    """
    if CUPY_AVAILABLE and isinstance(F_total, cp.ndarray):
        add_specified_flux_elementwise_kernel(F_i, cascade_idx_arr, i, F_total)
    else:
        F_total += (cascade_idx_arr == i) * F_i


def assign_fluxes(
    _cascade_idx_: ArrayLike,
    F_fallback: ArrayLike,
    G_fallback: ArrayLike,
    H_fallback: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
    H: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    nghost: int,
    mood_params: MOOD_Parameters,
    i_max_computed: int,
):
    """
    Update the flux arrays "F", ... based on the current "_cascade_idx_"
    """
    if len(mood_params.fallback_cascade) == 1:
        assign_blended_fluxes(
            _cascade_idx_,
            F_fallback,
            G_fallback,
            H_fallback,
            F,
            G,
            H,
            active_dims,
            nghost,
            mood_params.blend_troubles,
            i_max_computed,
        )
        return

    xp = cp if CUPY_AVAILABLE and isinstance(_cascade_idx_, cp.ndarray) else np

    # Assign fluxes based on cascade index
    for dim in active_dims:
        F_in = {"x": F_fallback, "y": G_fallback, "z": H_fallback}[dim]
        F_out = {"x": F, "y": G, "z": H}[dim]

        cascade_face_mask = get_face_mask(_cascade_idx_, dim, active_dims, nghost)
        combined_fluxes = xp.zeros_like(F_in[0, ...])
        for i in range(i_max_computed + 1):
            add_specified_fluxes(cascade_face_mask, F_in[i, ...], combined_fluxes, i)
        F_out[...] = combined_fluxes


def mood_loop(
    _uold_: ArrayLike,
    _wold_: ArrayLike,
    _unew_: ArrayLike,
    _wnew_: ArrayLike,
    _alpha_: ArrayLike,
    _theta_,
    _qcc_,
    _troubles_: ArrayLike,
    revisable_troubles: ArrayLike,
    _cascade_idx_: ArrayLike,
    _F_: ArrayLike,
    _G_: ArrayLike,
    _H_: ArrayLike,
    _F_fallback_: ArrayLike,
    _G_fallback_: ArrayLike,
    _H_fallback_: ArrayLike,
    S: ArrayLike,
    idx: VariableIndexMap,
    t: float,
    dt: float,
    mesh: UniformFiniteVolumeMesh,
    base_scheme: FV_SchemeParameters,
    bc_params: BoundaryConditionParameters,
    hydro_params: HydroParameters,
    substep_summary: SubstepSummary,
    timer: Optional[MultiTimer] = None,
):
    """
    Revise flux arrays "F", .. until no more revisable troubled cells are detected or the maximum
    number of revisions is reached. Update substep summary.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_uold_, cp.ndarray)
    xp = cp if using_cupy else np

    mood_params = base_scheme.mood_params
    n_cascade = len(mood_params.fallback_cascade)
    active_dims = mesh.active_dims
    using_x = "x" in active_dims
    using_y = "y" in active_dims
    using_z = "z" in active_dims
    interior = get_interior_view(active_dims, mesh.nghost)
    Finterior = replace_slice(interior, DIM_TO_AXIS["x"], slice(None))
    Ginterior = replace_slice(interior, DIM_TO_AXIS["y"], slice(None))
    Hinterior = replace_slice(interior, DIM_TO_AXIS["z"], slice(None))

    # Initialize MOOD arrays
    init_mood(_F_, _G_, _H_, _F_fallback_, _G_fallback_, _H_fallback_, _cascade_idx_, active_dims)
    i_max_computed: int = 0

    for _ in range(mood_params.max_revs):
        timer is not None and timer.start("candidate_solution", using_cupy)  # TIMER START
        compute_candidate_solution(
            _uold_,
            _F_[Finterior] if using_x else np.array([]),
            _G_[Ginterior] if using_y else np.array([]),
            _H_[Hinterior] if using_z else np.array([]),
            S,
            _unew_,
            _wnew_,
            idx,
            t,
            dt,
            mesh,
            bc_params,
            hydro_params,
        )
        timer is not None and timer.stop("candidate_solution", using_cupy)  # TIMER STOP
        timer is not None and timer.start("detect_troubles", using_cupy)  # TIMER START
        n_troubles = detect_troubled_cells(
            _uold_,
            _wold_,
            _unew_,
            _wnew_,
            _troubles_,
            _cascade_idx_,
            revisable_troubles,
            _alpha_,
            idx,
            mesh,
            base_scheme,
            bc_params,
        )
        substep_summary.n_troubles_hist.append(n_troubles)
        timer is not None and timer.stop("detect_troubles", using_cupy)  # TIMER STOP

        # Do not revise if no revisable troubled cells
        if n_troubles == 0:
            return

        # Update cascade index and compute new fluxes
        xp.minimum(_cascade_idx_ + _troubles_, n_cascade, out=_cascade_idx_)

        i_max = xp.max(_cascade_idx_).item()
        if i_max == i_max_computed + 1:
            next_fallback_scheme = mood_params.fallback_cascade[i_max - 1]
            timer is not None and timer.start("fallback_fluxes", using_cupy)  # TIMER START
            update_fv_fluxes(
                _uold_,
                _wold_,
                _F_fallback_[i_max, ...] if "x" in active_dims else np.array([]),
                _G_fallback_[i_max, ...] if "y" in active_dims else np.array([]),
                _H_fallback_[i_max, ...] if "z" in active_dims else np.array([]),
                _theta_,
                _qcc_,
                _alpha_,
                idx,
                active_dims,
                mesh,
                next_fallback_scheme,
                hydro_params,
            )
            timer is not None and timer.stop("fallback_fluxes", using_cupy)  # TIMER STOP
            i_max_computed += 1
        elif i_max != i_max_computed:
            raise RuntimeError(f"Unexpected {i_max=} with {i_max_computed=}.")

        # Update state
        substep_summary.n_MOOD_revisions += 1
        timer is not None and timer.start("assign_fluxes", using_cupy)  # TIMER START
        assign_fluxes(
            _cascade_idx_,
            _F_fallback_[insert_slice(Finterior, 0, slice(None))] if using_x else np.array([]),
            _G_fallback_[insert_slice(Ginterior, 0, slice(None))] if using_y else np.array([]),
            _H_fallback_[insert_slice(Hinterior, 0, slice(None))] if using_z else np.array([]),
            _F_[Finterior] if using_x else np.array([]),
            _G_[Ginterior] if using_y else np.array([]),
            _H_[Hinterior] if using_z else np.array([]),
            active_dims,
            mesh.nghost,
            base_scheme.mood_params,
            i_max_computed,
        )
        timer is not None and timer.stop("assign_fluxes", using_cupy)  # TIMER STOP

    if mood_params.detect_closing_troubles:
        timer is not None and timer.start("candidate_solution", using_cupy)  # TIMER START
        compute_candidate_solution(
            _uold_,
            _F_[Finterior] if using_x else np.array([]),
            _G_[Ginterior] if using_y else np.array([]),
            _H_[Hinterior] if using_z else np.array([]),
            S,
            _unew_,
            _wnew_,
            idx,
            t,
            dt,
            mesh,
            bc_params,
            hydro_params,
        )
        timer is not None and timer.stop("candidate_solution", using_cupy)  # TIMER STOP
        timer is not None and timer.start("detect_troubles", using_cupy)  # TIMER START
        n_closing_troubles = detect_troubled_cells(
            _uold_,
            _wold_,
            _unew_,
            _wnew_,
            _troubles_,
            _cascade_idx_,
            revisable_troubles,
            _alpha_,
            idx,
            mesh,
            base_scheme,
            bc_params,
        )
        substep_summary.n_troubles_hist.append(n_closing_troubles)
        timer is not None and timer.stop("detect_troubles", using_cupy)  # TIMER STOP


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    blend_fluxes_elementwise_kernel = cp.ElementwiseKernel(
        in_params="float64 F_high_order, float64 F_fallback, float64 theta",
        out_params="float64 F_blended",
        operation="F_blended = (1 - theta) * F_high_order + theta * F_fallback;",
        name="blend_fluxes_elementwise_kernel",
        no_return=True,
    )

    add_specified_flux_elementwise_kernel = cp.ElementwiseKernel(
        in_params="float64 F_i, int32 cascade_idx_arr, int32 target_idx",
        out_params="float64 F_total",
        operation="""
        if (cascade_idx_arr == target_idx) {
            F_total += F_i;
        }
        """,
        name="add_specified_flux_elementwise_kernel",
        no_return=True,
    )

    detect_troubles_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void detect_troubles_kernel(
            const double* qnew,
            const double* __restrict__ M,
            const double* __restrict__ m,
            const int* __restrict__ NAD_mask,
            const double* __restrict__ alpha,
            const double* wnew,
            const double* __restrict__ physical_bounds,
            int* __restrict__ troubles,
            const bool use_NAD,
            const bool use_SED,
            const bool use_PAD,
            const bool delta_NAD,
            const double rtol,
            const double atol,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ) {
            // qnew                     (nvars, nx, ny, nz), NAD argument
            // M                        (nvars, nx, ny, nz), NAD argument
            // m                        (nvars, nx, ny, nz), NAD argument
            // NAD_mask                 (nvars,), NAD argument
            // alpha                    (nvars, nx, ny, nz), NAD argument
            // wnew                     (nvars, nx, ny, nz), PAD argument
            // physical_bounds          (nvars, 2), PAD argument
            // troubles                 (1, nx, ny, nz), output

            const long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long nxyz = (long long)nx * (long long)ny * (long long)nz;

            for (long long ixyz = tid; ixyz < nxyz; ixyz += stride) {
                long long t = ixyz;
                const int iz = t % nz; t /= nz;
                const int iy = t % ny; t /= ny;
                const int ix = t % nx; t /= nx;

                bool found_trouble = false;

                for (int iv = 0; iv < nvars; iv++) {
                    const long long i = (((long long)iv * nx + ix) * ny + iy) * nz + iz;

                    // - - - compute NAD violations - - -
                    bool violates_NAD = false;

                    if (use_NAD && NAD_mask[iv] == 1) {
                        double lower_bound;
                        double upper_bound;

                        if (delta_NAD) {
                            double delta = M[i] - m[i];
                            lower_bound = m[i] - rtol * delta - atol;
                            upper_bound = M[i] + rtol * delta + atol;
                        } else {
                            lower_bound = m[i] - rtol * fabs(m[i]) - atol;
                            upper_bound = M[i] + rtol * fabs(M[i]) + atol;
                        }

                        if (qnew[i] < lower_bound || qnew[i] > upper_bound) {
                            violates_NAD = true;
                        }

                        // - - - apply SED relaxation - - -
                        if (use_SED && alpha[i] >= 1.0) {
                            violates_NAD = false;
                        }
                    }

                    // - - - compute PAD violations - - -
                    bool violates_PAD = false;

                    if (use_PAD) {
                        double lower_PAD_bound = physical_bounds[iv * 2];
                        double upper_PAD_bound = physical_bounds[iv * 2 + 1];


                        if (wnew[i] < lower_PAD_bound || wnew[i] > upper_PAD_bound) {
                            violates_PAD = true;
                        }
                    }

                    // update troubled status
                    if (violates_NAD || violates_PAD) {
                        found_trouble = true;
                        break;
                    }
                }
                troubles[ixyz] = found_trouble ? 1 : 0;
            }
        }
        """,
        name="detect_troubles_kernel",
    )

    def detect_troubles_kernel_helper(
        qnew: cp.ndarray,
        M: cp.ndarray,
        m: cp.ndarray,
        NAD_mask: cp.ndarray,
        alpha: cp.ndarray,
        wnew: cp.ndarray,
        physical_bounds: cp.ndarray,
        troubles: cp.ndarray,
        use_NAD: bool,
        use_SED: bool,
        use_PAD: bool,
        delta_NAD: bool,
        rtol: float,
        atol: float,
    ) -> Tuple[slice, ...]:
        if not qnew.flags.c_contiguous or qnew.dtype != cp.float64:
            raise ValueError("qnew must be a C-contiguous array of dtype float64.")
        if not M.flags.c_contiguous or M.dtype != cp.float64 or M.shape != qnew.shape:
            raise ValueError("M must be a C-contiguous float64 array with same shape as qnew.")
        if not m.flags.c_contiguous or m.dtype != cp.float64 or m.shape != qnew.shape:
            raise ValueError("M must be a C-contiguous float64 array with same shape as qnew.")
        if (
            not NAD_mask.flags.c_contiguous
            or NAD_mask.dtype != cp.int32
            or NAD_mask.shape != (qnew.shape[0],)
        ):
            raise ValueError(
                "NAD_mask must be a C-contiguous array of shape (nvars,) and dtype int32, where "
                "nvars is the first dimension of qnew."
            )
        if not alpha.flags.c_contiguous or alpha.dtype != cp.float64 or alpha.shape != qnew.shape:
            raise ValueError(
                "alpha must be a C-contiguous array of dtype float64 with same shape as qnew."
            )
        if not wnew.flags.c_contiguous or wnew.dtype != cp.float64 or wnew.shape != qnew.shape:
            raise ValueError(
                "wnew must be a C-contiguous array of dtype float64 with same shape as qnew."
            )
        if (
            not physical_bounds.flags.c_contiguous
            or physical_bounds.dtype != cp.float64
            or physical_bounds.shape != (qnew.shape[0], 2)
        ):
            raise ValueError(
                "physical_bounds must be a C-contiguous array of shape (nvars, 2) and dtype float64, where "
                "nvars is the first dimension of qnew."
            )
        if (
            not troubles.flags.c_contiguous
            or troubles.dtype != cp.int32
            or troubles.shape != (1, *qnew.shape[1:4])
        ):
            raise ValueError(
                "troubles must be a C-contiguous array of shape (1, nx, ny, nz) and dtype int32, where "
                "(nx, ny, nz) are the spatial dimensions of qnew."
            )

        # launch kernel
        nvars, nx, ny, nz = wnew.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        n_blocks = (nx * ny * nz + threads_per_block - 1) // threads_per_block

        detect_troubles_kernel(
            (n_blocks,),
            (threads_per_block,),
            (
                qnew,
                M,
                m,
                NAD_mask,
                alpha,
                wnew,
                physical_bounds,
                troubles,
                use_NAD,
                use_SED,
                use_PAD,
                delta_NAD,
                rtol,
                atol,
                nvars,
                nx,
                ny,
                nz,
            ),
        )

        return (slice(None), slice(None), slice(None), slice(None))
