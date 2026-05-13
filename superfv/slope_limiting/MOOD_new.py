from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

from superfv.configs import (
    FluxRecipe,
    FV_SchemeParameters,
    NumericalAdmissibilityParameters,
)
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import compute_alpha
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
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
    _wnew_ = sim.xp.empty_like(_uold_)

    # Reset troubled cells
    _troubles_[...] = 0.0

    # Compute candidate solution qnew
    _qnew_[interior] = _uold_[interior] + dt * sim.compute_time_derivative()
    sim.apply_bc(_qnew_, t, params.fv_scheme.p)
    sim.conservatives_to_primitives(_qnew_, _wnew_)

    # Assign _qold_ and _qnew_, the NAD arrays
    if params.fv_scheme.flux_recipe != FluxRecipe.CONS_LIM_PRIM:
        # NAD is applied to primitives
        sim.conservatives_to_primitives(_uold_, _qold_)
        _qnew_[...] = _wnew_
    else:
        # NAD is applied to conservatives
        _qold_[...] = _uold_[...]
        # _qnew_ is already conservative

    # Update smooth extrema
    if mood_params.NAD_params.use_NAD and mood_params.NAD_params.SED_params.use_SED:
        compute_alpha(_qold_, _alpha_, active_dims)

    if params.cupy:
        # Detect troubled cells with CuPy using a custom kernel
        _M_ = cp.empty_like(_qold_)
        _m_ = cp.empty_like(_qold_)

        if "NAD_mask" not in arrays:
            omit_vars_idxs = [idx(v) for v in mood_params.NAD_params.omit_vars]
            NAD_mask = cp.array(
                [int(i not in omit_vars_idxs) for i in range(idx.nvars)], dtype=np.int32
            )
            arrays["NAD_mask"] = NAD_mask

        if "PAD_bounds" not in arrays:
            physical_bounds = cp.empty((idx.nvars, 2), dtype=np.float64)
            idx_bound_map = {
                idx(v): (lb, ub) for v, (lb, ub) in mood_params.PAD_params.physical_bounds.items()
            }
            for i in range(idx.nvars):
                lb = idx_bound_map[i][0] if i in idx_bound_map else None
                ub = idx_bound_map[i][1] if i in idx_bound_map else None

                physical_bounds[i, 0] = lb if lb is not None else -cp.inf
                physical_bounds[i, 1] = ub if ub is not None else cp.inf
            arrays["PAD_bounds"] = physical_bounds

        compute_dmp(_qold_, _M_, _m_, active_dims, mood_params.NAD_params.include_corners)

        detect_troubles_kernel_helper(
            _qnew_,
            _M_,
            _m_,
            arrays["NAD_mask"],
            _alpha_,
            _wnew_,
            arrays["PAD_bounds"],
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
            for v, (lb, ub) in mood_params.PAD_params.physical_bounds.items():
                if lb is not None:
                    np.maximum(_troubles_, _wnew_[idx[v]] < lb, out=_troubles_)
                if ub is not None:
                    np.maximum(_troubles_, _wnew_[idx[v]] > ub, out=_troubles_)

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


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

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
