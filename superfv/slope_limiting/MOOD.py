from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.boundary_conditions import BCs, apply_bc
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.interpolation_schemes import InterpolationScheme, LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.physical_admissibility_detection import (
    detect_PAD_violations,
)
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

# custom type for fluxes
Fluxes = Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]


@dataclass(frozen=True, slots=True)
class MOODConfig(LimiterConfig):
    """
    Configuration for the MOOD framework.

    Attributes:
        shock_detection: Whether to enable shock detection.
        smooth_extrema_detection: Whether to enable smooth extrema detection.
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions if smooth
            extrema detection is enabled. Uniform regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        uniformity_tol: Tolerance for uniformity check when check_uniformity is True.
        numerical_admissibility_detection: Whether to enable numerical admissibility
            detection (NAD).
        delta: Whether to compute NAD violations using the range of the local maximum
            principle (DMP) to relax the bounds. If False, only rtol is used to relax
            the bounds.
        cascade: The list of interpolation schemes to use in the MOOD framework.
        blend: Whether to blend troubled cells when using a fallback scheme. Only valid
            for cascades of length 2.
        max_iters: The maximum number of iterations to perform.
        include_corners: Whether to include corner cells when computing the local
            minima and maxima.
        NAD_rtol: Factor by which the local DMP range is multipliedf, forming a product
            that is used to widen the bounds for numerical admissibility.
        NAD_atol: Absolute tolerance used to widen the bounds for numerical
            admissibility. Ignored if `delta` is False.
        scale_NAD_rtol_by_dt: Whether to scale the NAD rtol by the time step size dt.
        skip_trouble_counts: Whether to skip counting the number of troubled cells. If
            True, `detect_troubled_cells` will return (-1, -1) always. This can be used
            to avoid CUDA synchronization overhead when the troubled cell count is not
            needed.
        detect_closing_troubles: Whether to detect closing troubles at the end of the
            MOOD loop if revisable troubled cells were found during the last iteration.
            If False, the troubles array will represent the troubled cells that
            determined the closing cascade index.
    """

    numerical_admissibility_detection: bool = False
    delta: bool = True
    cascade: List[InterpolationScheme] = field(default_factory=list)
    blend: bool = False
    max_iters: int = 0
    include_corners: bool = False
    NAD_rtol: float = 1e-5
    NAD_atol: float = 0.0
    scale_NAD_rtol_by_dt: bool = False
    skip_trouble_counts: bool = False
    detect_closing_troubles: bool = True

    def __post_init__(self):
        LimiterConfig.__post_init__(self)
        if self.smooth_extrema_detection and not self.numerical_admissibility_detection:
            raise ValueError("SED requires NAD to be enabled.")
        if self.blend and len(self.cascade) != 2:
            raise ValueError("Blending is only supported for cascades of length 2")

    def key(self) -> str:
        cascade_keys = ", ".join([scheme.key() for scheme in self.cascade])
        return f"MOOD: [{cascade_keys}]"

    def to_dict(self) -> dict:
        out = LimiterConfig.to_dict(self)
        out.update(
            dict(
                numerical_admissibility_detection=self.numerical_admissibility_detection,
                delta=self.delta,
                cascade=[scheme.to_dict() for scheme in self.cascade],
                blend=self.blend,
                max_iters=self.max_iters,
                include_corners=self.include_corners,
                NAD_rtol=self.NAD_rtol,
                NAD_atol=self.NAD_atol,
                scale_NAD_rtol_by_dt=self.scale_NAD_rtol_by_dt,
                skip_trouble_counts=self.skip_trouble_counts,
                detect_closing_troubles=self.detect_closing_troubles,
            )
        )
        return out


@dataclass
class MOODState:
    """
    Class that describes the state of the MOOD iterator.

    Attributes:
        iter_idx: Current iteration index in the MOOD loop.
        iter_count: Total number of iterations across all MOOD loops in a step.
        fine_iter_count: Number of iterations in the current MOOD loop.
        cascade_idx: Current index in the cascade of schemes.
    """

    def __post_init__(self):
        self.reset_MOOD_loop()
        self.reset_stepwise_counters()

    def reset_stepwise_counters(self):
        """
        Reset logs which track information over the course of whole steps.
        """
        self.iter_count = 0
        self.iter_count_hist: List[int] = []
        self.troubled_cell_count = 0
        self.troubled_cell_count_hist: List[int] = []

    def increment_substep_hists(self):
        """
        Increment the substep history logs.
        """
        self.iter_count_hist.append(self.fine_iter_count)
        self.troubled_cell_count_hist.append(self.fine_troubled_cell_count)

    def reset_MOOD_loop(self):
        """
        Reset the parameters which change across every MOOD iteration.
        """
        self.iter_idx = 0
        self.fine_iter_count = 0
        self.fine_troubled_cell_count = 0
        self.cascade_idx = 0

    def increment_MOOD_iteration(self):
        """
        Increment the parameters which change across every MOOD iteration.
        """
        self.iter_idx += 1
        self.iter_count += 1
        self.fine_iter_count += 1

    def update_troubled_cell_count(self, n: int):
        """
        Update the number of troubled cells:

        Args:
            n: The number of troubled cells to add.
        """
        self.fine_troubled_cell_count = n
        self.troubled_cell_count += n


def compute_fallback_fluxes(fv_solver: FiniteVolumeSolver, t: float):
    """
    Compute fallback fluxes for all schemes in the MOOD cascade.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.
    """
    cascade = fv_solver.MOOD_config.cascade
    active_dims = fv_solver.active_dims
    flux_names = fv_solver.flux_names
    arrays = fv_solver.arrays

    working_fluxes = {flux_names[dim]: arrays[flux_names[dim]] for dim in active_dims}

    for i, scheme in enumerate(cascade):
        if i > 0:
            fv_solver.compute_fluxes(t, scheme)

        for name, arr in working_fluxes.items():
            arrays[f"{name}_{scheme.key()}"] = arr

    high_order_key = cascade[0].key()
    for name in working_fluxes:
        working_fluxes[name][...] = arrays[f"{name}_{high_order_key}"]


def detect_troubled_cells(fv_solver: FiniteVolumeSolver, t: float) -> Tuple[int, int]:
    """
    Detect troubled cells in the finite volume solver.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.

    Returns:
        A tuple containing:
        - The number of revisable troubled cells detected.
        - The total number of troubled cells detected, including non-revisable ones.
    """
    # gather solver parameters
    idx = fv_solver.variable_index_map
    config = fv_solver.MOOD_config
    MOOD_state = fv_solver.MOOD_state
    xp = fv_solver.xp
    scheme = fv_solver.base_scheme
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    lim_slc = fv_solver.variable_index_map("limiting", keepdims=True)
    interior = fv_solver.interior
    dt = fv_solver.substep_dt

    # gather MOOD parameters
    iter_idx = MOOD_state.iter_idx
    cascade = config.cascade
    NAD = config.numerical_admissibility_detection
    delta = config.delta
    NAD_rtol = config.NAD_rtol
    NAD_atol = config.NAD_atol
    scale_NAD_rtol_by_dt = config.scale_NAD_rtol_by_dt
    include_corners = config.include_corners
    PAD = config.physical_admissibility_detection
    PAD_bounds = config.PAD_bounds
    SED = config.smooth_extrema_detection
    skip_trouble_counts = config.skip_trouble_counts

    # determine limiting style
    primNAD = scheme.flux_recipe in (2, 3)
    max_idx = len(cascade) - 1

    # allocate arrays
    uold = arrays["_u_"]
    wold = arrays["_w_"]
    unew = arrays["_unew_"]
    wnew = arrays["_wnew_"]
    dmp_M = arrays["_M_"]
    dmp_m = arrays["_m_"]
    alpha = arrays["_alpha_"]
    troubles = arrays["_troubles_"]
    troubles_temp = arrays["_troubles2_"]
    cascade_idx = arrays["_cascade_idx_"]
    NAD_violations = arrays["_NAD_violations_"]
    PAD_violations = arrays["_PAD_violations_"]

    # reset troubles / cascade index array
    troubles[...] = False
    if iter_idx == 0:
        cascade_idx[...] = 0

    # compute candidate solution
    unew[interior] = uold[interior] + dt * fv_solver.compute_RHS()
    fv_solver.apply_bc(t, unew, scheme=fv_solver.base_scheme)
    fv_solver.conservatives_to_primitives(unew, wnew)

    if fv_solver.cupy:
        # get lim_slc mask
        nvars = wnew.shape[0]
        if "NAD_mask" in arrays:
            NAD_mask = arrays["NAD_mask"]
        else:
            lim_idxs_list: List[int]

            lim_idxs = idx("limiting")
            if isinstance(lim_idxs, int):
                lim_idxs_list = [lim_idxs]
            elif isinstance(lim_idxs, slice):
                lim_idxs_list = list(range(*lim_idxs.indices(nvars)))
            elif isinstance(lim_idxs, np.ndarray):
                lim_idxs_list = [x.item() for x in lim_idxs]
            else:
                raise ValueError(
                    f"Unsupported slice type for lim_slc: {type(lim_idxs)}"
                )

            NAD_mask = xp.array(
                [1 if i in lim_idxs_list else 0 for i in range(nvars)], dtype=cp.int32
            )
            arrays.add("NAD_mask", NAD_mask)

        # detect troubled cells
        if SED:
            fv_solver.detect_smooth_extrema(wnew if primNAD else unew, scheme)
        compute_dmp(
            wold if primNAD else uold,
            dmp_M,
            dmp_m,
            active_dims,
            include_corners,
        )
        detect_troubles_kernel_helper(
            unew,
            wnew,
            dmp_M,
            dmp_m,
            NAD_mask,
            alpha,
            PAD_bounds if PAD_bounds is not None else xp.empty((nvars, 2)),
            NAD_violations,
            PAD_violations,
            troubles,
            NAD,
            PAD,
            SED,
            primNAD,
            delta,
            NAD_rtol * dt if scale_NAD_rtol_by_dt else NAD_rtol,
            NAD_atol,
        )
    else:
        if NAD:
            fv_solver.stepper_timer.start("detect_troubles:NAD")

            detect_NAD_violations(
                (wnew if primNAD else unew),
                (wold if primNAD else uold),
                active_dims=active_dims,
                delta=delta,
                rtol=NAD_rtol * dt if scale_NAD_rtol_by_dt else NAD_rtol,
                atol=NAD_atol,
                include_corners=include_corners,
                out=NAD_violations,
                M=dmp_M,
                m=dmp_m,
            )

            # compute smooth extrema
            if SED:
                fv_solver.detect_smooth_extrema((wnew if primNAD else unew), scheme)
                NAD_violations *= alpha < 1

            # update troubles
            np.any(NAD_violations[lim_slc], axis=0, keepdims=True, out=troubles)

            fv_solver.stepper_timer.stop("detect_troubles:NAD")

        # compute PAD violations
        if PAD:
            fv_solver.stepper_timer.start("detect_troubles:PAD")

            detect_PAD_violations(
                wnew,
                cast(ArrayLike, PAD_bounds),
                PAD_violations,
                troubles_temp,
            )
            troubles |= troubles_temp

            fv_solver.stepper_timer.stop("detect_troubles:PAD")

    # update troubles workspace
    apply_bc(
        xp,
        troubles,
        pad_width=fv_solver.bc_pad_width,
        mode=normalize_troubles_bc(fv_solver.bc_mode),
    )

    # trouble counts
    if skip_trouble_counts:
        n_troubled_cells = -1
        n_revisable_troubled_cells = -1
    else:
        n_troubled_cells = xp.sum(troubles[interior]).item()
        revisable_troubles = xp.logical_and(troubles, cascade_idx < max_idx)
        n_revisable_troubled_cells = xp.sum(revisable_troubles[interior]).item()

    return n_revisable_troubled_cells, n_troubled_cells


def detect_NAD_violations(
    u_new: np.ndarray,
    u_old: np.ndarray,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    delta: bool = True,
    rtol: float = 1e-5,
    atol: float = 0.0,
    include_corners: bool = False,
    *,
    out: np.ndarray,
    M: np.ndarray,
    m: np.ndarray,
) -> Tuple[slice, ...]:
    """
    Compute the numerical admissibility detection (NAD) violations, which violate the
    following criterion if delta is True:

    m - delta <= u_new <= M + delta

    where m and M are the local minima and maxima of u_old, respectively, and
    delta = rtol*(M-m) + gtol*(Mglob-mglob) + atol, where Mglob and mglob are the
    global maxima and minima of u_old for each variable.

    If delta is False, the criterion becomes:

    m - rtol * |m| <= u_new <= M + rtol * |M|

    Args:
        u_new: New solution array. Has shape (nvars, nx, ny, nz).
        u_old: Old solution array. Has shape (nvars, nx, ny, nz).
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        delta: Whether to compute NAD violations using the range of the local maximum
            principle (DMP) to relax the bounds. If False, only rtol is used to relax
            the bounds.
        rtol: Factor by which the local DMP range or absolute values are multiplied,
            forming a product that is used to widen the bounds for numerical
            admissibility.
        atol: Absolute tolerance used to widen the bounds for numerical admissibility.
            Ignored if `delta` is False.
        include_corners: Whether to include corner values in the DMP computation.
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
            Negative values in `out` indicate violations of the NAD criterion.
        M: Array to store the local maxima. Has shape (nvars, nx, ny, nz).
        m: Array to store the local minima. Has shape (nvars, nx, ny, nz).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # allocate arrays
    dmp_range = np.empty_like(u_old)
    lower_bound = np.empty_like(u_old)
    upper_bound = np.empty_like(u_old)
    lower_violations = np.empty_like(u_old)
    upper_violations = np.empty_like(u_old)
    delta_arr = np.zeros_like(u_old)

    # compute discrete maximum principle (dmp)
    modified = compute_dmp(u_old, M, m, active_dims, include_corners)

    if delta:
        dmp_range[...] = M - m
        delta_arr += rtol * dmp_range + atol

        lower_bound[...] = m - delta_arr
        upper_bound[...] = M + delta_arr
    else:
        lower_bound[...] = m - rtol * np.abs(m)
        upper_bound[...] = M + rtol * np.abs(M)

    lower_violations[...] = u_new - lower_bound
    upper_violations[...] = upper_bound - u_new
    out[modified] = 0.0
    np.minimum(lower_violations[modified], out[modified], out=out[modified])
    np.minimum(upper_violations[modified], out[modified], out=out[modified])

    return modified


@lru_cache(maxsize=None)
def normalize_troubles_bc(
    bc_tuple: Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]],
) -> Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]]:
    """
    Normalize the boundary conditions for the trouble detection.

    Args:
        bc_tuple: A tuple of boundary conditions for each dimension, where each
            boundary condition is a tuple of (left, right) BCs.

    Returns:
        A normalized tuple of boundary conditions for each dimension.
        - "none" for no boundary condition
        - "periodic" for periodic boundary condition
        - "zeros" for no-trouble boundary condition
    """

    def map_bc(bc: BCs) -> BCs:
        if bc == "none":
            return "none"
        elif bc == "periodic":
            return "periodic"
        else:
            return "zeros"

    return (
        (map_bc(bc_tuple[0][0]), map_bc(bc_tuple[0][1])),
        (map_bc(bc_tuple[1][0]), map_bc(bc_tuple[1][1])),
        (map_bc(bc_tuple[2][0]), map_bc(bc_tuple[2][1])),
    )


def revise_fluxes(fv_solver: FiniteVolumeSolver, t: float):
    """
    In-place revision of fluxes based on the current time step.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.
    """
    # gather solver parameters
    config = fv_solver.MOOD_config
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    interior = fv_solver.interior
    flux_names = fv_solver.flux_names
    cascade = config.cascade
    max_idx = len(cascade) - 1

    # early escape for single fallback scheme case
    if len(cascade) == 2:
        revise_fluxes_with_one_fallback_scheme(fv_solver, t)
        return

    # allocate arrays
    working_fluxes = {dim: arrays[flux_names[dim]] for dim in active_dims}
    _Fmask_ = arrays["_mask_"][:, :, :-1, :-1]
    _Gmask_ = arrays["_mask_"][:, :-1, :, :-1]
    _Hmask_ = arrays["_mask_"][:, :-1, :-1, :]
    masks = {"F": _Fmask_, "G": _Gmask_, "H": _Hmask_}
    _troubles_ = arrays["_troubles_"]
    _cascade_idx_ = arrays["_cascade_idx_"]

    # assuming `troubles` has just been updated, update the cascade index arrays
    xp.minimum(_cascade_idx_ + _troubles_, max_idx, out=_cascade_idx_)

    # broadcast cascade index to each face
    for dim, flux in working_fluxes.items():
        axis = DIM_TO_AXIS[dim]
        flux_name = flux_names[dim]
        mask = masks[flux_name]

        map_cells_values_to_face_values(xp, _cascade_idx_, axis, out=mask)

        flux[...] = 0
        for i, scheme in enumerate(cascade):
            in_mask = mask[interior] == i
            flux += in_mask * arrays[f"{flux_name}_{scheme.key()}"]


def revise_fluxes_with_one_fallback_scheme(fv_solver: FiniteVolumeSolver, t: float):
    """
    In-place revision of fluxes based on the current time step using a single fallback
    scheme.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.
    """
    # gather solver parameters
    config = fv_solver.MOOD_config
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    interior = fv_solver.interior
    flux_names = fv_solver.flux_names
    blend = config.blend
    cascade = config.cascade

    # parse schemes
    if len(cascade) != 2:
        raise ValueError(
            "Fallback scheme revision requires a cascade of exactly 2 schemes."
        )
    base_scheme = cascade[0]
    fallback_scheme = cascade[1]

    # allocate arrays
    working_fluxes = {dim: arrays[flux_names[dim]] for dim in active_dims}
    _Fmask_ = arrays["_fmask_"][:, :, :-1, :-1]
    _Gmask_ = arrays["_fmask_"][:, :-1, :, :-1]
    _Hmask_ = arrays["_fmask_"][:, :-1, :-1, :]
    masks = {"F": _Fmask_, "G": _Gmask_, "H": _Hmask_}
    _troubles_ = arrays["_troubles_"]
    _cascade_idx_ = arrays["_cascade_idx_"]
    _blended_cascade_idx_ = arrays["_blended_cascade_idx_"]  # float64 dtype
    _theta_ = arrays["_buffer1_"][..., 0]

    # assuming `troubles` has just been updated, update the cascade index arrays
    xp.minimum(_cascade_idx_ + _troubles_, 1, out=_cascade_idx_)
    _blended_cascade_idx_[...] = _cascade_idx_

    # blend cascade index
    if blend:
        blend_troubled_cells(
            xp,
            _blended_cascade_idx_,
            active_dims,
            theta=_theta_,
        )

    # broadcast cascade index to each face
    for dim, flux in working_fluxes.items():
        axis = DIM_TO_AXIS[dim]
        flux_name = flux_names[dim]
        mask = masks[flux_name]

        map_cells_values_to_face_values(xp, _blended_cascade_idx_, axis, out=mask)
        in_mask = mask[interior]
        flux[...] = 0
        flux += in_mask * arrays[f"{flux_name}_{fallback_scheme.key()}"]
        flux += (1 - in_mask) * arrays[f"{flux_name}_{base_scheme.key()}"]


def map_cells_values_to_face_values(
    xp: ModuleType, cv: ArrayLike, axis: int, *, out: ArrayLike
) -> Tuple[slice, ...]:
    """
    Map cell values to face values by taking the maximum of adjacent cell values.

    Args:
        xp: `np` namespace or equivalent.
        cv: Array of cell values. Has shape (nvars, nx, ny, nz).
        axis: Axis along which to map the cell values to face values:
            - 1 for "x" (nx+1, ny, nz, ...)
            - 2 for "y" (nx, ny+1, nz, ...)
            - 3 for "z" (nx, ny, nz+1, ...)
        out: Output array to store the face values. Has shape:
            - (nvars, nx+1, ny, nz) if axis == 1
            - (nvars, nx, ny+1, nz) if axis == 2
            - (nvars, nx, ny, nz+1) if axis == 3

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    lft_slc = crop(axis, (None, -1), ndim=4)
    rgt_slc = crop(axis, (1, None), ndim=4)
    modified = crop(axis, (1, -1), ndim=4)

    out[modified] = cv[lft_slc]
    xp.maximum(out[modified], cv[rgt_slc], out=out[modified])

    return modified


def blend_troubled_cells(
    xp: ModuleType,
    troubles: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    theta: ArrayLike,
    out: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Compute the blended troubled cell indicators.

    Args:
        xp: `np` namespace or equivalent.
        troubles: Array of troubled cell indicators. Has shape (1, nx, ny, nz). Must
            have float dtype.
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        theta: Buffer array with shape (1, nx, ny, nz) to store intermediate values.
        out: Output array to store the blended troubled cell indicators. If None,
            the `troubles` array will be modified in place. Has shape
            (1, nx, ny, nz).

    Returns:
        Slice objects indicating the modified regions in the troubles array.
    """
    ndim = len(active_dims)

    # initialize theta
    theta[...] = troubles

    if ndim == 1:
        axis = DIM_TO_AXIS[active_dims[0]]
        lft_slc = crop(axis, (None, -1), ndim=4)
        rgt_slc = crop(axis, (1, None), ndim=4)
        modified = crop(axis, (1, -1), ndim=4)

        # First neighbors
        theta[lft_slc] = xp.maximum(0.75 * troubles[rgt_slc], theta[lft_slc])
        theta[rgt_slc] = xp.maximum(0.75 * troubles[lft_slc], theta[rgt_slc])

        # Second neighbors
        theta[lft_slc] = xp.maximum(0.25 * (theta[rgt_slc] > 0), theta[lft_slc])
        theta[rgt_slc] = xp.maximum(0.25 * (theta[lft_slc] > 0), theta[rgt_slc])
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
        modified = merge_slices(
            crop(axis1, (1, -1), ndim=4), crop(axis2, (1, -1), ndim=4)
        )

        # First neighbors
        theta[lft_slc1] = xp.maximum(0.75 * troubles[rgt_slc1], theta[lft_slc1])
        theta[rgt_slc1] = xp.maximum(0.75 * troubles[lft_slc1], theta[rgt_slc1])
        theta[lft_slc2] = xp.maximum(0.75 * troubles[rgt_slc2], theta[lft_slc2])
        theta[rgt_slc2] = xp.maximum(0.75 * troubles[lft_slc2], theta[rgt_slc2])

        # Second neighbors
        theta[lft_lft] = xp.maximum(0.5 * troubles[rgt_rgt], theta[lft_lft])
        theta[lft_rgt] = xp.maximum(0.5 * troubles[rgt_lft], theta[lft_rgt])
        theta[rgt_lft] = xp.maximum(0.5 * troubles[lft_rgt], theta[rgt_lft])
        theta[rgt_rgt] = xp.maximum(0.5 * troubles[lft_lft], theta[rgt_rgt])

        # Third neighbors
        theta[lft_slc2] = xp.maximum(0.25 * (theta[rgt_slc2] > 0), theta[lft_slc2])
        theta[rgt_slc2] = xp.maximum(0.25 * (theta[lft_slc2] > 0), theta[rgt_slc2])
        theta[lft_slc1] = xp.maximum(0.25 * (theta[rgt_slc1] > 0), theta[lft_slc1])
        theta[rgt_slc1] = xp.maximum(0.25 * (theta[lft_slc1] > 0), theta[rgt_slc1])

    elif ndim == 3:
        raise NotImplementedError("3D blending is not implemented yet.")

    if out is None:
        troubles[modified] = theta[modified]
    else:
        out[modified] = theta[modified]

    return modified


def init_troubled_cell_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Initialize troubled cell statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    config = fv_solver.MOOD_config
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_troubles"] = []

    step_log["nfine_NAD_violations"] = []
    step_log["nfine_PAD_violations"] = []
    for var in idx.var_idx_map.keys():
        step_log[f"nfine_NAD_violations_{var}"] = []
        step_log[f"nfine_PAD_violations_{var}"] = []

    for i, _ in enumerate(config.cascade[1:], start=1):
        step_log[f"nfine_cascade_idx{i}"] = []


def clear_troubled_cell_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Clear troubled cell statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    config = fv_solver.MOOD_config
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_troubles"].clear()

    step_log["nfine_NAD_violations"].clear()
    step_log["nfine_PAD_violations"].clear()
    for var in idx.var_idx_map.keys():
        step_log[f"nfine_NAD_violations_{var}"].clear()
        step_log[f"nfine_PAD_violations_{var}"].clear()

    for i, _ in enumerate(config.cascade[1:], start=1):
        step_log[f"nfine_cascade_idx{i}"].clear()


def append_troubled_cell_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Append troubled cell statistics from the finite volume solver's arrays to the step
    log. Specifically, log the sum of cells with 1 - theta > 0 using various criteria:
        `nfine_troubles`: Number of troubled cells in the interior of the troubles
                            array.
        `nfine_cascade_idx{i}`: Number of cells in the interior of the troubles array
                                with cascade index i, for i in the cascade indices
                                corresponding to fallback schemes (i.e. excluding 0).

    Args:
        fv_solver: The finite volume solver instance.
    """
    config = fv_solver.MOOD_config
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    interior = fv_solver.interior
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    troubles = arrays["_troubles_"][interior]
    NAD_violations = arrays["_NAD_violations_"][interior]
    PAD_violations = arrays["_PAD_violations_"][interior]
    cascade_idx = arrays["_cascade_idx_"][interior]

    # track scalar quantities
    step_log["nfine_troubles"].append(xp.sum(troubles).item())

    nfine_NAD_violations = xp.sum(xp.any(NAD_violations, axis=0)).item()
    nfine_PAD_violations = xp.sum(xp.any(PAD_violations, axis=0)).item()
    step_log["nfine_NAD_violations"].append(nfine_NAD_violations)
    step_log["nfine_PAD_violations"].append(nfine_PAD_violations)
    for var in idx.var_idx_map.keys():
        nfine_NAD_violations_var = xp.sum(NAD_violations[idx(var)]).item()
        nfine_PAD_violations_var = xp.sum(PAD_violations[idx(var)]).item()
        step_log[f"nfine_NAD_violations_{var}"].append(nfine_NAD_violations_var)
        step_log[f"nfine_PAD_violations_{var}"].append(nfine_PAD_violations_var)

    for i, _ in enumerate(config.cascade[1:], start=1):
        n_cascade_idx_i = xp.sum(cascade_idx == i).item()
        step_log[f"nfine_cascade_idx{i}"].append(n_cascade_idx_i)


def log_troubled_cell_scalar_statistics(
    fv_solver: FiniteVolumeSolver, data: Dict[str, Any]
):
    """
    Log troubled cell statistics from the finite volume solver's step log into the
    provided data dictionary.

    Args:
        fv_solver: The finite volume solver instance.
        data: The dictionary to which statistics are logged.
    """
    step_log = fv_solver.step_log
    idx = fv_solver.variable_index_map
    config = fv_solver.MOOD_config
    state = fv_solver.MOOD_state

    def zero_max(lst: list[float]) -> float:
        return max(lst) if lst else 0.0

    new_data = {
        "nfine_troubles": step_log["nfine_troubles"].copy(),
        "nfine_NAD_violations": step_log["nfine_NAD_violations"].copy(),
        "nfine_PAD_violations": step_log["nfine_PAD_violations"].copy(),
        "n_troubles": zero_max(step_log["nfine_troubles"]),
        "n_NAD_violations": zero_max(step_log["nfine_NAD_violations"]),
        "n_PAD_violations": zero_max(step_log["nfine_PAD_violations"]),
    }

    for var in idx.var_idx_map.keys():
        for key in ["NAD_violations", "PAD_violations"]:
            new_data[f"nfine_{key}_{var}"] = step_log[f"nfine_{key}_{var}"].copy()
            new_data[f"n_{key}_{var}"] = zero_max(step_log[f"nfine_{key}_{var}"])

    for i, _ in enumerate(config.cascade[1:], start=1):
        new_data[f"nfine_cascade_idx{i}"] = step_log[f"nfine_cascade_idx{i}"].copy()
        new_data[f"n_cascade_idx{i}"] = zero_max(step_log[f"nfine_cascade_idx{i}"])

    data.update(new_data)

    # some more MOOD statistics from the MOOD state
    if sum(state.iter_count_hist) != state.iter_count:
        raise ValueError(
            "MOOD iteration count mismatch: "
            f"{state.iter_count_hist} != {state.iter_count}"
        )
    nfine = state.troubled_cell_count_hist
    if sum(nfine) != state.troubled_cell_count:
        raise ValueError(
            "MOOD troubled cell count mismatch: "
            f"{nfine} != {state.troubled_cell_count}"
        )
    if not config.skip_trouble_counts and nfine != step_log["nfine_troubles"]:
        raise ValueError(
            "MOOD troubled cell history mismatch: "
            f"{nfine} != {step_log['nfine_troubles_max']}"
        )
    data.update(
        {"n_MOOD_iters": state.iter_count, "nfine_MOOD_iters": state.iter_count_hist}
    )


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    detect_troubles_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void detect_troubles_kernel(
            const double* __restrict__ unew,
            const double* __restrict__ wnew,
            const double* __restrict__ M,
            const double* __restrict__ m,
            const int* __restrict__ NAD_mask,
            const double* __restrict__ alpha,
            const double* __restrict__ physical_bounds,
            double* __restrict__ NAD_violation_amounts,
            double* __restrict__ PAD_violation_amounts,
            int* __restrict__ troubles,
            const bool NAD,
            const bool SED,
            const bool PAD,
            const bool primNAD,
            const bool delta,
            const double NAD_rtol,
            const double NAD_atol,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ) {
            // unew                     (nvars, nx, ny, nz)
            // wnew                     (nvars, nx, ny, nz)
            // M                        (nvars, nx, ny, nz)
            // m                        (nvars, nx, ny, nz)
            // NAD_mask                 (nvars,)
            // alpha                    (nvars, nx, ny, nz)
            // physical_bounds          (nvars, 2)
            // NAD_violation_amounts    (nvars, nx, ny, nz)
            // PAD_violation_amounts    (nvars, nx, ny, nz)
            // troubles                 (1, nx, ny, nz)

            const long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long nxyz = (long long)nx * (long long)ny * (long long)nz;

            for (long long ixyz = tid; ixyz < nxyz; ixyz += stride) {
                long long t = ixyz;
                const int iz = t % nz; t /= nz;
                const int iy = t % ny; t /= ny;
                const int ix = t % nx; t /= nx;

                bool troubled = false;

                for (int iv = 0; iv < nvars; iv++) {
                    const long long i = (((long long)iv * nx + ix) * ny + iy) * nz + iz;

                    // - - - compute NAD violations - - -
                    double NAD_violation = 0.0;

                    if (NAD && NAD_mask[iv] == 1) {
                        double lower_NAD_bound, upper_NAD_bound;

                        if (delta) {
                            double delta_amount = NAD_rtol * (M[i] - m[i]) + NAD_atol;
                            lower_NAD_bound = m[i] - delta_amount;
                            upper_NAD_bound = M[i] + delta_amount;
                        } else {
                            lower_NAD_bound = m[i] - NAD_rtol * fabs(m[i]);
                            upper_NAD_bound = M[i] + NAD_rtol * fabs(M[i]);
                        }

                        double val2check = primNAD ? wnew[i] : unew[i];
                        NAD_violation = fmin(NAD_violation, val2check - lower_NAD_bound);
                        NAD_violation = fmin(NAD_violation, upper_NAD_bound - val2check);
                    }

                    // - - - apply SED relaxation - - -
                    if (SED && alpha[i] >= 1.0) {
                        NAD_violation = 0.0;
                    }

                    NAD_violation_amounts[i] = NAD_violation;

                    // - - - compute PAD violations - - -
                    double PAD_violation = 0.0;

                    if (PAD) {
                        double lower_PAD_bound = physical_bounds[iv * 2];
                        double upper_PAD_bound = physical_bounds[iv * 2 + 1];

                        PAD_violation = fmin(PAD_violation, wnew[i] - lower_PAD_bound);
                        PAD_violation = fmin(PAD_violation, upper_PAD_bound - wnew[i]);

                        PAD_violation_amounts[i] = PAD_violation;
                    }

                    // update troubled status
                    if (NAD_violation < 0.0 || PAD_violation < 0.0) {
                        troubled = true;
                    }
                }
                troubles[ixyz] = troubled ? 1 : 0;
            }
        }
        """,
        name="detect_troubles_kernel",
    )

    def detect_troubles_kernel_helper(
        unew: cp.ndarray,
        wnew: cp.ndarray,
        M: cp.ndarray,
        m: cp.ndarray,
        NAD_mask: cp.ndarray,
        alpha: cp.ndarray,
        physical_bounds: cp.ndarray,
        NAD_violation_amounts: cp.ndarray,
        PAD_violation_amounts: cp.ndarray,
        troubles: cp.ndarray,
        NAD: bool,
        PAD: bool,
        SED: bool,
        primNAD: bool,
        delta: bool,
        NAD_rtol: float,
        NAD_atol: float,
    ) -> Tuple[slice, ...]:
        if not wnew.flags.c_contiguous or wnew.dtype != cp.float64:
            raise ValueError("wnew must be a C-contiguous array of dtype float64.")
        if not M.flags.c_contiguous or M.dtype != cp.float64:
            raise ValueError("M must be a C-contiguous array of dtype float64.")
        if not m.flags.c_contiguous or m.dtype != cp.float64:
            raise ValueError("m must be a C-contiguous array of dtype float64.")
        if (
            not physical_bounds.flags.c_contiguous
            or physical_bounds.dtype != cp.float64
            or physical_bounds.ndim != 2
            or physical_bounds.shape[1] != 2
        ):
            raise ValueError(
                "physical_bounds must be a C-contiguous array of shape (nvars, 2) and "
                "dtype float64."
            )
        if (
            not NAD_violation_amounts.flags.c_contiguous
            or NAD_violation_amounts.dtype != cp.float64
        ):
            raise ValueError(
                "NAD_violation_amounts must be a C-contiguous array of dtype float64."
            )
        if (
            not PAD_violation_amounts.flags.c_contiguous
            or PAD_violation_amounts.dtype != cp.float64
        ):
            raise ValueError(
                "PAD_violation_amounts must be a C-contiguous array of dtype float64."
            )
        if not troubles.flags.c_contiguous or troubles.dtype != cp.int32:
            raise ValueError("troubles must be a C-contiguous array of dtype int32.")
        if troubles.shape != (1, wnew.shape[1], wnew.shape[2], wnew.shape[3]):
            raise ValueError(
                "troubles must have shape (1, nx, ny, nz) where (nx, ny, nz) are the "
                "spatial dimensions of wnew."
            )
        if not alpha.flags.c_contiguous or alpha.dtype != cp.float64:
            raise ValueError("alpha must be a C-contiguous array of dtype float64.")
        if alpha.shape != wnew.shape:
            raise ValueError("alpha must have the same shape as wnew.")

        # launch kernel
        nvars, nx, ny, nz = wnew.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        n_blocks = (nx * ny * nz + threads_per_block - 1) // threads_per_block

        detect_troubles_kernel(
            (n_blocks,),
            (threads_per_block,),
            (
                unew,
                wnew,
                M,
                m,
                NAD_mask,
                alpha,
                physical_bounds,
                NAD_violation_amounts,
                PAD_violation_amounts,
                troubles,
                NAD,
                SED,
                PAD,
                primNAD,
                delta,
                NAD_rtol,
                NAD_atol,
                nvars,
                nx,
                ny,
                nz,
            ),
        )

        return (slice(None), slice(None), slice(None), slice(None))
