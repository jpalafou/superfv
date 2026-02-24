from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, cast

from superfv.axes import DIM_TO_AXIS
from superfv.boundary_conditions import BCs, apply_bc
from superfv.interpolation_schemes import InterpolationScheme, LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.physical_admissibility_detection import (
    detect_PAD_violations,
)
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import ArrayLike
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
        PAD_atol: Absolute tolerance for physical admissibility detection if
            physical_admissibility_detection is True.
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
            that is used to widen the bounds for numerical admissibility. Has shape
            (nvars,) or is treated as 0s if None.
        NAD_gtol: Factor by which the global range is multiplied, forming a product
            that is used to widen the bounds for numerical admissibility. Has shape
            (nvars,) or is treated as 0s if None.
        NAD_atol: Absolute tolerance used to widen the bounds for numerical
            admissibility. Has shape (nvars,) or is treated as 0s if None.
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
    NAD_rtol: Optional[ArrayLike] = None
    NAD_gtol: Optional[ArrayLike] = None
    NAD_atol: Optional[ArrayLike] = None
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
                NAD_rtol=None if self.NAD_rtol is None else self.NAD_rtol.tolist(),
                NAD_gtol=None if self.NAD_gtol is None else self.NAD_gtol.tolist(),
                NAD_atol=None if self.NAD_atol is None else self.NAD_atol.tolist(),
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
    active_dims = fv_solver.mesh.active_dims
    arrays = fv_solver.arrays

    F = arrays["F"]
    G = arrays["G"]
    H = arrays["H"]

    for i, scheme in enumerate(cascade):
        if i == 0:
            pass  # already computed high-order fluxes
        else:
            fv_solver.compute_fluxes(t, scheme)  # overwrites F, G, H

        if "x" in active_dims:
            arrays["F_" + scheme.key()] = F.copy()
        if "y" in active_dims:
            arrays["G_" + scheme.key()] = G.copy()
        if "z" in active_dims:
            arrays["H_" + scheme.key()] = H.copy()

    # copy high-order fluxes back to F, G, H
    if "x" in active_dims:
        F[...] = arrays["F_" + cascade[0].key()]
    if "y" in active_dims:
        G[...] = arrays["G_" + cascade[0].key()]
    if "z" in active_dims:
        H[...] = arrays["H_" + cascade[0].key()]


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

    Notes:
        - The required buffer shape depends on whether smooth extrema detection (SED)
            is enabled and on the number of active dimensions:
            - SED is not enabled: (nvars, nx, ny, nz, 2)
    """
    # gather solver parameters
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
    NAD_gtol = config.NAD_gtol
    NAD_atol = config.NAD_atol
    scale_NAD_rtol_by_dt = config.scale_NAD_rtol_by_dt
    include_corners = config.include_corners
    PAD = config.physical_admissibility_detection
    PAD_bounds = config.PAD_bounds
    PAD_atol = config.PAD_atol
    SED = config.smooth_extrema_detection
    skip_trouble_counts = config.skip_trouble_counts

    # determine limiting style
    primitive_NAD = scheme.flux_recipe in (2, 3)
    max_idx = len(cascade) - 1

    # allocate arrays
    _u_old_ = arrays["_u_"]
    _w_old_ = arrays["_w_"]
    _u_new_ = arrays["_unew_"]
    _w_new_ = arrays["_wnew_"]
    _M_ = arrays["_M_"]
    _m_ = arrays["_m_"]
    _NAD_violations_ = arrays["_NAD_violations_"]
    _alpha_ = arrays["_alpha_"][lim_slc]
    _buff_ = arrays["_buffer_"]
    _troubles_ = arrays["_troubles_"]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_ = arrays["_cascade_idx_"]
    _var_PAD_ = arrays["_var_PAD_"]
    _any_PAD_ = arrays["_any_PAD_"]
    troubles = arrays["troubles"]
    revisable_troubles = arrays["revisable_troubles"]
    cascade_idx = arrays["cascade_idx"]

    # reset troubles / cascade index array
    troubles[...] = False
    if iter_idx == 0:
        _cascade_idx_[...] = 0
        cascade_idx[...] = 0

    # compute candidate solution
    _u_new_[interior] = _u_old_[interior] + dt * fv_solver.compute_RHS()
    fv_solver.apply_bc(t, _u_new_, scheme=fv_solver.base_scheme)
    _w_new_[...] = fv_solver.primitives_from_conservatives(_u_new_)

    if NAD:
        NAD_rtol_local: Optional[ArrayLike] = None
        NAD_gtol_local: Optional[ArrayLike] = None
        NAD_atol_local: Optional[ArrayLike] = None

        if NAD_rtol is not None:
            NAD_rtol_local = NAD_rtol.copy()
            if scale_NAD_rtol_by_dt:
                NAD_rtol_local *= dt
        if NAD_gtol is not None:
            NAD_gtol_local = NAD_gtol
        if NAD_atol is not None:
            NAD_atol_local = NAD_atol

        detect_NAD_violations(
            xp,
            (_w_new_ if primitive_NAD else _u_new_),
            (_w_old_ if primitive_NAD else _u_old_),
            active_dims=active_dims,
            delta=delta,
            rtol=NAD_rtol_local,
            gtol=NAD_gtol_local,
            atol=NAD_atol_local,
            include_corners=include_corners,
            out=_NAD_violations_,
            M=_M_,
            m=_m_,
            buffer=_buff_,
        )
        NAD_violations = _NAD_violations_[interior][lim_slc]

        # compute smooth extrema
        if SED:
            fv_solver.detect_smooth_extrema(
                (_w_new_ if primitive_NAD else _u_new_), scheme
            )
            alpha = _alpha_[..., 0][interior]
            troubles[lim_slc] = xp.logical_and(NAD_violations < 0, alpha < 1)
        else:
            troubles[lim_slc] = NAD_violations < 0

    # compute PAD violations
    if PAD:
        detect_PAD_violations(
            xp,
            _w_new_,
            cast(ArrayLike, PAD_bounds),
            PAD_atol,
            violated_vars=_var_PAD_,
            violated_cells=_any_PAD_,
        )
        xp.logical_or(troubles, _var_PAD_[interior], out=troubles)

    # update troubles workspace
    _troubles_[interior] = troubles
    apply_bc(
        xp,
        _troubles_,
        pad_width=fv_solver.bc_pad_width,
        mode=normalize_troubles_bc(fv_solver.bc_mode),
    )

    # trouble counts
    if skip_trouble_counts:
        n_troubled_cells = -1
        n_revisable_troubled_cells = -1
    else:
        _any_troubles_[...] = xp.any(_troubles_, axis=0, keepdims=True)
        any_troubles = _any_troubles_[interior]

        n_troubled_cells = xp.sum(any_troubles).item()
        revisable_troubles[...] = any_troubles & (cascade_idx < max_idx)
        n_revisable_troubled_cells = xp.sum(revisable_troubles).item()

    return n_revisable_troubled_cells, n_troubled_cells


def detect_NAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    u_old: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    delta: bool = True,
    rtol: Optional[ArrayLike] = None,
    gtol: Optional[ArrayLike] = None,
    atol: Optional[ArrayLike] = None,
    include_corners: bool = False,
    *,
    out: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    buffer: ArrayLike,
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
        xp: `np` namespace or equivalent.
        u_new: New solution array. Has shape (nvars, nx, ny, nz).
        u_old: Old solution array. Has shape (nvars, nx, ny, nz).
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        delta: Whether to compute NAD violations using the range of the local maximum
            principle (DMP) to relax the bounds. If False, only rtol is used to relax
            the bounds.
        rtol: Factor by which the local DMP range or absolute values are multiplied,
            forming a product that is used to widen the bounds for numerical
            admissibility. Has shape (nvars,) or is treated as 0s if None
        gtol: Factor by which the global range is multiplied, forming a product that is
            used to widen the bounds for numerical admissibility. Has shape (nvars,)
            or is treated as 0s if None. Ignored if delta is False.
        atol: Absolute tolerance used to widen the bounds for numerical admissibility.
            Has shape (nvars,) or is treated as 0s if None. Ignored if delta is False.
        include_corners: Whether to include corner values in the DMP computation.
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
        M: Array to store the local maxima. Has shape (nvars, nx, ny, nz).
        m: Array to store the local minima. Has shape (nvars, nx, ny, nz).
        buffer: Buffer array with shape (nvars, nx, ny, nz, >=6) to store intermediate
            values.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    na = xp.newaxis

    check_buffer_slots(buffer, required=6)
    dmp_range = buffer[..., 0]
    delta_arr = buffer[..., 1]
    lower_bound = buffer[..., 2]
    upper_bound = buffer[..., 3]
    lower_violations = buffer[..., 4]
    upper_violations = buffer[..., 5]

    delta_arr[...] = 0.0

    # compute discrete maximum principle (dmp)
    modified = compute_dmp(xp, u_old, active_dims, include_corners, M=M, m=m)

    if delta:
        # relax bounds with local dmp range
        if rtol is not None:
            dmp_range[...] = M - m
            delta_arr += rtol[:, na, na, na] * dmp_range

        # relax bounds with global range
        if gtol is not None:
            global_min = xp.min(u_old, axis=(1, 2, 3), keepdims=True)
            global_max = xp.max(u_old, axis=(1, 2, 3), keepdims=True)
            global_range = global_max - global_min
            delta_arr += gtol[:, na, na, na] * global_range

        # relax bounds with absolute tolerance
        if atol is not None:
            delta_arr += atol[:, na, na, na]

        # compute violations
        lower_bound[...] = m - delta_arr
        upper_bound[...] = M + delta_arr
    elif rtol is None:
        raise ValueError("rtol must be provided if delta is False.")
    else:
        lower_bound[...] = m - rtol[:, na, na, na] * xp.abs(m)
        upper_bound[...] = M + rtol[:, na, na, na] * xp.abs(M)

    lower_violations[...] = u_new - lower_bound
    upper_violations[...] = upper_bound - u_new
    out[modified] = xp.minimum(lower_violations[modified], upper_violations[modified])

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
    cascade = config.cascade
    max_idx = len(cascade) - 1

    # early escape for single fallback scheme case
    if len(cascade) == 2:
        revise_fluxes_with_one_fallback_scheme(fv_solver, t)
        return

    # allocate arrays
    F = arrays["F"]
    G = arrays["G"]
    H = arrays["H"]
    _Fmask_ = arrays["_mask_"][:, :, :-1, :-1]
    _Gmask_ = arrays["_mask_"][:, :-1, :, :-1]
    _Hmask_ = arrays["_mask_"][:, :-1, :-1, :]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_ = arrays["_cascade_idx_"]
    cascade_idx = arrays["cascade_idx"]

    # assuming `troubles` has just been updated, update the cascade index arrays
    xp.minimum(_cascade_idx_ + _any_troubles_, max_idx, out=_cascade_idx_)
    cascade_idx[...] = _cascade_idx_[interior]

    # broadcast cascade index to each face
    if "x" in active_dims:
        F[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 1, out=_Fmask_)
        for i, scheme in enumerate(cascade):
            mask = _Fmask_[interior] == i
            xp.add(F, mask * arrays["F_" + scheme.key()], out=F)

    if "y" in active_dims:
        G[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 2, out=_Gmask_)
        for i, scheme in enumerate(cascade):
            mask = _Gmask_[interior] == i
            xp.add(G, mask * arrays["G_" + scheme.key()], out=G)

    if "z" in active_dims:
        H[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 3, out=_Hmask_)
        for i, scheme in enumerate(cascade):
            mask = _Hmask_[interior] == i
            xp.add(H, mask * arrays["H_" + scheme.key()], out=H)


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
    F = arrays["F"]
    G = arrays["G"]
    H = arrays["H"]
    _Fmask_ = arrays["_fmask_"][:, :, :-1, :-1]
    _Gmask_ = arrays["_fmask_"][:, :-1, :, :-1]
    _Hmask_ = arrays["_fmask_"][:, :-1, :-1, :]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_ = arrays["_cascade_idx_"]
    cascade_idx = arrays["cascade_idx"]
    _blended_cascade_idx_ = arrays["_blended_cascade_idx_"]
    _troubles_buffer_ = arrays["_buffer_"][:1, ..., 1:]

    # assuming `troubles` has just been updated, update the cascade index arrays
    xp.minimum(_cascade_idx_ + _any_troubles_, 1, out=_cascade_idx_)
    cascade_idx[...] = _cascade_idx_[interior]  # does not include blending information

    # blend cascade index
    if blend:
        _blended_cascade_idx_[...] = _cascade_idx_
        blend_troubled_cells(
            xp,
            _blended_cascade_idx_,
            active_dims,
            buffer=_troubles_buffer_,
        )
        _cascade_idx_ = _blended_cascade_idx_

    # broadcast cascade index to each face
    if "x" in active_dims:
        F[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 1, out=_Fmask_)
        mask = _Fmask_[interior]
        xp.add(F, mask * arrays["F_" + fallback_scheme.key()], out=F)
        xp.add(F, (1 - mask) * arrays["F_" + base_scheme.key()], out=F)

    if "y" in active_dims:
        G[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 2, out=_Gmask_)
        mask = _Gmask_[interior]
        xp.add(G, mask * arrays["G_" + fallback_scheme.key()], out=G)
        xp.add(G, (1 - mask) * arrays["G_" + base_scheme.key()], out=G)

    if "z" in active_dims:
        H[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_, 3, out=_Hmask_)
        mask = _Hmask_[interior]
        xp.add(H, mask * arrays["H_" + fallback_scheme.key()], out=H)
        xp.add(H, (1 - mask) * arrays["H_" + base_scheme.key()], out=H)


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
    buffer: ArrayLike,
    out: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Compute the blended troubled cell indicators.

    Args:
        xp: `np` namespace or equivalent.
        troubles: Array of troubled cell indicators. Has shape (1, nx, ny, nz). Must
            have float dtype.
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        buffer: Buffer array with shape (1, nx, ny, nz, >= 1) to store intermediate
            values.
        out: Output array to store the blended troubled cell indicators. If None,
            the `troubles` array will be modified in place. Has shape
            (1, nx, ny, nz).

    Returns:
        Slice objects indicating the modified regions in the troubles array.
    """
    ndim = len(active_dims)

    # allocate arrays
    check_buffer_slots(buffer, required=1)
    theta = buffer[..., 0]

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

    step_log["nfine_troubles_mean"] = []
    step_log["nfine_troubles_max"] = []

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_troubles_{var}"] = []

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

    step_log["nfine_troubles_mean"].clear()
    step_log["nfine_troubles_max"].clear()

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_troubles_{var}"].clear()

    for i, _ in enumerate(config.cascade[1:], start=1):
        step_log[f"nfine_cascade_idx{i}"].clear()


def append_troubled_cell_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Append troubled cell statistics from the finite volume solver's arrays to the step
    log. Specifically, log the sum of cells with 1 - theta > 0 using various criteria:
        `nfine_troubles_mean`: Sum over all cells of the mean trouble in [0, 1] over
            all variables.
        `nfine_troubles_max`: Sum over all cells of the maximum trouble in {0, 1} over
            all variables.
        `nfine_troubles_{var}`: Sum over all cells of the trouble in {0, 1} for the
            variable `var`.

    Args:
        fv_solver: The finite volume solver instance.
    """
    config = fv_solver.MOOD_config
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    interior = fv_solver.interior
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    troubles = arrays["troubles"]
    cascade_idx = arrays["cascade_idx"]
    buffer = arrays["_buffer_"]

    check_buffer_slots(buffer, required=4)
    mean_troubles = buffer[interior][0, ..., 0]
    max_troubles = buffer[interior][0, ..., 1]

    # track scalar quantities
    mean_troubles[...] = xp.mean(troubles, axis=0)
    max_troubles[...] = xp.max(troubles, axis=0)

    n_mean = xp.sum(mean_troubles).item()
    n_max = xp.sum(max_troubles).item()

    step_log["nfine_troubles_mean"].append(n_mean)
    step_log["nfine_troubles_max"].append(n_max)

    for var in idx.var_idx_map.keys():
        n = xp.sum(troubles[idx(var), ...]).item()
        step_log[f"nfine_troubles_{var}"].append(n)

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
        "nfine_troubles_mean": step_log["nfine_troubles_mean"].copy(),
        "nfine_troubles_max": step_log["nfine_troubles_max"].copy(),
        "n_troubles_mean": zero_max(step_log["nfine_troubles_mean"]),
        "n_troubles_max": zero_max(step_log["nfine_troubles_max"]),
    }

    for var in idx.var_idx_map.keys():
        new_data[f"nfine_troubles_{var}"] = step_log[f"nfine_troubles_{var}"].copy()
        new_data[f"n_troubles_{var}"] = zero_max(step_log[f"nfine_troubles_{var}"])

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
    if not config.skip_trouble_counts and nfine != step_log["nfine_troubles_max"]:
        raise ValueError(
            "MOOD troubled cell history mismatch: "
            f"{nfine} != {step_log['nfine_troubles_max']}"
        )
    data.update(
        {"n_MOOD_iters": state.iter_count, "nfine_MOOD_iters": state.iter_count_hist}
    )
