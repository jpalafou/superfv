from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

from superfv.boundary_conditions import BCs, apply_bc
from superfv.fv import DIM_TO_AXIS
from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop

from .smooth_extrema_detection import smooth_extrema_detector

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from superfv.interpolation_schemes import InterpolationScheme

# custom type for fluxes
Fluxes = Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]


@dataclass(frozen=True, slots=True)
class MOODConfig(LimiterConfig):
    """
    Configuration for the MOOD framework.

    Attributes:
        cascade: The list of interpolation schemes to use in the MOOD framework.
        blend: Whether to blend troubled cells when using a fallback scheme. Only valid
            for cascades of length 2.
        max_iters: The maximum number of iterations to perform.
        NAD: Whether to use numerical admissibility detection to detect troubled cells.
        PAD: Whether to use physical admissibility detection to detect troubled cells.
        SED: Whether to use smooth extrema detection to forgive troubled cells.
        NAD_rtol: Relative tolerance for NAD violations.
        NAD_atol: Absolute tolerance for NAD violations.
        PAD_atol: Absolute tolerance for PAD violations.
        PAD_bounds: Physical bounds for each variable used in PAD. Has shape
            (nvars, 1, 1, 1, 2) with the minimum and maximum values for each variable
            stored in the first and second element of the last dimension, respectively.
        absolute_dmp: Whether to use the absolute values of the DMP instead of the
            range to set the NAD bounds. The NAD condition for each case is:
            - `absolute_dmp=False`:
                umin-rtol*(umax-umin)-atol <= u_new <= umax+rtol*(umax-umin)+atol
            - `absolute_dmp=True`:
                umin-rtol*|umin|-atol <= u_new <= umax+rtol*|umax|+atol
        include_corners: Whether to include corner cells when computing the local
            minima and maxima.
    """

    cascade: List[InterpolationScheme]
    blend: bool
    max_iters: int
    NAD: bool
    PAD: bool
    SED: bool
    NAD_rtol: float = 1.0
    NAD_atol: float = 0.0
    PAD_atol: float = 0.0
    PAD_bounds: Optional[ArrayLike] = None
    absolute_dmp: bool = False
    include_corners: bool = False

    def __post_init__(self):
        if self.PAD and self.PAD_bounds is None:
            raise ValueError(
                "PAD requires PAD_bounds to be set. "
                "Set PAD=False if you do not want to use PAD."
            )
        if self.SED and not self.NAD:
            raise ValueError(
                "SED requires NAD to be enabled. Please set NAD to a non-None value."
            )
        if self.blend and len(self.cascade) != 2:
            raise ValueError("Blending is only supported for cascades of length 2.")

    def key(self) -> str:
        cascade_keys = ", ".join([scheme.key() for scheme in self.cascade])
        return f"MOOD: [{cascade_keys}]"

    def to_dict(self) -> dict:
        return dict(
            cascade=[scheme.to_dict() for scheme in self.cascade],
            max_iters=self.max_iters,
            NAD=self.NAD,
            PAD=self.PAD,
            SED=self.SED,
            NAD_rtol=self.NAD_rtol,
            NAD_atol=self.NAD_atol,
            PAD_atol=self.PAD_atol,
            PAD_bounds=(
                None
                if self.PAD_bounds is None
                else self.PAD_bounds[:, 0, 0, 0, :].tolist()
            ),
            absolute_dmp=self.absolute_dmp,
            include_corners=self.include_corners,
        )


@dataclass
class MOODState:
    """
    Class that describes the state of the MOOD iterator.

    Attributes:
        config: The MOOD configuration.
        iter_idx: Current iteration index in the MOOD loop.
        iter_count: Total number of iterations across all MOOD loops in a step.
        fine_iter_count: Number of iterations in the current MOOD loop.
        cascade_status: List of boolean flags indicating the status of each
            interpolation scheme in the cascade.
    """

    config: MOODConfig

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
        self.cascade_status: List[bool] = [False] * len(self.config.cascade)

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

    def to_dict(self) -> dict:
        """
        Convert the MOODState to a dictionary.

        Returns:
            A dictionary representation of the MOODState.
        """
        return self.config.to_dict()


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
    xp = fv_solver.xp
    scheme = fv_solver.base_scheme
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    lim_slc = fv_solver.variable_index_map("limiting", keepdims=True)
    interior = fv_solver.interior
    dt = fv_solver.substep_dt
    MOOD_state = fv_solver.MOOD_state

    # gather MOOD parameters
    iter_idx = MOOD_state.iter_idx
    cascade_status = MOOD_state.cascade_status
    MOOD_config = MOOD_state.config
    cascade = MOOD_config.cascade
    NAD = MOOD_config.NAD
    NAD_rtol = MOOD_config.NAD_rtol
    NAD_atol = MOOD_config.NAD_atol
    absolute_dmp = MOOD_config.absolute_dmp
    include_corners = MOOD_config.include_corners
    PAD = MOOD_config.PAD
    PAD_bounds = MOOD_config.PAD_bounds
    PAD_atol = MOOD_config.PAD_atol
    SED = MOOD_config.SED

    # determine limiting style
    primitive_NAD = scheme.flux_recipe in (2, 3)
    max_idx = len(cascade) - 1

    # allocate arrays
    u_old = arrays["_u_"]
    u_old_interior = arrays["_u_"][interior]
    w_old = arrays["_w_"]
    u_new = arrays["buffer"][..., 0]
    buffer = arrays["buffer"]
    NAD_violations = buffer[..., 1][lim_slc]
    PAD_violations = buffer[..., 2][interior]
    alpha = buffer[..., 3:4][lim_slc]
    buffer = buffer[..., 4:]
    troubles = arrays["troubles"]
    _troubles_ = arrays["_troubles_"]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_array_ = arrays["_cascade_idx_array_"]

    # reset troubles / cascade index array
    troubles[...] = False
    if iter_idx == 0:
        _cascade_idx_array_[...] = 0

    # compute candidate solution
    u_new_interior = u_old_interior + dt * fv_solver.compute_RHS()
    u_new[interior] = u_new_interior
    fv_solver.apply_bc(t, u_new, scheme=fv_solver.base_scheme)
    w_new = fv_solver.primitives_from_conservatives(u_new)
    w_new_interior = w_new[interior]

    # compute NAD violations
    if NAD:
        detect_NAD_violations(
            xp,
            (w_new if primitive_NAD else u_new)[lim_slc],
            (w_old if primitive_NAD else u_old)[lim_slc],
            active_dims,
            out=NAD_violations,
            buffer=buffer[lim_slc],
            rtol=NAD_rtol,
            atol=NAD_atol,
            absolute_dmp=absolute_dmp,
            include_corners=include_corners,
        )

        # compute smooth extrema
        if SED:
            smooth_extrema_detector(
                xp,
                (w_new if primitive_NAD else u_new)[lim_slc],
                active_dims,
                out=alpha,
                buffer=buffer[lim_slc],
            )
            troubles[lim_slc] = xp.logical_and(
                NAD_violations[interior] < 0, alpha[..., 0][interior] < 1
            )
        else:
            troubles[lim_slc] = NAD_violations[interior]

    # compute PAD violations
    if PAD:
        detect_PAD_violations(
            xp,
            w_new_interior,
            cast(ArrayLike, PAD_bounds),
            physical_tols=PAD_atol,
            out=PAD_violations,
        )
        xp.logical_or(troubles, PAD_violations < 0, out=troubles)

    # update troubles workspace
    _troubles_[interior] = troubles
    apply_bc(
        xp,
        _troubles_,
        pad_width=fv_solver.bc_pad_width,
        mode=normalize_troubles_bc(fv_solver.bc_mode),
    )

    # trouble counts
    _any_troubles_[...] = xp.any(_troubles_, axis=0, keepdims=True)
    any_troubles = _any_troubles_[interior]

    n_troubled_cells = xp.sum(any_troubles).item()
    revisable_troubled_cells = any_troubles & (_cascade_idx_array_[interior] < max_idx)
    n_revisable_troubled_cells = xp.sum(revisable_troubled_cells).item()

    # early escape for no revisable troubles
    if n_revisable_troubled_cells == 0:
        return 0, n_troubled_cells

    # revisable troubles were detected. initialize cascade cache
    if iter_idx == 0:
        # store high-order fluxes
        scheme_key = cascade[0].key()
        if "x" in active_dims:
            arrays["F_" + scheme_key][...] = arrays["F"].copy()
        if "y" in active_dims:
            arrays["G_" + scheme_key][...] = arrays["G"].copy()
        if "z" in active_dims:
            arrays["H_" + scheme_key][...] = arrays["H"].copy()
        cascade_status[0] = True

    return n_revisable_troubled_cells, n_troubled_cells


def detect_NAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    u_old: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    rtol: float = 1.0,
    atol: float = 0.0,
    absolute_dmp: bool = False,
    include_corners: bool = False,
):
    """
    Compute the numerical admissibility detection (NAD) violations.

    Args:
        xp: `np` namespace or equivalent.
        u_new: New solution array. Has shape (nvars, nx, ny, nz).
        u_old: Old solution array. Has shape (nvars, nx, ny, nz).
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
        buffer: Buffer array with shape (nvars, nx, ny, nz, >= 2).
        rtol: Relative tolerance for the NAD violations.
        atol: Absolute tolerance for the NAD violations.
        absolute_dmp: Whether to use the absolute values of the DMP instead of the
            range to set the NAD bounds. The NAD condition for each case is:
            - `absolute_dmp=False`:
                umin-rtol*(umax-umin)-atol <= u_new <= umax+rtol*(umax-umin)+atol
            - `absolute_dmp=True`:
                umin-rtol*|umin|-atol <= u_new <= umax+rtol*|umax|+atol
        include_corners: Whether to include corner values in the DMP computation.
    """
    dmp = buffer[..., :2]
    dmp_min, dmp_max = dmp[..., 0], dmp[..., 1]
    compute_dmp(xp, u_old, active_dims, out=dmp, include_corners=include_corners)

    if absolute_dmp:
        upper_bound = dmp_max + rtol * xp.abs(dmp_max) + atol
        lower_bound = dmp_min - rtol * xp.abs(dmp_min) - atol
    else:
        dmp_range = dmp_max - dmp_min
        upper_bound = dmp_max + rtol * dmp_range + atol
        lower_bound = dmp_min - rtol * dmp_range - atol
    out[...] = xp.minimum(u_new - lower_bound, upper_bound - u_new)


def detect_PAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    physical_ranges: ArrayLike,
    physical_tols: float,
    *,
    out: ArrayLike,
):
    """
    Compute the physical admissibility detection (PAD) violations.

    Args:
        xp: `np` namespace or equivalent.
        u_new: New solution array. Has shape (nvars, nx, ny, nz).
        physical_ranges: Physical ranges for each variable. Has shape (nvars, 2).
        physical_tols: Physical tolerances for all variables as a single float.
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
    """
    physical_mins = physical_ranges[..., 0]
    physical_maxs = physical_ranges[..., 1]
    out[...] = xp.minimum(u_new - physical_mins, physical_maxs - u_new) + physical_tols


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
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    interior = fv_solver.interior
    MOOD_state = fv_solver.MOOD_state
    cascade = MOOD_state.config.cascade
    cascade_status = MOOD_state.cascade_status
    max_idx = len(cascade) - 1

    # early escape for single fallback scheme case
    if len(cascade) == 2:
        revise_fluxes_with_fallback_scheme(fv_solver, t)
        return

    # allocate arrays
    F = arrays["F"]
    G = arrays["G"]
    H = arrays["H"]
    F_mask = arrays["_mask_"][:, :, :-1, :-1]
    G_mask = arrays["_mask_"][:, :-1, :, :-1]
    H_mask = arrays["_mask_"][:, :-1, :-1, :]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_array_ = arrays["_cascade_idx_array_"]

    # assuming `troubles` has just been updated, update the cascade index array
    xp.minimum(_cascade_idx_array_ + _any_troubles_, max_idx, out=_cascade_idx_array_)
    current_max_idx = xp.max(_cascade_idx_array_).item()

    # update the cascade scheme fluxes
    if not cascade_status[current_max_idx]:
        scheme = cascade[current_max_idx]
        fv_solver.compute_fluxes(t, scheme)
        cascade_status[current_max_idx] = True

        if "x" in active_dims:
            arrays["F_" + scheme.key()] = F.copy()
        if "y" in active_dims:
            arrays["G_" + scheme.key()] = G.copy()
        if "z" in active_dims:
            arrays["H_" + scheme.key()] = H.copy()

    # broadcast cascade index to each face
    if "x" in active_dims:
        F[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 1, out=F_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = F_mask[interior] == i
            xp.add(F, mask * arrays["F_" + scheme.key()], out=F)

    if "y" in active_dims:
        G[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 2, out=G_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = G_mask[interior] == i
            xp.add(G, mask * arrays["G_" + scheme.key()], out=G)

    if "z" in active_dims:
        H[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 3, out=H_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = H_mask[interior] == i
            xp.add(H, mask * arrays["H_" + scheme.key()], out=H)


def revise_fluxes_with_fallback_scheme(fv_solver: FiniteVolumeSolver, t: float):
    """
    In-place revision of fluxes based on the current time step using a fallback scheme.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.
    """
    # gather solver parameters
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    interior = fv_solver.interior
    MOOD_state = fv_solver.MOOD_state
    cascade = MOOD_state.config.cascade
    cascade_status = MOOD_state.cascade_status

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
    F_mask = arrays["_fmask_"][:, :, :-1, :-1]
    G_mask = arrays["_fmask_"][:, :-1, :, :-1]
    H_mask = arrays["_fmask_"][:, :-1, :-1, :]
    _any_troubles_ = arrays["_any_troubles_"]
    _cascade_idx_array_ = arrays["_cascade_idx_array_"]
    _blended_cascade_idx_array_ = arrays["_blended_cascade_idx_array_"]
    troubles_buffer = arrays["buffer"][:1, ..., 1:]

    # assuming `troubles` has just been updated, update the cascade index array
    xp.minimum(_cascade_idx_array_ + _any_troubles_, 1, out=_cascade_idx_array_)
    # max is 1 since there should be at least 1 troubled cell at this point

    # blend cascade index
    if MOOD_state.config.blend:
        _blended_cascade_idx_array_[...] = _cascade_idx_array_
        blend_troubled_cells(
            xp, _blended_cascade_idx_array_, active_dims, buffer=troubles_buffer
        )
        _cascade_idx_array_ = _blended_cascade_idx_array_

    # compute fallback scheme fluxes
    if not cascade_status[1]:
        fv_solver.compute_fluxes(t, fallback_scheme)
        cascade_status[1] = True

        if "x" in active_dims:
            arrays["F_" + fallback_scheme.key()] = F.copy()
        if "y" in active_dims:
            arrays["G_" + fallback_scheme.key()] = G.copy()
        if "z" in active_dims:
            arrays["H_" + fallback_scheme.key()] = H.copy()

    # broadcast cascade index to each face
    if "x" in active_dims:
        F[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 1, out=F_mask)
        mask = F_mask[interior]
        xp.add(F, mask * arrays["F_" + fallback_scheme.key()], out=F)
        xp.add(F, (1 - mask) * arrays["F_" + base_scheme.key()], out=F)

    if "y" in active_dims:
        G[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 2, out=G_mask)
        mask = G_mask[interior]
        xp.add(G, mask * arrays["G_" + fallback_scheme.key()], out=G)
        xp.add(G, (1 - mask) * arrays["G_" + base_scheme.key()], out=G)

    if "z" in active_dims:
        H[...] = 0
        map_cells_values_to_face_values(xp, _cascade_idx_array_, 3, out=H_mask)
        mask = H_mask[interior]
        xp.add(H, mask * arrays["H_" + fallback_scheme.key()], out=H)
        xp.add(H, (1 - mask) * arrays["H_" + base_scheme.key()], out=H)


def map_cells_values_to_face_values(
    xp: ModuleType, cv: ArrayLike, axis: int, *, out: ArrayLike
):
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
    lft = crop(axis, (None, -1))
    rgt = crop(axis, (1, None))
    ctr = crop(axis, (1, -1))

    out[ctr] = cv[lft]
    xp.maximum(out[ctr], cv[rgt], out=out[ctr])

    return ctr


def blend_troubled_cells(
    xp: ModuleType,
    troubles: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    buffer: ArrayLike,
):
    """
    Overwrite the troubled cells array `troubles` with a blended value based on
    neighboring troubled cells.

    Args:
        xp: `np` namespace or equivalent.
        troubles: Array of troubled cell indicators. Has shape (1, nx, ny, nz). Must
            have float dtype.
        active_dims: Active dimensions of the problem as a tuple of "x", "y", "z".
        buffer: Buffer array with shape (1, nx, ny, nz, >= 1) to store intermediate
            values.
    """
    theta = buffer[..., 0]
    ndim = len(active_dims)

    # initialize theta
    theta[...] = troubles

    if ndim == 1:
        axis = DIM_TO_AXIS[active_dims[0]]
        lft_slc = crop(axis, (None, -1))
        rgt_slc = crop(axis, (1, None))

        # First neighbors
        theta[lft_slc] = xp.maximum(0.75 * troubles[rgt_slc], theta[lft_slc])
        theta[rgt_slc] = xp.maximum(0.75 * troubles[lft_slc], theta[rgt_slc])

        # Second neighbors
        theta[lft_slc] = xp.maximum(0.25 * (theta[rgt_slc] > 0), theta[lft_slc])
        theta[rgt_slc] = xp.maximum(0.25 * (theta[lft_slc] > 0), theta[rgt_slc])
    elif ndim == 2:
        raise NotImplementedError("2D blending is not implemented yet.")
    elif ndim == 3:
        raise NotImplementedError("3D blending is not implemented yet.")

    troubles[...] = theta
