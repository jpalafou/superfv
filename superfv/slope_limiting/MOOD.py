from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

from superfv.boundary_conditions import BCs, apply_bc
from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop

from .smooth_extrema_detection import inplace_smooth_extrema_detector

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
        global_dmp: Whether to use global minima and maxima to set the range for NAD
            violations instead of local minima and maxima.
        include_corners: When `global_dmp=False`, whether to include corner cells when
            computing the local minima and maxima.
    """

    cascade: List[InterpolationScheme]
    max_iters: int
    NAD: bool
    PAD: bool
    SED: bool
    NAD_rtol: float = 1.0
    NAD_atol: float = 0.0
    PAD_atol: float = 0.0
    PAD_bounds: Optional[ArrayLike] = None
    global_dmp: bool = False
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

    def key(self) -> str:
        cascade_keys = ", ".join([scheme.key() for scheme in self.cascade])
        return f"MOOD: [{cascade_keys}]"


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
        self.reset_iter_count()
        self.reset_iter_count_hist()
        self.reset_MOOD_loop()

    def reset_iter_count(self):
        """
        Reset the iteration count which accumulates the total number of iterations
        in a step across all MOOD loops in a step.
        """
        self.iter_count = 0

    def reset_iter_count_hist(self):
        """
        Reset the iter count history which accumulates the total number of iterations
        for each substep.
        """
        self.iter_count_hist: List[int] = []

    def increment_iter_count_hist(self):
        """
        Increment the iter count history at the end of each substep.
        """
        self.iter_count_hist.append(self.fine_iter_count)

    def reset_MOOD_loop(self):
        """
        Reset the iteration index and cascade status for the first MOOD iteration in a
        MOOD loop.
        """
        self.iter_idx = 0
        self.fine_iter_count = 0
        self.cascade_status: List[bool] = [False] * len(self.config.cascade)

    def increment_MOOD_iteration(self):
        """
        Increment the iteration index and count for a single MOOD iteration.
        """
        self.iter_idx += 1
        self.iter_count += 1
        self.fine_iter_count += 1


def detect_troubled_cells(fv_solver: FiniteVolumeSolver, t: float) -> bool:
    """
    Detect troubled cells in the finite volume solver.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Current time value.

    Returns:
        bool: Whether any troubles were detected or whether the maximum number of
            iterations for the MOOD algorithm has been reached.
    """
    # gather solver parameters
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    active_dims = fv_solver.active_dims
    lim_slc = fv_solver.variable_index_map("limiting", keepdims=True)
    interior = fv_solver.interior
    dt = fv_solver.substep_dt
    MOOD_state = fv_solver.MOOD_state

    # gather MOOD parameters
    iter_idx = MOOD_state.iter_idx
    max_iters = MOOD_state.config.max_iters
    cascade_status = MOOD_state.cascade_status
    MOOD_config = MOOD_state.config
    cascade = MOOD_config.cascade
    NAD = MOOD_config.NAD
    NAD_rtol = MOOD_config.NAD_rtol
    NAD_atol = MOOD_config.NAD_atol
    global_dmp = MOOD_config.global_dmp
    include_corners = MOOD_config.include_corners
    PAD = MOOD_config.PAD
    PAD_bounds = MOOD_config.PAD_bounds
    PAD_atol = MOOD_config.PAD_atol
    SED = MOOD_config.SED

    # early escape if max iter count reached
    if iter_idx >= max_iters:
        return False

    # allocate arrays
    u_old = arrays["_u_"]
    u_old_interior = arrays["_u_"][interior]
    u_new = arrays["buffer"][..., 0]
    buffer = arrays["buffer"][lim_slc]
    NAD_violations = buffer[..., 1]
    PAD_violations = buffer[..., 2][interior]
    alpha = buffer[..., 3:4]
    buffer = buffer[..., 4:]
    troubles = arrays["troubles"]
    _troubles_ = arrays["_troubles_"]
    _cascade_idx_array_ = arrays["_cascade_idx_array_"]

    # reset troubles / cascade index array
    troubles[...] = False

    # compute candidate solution
    u_new_interior = u_old_interior + dt * fv_solver.compute_RHS()
    u_new[interior] = u_new_interior
    fv_solver.inplace_apply_bc(t, u_new, scheme=fv_solver.base_scheme)

    # compute NAD violations
    if NAD:
        inplace_NAD_violations(
            xp,
            u_new[lim_slc],
            u_old[lim_slc],
            active_dims,
            out=NAD_violations,
            buffer=buffer,
            rtol=NAD_rtol,
            atol=NAD_atol,
            global_dmp=global_dmp,
            include_corners=include_corners,
        )

    # compute smooth extrema
    if SED:
        inplace_smooth_extrema_detector(
            xp,
            u_new[lim_slc],
            active_dims,
            out=alpha,
            buffer=buffer,
        )
        troubles[...] = xp.any(
            xp.logical_and(NAD_violations[interior] < 0, alpha[..., 0][interior] < 1),
            axis=0,
            keepdims=True,
        )
    else:
        troubles[...] = xp.any(NAD_violations[interior] < 0, axis=0, keepdims=True)

    # compute PAD violations
    if PAD:
        inplace_PAD_violations(
            xp,
            u_new_interior[lim_slc],
            cast(ArrayLike, PAD_bounds)[lim_slc],
            physical_tols=PAD_atol,
            out=PAD_violations,
        )
        xp.logical_or(
            troubles, xp.any(PAD_violations < 0, axis=0, keepdims=True), out=troubles
        )

    # update troubles workspace
    _troubles_[interior] = troubles
    apply_bc(
        _troubles_,
        pad_width=fv_solver.bc_pad_width,
        mode=normalize_troubles_bc(fv_solver.bc_mode),
    )

    # check for troubles
    has_troubles = xp.any(troubles)

    # reset some arrays
    if has_troubles and iter_idx == 0:
        # store high-order fluxes
        scheme_key = cascade[0].key()
        if "x" in active_dims:
            arrays["F_" + scheme_key][...] = arrays["F"].copy()
        if "y" in active_dims:
            arrays["G_" + scheme_key][...] = arrays["G"].copy()
        if "z" in active_dims:
            arrays["H_" + scheme_key][...] = arrays["H"].copy()
        cascade_status[0] = True

        # reset cascade index array
        _cascade_idx_array_[...] = 0

    return has_troubles


def inplace_NAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    u_old: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    buffer: Optional[ArrayLike] = None,
    rtol: float = 1.0,
    atol: float = 0.0,
    global_dmp: bool = False,
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
        buffer: Buffer array with shape (nvars, nx, ny, nz, >= 2) if `global_dmp` is
            False. If `global_dmp` is True, this parameter is ignored.
        rtol: Relative tolerance for the NAD violations.
        atol: Absolute tolerance for the NAD violations.
        global_dmp: Whether to use global DMP.
        include_corners: Whether to include corner values in the DMP computation.
    """

    if global_dmp:
        dmp_min = xp.min(u_old, axis=(1, 2, 3), keepdims=True)
        dmp_max = xp.max(u_old, axis=(1, 2, 3), keepdims=True)
    elif buffer is None:
        raise ValueError("Buffer must be provided if global_dmp is False.")
    else:
        dmp = buffer[..., :2]
        dmp_min, dmp_max = dmp[..., 0], dmp[..., 1]
        compute_dmp(xp, u_old, active_dims, out=dmp, include_corners=include_corners)
    dmp_range = dmp_max - dmp_min
    violations = xp.minimum(u_new - dmp_min, dmp_max - u_new) + rtol * dmp_range + atol

    out[...] = violations


def inplace_PAD_violations(
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


def inplace_revise_fluxes(fv_solver: FiniteVolumeSolver, t: float):
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

    # allocate arrays
    F = arrays["F"]
    G = arrays["G"]
    H = arrays["H"]
    F_mask = arrays["_mask_"][:, :, :-1, :-1]
    G_mask = arrays["_mask_"][:, :-1, :, :-1]
    H_mask = arrays["_mask_"][:, :-1, :-1, :]
    _troubles_ = arrays["_troubles_"]
    _cascade_idx_array_ = arrays["_cascade_idx_array_"]

    # assuming `troubles` has just been updated, update the cascade index array
    xp.minimum(_cascade_idx_array_ + _troubles_, max_idx, out=_cascade_idx_array_)
    current_max_idx = xp.max(_cascade_idx_array_)

    # update the cascade scheme fluxes
    if not cascade_status[current_max_idx]:
        scheme = cascade[current_max_idx]
        fv_solver.inplace_compute_fluxes(t, scheme)
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
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 1, out=F_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = F_mask[interior] == i
            xp.add(F, mask * arrays["F_" + scheme.key()], out=F)

    if "y" in active_dims:
        G[...] = 0
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 2, out=G_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = G_mask[interior] == i
            xp.add(G, mask * arrays["G_" + scheme.key()], out=G)

    if "z" in active_dims:
        H[...] = 0
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 3, out=H_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = H_mask[interior] == i
            xp.add(H, mask * arrays["H_" + scheme.key()], out=H)


def inplace_map_cells_values_to_face_values(
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
