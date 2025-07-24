from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop

from .smooth_extrema_detection import inplace_smooth_extrema_detector

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

from dataclasses import dataclass, field
from types import ModuleType

# custom type for fluxes
Fluxes = Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]


@dataclass
class MOODConfig:
    """
    Configuration options for the MOOD (Multidimensional Optimal Order Detection)
    algorithm.

    Attributes:
        iter_idx (int): Current iteration index.
        iter_count (int): Total number of iterations performed.
        max_iters (int): Maximum number of iterations for the MOOD algorithm.
        cascade (List[str]): List of schemes in the cascade.
        NAD (bool): Whether to enable Numerical Admissibility Detection.
        NAD_rtol (float): Relative tolerance for the NAD violations.
        NAD_atol (float): Absolute tolerance for the NAD violations.
        global_dmp (bool): Whether to use global DMP (Discrete Maximum Principle).
        include_corners (bool): Whether to include corner values in the DMP
            computation.
        PAD (bool): Whether to enable Physical Admissibility Detection.
        PAD_atol (float): Absolute tolerance for PAD violations.
        SED (bool): Whether to enable Smooth Extrema Detection.
    """

    iter_idx: int = 0
    iter_count: int = 0
    max_iters: int = 0
    cascade: List[str] = field(default_factory=list)
    cascade_status: List[bool] = field(default_factory=list)
    NAD: bool = False
    NAD_rtol: float = 1.0
    NAD_atol: float = 0.0
    global_dmp: bool = False
    include_corners: bool = False
    PAD: bool = False
    PAD_atol: float = 0.0
    SED: bool = False


def detect_troubled_cells(
    fv_solver: FiniteVolumeSolver,
    t: float,
) -> bool:
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
    mesh = fv_solver.mesh
    lim_slc = fv_solver.variable_index_map("limiting", keepdims=True)
    interior = fv_solver.interior
    dt = fv_solver.substep_dt
    MOOD_config = fv_solver.MOOD_config

    # gather MOOD parameters
    iter_idx = MOOD_config.iter_idx
    max_iters = MOOD_config.max_iters
    cascade = MOOD_config.cascade
    cascade_status = MOOD_config.cascade_status
    NAD = MOOD_config.NAD
    NAD_rtol = MOOD_config.NAD_rtol
    NAD_atol = MOOD_config.NAD_atol
    global_dmp = MOOD_config.global_dmp
    include_corners = MOOD_config.include_corners
    PAD = MOOD_config.PAD
    PAD_atol = MOOD_config.PAD_atol
    SED = MOOD_config.SED

    # early escape if max iter count reached
    if iter_idx >= max_iters:
        return False

    # allocate arrays
    u_old_interior = arrays["u"]
    u_old = arrays["_u_"]
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
    fv_solver.inplace_apply_bc(t, u_new, p=fv_solver.p)

    # compute NAD violations
    if NAD:
        inplace_NAD_violations(
            xp,
            u_new[lim_slc],
            u_old[lim_slc],
            fv_solver.mesh.active_dims,
            buffer,
            NAD_violations,
            rtol=NAD_rtol,
            atol=NAD_atol,
            global_dmp=global_dmp,
            include_corners=include_corners,
        )

    # compute smooth extrema
    if SED:
        if not NAD:
            raise ValueError(
                "SED requires NAD to be enabled. Please set NAD to a non-None value."
            )
        inplace_smooth_extrema_detector(
            xp, u_new[lim_slc], fv_solver.mesh.active_dims, buffer, alpha
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
            fv_solver.arrays["PAD_bounds"][lim_slc],
            physical_tols=PAD_atol,
            out=PAD_violations,
        )
        xp.logical_or(
            troubles, xp.any(PAD_violations < 0, axis=0, keepdims=True), out=troubles
        )

    # update troubles workspace
    _troubles_[interior] = troubles
    fv_solver.inplace_troubles_bc(_troubles_)

    # check for troubles
    has_troubles = xp.any(troubles)

    # reset some arrays
    if has_troubles and iter_idx == 0:
        # store high-order fluxes
        if "x" in mesh.active_dims:
            arrays["F_" + cascade[0]][...] = arrays["F"].copy()
        if "y" in mesh.active_dims:
            arrays["G_" + cascade[0]][...] = arrays["G"].copy()
        if "z" in mesh.active_dims:
            arrays["H_" + cascade[0]][...] = arrays["H"].copy()
        cascade_status[0] = True

        # reset cascade index array
        _cascade_idx_array_[...] = 0

    return has_troubles


def inplace_NAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    u_old: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    buffer: ArrayLike,
    out: ArrayLike,
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
        buffer: Buffer array with shape (nvars, nx, ny, nz, >= 2).
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
        rtol: Relative tolerance for the NAD violations.
        atol: Absolute tolerance for the NAD violations.
        global_dmp: Whether to use global DMP.
        include_corners: Whether to include corner values in the DMP computation.
    """

    if global_dmp:
        dmp_min = xp.min(u_old, axis=(1, 2, 3), keepdims=True)
        dmp_max = xp.max(u_old, axis=(1, 2, 3), keepdims=True)
    else:
        dmp = buffer[..., :2]
        dmp_min, dmp_max = dmp[..., 0], dmp[..., 1]
        compute_dmp(xp, u_old, active_dims, include_corners, dmp)
    dmp_range = dmp_max - dmp_min
    violations = xp.minimum(u_new - dmp_min, dmp_max - u_new) + rtol * dmp_range + atol

    out[...] = violations


def inplace_PAD_violations(
    xp: ModuleType,
    u_new: ArrayLike,
    physical_ranges: ArrayLike,
    physical_tols: ArrayLike,
    out: ArrayLike,
):
    """
    Compute the physical admissibility detection (PAD) violations.

    Args:
        xp: `np` namespace or equivalent.
        u_new: New solution array. Has shape (nvars, nx, ny, nz).
        physical_ranges: Physical ranges for each variable. Has shape (nvars, 2).
        physical_tols: Physical tolerances for each variable. Has shape (nvars, 2).
        out: Output array to store the violations. Has shape (nvars, nx, ny, nz).
    """
    physical_mins = physical_ranges[..., 0]
    physical_maxs = physical_ranges[..., 1]
    out[...] = xp.minimum(u_new - physical_mins, physical_maxs - u_new) + physical_tols


def inplace_revise_fluxes(fv_solver: FiniteVolumeSolver, t: float):
    """ """
    # gather solver parameters
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    mesh = fv_solver.mesh
    MOOD_config = fv_solver.MOOD_config
    cascade = MOOD_config.cascade
    cascade_status = MOOD_config.cascade_status
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
        fv_solver.inplace_compute_fluxes(t, **compute_fluxes_kwargs(scheme))
        cascade_status[current_max_idx] = True

        if "x" in mesh.active_dims:
            arrays["F_" + scheme] = F.copy()
        if "y" in mesh.active_dims:
            arrays["G_" + scheme] = G.copy()
        if "z" in mesh.active_dims:
            arrays["H_" + scheme] = H.copy()

    # broadcast cascade index to each face
    if "x" in mesh.active_dims:
        F[...] = 0
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 1, F_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = F_mask[fv_solver.interior] == i
            xp.add(F, mask * arrays["F_" + scheme], out=F)

    if "y" in mesh.active_dims:
        G[...] = 0
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 2, G_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = G_mask[fv_solver.interior] == i
            xp.add(G, mask * arrays["G_" + scheme], out=G)

    if "z" in mesh.active_dims:
        H[...] = 0
        inplace_map_cells_values_to_face_values(xp, _cascade_idx_array_, 3, H_mask)
        for i, scheme in enumerate(cascade[: (current_max_idx + 1)]):
            mask = H_mask[fv_solver.interior] == i
            xp.add(H, mask * arrays["H_" + scheme], out=H)


def compute_fluxes_kwargs(scheme: str) -> dict:
    if scheme.startswith("fv"):
        p = int(scheme[2:])
        return {"p": p}
    raise ValueError(f"Unknown scheme {scheme}")


def inplace_map_cells_values_to_face_values(
    xp: ModuleType, cv: ArrayLike, axis: int, out: ArrayLike
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
