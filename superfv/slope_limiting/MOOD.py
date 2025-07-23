from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, cast

from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop

from .smooth_extrema_detection import inplace_smooth_extrema_detector

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

from dataclasses import dataclass, field
from types import ModuleType

from .smooth_extrema_detection import compute_smooth_extrema_detector

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


def _cache_fluxes(fv_solver: FiniteVolumeSolver, fluxes: Fluxes, scheme: str):
    """
    Cache the fluxes for the given scheme in `fv_solver.MOOD_cache`.

    Args:
        fv_solver: FiniteVolumeSolver object.
        fluxes: The fluxes (F, G, H). None if the corresponding dimension is unused.
            Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        scheme: Name of the scheme used to compute the fluxes.
    """
    if fv_solver.using_xdim:
        fv_solver.MOOD_cache.add("F_" + scheme, cast(ArrayLike, fluxes[0]))
    if fv_solver.using_ydim:
        fv_solver.MOOD_cache.add("G_" + scheme, cast(ArrayLike, fluxes[1]))
    if fv_solver.using_zdim:
        fv_solver.MOOD_cache.add("H_" + scheme, cast(ArrayLike, fluxes[2]))


def init_MOOD(fv_solver: FiniteVolumeSolver, u: ArrayLike, fluxes: Fluxes):
    """
    Initialize the MOOD algorithm.

    Args:
        fv_solver: FiniteVolumeSolver object.
        u: Array of conservative FV cell-averaged variables. Has shape
            (nvars, nx, ny, nz, ...).
        fluxes: The fluxes (F, G, H). None if the corresponding dimension is unused.
            Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
    """
    xp = fv_solver.xp
    fv_solver.MOOD_cache.clear()
    fv_solver.MOOD_cache.add("violations", xp.empty_like(u[0], dtype=bool))
    initial_scheme = "fv" + str(fv_solver.p)
    if initial_scheme != fv_solver.MOOD_cascade[0]:
        raise ValueError(
            f"Starting scheme {initial_scheme} is not the first scheme in the cascade {fv_solver.MOOD_cascade}."
        )
    _cache_fluxes(fv_solver, fluxes, initial_scheme)
    fv_solver.MOOD_cache.add("cascade_idx_array", xp.full_like(u[:1], 0, dtype=int))
    fv_solver.MOOD_iter_count = 0


def detect_troubles(
    fv_solver: FiniteVolumeSolver,
    t: float,
    u: ArrayLike,
    fluxes: Fluxes,
    NAD: Optional[float] = None,
    PAD: Optional[ArrayLike] = None,
    PAD_tol: float = 0.0,
    SED: bool = False,
) -> bool:
    """
    Detect troubles in the solution.

    Args:
        fv_solver: FiniteVolumeSolver object. Is expected to have variable group
            "limiting_vars" defined in `fv_solver.variable_index_map.group_names`.
        t: Time value.
        u: Array of conservative FV cell-averaged variables. Has shape
            (nvars, nx, ny, nz, ...).
        fluxes: The fluxes (F, G, H). None if the corresponding dimension is unused.
            Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        NAD: Numerical admissibility detection tolerance. If not provided, numerical
            admissibility is not checked.
        PAD: Physical admissibility detection bounds as an array of shape (nvars, 2)
            where the first column is the lower bound and the second column is the
            upper bound. If None, physical admissibility is not checked.
        PAD_tol: Tolerance for the physical admissibility detection. Default is 0.0.
        SED: Whether to use the smooth extrema detector (SED) to detect troubles.

    Returns:
        Whether troubles were detected (bool). If True, the MOOD algorithm will attempt
            to revise the fluxes in the next step.
    """
    idx = fv_solver.variable_index_map
    xp = fv_solver.xp

    # early escape if max iter count reached
    if fv_solver.MOOD_iter_count >= fv_solver.MOOD_max_iter_count:
        return False

    # compute candidate solution
    dt = fv_solver.substep_dt
    ustar = u + dt * fv_solver.RHS(u, *fluxes)

    # compute NAD and/or PAD violations
    limiting_slice = idx("limiting_vars")
    possible_violations = xp.zeros_like(u[limiting_slice], dtype=bool)
    if PAD is None and NAD is None:
        return False
    if NAD is not None:
        dmp_min, dmp_max = compute_dmp(
            xp,
            fv_solver.apply_bc(u, 1),
            dims=fv_solver.dims,
            include_corners=True,
        )
        NAD_violations = compute_dmp_violations(xp, ustar, dmp_min, dmp_max, tol=NAD)
        possible_violations[...] = NAD_violations[limiting_slice] < 0
    if SED:
        alpha = compute_smooth_extrema_detector(
            xp,
            fv_solver.apply_bc(ustar, 3)[limiting_slice],
            fv_solver.axes,
        )
        possible_violations[...] = xp.logical_and(alpha < 1, possible_violations)
    if PAD is not None:
        possible_violations[...] = xp.logical_or.reduce(
            [
                possible_violations,
                ustar[limiting_slice] < PAD[limiting_slice][..., 0] - PAD_tol,
                ustar[limiting_slice] > PAD[limiting_slice][..., 1] + PAD_tol,
            ]
        )
    violations = xp.any(possible_violations, axis=0)

    # return bool
    if xp.any(violations):
        fv_solver.MOOD_cache["violations"] = violations
        fv_solver.MOOD_cache["cascade_idx_array"] = xp.where(
            violations,
            xp.minimum(
                fv_solver.MOOD_cache["cascade_idx_array"] + 1,
                len(fv_solver.MOOD_cascade) - 1,
            ),
            fv_solver.MOOD_cache["cascade_idx_array"],
        )
        return True
    else:
        return False


def compute_dmp_violations(
    xp: Any, arr: ArrayLike, dmp_min: ArrayLike, dmp_max: ArrayLike, tol: float
) -> ArrayLike:
    """
    Compute a boolean array indicating whether the given array violates the discrete
    maximum principle.

    Args:
        xp: `np` namespace or equivalent.
        arr: The array to check for violations. Has shape (nvars, nx, ny, nz).
        dmp_min: The minimum values of the array. Has shape (nvars, nx, ny, nz).
        dmp_max: The maximum values of the array. Has shape (nvars, nx, ny, nz).
        tol: DMP tolerance.

    Returns:
        Array of DMP violations as negative values. Has shape (nvars, nx, ny, nz).
    """
    dmp_range = xp.max(dmp_max, axis=(1, 2, 3), keepdims=True) - xp.min(
        dmp_min, axis=(1, 2, 3), keepdims=True
    )
    violations = xp.minimum(arr - dmp_min, dmp_max - arr) + tol * dmp_range
    return violations


def revise_fluxes(
    fv_solver: FiniteVolumeSolver,
    t: float,
    u: ArrayLike,
    fluxes: Fluxes,
    mode: Literal["transverse", "gauss-legendre"],
    slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
) -> Fluxes:
    """
    Revise the fluxes using the MOOD algorithm.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Time value.
        u: Array of conservative FV cell-averaged variables. Has shape
            (nvars, nx, ny, nz, ...).
        fluxes: The fluxes (F, G, H). None if the corresponding dimension is unused.
            Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        mode: The mode for interpolating nodes and integrals. Possible values:
            - "transverse": Use transverse interpolation.
            - "gauss-legendre": Use Gauss-Legendre interpolation.
        slope_limiter: Optional slope limiter to use if `scheme` is "muscl". Possible
            values:
            - "minmod": Use the minmod slope limiter.
            - "moncen": Use the Monotone Central slope limiter.

    Returns:
        fluxes: The revised fluxes (F, G, H). None if the corresponding dimension is
            unused. Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
    """
    fv_solver.MOOD_iter_count += 1
    xp = fv_solver.xp
    fallback_scheme = fv_solver.MOOD_cascade[
        min(fv_solver.MOOD_iter_count, len(fv_solver.MOOD_cascade) - 1)
    ]
    if "F_" + fallback_scheme not in fv_solver.MOOD_cache:
        # parse new scheme
        if fallback_scheme[:2] == "fv":
            p = int(fallback_scheme[2:])
            # limiting_scheme = None
            slope_limiter = None
        elif fallback_scheme == "muscl":
            p = None
            # limiting_scheme = "muscl"
            slope_limiter = slope_limiter
        elif fallback_scheme == "half-dt":
            raise NotImplementedError(
                "Half-dt scheme is not implemented in MOOD revision."
            )
        else:
            raise ValueError(f"Unknown fallback scheme {fallback_scheme}.")

        # compute fluxes needed
        fallback_fluxes = fv_solver.xp.empty_like(fluxes)
        fv_solver.inplace_compute_fluxes(u, p)
        _cache_fluxes(fv_solver, fallback_fluxes, fallback_scheme)

    cascade_idx_array = fv_solver.MOOD_cache["cascade_idx_array"]
    revised_F, revised_G, revised_H = None, None, None
    if fv_solver.using_xdim:
        F_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            cascade_idx_array,
            "x",
        )
        revised_F = xp.zeros_like(fluxes[0])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_F += fv_solver.MOOD_cache["F_" + scheme] * xp.where(
                F_cascade_idx_array == i, 1, 0
            )
    if fv_solver.using_ydim:
        G_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            cascade_idx_array,
            "y",
        )
        revised_G = xp.zeros_like(fluxes[1])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_G += fv_solver.MOOD_cache["G_" + scheme] * xp.where(
                G_cascade_idx_array == i, 1, 0
            )
    if fv_solver.using_zdim:
        H_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            cascade_idx_array,
            "z",
        )
        revised_H = xp.zeros_like(fluxes[2])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_H += fv_solver.MOOD_cache["H_" + scheme] * xp.where(
                H_cascade_idx_array == i, 1, 0
            )

    return revised_F, revised_G, revised_H


def map_cell_values_to_face_values(
    fv_solver: FiniteVolumeSolver,
    cell_values: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Map cell values to face values by taking the maximum of adjacent cell values.

    Args:
        fv_solver: FiniteVolumeSolver object.
        cell_values: Array of cell values. Has shape (nx, ny, nz, ...).
        dim: Dimension along which to map the cell values to face values: "x", "y", or
            "z".

    Returns:
        Array of face values with shape:
            - (nx+1, ny, nz, ...) if dim == "x"
            - (nx, ny+1, nz, ...) if dim == "y"
            - (nx, ny, nz+1, ...) if dim == "z"
    """
    padded_cell_values = fv_solver.bc_for_troubled_cell_mask(
        cell_values,
        {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[dim],
    )
    axis = "xyz".index(dim) + 1
    left_slice = crop(axis, (None, -1))
    right_slice = crop(axis, (1, None))
    face_values = fv_solver.xp.maximum(
        padded_cell_values[left_slice], padded_cell_values[right_slice]
    )
    return face_values


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
