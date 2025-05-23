from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, cast

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import ArrayLike

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

from .smooth_extrema_detection import compute_smooth_extrema_detector


def _cache_fluxes(
    fv_solver: FiniteVolumeSolver,
    fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
    scheme: str,
):
    """
    Cache the fluxes for the given scheme in `fv_solver.MOOD_cache`.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        fluxes (Tuple[Optional[ArrayLike], ...]): The fluxes (F, G, H). None if the
            corresponding dimension is unused. Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        scheme (str): The scheme name.
    """
    if fv_solver.using_xdim:
        fv_solver.MOOD_cache.add("F_" + scheme, cast(ArrayLike, fluxes[0]))
    if fv_solver.using_ydim:
        fv_solver.MOOD_cache.add("G_" + scheme, cast(ArrayLike, fluxes[1]))
    if fv_solver.using_zdim:
        fv_solver.MOOD_cache.add("H_" + scheme, cast(ArrayLike, fluxes[2]))


def init_MOOD(
    fv_solver: FiniteVolumeSolver,
    u: ArrayLike,
    fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
):
    """
    Initialize the MOOD algorithm.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[Optional[ArrayLike], ...]): The fluxes (F, G, H). None if the
            corresponding dimension is unused. Otherwise, is an array with shape:
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
    dt: float,
    u: ArrayLike,
    fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
    NAD: Optional[float] = None,
    PAD: Optional[ArrayLike] = None,
    PAD_tol: float = 0.0,
    SED: bool = False,
) -> bool:
    """
    Detect troubles in the solution.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object. Is expected to
            have variable group "limiting_vars" defined in
            `fv_solver.array_slicer.group_names`.
        t (float): The current time.
        dt (float): The time step.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[Optional[ArrayLike], ...]): The fluxes (F, G, H). None if the
            corresponding dimension is unused. Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        NAD (Optional[float]): Numerical admissibility detection tolerance. Numerical
            admissibility is not checked if None.
        PAD (Optional[ArrayLike]): Physical admissibility detection bounds. Is an array
            of shape (nvars, 2) if not None, where the first column is the lower bound
            and the second column is the upper bound. Physical admissibility is not
            checked if None.
        PAD_tol (float): Tolerance for the physical admissibility detection. Default is
            0.0.
        SED whether to apply smooth extrema detection.
    Returns:
        bool: Whether troubles were detected.
    """
    _slc = fv_solver.array_slicer
    xp = fv_solver.xp

    # early escape if max iter count reached
    if fv_solver.MOOD_iter_count >= fv_solver.MOOD_max_iter_count:
        return False

    # compute candidate solution
    ustar = u + dt * fv_solver.RHS(u, *fluxes)

    # compute NAD and/or PAD violations
    __limiting_slc__ = _slc("limiting_vars", keepdims=True)
    possible_violations = xp.zeros_like(u[__limiting_slc__], dtype=bool)
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
        possible_violations[...] = NAD_violations[__limiting_slc__] < 0
    if SED:
        alpha = compute_smooth_extrema_detector(
            xp,
            fv_solver.apply_bc(ustar, 3)[__limiting_slc__],
            fv_solver.axes,
        )
        possible_violations[...] = xp.logical_and(alpha < 1, possible_violations)
    if PAD is not None:
        possible_violations[...] = xp.logical_or.reduce(
            [
                possible_violations,
                ustar[__limiting_slc__] < PAD[__limiting_slc__][..., 0] - PAD_tol,
                ustar[__limiting_slc__] > PAD[__limiting_slc__][..., 1] + PAD_tol,
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
        xp (Any): `np` namespace or equivalent.
        arr (ArrayLike): The array to check. Has shape (nvars, nx, ny, nz).
        dmp_min (ArrayLike): The minimum values of the array. Has shape
            (nvars, nx, ny, nz).
        dmp_max (ArrayLike): The maximum values of the array. Has shape
            (nvars, nx, ny, nz).
        tol (float): The tolerance.

    Returns:
        ArrayLike: The DMP violations as negative values. Has shape
            (nvars, nx, ny, nz).
    """
    dmp_range = xp.max(dmp_max, axis=(1, 2, 3), keepdims=True) - xp.min(
        dmp_min, axis=(1, 2, 3), keepdims=True
    )
    violations = xp.minimum(arr - dmp_min, dmp_max - arr) + tol * dmp_range
    return violations


def revise_fluxes(
    fv_solver: FiniteVolumeSolver,
    t: float,
    dt: float,
    u: ArrayLike,
    fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
    mode: Literal["transverse", "gauss-legendre"],
    slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
) -> Tuple[float, Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]]:
    """
    Revise the fluxes using the MOOD algorithm.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        t (float): The current time.
        dt (float): The time step.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[Optional[ArrayLike], ...]): The fluxes (F, G, H). None if the
            corresponding dimension is unused. Otherwise, is an array with shape:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        mode (Literal["transverse", "gauss-legendre"]): The mode for interpolating
            nodes and integrals.
        slope_limiter (Optional[Literal["minmod", "moncen"]]): The slope limiter to use
            if `scheme` is "muscl".

    Returns:
        Tuple[float, Tuple[Optional[ArrayLike]], ...]: Tuple composed of:
            - The revised time step.
            - The revised fluxes (F, G, H). None if the corresponding dimension is
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
            limiting_scheme = None
            slope_limiter = None
        elif fallback_scheme == "muscl":
            p = None
            limiting_scheme = "muscl"
            slope_limiter = slope_limiter
        elif fallback_scheme == "half-dt":
            return dt / 2, fluxes  # early escape for time-step halving
        else:
            raise ValueError(f"Unknown fallback scheme {fallback_scheme}.")

        # compute fluxes needed
        _, fallback_fluxes = fv_solver.compute_dt_and_fluxes(
            t, u, mode, p, limiting_scheme, slope_limiter
        )
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

    return dt, (revised_F, revised_G, revised_H)


def map_cell_values_to_face_values(
    fv_solver: FiniteVolumeSolver,
    cell_values: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Map cell values to face values by taking the maximum of adjacent cell values.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        cell_values (ArrayLike): The cell values. Has shape (nx, ny, nz, ...).
        dim (Literal["x", "y", "z"]): The dimension.

    Returns:
        ArrayLike: The face values. Has shape
            - (nx+1, ny, nz, ...) if dim == "x"
            - (nx, ny+1, nz, ...) if dim == "y"
            - (nx, ny, nz+1, ...) if dim == "z"
    """
    _slc = fv_solver.array_slicer
    padded_cell_values = fv_solver.bc_for_troubled_cell_mask(
        cell_values,
        {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[dim],
    )
    leftslc = _slc(axis="xyz".index(dim) + 1, cut=(None, -1))
    rightslc = _slc(axis="xyz".index(dim) + 1, cut=(1, None))
    face_values = fv_solver.xp.maximum(
        padded_cell_values[leftslc], padded_cell_values[rightslc]
    )
    return face_values
