from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Tuple

import numpy as np

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import ArrayLike

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver


def _cache_fluxes(
    fv_solver: FiniteVolumeSolver,
    fluxes: Tuple[ArrayLike, ArrayLike, ArrayLike],
    scheme: str,
):
    """
    Cache the fluxes for the given scheme in `fv_solver.MOOD_cache`.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): The fluxes.
        scheme (str): The scheme name.
    """
    fv_solver.MOOD_cache.add("F_" + scheme, fluxes[0])
    fv_solver.MOOD_cache.add("G_" + scheme, fluxes[1])
    fv_solver.MOOD_cache.add("H_" + scheme, fluxes[2])


def init_MOOD(
    fv_solver: FiniteVolumeSolver,
    u: ArrayLike,
    fluxes: Tuple[ArrayLike, ArrayLike, ArrayLike],
):
    """
    Initialize the MOOD algorithm.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): The fluxes (F, G, H). Has
            shapes:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
    """
    fv_solver.MOOD_cache.clear()
    fv_solver.MOOD_cache.add("violations", np.empty_like(u[0]))
    initial_scheme = "fv" + str(fv_solver.p)
    if initial_scheme != fv_solver.MOOD_cascade[0]:
        raise ValueError(
            f"Starting scheme {initial_scheme} is not the first scheme in the cascade {fv_solver.MOOD_cascade}."
        )
    _cache_fluxes(fv_solver, fluxes, initial_scheme)
    fv_solver.MOOD_cache.add("cascade_idx_array", np.full_like(u[0], 0, dtype=int))
    fv_solver.MOOD_iter_count = 0


def detect_troubles(
    fv_solver: FiniteVolumeSolver,
    t: float,
    dt: float,
    u: ArrayLike,
    fluxes: Tuple[ArrayLike, ArrayLike, ArrayLike],
    NAD: Optional[float] = None,
    NAD_vars: Optional[Tuple[str]] = None,
    PAD: Optional[Dict[str, Tuple[float, float]]] = None,
) -> bool:
    """
    Detect troubles in the solution.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        t (float): The current time.
        dt (float): The time step.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): The fluxes (F, G, H). Has
            shapes:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        NAD (Optional[float]): The NAD tolerance. If None, NAD is not checked.
        NAD_vars (Optional[Tuple[str]]): The variables to check for NAD. If None, all
            "active" variables are checked.
        PAD (Optional[Dict[str, Tuple[float, float]]]): Dict of variable names to
            (lower, upper) bounds. If None, PAD is not checked.

    Returns:
        bool: Whether troubles were detected.
    """
    _slc = fv_solver.array_slicer

    if fv_solver.MOOD_iter_count >= fv_solver.MOOD_max_iter_count:
        return False

    # compute candidate solution
    ustar = u + dt * fv_solver.RHS(u, *fluxes)

    # compute NAD and/or PAD violations
    violations = np.zeros_like(u[0], dtype=bool)
    if PAD is None and NAD is None:
        return False
    if NAD is not None:
        _NAD_var_slc = _slc("actives") if NAD_vars is None else _slc(NAD_vars)
        dmp_min, dmp_max = compute_dmp(
            fv_solver.apply_bc(u, 1)[_NAD_var_slc],
            dims=fv_solver.dims,
            include_corners=True,
        )
        NAD_violations = compute_dmp_violations(
            ustar[_NAD_var_slc], dmp_min, dmp_max, tol=NAD
        )
        violations[...] = np.any(NAD_violations < 0, axis=0)
    if PAD is not None:
        _PAD_var_slc = _slc(tuple(PAD.keys()))
        lower = np.full_like(u[:, :1, :1, :1], -np.inf)
        upper = np.full_like(u[:, :1, :1, :1], np.inf)
        for v, (m, M) in PAD.items():
            lower[_slc(v)] = m
            upper[_slc(v)] = M
        violations[...] = np.logical_or(
            np.any(
                np.logical_or(
                    ustar[_PAD_var_slc] < lower[_PAD_var_slc],
                    ustar[_PAD_var_slc] > upper[_PAD_var_slc],
                ),
                axis=0,
            ),
            violations,
        )

    # return bool
    if np.any(violations):
        fv_solver.MOOD_cache["violations"] = violations
        fv_solver.MOOD_cache["cascade_idx_array"][...] = np.where(
            violations,
            np.minimum(
                fv_solver.MOOD_cache["cascade_idx_array"] + 1,
                len(fv_solver.MOOD_cascade) - 1,
            ),
            fv_solver.MOOD_cache["cascade_idx_array"],
        )
        return True
    else:
        return False


def compute_dmp_violations(
    arr: ArrayLike, dmp_min: ArrayLike, dmp_max: ArrayLike, tol: float
) -> ArrayLike:
    """
    Compute a boolean array indicating whether the given array violates the discrete
    maximum principle.

    Args:
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
    dmp_range = np.max(dmp_max, axis=(1, 2, 3), keepdims=True) - np.min(
        dmp_min, axis=(1, 2, 3), keepdims=True
    )
    violations = np.minimum(arr - dmp_min, dmp_max - arr) + tol * dmp_range
    return violations


def revise_fluxes(
    fv_solver: FiniteVolumeSolver,
    t: float,
    dt: float,
    u: ArrayLike,
    fluxes: Tuple[ArrayLike, ArrayLike, ArrayLike],
    mode: Literal["transverse", "gauss-legendre"],
    slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """
    Revise the fluxes using the MOOD algorithm.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        t (float): The current time.
        dt (float): The time step.
        u (ArrayLike): The solution array. Has shape (nvars, nx, ny, nz, ...).
        fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): The fluxes (F, G, H). Has
            shapes:
            - F: (nvars, nx+1, ny, nz, ...)
            - G: (nvars, nx, ny+1, nz, ...)
            - H: (nvars, nx, ny, nz+1, ...)
        mode (Literal["transverse", "gauss-legendre"]): The mode for interpolating
            nodes and integrals.
        slope_limiter (Optional[Literal["minmod", "moncen"]]): The slope limiter to use
            if `scheme` is "muscl".

    Returns:
        Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]: Tuple composed of:
            - The revised time step.
            - The revised fluxes (F, G, H). Has shapes:
                - F: (nvars, nx+1, ny, nz, ...)
                - G: (nvars, nx, ny+1, nz, ...)
                - H: (nvars, nx, ny, nz+1, ...)
    """
    fv_solver.MOOD_iter_count += 1
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
    revised_F, revised_G, revised_H = np.array([]), np.array([]), np.array([])
    if fv_solver.using_xdim:
        F_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            np.maximum,
            cascade_idx_array,
            "x",
            periodic=fv_solver.bc.bcx == ("periodic", "periodic"),
        )
        revised_F = np.zeros_like(fluxes[0])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_F += fv_solver.MOOD_cache["F_" + scheme] * np.where(
                F_cascade_idx_array == i, 1, 0
            )
    if fv_solver.using_ydim:
        G_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            np.maximum,
            cascade_idx_array,
            "y",
            periodic=fv_solver.bc.bcy == ("periodic", "periodic"),
        )
        revised_G = np.zeros_like(fluxes[1])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_G += fv_solver.MOOD_cache["G_" + scheme] * np.where(
                G_cascade_idx_array == i, 1, 0
            )
    if fv_solver.using_zdim:
        H_cascade_idx_array = map_cell_values_to_face_values(
            fv_solver,
            np.maximum,
            cascade_idx_array,
            "z",
            periodic=fv_solver.bc.bcz == ("periodic", "periodic"),
        )
        revised_H = np.zeros_like(fluxes[2])
        for i, scheme in enumerate(fv_solver.MOOD_cascade):
            revised_H += fv_solver.MOOD_cache["H_" + scheme] * np.where(
                H_cascade_idx_array == i, 1, 0
            )

    return dt, (revised_F, revised_G, revised_H)


def map_cell_values_to_face_values(
    fv_solver: FiniteVolumeSolver,
    f: Callable[[ArrayLike, ArrayLike], ArrayLike],
    cell_values: ArrayLike,
    dim: Literal["x", "y", "z"],
    periodic: bool = False,
) -> ArrayLike:
    """
    Map cell values to face values using a function `f`.

    Args:
        fv_solver (FiniteVolumeSolver): The finite volume solver object.
        f (Callable[[ArrayLike, ArrayLike], ArrayLike]): The function to use.
        cell_values (ArrayLike): The cell values. Has shape (nx, ny, nz, ...).
        dim (Literal["x", "y", "z"]): The dimension.
        periodic (bool): Whether the dimension is periodic.

    Returns:
        ArrayLike: The face values. Has shape
            - (nx+1, ny, nz, ...) if dim == "x"
            - (nx, ny+1, nz, ...) if dim == "y"
            - (nx, ny, nz+1, ...) if dim == "z"
    """
    padded_cell_values = np.pad(
        cell_values,
        pad_width=[
            (1, 1) if dim == "x" else (0, 0),
            (1, 1) if dim == "y" else (0, 0),
            (1, 1) if dim == "z" else (0, 0),
        ],
        mode="wrap" if periodic else "edge",
    )
    leftslc, rightslc = [slice(None)] * 3, [slice(None)] * 3
    leftslc["xyz".index(dim)] = slice(None, -1)
    rightslc["xyz".index(dim)] = slice(1, None)
    face_values = f(padded_cell_values[(*leftslc,)], padded_cell_values[(*rightslc,)])
    return face_values
