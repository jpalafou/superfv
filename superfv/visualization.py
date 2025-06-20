from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

warnings.simplefilter("always")


def _get_nearest_index(
    array: np.ndarray,
    coord: float,
) -> int:
    """
    Get the index of the nearest value in an array to a given coordinate.

    Args:
        array: 1D array of coordinates.
        coord: Coordinate to find in the array.

    Returns:
        Index of the nearest value in the array to the given coordinate.

    """
    idx = np.argmin(np.abs(array - coord)).item()
    if coord not in array:
        warnings.warn(
            f"Coordinate {coord} not found in array. Using nearest: {array[idx]}."
        )
    return idx


def _parse_txyz_slices(
    fv_solver: FiniteVolumeSolver,
    t: Optional[float],
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
) -> Tuple[float, Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]]:
    """
    Parse the time and spatial slices for a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver object.
        t: Desired time. If provided, the snapshot with the closest time will be
            selected. If None, the latest available snapshot is used.
        x, y, z : Desired spatial location(s) along the x, y, or z axis. Each can be:
            - A float: selects the grid index nearest to that coordinate.
            - A tuple (start, end): selects a slice between the nearest grid points to
            `start` and `end`. Either bound may be None to indicate an open interval.
            - None: selects the full range along that dimension.

    Returns:
        nearest_t : The snapshot time closest to the requested `t`.
        tuple_of_slices : A tuple representing slices or indices for the x, y, and z
            dimensions. Each entry is either:
            - An integer index (if a single coordinate was requested),
            - A slice object (if a range or full axis was requested).

    Raises:
        ValueError: x, y, or z is not None, a float, or a tuple of length 2.
    """
    # get nearest time
    t_array = np.sort(np.array(fv_solver.snapshots.times()))
    n = -1 if t is None else _get_nearest_index(t_array, t)
    nearest_t = t_array[n]

    # get xyz slices
    slices: Dict[str, Union[int, slice]] = {
        "x": slice(None),
        "y": slice(None),
        "z": slice(None),
    }
    for dim, coord in zip("xyz", [x, y, z]):
        x_array = getattr(fv_solver.mesh, dim + "_centers")
        if coord is None:
            continue
        elif isinstance(coord, float):
            slices[dim] = _get_nearest_index(x_array, coord)
        elif isinstance(coord, tuple):
            if len(coord) != 2:
                raise ValueError(f"Invalid length for {dim}: {len(coord)}")
            if coord[0] is not None and coord[1] is not None and coord[0] > coord[1]:
                raise ValueError(f"Invalid range for {dim}: {coord}")
            lim1, lim2 = None, None
            if coord[0] is not None:
                lim1 = _get_nearest_index(x_array, coord[0])
            if coord[1] is not None:
                lim2 = _get_nearest_index(x_array, coord[1]) + 1
            slices[dim] = slice(lim1, lim2)
        else:
            raise ValueError(f"Invalid type for {dim}: {type(coord)}")
    tuple_of_slices = (slices["x"], slices["y"], slices["z"])

    return nearest_t, tuple_of_slices


def _extract_variable_data(
    fv_solver: FiniteVolumeSolver, nearest_t: float, variable: str, cell_averaged: bool
) -> np.ndarray:
    """
    Extract the data for a given variable at the nearest time.

    Args:
        fv_solver: FiniteVolumeSolver object.
        nearest_t: Nearest time.
        variable: Name of the variable to extract from the snapshots.
        cell_averaged: Whether to extract the cell-averaged data. If False, the
            variable is extracted using its cell-centered values.
    Returns:
        Array of data for the variable at the nearest time.

    Raises:
        ValueError: Variable not found in snapshots.
    """
    idx = fv_solver.variable_index_map

    # choose the snapshot with the nearest time
    snapshot = fv_solver.snapshots(nearest_t)

    # determine the key for the variable
    if variable in idx.group_var_map["primitives"]:
        key = "w"
    elif variable in idx.group_var_map["conservatives"]:
        key = "u"
    elif variable in idx.group_var_map["passives"]:
        key = "w"
    else:
        raise ValueError(
            f"Variable {variable} not found in groups 'primitives' 'conservatives', or 'passives'."
        )
    if not cell_averaged:
        key += "cc"

    # extract the data
    return snapshot[key][idx(variable)]


def plot_1d_slice(
    fv_solver: FiniteVolumeSolver,
    ax: Axes,
    variable: str,
    cell_averaged: bool = False,
    t: Optional[float] = None,
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    xlabel: bool = False,
    **kwargs,
):
    """
    Plot a 1D slice of a variable at a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver object.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        cell_averaged: Whether to plot the cell average of the variable. If False, the
            variable is plotted using its cell-centered values.
        t: Desired time. If provided, the snapshot with the closest time will be
            selected. If None, the latest available snapshot is used.
        x, y, z : Desired spatial location(s) along the x, y, or z axis. Defaults to
            a floating-point value of 0.5, but each can be:
            - A float: selects the grid index nearest to that coordinate.
            - A tuple (start, end): selects a slice between the nearest grid points to
            `start` and `end`. Either bound may be None to indicate an open interval.
            - None: selects the full range along that dimension.
        xlabel: Whether to show the x-axis label. Defaults to False.
        **kwargs: Keyword arguments for the plot.

    Raises:
        ValueError: If not exactly one of x, y, z is None or a tuple.
    """
    # find the dimension and slices
    if sum(coord is None or isinstance(coord, tuple) for coord in (x, y, z)) != 1:
        raise ValueError("Exactly one of x, y, z must be None or a tuple.")
    dim = next(
        dim
        for dim, val in zip("XYZ", [x, y, z])
        if val is None or isinstance(val, tuple)
    )
    nearest_t, slices = _parse_txyz_slices(fv_solver, t, x, y, z)

    # gather data
    x_arr = getattr(fv_solver.mesh, dim)[slices[0], slices[1], slices[2]]
    f_arr = _extract_variable_data(fv_solver, nearest_t, variable, cell_averaged)[
        slices[0], slices[1], slices[2]
    ]

    # plot
    ax.plot(x_arr, f_arr, **kwargs)
    if xlabel:
        ax.set_xlabel(rf"${dim.lower()}$")


def plot_2d_slice(
    fv_solver: FiniteVolumeSolver,
    ax: Axes,
    variable: str,
    cell_averaged: bool = False,
    t: Optional[float] = None,
    x: Union[float, Tuple[Optional[float], Optional[float]]] = 0.5,
    y: Union[float, Tuple[Optional[float], Optional[float]]] = 0.5,
    z: Union[float, Tuple[Optional[float], Optional[float]]] = 0.5,
    levels: Optional[Union[int, np.ndarray]] = None,
    **kwargs,
):
    """
    Plot a 2D slice of a variable at a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver object.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        cell_averaged: Whether to plot the cell average of the variable. If False, the
            variable is plotted using its cell-centered values.
        t: Desired time. If provided, the snapshot with the closest time will be
            selected. If None, the latest available snapshot is used.
        x, y, z : Desired spatial location(s) along the x, y, or z axis. Defaults to
            a floating-point value of 0.5, but each can be:
            - A float: selects the grid index nearest to that coordinate.
            - A tuple (start, end): selects a slice between the nearest grid points to
            `start` and `end`. Either bound may be None to indicate an open interval.
            - None: selects the full range along that dimension.
        levels: Contour levels to plot. If None, uses imshow instead of contour.
        **kwargs: Keyword arguments for the plot.

    Raises:
        ValueError: If not exactly two of x, y, z is None or a tuple.
    """
    # find the dimensions and slices
    if sum(coord is None or isinstance(coord, tuple) for coord in (x, y, z)) != 2:
        raise ValueError("Exactly two of x, y, z must be None or a tuple.")
    dim1, dim2 = [
        dim
        for dim, val in zip("XYZ", [x, y, z])
        if val is None or isinstance(val, tuple)
    ]
    nearest_t, slices = _parse_txyz_slices(fv_solver, t, x, y, z)

    # determine plotting mode
    using = "imshow" if levels is None else "contour"

    if using == "contour":
        x_arr = getattr(fv_solver.mesh, dim1)[slices[0], slices[1], slices[2]]
        y_arr = getattr(fv_solver.mesh, dim2)[slices[0], slices[1], slices[2]]
    else:
        x_arr = getattr(fv_solver.mesh, dim1.lower() + "_centers")[
            slices["XYZ".index(dim1)]
        ]
        y_arr = getattr(fv_solver.mesh, dim2.lower() + "_centers")[
            slices["XYZ".index(dim2)]
        ]
    f_arr = _extract_variable_data(fv_solver, nearest_t, variable, cell_averaged)[
        slices[0], slices[1], slices[2]
    ]

    # rotate for imshow
    f_arr = np.rot90(f_arr, 1)
    if using == "contour":
        x_arr = np.rot90(x_arr, 1)
        y_arr = np.rot90(y_arr, 1)

    # plot
    if using == "imshow":
        ax.imshow(
            f_arr,
            extent=(
                cast(float, x_arr[0]),
                cast(float, x_arr[-1]),
                cast(float, y_arr[0]),
                cast(float, y_arr[-1]),
            ),
            **kwargs,
        )
    elif using == "contour":
        ax.contour(x_arr, y_arr, f_arr, levels=levels, **kwargs)

    ax.set_xlabel(rf"${dim1}$")
    ax.set_ylabel(rf"${dim2}$")


def power_law(
    x0: float, f0: float, x1: float, f1: float
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a power law function `f(x) = f0 * (x  / x0) ** r` from two points (x0, f0)
    and (x1, f1) where `r = log(f1 / f0) / log(x1 / x0)`.

    Args:
        x0: First x-coordinate.
        f0: First y-coordinate.
        x1: Second x-coordinate.
        f1: Second y-coordinate.

    Returns:
        Power law function.
    """
    r = np.log(f1 / f0) / np.log(x1 / x0)
    return lambda x: f0 * (x / x0) ** r


def plot_power_law_fit(ax: Axes, x: np.ndarray, f: np.ndarray, **kwargs):
    """
    Plot a power law function on a given axes from the two points (x[0], f[0]) and
    (x[-1], f[-1]).

    Args:
        ax: Matplotlib axes object.
        x: x-coordinates.
        f: y-coordinates.
        **kwargs: Keyword arguments for the plot.
    """
    p_law = power_law(x[0], f[0], x[-1], f[-1])
    ax.plot(x, p_law(x), **kwargs)
