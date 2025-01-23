from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

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
        array (np.ndarray): 1D array of coordinates.
        coord (float): Coordinate to find in the array.

    Returns:
        int: Index of the nearest value in the array to the given coordinate.

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
        fv_solver (FiniteVolumeSolver): Solver object.
        t (Optional[float]): Time to get the nearest snapshot. Defaults to None, which
            uses the last snapshot time.
        x (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            x-coordinate or range.
        y (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            y-coordinate or range.
        z (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            z-coordinate or range.

    Returns:
        Tuple[float, Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]]:
            nearest_t: Nearest time.
            tuple_of_slices: Tuple of slices for x, y, z.

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
        x_array = getattr(fv_solver, dim)
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
    fv_solver: FiniteVolumeSolver, nearest_t: float, variable: str
) -> np.ndarray:
    """
    Extract the data for a given variable at the nearest time.

    Args:
        fv_solver (FiniteVolumeSolver): Solver object.
        nearest_t (float): Nearest time.
        variable (str): Variable to extract.

    Returns:
        np.ndarray: Data for the variable at the nearest time.

    Raises:
        ValueError: Variable not found in snapshots.
    """
    snapshot = fv_solver.snapshots(nearest_t)
    _slc = fv_solver.array_slicer
    if variable in snapshot:
        return snapshot[variable]
    if variable in _slc.var_names:
        return snapshot["u"][_slc(variable)]
    raise ValueError(f"Variable {variable} not found in snapshots.")


def plot_1d_slice(
    fv_solver: FiniteVolumeSolver,
    ax: Axes,
    variable: str,
    t: Optional[float] = None,
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = 0.5,
    **kwargs,
):
    """
    Plot a 1D slice of a variable at a given time and coordinates.

    Args:
        fv_solver (FiniteVolumeSolver): Solver object.
        ax (Axes): Matplotlib axes object.
        variable (str): Variable to plot.
        t (Optional[float]): Time to get the nearest snapshot. Defaults to None, which
            uses the last snapshot time.
        x (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            x-coordinate or range. Defaults to 0.5.
        y (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            y-coordinate or range. Defaults to 0.5.
        z (Optional[Union[float, Tuple[Optional[float], Optional[float]]]]):
            z-coordinate or range. Defaults to 0.5.
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
    _x = getattr(fv_solver, dim)[slices[0], slices[1], slices[2]]
    _y = _extract_variable_data(fv_solver, nearest_t, variable)[
        slices[0], slices[1], slices[2]
    ]

    # plot
    ax.plot(_x, _y, **kwargs)
    ax.set_xlabel(rf"${dim.lower()}$")


def plot_2d_slice(
    fv_solver: FiniteVolumeSolver,
    ax: Axes,
    variable: str,
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
        fv_solver (FiniteVolumeSolver): Solver object.
        ax (Axes): Matplotlib axes object.
        variable (str): Variable to plot.
        t (Optional[float]): Time to get the nearest snapshot. Defaults to None, which
            uses the last snapshot time.
        x (Union[float, Tuple[Optional[float], Optional[float]]]):
            x-coordinate or range. Defaults to 0.5.
        y (Union[float, Tuple[Optional[float], Optional[float]]]):
            y-coordinate or range. Defaults to 0.5.
        z (Union[float, Tuple[Optional[float], Optional[float]]]):
            z-coordinate or range. Defaults to 0.5.
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
        _x = getattr(fv_solver, dim1)[slices[0], slices[1], slices[2]]
        _y = getattr(fv_solver, dim2)[slices[0], slices[1], slices[2]]
    else:
        _x = getattr(fv_solver, dim1.lower())[slices["XYZ".index(dim1)]]
        _y = getattr(fv_solver, dim2.lower())[slices["XYZ".index(dim2)]]
    _z = _extract_variable_data(fv_solver, nearest_t, variable)[
        slices[0], slices[1], slices[2]
    ]

    # rotate for imshow
    _z_rotated = np.rot90(_z, 1)
    if using == "contour":
        _x_rotated = np.rot90(_x, 1)
        _y_rotated = np.rot90(_y, 1)

    # plot
    if using == "imshow":
        ax.imshow(
            _z_rotated,
            extent=(
                cast(float, _x[0]),
                cast(float, _x[-1]),
                cast(float, _y[0]),
                cast(float, _y[-1]),
            ),
            **kwargs,
        )
    elif using == "contour":
        ax.contour(_x_rotated, _y_rotated, _z_rotated, levels=levels, **kwargs)

    ax.set_xlabel(rf"${dim1}$")
    ax.set_ylabel(rf"${dim2}$")
