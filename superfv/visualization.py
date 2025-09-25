from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver

from .tools.loader import OutputLoader

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
            f"Cell-centered coordinate {coord} not exactly matched in mesh;"
            f" using nearest: {array[idx]:.6g}"
        )
    return idx


def _get_time_array(fv_solver: Union[FiniteVolumeSolver, OutputLoader]) -> np.ndarray:
    """
    Get the array of available snapshot times.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.

    Returns:
        Array of available snapshot times.
    """
    t_list = fv_solver.snapshots.times()
    t_array = np.sort(np.array(t_list))
    return t_array


def _parse_txyz_slices(
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    t: Optional[float],
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
) -> Tuple[float, Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]]:
    """
    Parse the time and spatial slices for a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.
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
    t_array = _get_time_array(fv_solver)
    n = -1 if t is None else _get_nearest_index(t_array, t)
    nearest_t = t_array[n]

    # get xyz slices
    slices: Dict[str, Union[int, slice]] = {}
    for dim, coord in zip("xyz", [x, y, z]):
        x_array = getattr(fv_solver.mesh, dim + "_centers")
        if dim not in fv_solver.active_dims:
            if coord is not None:
                warnings.warn(
                    f"Dimension {dim} is not active in the solver. "
                    f"Ignoring coordinate {coord}."
                )
            slices[dim] = 0
        elif coord is None:
            slices[dim] = slice(None)
        elif isinstance(coord, (int, float)):
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
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    nearest_t: float,
    variable: str,
    cell_averaged: bool = False,
    theta: bool = False,
    troubles: bool = False,
) -> np.ndarray:
    """
    Extract the data for a given variable at the nearest time.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.
        nearest_t: Nearest time.
        variable: Name of the variable to extract from the snapshots.
        cell_averaged: Whether to extract the cell-averaged data. If False, the
            variable is extracted using its cell-centered values.
        theta: Whether to plot the Zhang-Shu slope limiter of a specific variable.
    Returns:
        Array of data for the variable at the nearest time.

    Raises:
        ValueError: Variable not found in snapshots.
    """
    idx = fv_solver.variable_index_map

    # choose the snapshot with the nearest time
    snapshot = fv_solver.snapshots(nearest_t)

    # plot troubles/cascade
    if troubles:
        # troubled to be overlaid on variable plot
        return snapshot["troubles"][idx(variable)]
    if variable == "troubles":
        # troubles to be plotted as their own variable
        return np.max(snapshot["troubles"], axis=0)
    if variable == "cascade":
        return snapshot["cascade"][0]

    # determine the key for the variable
    if theta:
        if "theta" not in snapshot:
            raise ValueError("Theta not found in snapshots.")
        key = "theta"
    elif variable in idx.group_var_map["conservatives"]:
        key = "u"
    elif variable in idx.group_var_map["primitives"]:
        key = "w"
    elif variable in idx.group_var_map["passives"]:
        key = "w"
    else:
        raise ValueError(
            f"Variable {variable} not found in groups 'primitives' 'conservatives', or 'passives'."
        )

    if not theta and not cell_averaged:
        key += "cc"

    # extract the data
    return snapshot[key][idx(variable)]


def _is_None_or_tuple(
    value: Optional[Union[float, Tuple[Optional[float], Optional[float]]]],
) -> bool:
    """
    Check if the value is None or a tuple of length 2.

    Args:
        value: Value to check.

    Returns:
        True if value is None or a tuple of length 2, False otherwise.
    """
    return value is None or (isinstance(value, tuple) and len(value) == 2)


def plot_1d_slice(
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    ax: Axes,
    variable: str,
    cell_averaged: bool = False,
    theta: bool = False,
    trouble_marker: Optional[str] = None,
    trouble_color: str = "red",
    trouble_size_rate: float = 0.5,
    t: Optional[float] = None,
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    xlabel: bool = False,
    **kwargs,
):
    """
    Plot a 1D slice of a variable at a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        cell_averaged: Whether to plot the cell average of the variable. If False, the
            variable is plotted using its cell-centered values.
        theta: Whether to plot the Zhang-Shu slope limiter of a specific variable.
        trouble_marker: Style of marker overlaid on troubled cells. If None, no markers
            are overlaid. Only valid for solvers that use MOOD.
        trouble_color: Color of the trouble markers.
        trouble_size_rate: Float in [0, 1] controlling how the size of trouble markers
            scales with trouble level (which ranges from 0 to 1). A value of 0 makes
            all trouble markers the same size as the variable markers, regardless of
            trouble level. A value of 1 makes the smallest trouble markers vanish
            (size 0) and the largest match the size of the variable markers.
            Intermediate values interpolate linearly between these extremes.
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

    # find the dimensions to plot
    active_dims = fv_solver.active_dims
    slicing_dims = [
        dim
        for dim, coord in zip("xyz", (x, y, z))
        if _is_None_or_tuple(coord) and dim in active_dims
    ]
    if len(slicing_dims) != 1:
        raise ValueError(
            "Exactly one of x, y, z must be None or a length-two tuple for a 1D slice, "
            "and it must match an active mesh dimension."
        )
    dim = slicing_dims[0]

    # find the nearest slices in time and space
    nearest_t, slices = _parse_txyz_slices(fv_solver, t, x, y, z)

    # gather data
    x_arr = getattr(fv_solver.mesh, dim.upper())[slices[0], slices[1], slices[2]]
    f_arr = _extract_variable_data(
        fv_solver,
        nearest_t,
        variable,
        cell_averaged=cell_averaged,
        theta=theta,
    )[slices[0], slices[1], slices[2]]

    # plot
    (line,) = ax.plot(x_arr, f_arr, **kwargs)

    # optionally plot troubles
    if trouble_marker is not None:
        ms = line.get_markersize()
        min_trouble_size = (1 - trouble_size_rate) * ms
        trouble_size_rate = trouble_size_rate * ms

        troubles_arr = _extract_variable_data(
            fv_solver, nearest_t, variable, troubles=True
        )[slices[0], slices[1], slices[2]]
        trouble_levels = np.unique(troubles_arr)

        for trouble_level in trouble_levels:
            if trouble_level == 0:
                continue
            trouble_size = min_trouble_size + trouble_size_rate * trouble_level
            troubles_idx = troubles_arr == trouble_level
            ax.plot(
                x_arr[troubles_idx],
                f_arr[troubles_idx],
                marker=trouble_marker,
                color=trouble_color,
                markersize=trouble_size,
                linestyle="none",
            )

    # optionally add x label
    if xlabel:
        ax.set_xlabel(rf"${dim}$")


def plot_2d_slice(
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    ax: Axes,
    variable: str,
    cell_averaged: bool = False,
    theta: bool = False,
    t: Optional[float] = None,
    x: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    y: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    z: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None,
    levels: Optional[Union[int, np.ndarray]] = None,
    cmap: Optional[str] = None,
    colorbar: bool = False,
    overlay_troubles: bool = False,
    troubles_cmap: str = "Reds",
    troubles_alpha: float = 0.5,
    xlabel: bool = False,
    ylabel: bool = False,
    **kwargs,
) -> Tuple[Union[AxesImage, QuadContourSet], Optional[Colorbar]]:
    """
    Plot a 2D slice of a variable at a given time and coordinates.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        cell_averaged: Whether to plot the cell average of the variable. If False, the
            variable is plotted using its cell-centered values.
        theta: Whether to plot the Zhang-Shu slope limiter of a specific variable.
        t: Desired time. If provided, the snapshot with the closest time will be
            selected. If None, the latest available snapshot is used.
        x, y, z : Desired spatial location(s) along the x, y, or z axis. Defaults to
            a floating-point value of 0.5, but each can be:
            - A float: selects the grid index nearest to that coordinate.
            - A tuple (start, end): selects a slice between the nearest grid points to
            `start` and `end`. Either bound may be None to indicate an open interval.
            - None: selects the full range along that dimension.
        levels: Contour levels to plot. If None, uses imshow instead of contour.
        cmap: Colormap to use for the plot. If None, no colormap is applied.
        colorbar: Whether to add a colorbar to the plot.
        overlay_troubles: Whether to overlay troubled cells on top of the variable
            plot. Only valid for solvers that use MOOD.
        troubles_cmap: Colormap to use for the troubled cells overlay.
        troubles_alpha: Alpha value for the troubled cells overlay.
        xlabel: Whether to show the x-axis label.
        ylabel: Whether to show the y-axis label.
        **kwargs: Keyword arguments for the plot.

    Returns:
        artist: The AxesImage or QuadContourSet created by the plot.
        cbar: The Colorbar object if colorbar is True, else None.

    Raises:
        ValueError: If not exactly two of x, y, z is None or a tuple.
    """
    artist: Union[AxesImage, QuadContourSet]

    # find the dimensions to plot
    is_valid = _is_None_or_tuple
    active_dims = fv_solver.active_dims
    slicing_dims = [
        dim
        for dim, coord in zip("xyz", (x, y, z))
        if is_valid(coord) and dim in active_dims
    ]
    if len(slicing_dims) != 2:
        raise ValueError(
            "Exactly two of x, y, z must be None or a length-two tuple for a 2D slice "
            "and they must match the active mesh dimensions."
        )
    dim1, dim2 = slicing_dims

    # find the nearest slices in time and space
    nearest_t, slices = _parse_txyz_slices(fv_solver, t, x, y, z)

    # determine plotting mode
    using = "imshow" if levels is None else "contour"

    if using == "contour":
        x_arr = getattr(fv_solver.mesh, dim1.upper())[slices[0], slices[1], slices[2]]
        y_arr = getattr(fv_solver.mesh, dim2.upper())[slices[0], slices[1], slices[2]]
    else:
        x_arr = getattr(fv_solver.mesh, dim1 + "_centers")[slices["xyz".index(dim1)]]
        y_arr = getattr(fv_solver.mesh, dim2 + "_centers")[slices["xyz".index(dim2)]]
    f_arr = _extract_variable_data(
        fv_solver, nearest_t, variable, cell_averaged=cell_averaged, theta=theta
    )[slices[0], slices[1], slices[2]]

    # rotate for imshow
    f_arr = np.rot90(f_arr, 1)
    if using == "contour":
        x_arr = np.rot90(x_arr, 1)
        y_arr = np.rot90(y_arr, 1)

    # plot
    if using == "imshow":
        extent = (
            cast(float, x_arr[0]),
            cast(float, x_arr[-1]),
            cast(float, y_arr[0]),
            cast(float, y_arr[-1]),
        )

        artist = ax.imshow(f_arr, extent=extent, cmap=cmap, **kwargs)

        if overlay_troubles:
            troubles_arr = _extract_variable_data(
                fv_solver, nearest_t, variable, troubles=True
            )[slices[0], slices[1], slices[2]]
            troubles_arr = np.rot90(troubles_arr, 1)
            ax.imshow(
                np.where(troubles_arr > 0, troubles_arr, np.nan),
                extent=extent,
                alpha=troubles_alpha,
                cmap=troubles_cmap,
                vmin=0,
                vmax=1,
            )
    elif using == "contour":
        artist = ax.contour(x_arr, y_arr, f_arr, levels=levels, cmap=cmap, **kwargs)

    # add colorbar
    if colorbar:
        cbar = plt.colorbar(
            ax.images[0] if using == "imshow" else ax.collections[0], ax=ax
        )
    else:
        cbar = None

    # add axis labels
    if xlabel:
        ax.set_xlabel(rf"${dim1}$")
    if ylabel:
        ax.set_ylabel(rf"${dim2}$")

    return artist, cbar


def plot_spacetime(
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    ax: Axes,
    variable: str,
    cell_averaged: bool = False,
    theta: bool = False,
    cmap: Optional[str] = None,
    colorbar: bool = False,
    overlay_troubles: bool = False,
    troubles_cmap: str = "Reds",
    troubles_alpha: float = 0.5,
    xlabel: bool = False,
    tlabel: bool = False,
    **kwargs,
) -> Tuple[AxesImage, Optional[Colorbar]]:
    """
    Plot a spacetime diagram of a variable.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object. Must be 1D.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        cell_averaged: Whether to plot the cell average of the variable. If False, the
            variable is plotted using its cell-centered values.
        theta: Whether to plot the Zhang-Shu slope limiter of a specific variable.
        cmap: Colormap to use for the plot. If None, no colormap is applied
        colorbar: Whether to add a colorbar to the plot.
        overlay_troubles: Whether to overlay troubled cells on top of the variable
            plot. Only valid for solvers that use MOOD.
        troubles_cmap: Colormap to use for the troubled cells overlay.
        troubles_alpha: Alpha value for the troubled cells overlay.
        xlabel: Whether to show the x-axis (vertical axis) label.
        tlabel: Whether to show the t-axis (horizontal axis) label.
        **kwargs: Keyword arguments for the plot.
    """
    active_dims = fv_solver.active_dims
    if len(active_dims) != 1:
        raise ValueError(
            "Spacetime plots are only supported for 1D problems (exactly one active "
            "dimension)."
        )
    dim = active_dims[0]

    _, slices = _parse_txyz_slices(fv_solver, 0, None, None, None)
    t_arr = _get_time_array(fv_solver)
    x_arr = getattr(fv_solver.mesh, dim + "_centers")

    f_arr = np.empty((len(x_arr), len(t_arr))) * np.nan
    for i, t in enumerate(t_arr):
        f_arr[:, i] = _extract_variable_data(
            fv_solver, t, variable, cell_averaged=cell_averaged, theta=theta
        )[*slices]

    extent = (
        cast(float, t_arr[0]),
        cast(float, t_arr[-1]),
        cast(float, x_arr[0]),
        cast(float, x_arr[-1]),
    )
    im = ax.imshow(
        f_arr, extent=extent, cmap=cmap, origin="lower", aspect="auto", **kwargs
    )

    if overlay_troubles:
        troubles_arr = np.empty((len(x_arr), len(t_arr))) * np.nan
        for i, t in enumerate(t_arr):
            current_troubles = _extract_variable_data(
                fv_solver, t, variable, troubles=True
            )[*slices]
            troubles_arr[:, i] = np.where(
                current_troubles > 0, current_troubles, np.nan
            )

        ax.imshow(
            troubles_arr,
            extent=extent,
            cmap=troubles_cmap,
            alpha=troubles_alpha,
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="auto",
        )

    if colorbar:
        cbar = plt.colorbar(ax.images[0], ax=ax)
    else:
        cbar = None

    if tlabel:
        ax.set_xlabel(r"$t$")
    if xlabel:
        ax.set_ylabel(rf"${dim}$")

    return im, cbar


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


def plot_timeseries(
    fv_solver: Union[FiniteVolumeSolver, OutputLoader],
    ax: Axes,
    variable: str,
    **kwargs,
):
    """
    Plot a timeseries logged in `fv_solver.minisnapshots`.

    Args:
        fv_solver: FiniteVolumeSolver or OutputLoader object.
        ax: Matplotlib axes object.
        variable: Name of the variable to plot.
        **kwargs: Keyword arguments for the plot.

    Raises:
        ValueError: `variable` is not in `fv_solver.minisnapshots`.
    """
    if variable not in fv_solver.minisnapshots:
        raise ValueError(
            f"Variable {variable} not found in `FiniteVolumeSolver.minisnapshots`."
        )

    t = fv_solver.minisnapshots["t"]
    f = fv_solver.minisnapshots[variable]

    # unpack the data and plot along substeps if it is a list of lists
    if isinstance(f[0], list):
        t_big = []
        f_big = []

        for t, dt, f_packet in zip(t, fv_solver.minisnapshots["dt"], f):
            n = len(f_packet)
            t_big.extend([t - (i / n) * dt for i in range(n - 1, -1, -1)])
            f_big.extend(f_packet)

        t = t_big
        f = f_big

    ax.plot(t, f, **kwargs)
