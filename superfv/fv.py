from collections.abc import Iterable
from functools import lru_cache
from itertools import product
from types import ModuleType
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union, cast

import numpy as np

from .stencil import (
    conservative_interpolation_weights,
    inplace_stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.array_management import ArrayLike, merge_slices

InterpCoord = Union[int, float]
InterpCoords = Union[Sequence[InterpCoord], np.ndarray]
StencilWeights = Union[Sequence[float], np.ndarray]

DIM_TO_AXIS = {"x": 1, "y": 2, "z": 3}


def _scaled_gauss_legendre_points_and_weights(p: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute Gauss-Legendre quadrature points and weights scaled to the interval
    [-0.5, 0.5].

    Args:
        p: Polynomial degree of quadrature rule.

    Returns:
        x: Quadrature points, has shape (n,).
        w: Quadrature weights, has shape (n,).
    """
    x, w = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))
    return 0.5 * x, 0.5 * w


@lru_cache(maxsize=None)
def _gauss_legendre_for_finite_volume(
    px: int, py: int, pz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_pts, x_wts = _scaled_gauss_legendre_points_and_weights(px)
    y_pts, y_wts = _scaled_gauss_legendre_points_and_weights(py)
    z_pts, z_wts = _scaled_gauss_legendre_points_and_weights(pz)
    xp, yp, zp = np.meshgrid(x_pts, y_pts, z_pts, indexing="ij")
    xw, yw, zw = np.meshgrid(x_wts, y_wts, z_wts, indexing="ij")
    xp = xp.flatten()
    yp = yp.flatten()
    zp = zp.flatten()
    w = (xw * yw * zw).flatten()
    return xp, yp, zp, w


def gauss_legendre_for_finite_volume(
    px: int, py: int, pz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gauss-Legendre quadrature points and weights for a finite volume with up to
    three dimensions, where the quadrature points are scaled to the 3D unit cube
    [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5].

    Args:
        px: Polynomial degree of quadrature rule in x dimension.
        py: Polynomial degree of quadrature rule in y dimension.
        pz: Polynomial degree of quadrature rule in z dimension.

    Returns:
        xp, yp, zp: Quadrature points in x, y, and z dimensions. Each has shape
            (n_quadrature), where `n_quadrature` is the total number of quadrature
            points flattened across the three dimensions.
        w: Weights for the quadrature points, has shape (n_quadrature,).
    """
    return _gauss_legendre_for_finite_volume(px, py, pz)


def gauss_legendre_mesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h: Tuple[float, float, float],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gauss-Legendre quadrature points and weights for a finite volume mesh.

    Args:
        x: x-coordinates, has shape (nx, ny, nz).
        y: y-coordinates, has shape (nx, ny, nz).
        z: z-coordinates, has shape (nx, ny, nz).
        h: Mesh spacings (hx, hy, hz).
        p: Polynomial degree of quadrature rule in each dimension (px, py, pz).

    Returns:
        xp, yp, zp: Quadrature points in x, y, and z dimensions. Each has shape
            (nx, ny, nz, n_quadrature), where `n_quadrature` is the total number of
            quadrature points flattened across the three dimensions.
        w: Weights for the quadrature points, has shape (1, 1, 1, n_quadrature).
    """
    hx, hy, hz = h
    px, py, pz = p
    xp, yp, zp, w = gauss_legendre_for_finite_volume(px, py, pz)

    # Compute the evaluation points for the quadrature rule
    na = np.newaxis
    x_eval = x[..., na] + xp[na, na, na, :] * hx
    y_eval = y[..., na] + yp[na, na, na, :] * hy
    z_eval = z[..., na] + zp[na, na, na, :] * hz

    return x_eval, y_eval, z_eval, w


def fv_average(
    f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    h: Tuple[float, float, float],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> ArrayLike:
    """
    Compute finite volume average of f over 3D domain.

    Args:
        f: Function at which to evaluate quadrature points. It should accept the
            following arguments:
            - x: x-coordinate array. Has shape (nx, ny, nz).
            - y: y-coordinate array. Has shape (nx, ny, nz).
            - z: z-coordinate array. Has shape (nx, ny, nz).
            and return an array of shape (nvar, nx, ny, nz).
        x: x-coordinates, has shape (nx, ny, nz).
        y: y-coordinates, has shape (nx, ny, nz).
        z: z-coordinates, has shape (nx, ny, nz).
        h: Mesh spacings (hx, hy, hz).
        p: Polynomial degree of quadrature rule in each dimension.

    Returns:
        ArrayLike: Finite volume average.
    """
    x_eval, y_eval, z_eval, weights = gauss_legendre_mesh(x, y, z, h, p)
    vals = f(x_eval, y_eval, z_eval)
    return np.sum(weights * vals, axis=4)


Union[int, float, Sequence[Union[int, float]]]


def fv_interpolate(
    xp: ModuleType,
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Interpolate a nodal value from an array of finite volume averages.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        nodes: Dictionary with keys 'x', 'y', 'z' and values as the nodal coordinates.
            Each value may be a float or an array of floats representing the
            coordinates to interpolate along the respective dimension on [-1, 1].
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout). Results are written to
            `out[..., :ninterpolations]`, where `ninterpolations` is the flattened
            number of coordinates across the specified dimensions. The order of the
            coordinates is determined by the order of the keys in `nodes`.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    return _fv_interpolate_direct(
        xp,
        conservative_interpolation_weights,
        u,
        nodes,
        p,
        buffer,
        out,
    )


def fv_integrate(
    xp: ModuleType,
    u: ArrayLike,
    dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Compute an average from an array of cell-centered or face-centered nodal values.

    Args:
        xp: `np` namespace.
        u: Array of central nodes to integrate, has shape (nvars, nx, ny, nz).
        dims: Tuple of strings indicating the dimensions to integrate over with
            dimensions indicated as 'x', 'y', or 'z'. Can have length 1, 2, or 3
            and cannot be empty or contain duplicates.
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout). Results are written to `out[..., 0]`.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    return _fv_interpolate_direct(
        xp,
        lambda p, x: uniform_quadrature_weights(p),
        u,
        {dim: 0 for dim in dims},
        p,
        buffer,
        out,
    )


def _fv_interpolate_direct(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Interpolate a nodal value from an array of finite volume averages.

    Args:
        xp: `np` namespace.
        stencil_func: Function to compute the interpolation stencil weights.
            It should accept the polynomial degree and a coordinate value, and return
            an array of stencil weights.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        nodes: Dictionary with keys 'x', 'y', 'z' and values as the nodal coordinates.
            Each value may be a float or an array of floats representing the
            coordinates to interpolate along the respective dimension on [-1, 1].
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    if u.ndim != 4:
        raise ValueError(
            f"Input array u must have 4 dimensions, but has {u.ndim} dimensions."
        )
    if out.ndim != 5:
        raise ValueError(
            f"Output array out must have 5 dimensions, but has {out.ndim} dimensions."
        )
    if buffer.ndim != 5:
        raise ValueError(
            f"Buffer array must have 5 dimensions, but has {buffer.ndim} dimensions."
        )

    # one-dimensional interpolation
    if len(nodes) == 1:
        return _fv_interpolate_1sweep(xp, stencil_func, u, nodes, p, out)

    # two-dimensional interpolation
    if len(nodes) == 2:
        return _fv_interpolate_2sweeps(xp, stencil_func, u, nodes, p, buffer, out)

    # three-dimensional interpolation
    if len(nodes) == 3:
        return _fv_interpolate_3sweeps(xp, stencil_func, u, nodes, p, buffer, out)

    raise ValueError(
        f"Invalid number of dimensions in nodes: {len(nodes)}. Expected 1, 2, or 3."
    )


def _fv_interpolate_1sweep(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    out: ArrayLike,
):
    dim1 = next(iter(nodes))
    coords1 = _to_iter(nodes[dim1])
    if out.shape[4] < len(coords1):
        raise ValueError(
            f"Output interpolation axis size {out.shape[4]} is smaller than the number"
            f" number of coordinates {len(coords1)} for dimension '{dim1}'."
        )

    slices = []
    for i, coord in enumerate(coords1):
        stencil = stencil_func(p, coord)
        slices.append(
            inplace_stencil_sweep(xp, u, stencil, DIM_TO_AXIS[dim1], out=out[..., i])
        )

    return merge_slices(*slices, union=True)


def _fv_interpolate_2sweeps(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
):
    dim1, dim2 = list(nodes)
    coords1 = _to_iter(nodes[dim1])
    coords2 = _to_iter(nodes[dim2])
    layer1_len = len(coords1)
    layer2_len = len(coords2)

    if out.shape[4] < layer1_len * layer2_len:
        raise ValueError(
            f"Output interpolation axis size {out.shape[4]} is smaller than the number"
            f" of coordinates {layer1_len * layer2_len} for dimensions '{dim1}' and "
            f"'{dim2}'."
        )
    if buffer.shape[4] < layer1_len:
        raise ValueError(
            f"Buffer size {buffer.shape[4]} is smaller than the number of "
            f"coordinates {layer1_len} for dimension '{dim1}'."
        )

    # fill buffer
    slices = []
    slices1 = []
    for i in range(layer1_len):
        stencil1 = stencil_func(p, coords1[i])
        modified = inplace_stencil_sweep(
            xp, u, stencil1, DIM_TO_AXIS[dim1], out=buffer[:, :, :, :, i]
        )
        slices1.append(modified)
    slices.append(merge_slices(*slices1, union=True))

    # interpolate nodes from buffer
    slices2 = []
    for i, j in product(range(layer1_len), range(layer2_len)):
        stencil2 = stencil_func(p, coords2[j])
        modified = inplace_stencil_sweep(
            xp,
            buffer[:, :, :, :, i],
            stencil2,
            DIM_TO_AXIS[dim2],
            out=out[:, :, :, :, layer2_len * i + j],
        )
        slices2.append(modified)
    slices.append(merge_slices(*slices2, union=True))

    return merge_slices(*slices)


def _fv_interpolate_3sweeps(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
):
    dim1, dim2, dim3 = list(nodes)
    coords1 = _to_iter(nodes[dim1])
    coords2 = _to_iter(nodes[dim2])
    coords3 = _to_iter(nodes[dim3])
    layer1_len = len(coords1)
    layer2_len = len(coords2)
    layer3_len = len(coords3)

    if out.shape[4] < layer1_len * layer2_len * layer3_len:
        raise ValueError(
            f"Output interpolation axis size {out.shape[4]} is smaller than the number"
            f" of coordinates {layer1_len}, {layer2_len}, and {layer3_len} "
            f"for dimensions '{dim1}', '{dim2}', and '{dim3}'."
        )
    if buffer.shape[4] < layer1_len + layer1_len * layer2_len:
        raise ValueError(
            f"Buffer size {buffer.shape[4]} is smaller than the product of "
            f"the number of coordinates {layer1_len} and {layer2_len} for "
            f"dimensions '{dim1}' and '{dim2}'."
        )

    # fill buffer layer 1
    slices = []
    slices1 = []
    for i in range(layer1_len):
        stencil1 = stencil_func(p, coords1[i])
        modified = inplace_stencil_sweep(
            xp, u, stencil1, DIM_TO_AXIS[dim1], out=buffer[:, :, :, :, i]
        )
        slices1.append(modified)
    slices.append(merge_slices(*slices1, union=True))

    # fill buffer layer 2
    slices2 = []
    for i, j in product(range(layer1_len), range(layer2_len)):
        stencil2 = stencil_func(p, coords2[j])
        flat_idx = layer1_len + i * layer2_len + j
        modified = inplace_stencil_sweep(
            xp,
            buffer[:, :, :, :, i],
            stencil2,
            DIM_TO_AXIS[dim2],
            out=buffer[:, :, :, :, flat_idx],
        )
        slices2.append(modified)
    slices.append(merge_slices(*slices2, union=True))

    # interpolate nodes from buffer
    slices3 = []
    for i, j, k in product(range(layer1_len), range(layer2_len), range(layer3_len)):
        stencil3 = stencil_func(p, coords3[k])
        in_idx = layer1_len + i * layer2_len + j
        out_idx = (i * layer2_len + j) * layer3_len + k
        modified = inplace_stencil_sweep(
            xp,
            buffer[:, :, :, :, in_idx],
            stencil3,
            DIM_TO_AXIS[dim3],
            out=out[:, :, :, :, out_idx],
        )
        slices3.append(modified)
    slices.append(merge_slices(*slices3))

    return merge_slices(*slices)


def _to_iter(x):
    return x if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) else [x]


def _fv_interpolate_recursive(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    nodes: Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Recursive implementation of interpolating a nodal value from an array of finite
    volume averages.

    Args:
        xp: `np` namespace.
        stencil_func: Function to compute the interpolation stencil weights.
            It should accept the polynomial degree and a coordinate value, and return
            an array of stencil weights.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        nodes: Dictionary with keys 'x', 'y', 'z' and values as the nodal coordinates.
            Each value may be a float or an array of floats representing the
            coordinates to interpolate along the respective dimension on [-1, 1].
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    coords = [_to_iter(nodes[dim]) for dim in nodes.keys()]
    modified = _fv_interpolate_recursive_helper(
        xp,
        stencil_func,
        u,
        list(nodes.keys()),
        coords,
        p,
        buffer,
        out,
        layers=tuple(len(c) for c in coords),
    )
    return modified


def _fv_interpolate_recursive_helper(
    xp: ModuleType,
    stencil_func: Callable[[int, InterpCoord], StencilWeights],
    u: ArrayLike,
    dims: List[str],
    coords: List[List[float]],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
    layers: Tuple[int, ...],
    path: Tuple[int, ...] = tuple(),
):
    depth = len(path)
    dim = dims[depth]

    if depth == 0:
        parent_idx = None
        parent = u
    else:
        parent_idx = _buffer_flat_index(layers, path)
        parent = buffer[:, :, :, :, parent_idx]

    FOUND_NODE = depth == len(dims) - 1
    child_slices = []
    for i, x in enumerate(coords[depth]):
        stencil = stencil_func(p, x)
        next_path = path + (i,)

        if FOUND_NODE:
            child_idx = _flattened_index(layers, next_path)
            child = out[:, :, :, :, child_idx]
        else:
            child_idx = _buffer_flat_index(layers, next_path)
            child = buffer[:, :, :, :, child_idx]

        modified = inplace_stencil_sweep(
            xp, parent, stencil, DIM_TO_AXIS[dim], out=child
        )

        if FOUND_NODE:
            child_slices.append(modified)
        else:
            downstream = _fv_interpolate_recursive_helper(
                xp, stencil_func, u, dims, coords, p, buffer, out, layers, next_path
            )
            effective = merge_slices(modified, downstream)
            child_slices.append(effective)

    return merge_slices(*child_slices, union=True)


@lru_cache(maxsize=None)
def _flattened_index(shape: Tuple[int, ...], idx: Tuple[int, ...]) -> int:
    """
    Compute the flattened index for a given multi-dimensional index.

    Args:
        shape: Shape of the multi-dimensional array.
        idx: Multi-dimensional index as a tuple of integers.

    Returns:
        int: The flattened index corresponding to the multi-dimensional index.
    """
    flat = 0
    stride = 1
    for i, w in reversed(list(zip(idx, shape))):
        if i < 0 or i >= w:
            raise IndexError(f"Index {i} out of bounds for shape {shape}.")
        flat += i * stride
        stride *= w
    return flat


@lru_cache(maxsize=None)
def _buffer_flat_index(shape: Tuple[int, ...], idx: Tuple[int, ...]) -> int:
    """
    Compute the flattened index for a given multi-dimensional index where the
    intermediate dimension pairings are also stored.

    Args:
        shape: Shape of the multi-dimensional array.
        idx: Multi-dimensional index as a tuple of integers.

    Returns:
        int: The flattened index corresponding to the multi-dimensional index.
    """
    base_idx = 0
    if len(idx) > 1:
        base_idx += np.sum(np.cumprod(shape[: len(idx) - 1])).item()
    return _flattened_index(shape, idx) + base_idx


def interpolate_cell_centers(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Interpolate cell-centered nodes from an array of finite volume averages.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for a
            two-dimensional interpolation.
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout). Result is stored in out[..., 0].

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    return fv_interpolate(
        xp,
        u,
        {d: 0 for d in active_dims},
        p,
        buffer,
        out,
    )


def integrate_fv_averages(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Integrate the finite volume averages from an array of cell-centered values.

    Args:
        xp: `np` namespace.
        u: Array of cell-centered nodal values to integrate, has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple indicating the active dimensions for integration. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            integration of a cell in a two-dimensional grid.
        p: Polynomial degree of the integration stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the integrated values. Has shape
            (nvars, nx, ny, nz, nout). Result is stored in out[..., 0].

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    return fv_integrate(
        xp,
        u,
        active_dims,
        p,
        buffer,
        out,
    )


def interpolate_GaussLegendre_nodes(
    xp: ModuleType,
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Interpolate opposing Gauss-Legendre nodes from an array of finite volume averages.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the Gauss-Legendre interpolation is performed.
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of nodes along the face of a cell on a two-dimensional grid.
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout). The "left" Gauss-Legendre node is stored in
            out[..., :n_gauss_legendre] and the "right" Gauss-Legendre node is stored
            in out[..., n_gauss_legendre:2*n_gauss_legendre].
    """
    nodes, _ = _get_GaussLegendre_nodes_and_weights(xp, face_dim, active_dims, p)
    return fv_interpolate(
        xp,
        u,
        nodes,
        p,
        buffer,
        out,
    )


def integrate_GaussLegendre_nodes(
    xp: ModuleType,
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    out: ArrayLike,
):
    """
    Integrate the finite volume averages at Gauss-Legendre nodes across the specified
    face dimension.

    Args:
        xp: `np` namespace.
        u: Array of face-centered nodal values to integrate, has shape
            (nvars, nx, ny, nz, ninterpolations) with the Gauss-Legendre nodes stored
            along u[..., :n_GaussLegendre_nodes].
        face_dim: Dimension along which the integration is performed.
        active_dims: Tuple indicating the active dimensions for integration. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for a
            two-dimensional integration.
        p: Polynomial degree of the integration stencil.
        out: Output array to store the integrated values. Has shape
            (nvars, nx, ny, nz).
    """
    if len(active_dims) < 2:
        raise ValueError(
            "At least two active dimensions are required for Gauss-Legendre node "
            "integration."
        )
    _, w2 = _get_GaussLegendre_nodes_and_weights(xp, face_dim, active_dims, p)
    n_GaussLegendre_nodes = w2.shape[-1]
    w = w2[:, :, :, :, : n_GaussLegendre_nodes // 2]
    out[...] = xp.sum(u[..., :n_GaussLegendre_nodes] * w, axis=-1)


@lru_cache(maxsize=None)
def _get_GaussLegendre_nodes_and_weights(
    xp: ModuleType,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
) -> Tuple[Dict[Literal["x", "y", "z"], Union[InterpCoord, InterpCoords]], np.ndarray]:
    """
    Get Gauss-Legendre nodes and weights for two opposing cells along a specified face
    dimension.

    Args:
        xp: `np` namespace.
        face_dim: Dimension along which the Gauss-Legendre nodes are defined.
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of nodes along the face of a cell on a two-dimensional grid.
        p: Polynomial degree of the interpolation stencil.

    Returns:
        nodes: Dictionary with keys 'x', 'y', 'z' and values as the Gauss-Legendre
            nodes for the specified face dimension. The nodes corresponding to the
            `dim` key are the left and right nodes of the face, represented as
            `[-1, 1]`.
        weights: Array of shape (1, 1, 1, 1, 2 * n_GaussLegendre_nodes) containing the
            Gauss-Legendre weights for the left and right nodes of the face, written to
            `weights[..., :n_GaussLegendre_nodes]` and
            `weights[..., n_GaussLegendre_nodes:]`, respectively. The weights are
            scaled to the interval [-0.5, 0.5].
    """
    if face_dim not in active_dims:
        raise ValueError(
            f"face_dim '{face_dim}' must be one of the active dimensions: {active_dims}"
        )
    nodes = {face_dim: cast(Union[InterpCoord, InterpCoords], [-1, 1])}
    weights = {face_dim: xp.array([1, 1])}
    for dim in active_dims:
        if dim == face_dim:
            continue
        x, w = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))
        nodes[dim] = x.tolist()
        weights[dim] = xp.asarray(w) / 2  # scale to [-0.5, 0.5] interval
    wmesh = xp.meshgrid(*weights.values(), indexing="ij")
    flattened_weights = xp.prod(xp.array(wmesh), axis=0).reshape(1, 1, 1, 1, -1)
    return nodes, flattened_weights


def interpolate_face_centers(
    xp: ModuleType,
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Interpolate opposing face-centered nodes from an array of finite volume averages.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to interpolate, has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the face-centered interpolation is performed.
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of the face center of a cell on a two-dimensional grid.
        p: Polynomial degree of the interpolation stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the interpolated values. Has shape
            (nvars, nx, ny, nz, nout). The "left" face-centered node is stored in
            out[..., 0] and the "right" face-centered node is stored in out[..., 1].

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    transverse_dims = [d for d in active_dims if d != face_dim]
    return fv_interpolate(
        xp,
        u,
        {face_dim: cast(Union[InterpCoord, InterpCoords], [-1, 1])}
        | {d: 0 for d in transverse_dims},
        p,
        buffer,
        out,
    )


def transversely_integrate_nodes(
    xp: ModuleType,
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    buffer: ArrayLike,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Integrate the finite volume averages transversely across the specified face
    dimension using their central nodes.

    Args:
        xp: `np` namespace.
        u: Array of face-centered nodal values to integrate, has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the integration is performed.
        active_dims: Tuple indicating the active dimensions for integration. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            integration of the face of a cell in a two-dimensional grid.
        p: Polynomial degree of the integration stencil.
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, nbuffer).
        out: Output array to store the integrated values. Has shape
            (nvars, nx, ny, nz, nout). The result is stored in out[..., 0].
    """
    if len(active_dims) < 2:
        raise ValueError(
            "At least two active dimensions are required for transverse integration."
        )
    return fv_integrate(
        xp,
        u,
        tuple(d for d in active_dims if d != face_dim),
        p,
        buffer,
        out,
    )
