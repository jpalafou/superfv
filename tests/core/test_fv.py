from typing import Callable, Dict

import numpy as np
import pytest

from superfv.fv import (
    _fv_interpolate_direct,
    _fv_interpolate_recursive,
    conservative_interpolation_weights,
    inplace_stencil_sweep,
)
from superfv.tools.array_management import linf_norm


@pytest.mark.parametrize("stencil_size", [1, 3, 5])
@pytest.mark.parametrize("axis", [1, 2, 3])
def test_trivial_stencil(stencil_size: int, axis: int):
    """
    Test that a trivial stencil (identity) returns the input array unchanged.
    """
    N = 64
    u = np.random.rand(5, N, N, N)
    out = np.zeros_like(u)
    out_original = out.copy()

    trivial_stencil = np.zeros(stencil_size)
    trivial_stencil[stencil_size // 2] = 1.0
    modified = inplace_stencil_sweep(np, u, trivial_stencil, axis, out)

    # assert that the output is identical to the input
    assert np.array_equal(out[modified], u[modified])

    # assert that the output is unchanged outside the modified region
    out[modified] = out_original[modified]
    assert np.array_equal(out, out_original)


@pytest.mark.parametrize("stencil_size", [1, 2, 5])
@pytest.mark.parametrize("axis", [1, 2, 3])
def test_ones_stencil(stencil_size: int, axis: int):
    """
    Test that a stencil of ones returns the average of the input array over the stencil
    size.
    """
    N = 64
    u = np.ones((5, N, N, N))
    out = np.zeros_like(u)
    out_original = out.copy()

    ones_stencil = np.ones(stencil_size) / stencil_size
    slc = inplace_stencil_sweep(np, u, ones_stencil, axis, out)

    # assert that the output is the average of the input over the stencil size
    assert np.all(out[slc] == 1)
    assert np.array_equal(out[slc], u[slc])

    # assert that the output is unchanged outside the modified region
    out[slc] = out_original[slc]
    assert np.array_equal(out, out_original)


@pytest.mark.parametrize("x_coord", [{"x": 0}, {"x": [-0.5, 0.5]}])
@pytest.mark.parametrize("y_coord", [{}, {"y": 0}, {"y": [-np.nan, np.nan]}])
@pytest.mark.parametrize("z_coord", [{}, {"z": 0}, {"z": [-0.5, 0, 0.5]}])
@pytest.mark.parametrize(
    "fv_interpolate", [_fv_interpolate_direct, _fv_interpolate_recursive]
)
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
def test_interpolate_node_from_uniform_field(
    x_coord: Dict[str, float],
    y_coord: Dict[str, float],
    z_coord: Dict[str, float],
    fv_interpolate: Callable,
    p: int,
):
    """
    Test interpolation of a uniform field at specified coordinates.
    """
    # Merge coordinate specs
    y_coord_GL = y_coord.copy()
    if isinstance(y_coord_GL.get("y", None), list) and len(y_coord_GL["y"]) == 2:
        y_coord_GL["y"][0] = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))[0][0]
        y_coord_GL["y"][1] = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))[0][-1]
    nodes = x_coord | y_coord | z_coord
    dims = list(nodes)

    # Count output nodes and intermediate buffer size
    shape_per_dim = [len(nodes[d]) if isinstance(nodes[d], list) else 1 for d in dims]
    n_nodes = np.prod(shape_per_dim).item()
    n_buffer = np.sum(np.cumprod(shape_per_dim)).item()

    # Define a uniform field (u ≡ 1)
    N = 32
    Nx = N if "x" in nodes else 1
    Ny = N if "y" in nodes else 1
    Nz = N if "z" in nodes else 1
    u = np.ones((5, Nx, Ny, Nz))

    # Allocate output and buffer
    out = np.zeros((5, Nx, Ny, Nz, n_nodes))
    out_original = out.copy()
    buffer = np.zeros((5, Nx, Ny, Nz, n_buffer)) * np.nan

    # Perform interpolation
    modified = fv_interpolate(
        np, conservative_interpolation_weights, u, nodes, p=p, buffer=buffer, out=out
    )

    # Assert output equals the uniform input on the modified region
    for i in range(n_nodes):
        node = out[..., i]
        assert linf_norm(node[modified] - u[modified]) < 1e-15

    # assert that the output is unchanged outside the modified region
    out[modified] = out_original[modified]
    np.array_equal(out, out_original)
