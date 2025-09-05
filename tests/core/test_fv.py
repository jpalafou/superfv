from typing import Dict

import numpy as np
import pytest

from superfv.fv import _fv_interpolate_direct, gather_multistencils
from superfv.stencil import inplace_multistencil_sweep, inplace_stencil_sweep
from superfv.tools.norms import linf_norm


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
    modified = inplace_stencil_sweep(np, u, trivial_stencil, axis, out=out)

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
    slc = inplace_stencil_sweep(np, u, ones_stencil, axis, out=out)

    # assert that the output is the average of the input over the stencil size
    assert np.all(out[slc] == 1)
    assert np.array_equal(out[slc], u[slc])

    # assert that the output is unchanged outside the modified region
    out[slc] = out_original[slc]
    assert np.array_equal(out, out_original)


def test_inplace_multistencil_sweep():
    """
    Test that inplace_multistencil_sweep produces the expected output.
    """
    N = 64
    stencil1 = np.array([0.0, 1.0, 0.0])
    stencil2 = np.array([0.1, 0.8, 0.1])
    multistencil = np.array([stencil1, stencil2])

    # allocate arrays
    arr = np.random.rand(5, N, N, N)
    out1 = np.empty((5, N, N, N, 2))
    out2 = np.empty((5, N, N, N, 2))

    # perform serial stencil sweeps
    modified1 = inplace_stencil_sweep(np, arr, stencil1, axis=1, out=out1[..., 0])
    _ = inplace_stencil_sweep(np, arr, stencil2, axis=1, out=out1[..., 1])

    # perform multistencil sweep
    modified2 = inplace_multistencil_sweep(np, arr, multistencil, axis=1, out=out2)

    # assert that the modified regions are the same
    l2 = np.sqrt(np.mean(np.square(out1[modified1] - out2[modified2])))
    assert np.array_equal(out1[modified1], out2[modified2]), f"L2 norm: {l2}"


@pytest.mark.parametrize("x_coord", [{"x": 0}, {"x": [-1, 1]}])
@pytest.mark.parametrize("y_coord", [{}, {"y": 0}, {"y": np.nan}])
@pytest.mark.parametrize("z_coord", [{}, {"z": 0}, {"z": [-1.0, -0.5, 0.0, 0.5, 1.0]}])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
def test__fv_interpolate_direct_interpolate_node_from_uniform_field(
    x_coord: Dict[str, float],
    y_coord: Dict[str, float],
    z_coord: Dict[str, float],
    p: int,
):
    """
    Test that _fv_interpolate_direct can interpolate a node from a uniform field.
    """
    # Merge coordinate specs
    y_coord_copy = y_coord.copy()
    if np.isnan(y_coord_copy.get("y", 1)):
        y_coord_copy["y"] = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))[
            0
        ].tolist()
    print(y_coord_copy)
    nodes = x_coord | y_coord_copy | z_coord
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
    out = np.empty((5, Nx, Ny, Nz, n_nodes))
    out.fill(np.nan)
    out_original = out.copy()
    buffer = np.empty((5, Nx, Ny, Nz, n_buffer))
    buffer.fill(np.nan)

    # Perform interpolation
    modified = _fv_interpolate_direct(
        np,
        lambda p, x: gather_multistencils(np, "conservative-interpolation", p, x),
        u,
        nodes,
        p,
        out=out,
        buffer=buffer,
    )

    # Assert output equals the uniform input on the modified region
    for i in range(n_nodes):
        node = out[..., i]
        assert linf_norm(node[modified[:-1]] - u[modified[:-1]]) < 1e-15

    # assert that the output is unchanged outside the modified region
    out[modified] = out_original[modified]
    np.array_equal(out, out_original)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("nodes", [{"x": [0]}, {"x": [-1, 1]}])
def test__fv_interpolate_direct_1sweep_modified_slice(p: int, nodes: Dict[str, float]):
    """
    Test that _fv_interpolate_direct returns the correct modified slice when performing
    a single-sweep interpolation,
    """
    N = 16

    # allocate arrays
    u = np.zeros((5, N, 1, 1))
    out = np.empty((5, N, 1, 1, 10))
    out.fill(np.nan)

    # perform interpolation
    modified = _fv_interpolate_direct(
        np,
        lambda p, x: gather_multistencils(np, "conservative-interpolation", p, x),
        u,
        nodes,
        p,
        out=out,
        buffer=np.empty((0,)),
    )

    # assert that the modified region is correctly set
    assert np.all(out[modified] == 0)
    out[modified].fill(np.nan)
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "nodes",
    [{"x": [0], "y": [-1, 1]}, {"x": [-1, 1], "y": [0]}, {"x": [-1, 1], "y": [-1, 1]}],
)
def test__fv_interpolate_direct_2sweep_modified_slice(p: int, nodes: Dict[str, float]):
    """
    Test that _fv_interpolate_direct returns the correct modified slice when performing
    a double-sweep interpolation.
    """
    N = 16

    # allocate arrays
    u = np.zeros((5, N, N, 1))
    out = np.empty((5, N, N, 1, 10))
    out.fill(np.nan)
    buffer = np.empty((5, N, N, 1, 10))
    buffer.fill(np.nan)

    # perform interpolation
    modified = _fv_interpolate_direct(
        np,
        lambda p, x: gather_multistencils(np, "conservative-interpolation", p, x),
        u,
        nodes,
        p,
        out=out,
        buffer=buffer,
    )

    # assert that the modified region is correctly set
    assert np.all(out[modified] == 0)
    out[modified].fill(np.nan)
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "nodes",
    [
        {"x": [0], "y": [-1, 1], "z": [-0.5, 0.0, 0.5]},
        {"x": [-0.5, 0.0, 0.5], "y": [-1, 1], "z": [0]},
        {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]},
    ],
)
def test__fv_interpolate_direct_3sweep_modified_slice(p: int, nodes: Dict[str, float]):
    """
    Test that _fv_interpolate_direct returns the correct modified slice when performing
    a triple-sweep interpolation.
    """
    N = 16

    # allocate arrays
    u = np.zeros((5, N, N, N))
    out = np.empty((5, N, N, N, 10))
    out.fill(np.nan)
    buffer = np.empty((5, N, N, N, 10))
    buffer.fill(np.nan)

    # perform interpolation
    modified = _fv_interpolate_direct(
        np,
        lambda p, x: gather_multistencils(np, "conservative-interpolation", p, x),
        u,
        nodes,
        p,
        out=out,
        buffer=buffer,
    )

    # assert that the modified region is correctly set
    assert np.all(out[modified] == 0)
    # out[modified].fill(np.nan)
    # assert np.all(np.isnan(out))
