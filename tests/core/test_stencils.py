import pytest

from superfv.stencils import transverse_integration
from superfv.stencils.conservative_interpolation import (
    cell_center,
    gauss_legendre_nodes,
    left_right,
)
from superfv.sweep import stencil_sweep
from superfv.tools.device_management import xp
from superfv.tools.norms import linf_norm


@pytest.mark.parametrize("interp_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "yz", "xz", "xyz"])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize(
    "stencil", ["ci:center", "ci:left_right", "ci:gauss_legendre", "transverse"]
)
@pytest.mark.parametrize("ninterps", [1, 2])
def test_trivial_interpolation(interp_dim, active_dims, p, stencil, ninterps):
    if interp_dim not in active_dims:
        pytest.skip("Interpolation dimension must be among active dimensions")

    N = 64
    shape = (
        1,
        N if "x" in active_dims else 1,
        N if "y" in active_dims else 1,
        N if "z" in active_dims else 1,
    )

    match stencil:
        case "ci:center":
            weights = cell_center(p)
        case "ci:left_right":
            weights = left_right(p)
        case "ci:gauss_legendre":
            weights = gauss_legendre_nodes(p)
        case "transverse":
            weights = transverse_integration(p)
        case _:
            raise ValueError(f"Unknown stencil: {stencil}")
    nouterps, _ = weights.shape
    weights = xp.asarray(weights)

    u = xp.ones(shape + (ninterps,))
    uj = xp.empty(shape + (ninterps * nouterps,))

    modified = stencil_sweep(u, weights, uj, interp_dim)

    assert linf_norm(uj[modified] - 1.0) < 1e-15
