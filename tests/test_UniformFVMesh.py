from typing import Literal, Tuple

import pytest

from superfv.mesh import UniformFiniteVolumeMesh


@pytest.mark.parametrize(
    "active_dims", [("x",), ("y",), ("z",), ("x", "y"), ("x", "z"), ("y", "z"), ("x", "y", "z")]
)
@pytest.mark.parametrize("nghost", [0, 1])
def test_mesh_initialization(active_dims: Tuple[Literal["x", "y", "z"], ...], nghost: int):
    """
    Test the initialization of UniformFiniteVolumeMesh with various dimensions and ghost cell counts.
    """
    N = 64

    UniformFiniteVolumeMesh(
        xlims=(0, 1),
        ylims=(0, 1),
        zlims=(0, 1),
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        nghost=nghost,
        active_dims=active_dims,
    )
