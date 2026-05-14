import pytest

from superfv.mesh import UniformFVMesh


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("nghost", [0, 1])
def test_mesh_initialization(dims: str, nghost: int):
    """
    Test the initialization of UniformFVMesh with various dimensions and ghost cell counts.
    """
    N = 64

    UniformFVMesh(
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        nghost=nghost,
        xlims=(0, 1),
        ylims=(0, 1),
        zlims=(0, 1),
        active_dims=tuple(dims),
    )
