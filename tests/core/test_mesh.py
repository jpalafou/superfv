import pytest

from superfv.mesh import UniformFVMesh


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("slab_depth", [0, 1])
def test_mesh_initialization(dims: str, slab_depth: int):
    """
    Test the initialization of UniformFVMesh with various dimensions and slab depths.
    """
    N = 64

    UniformFVMesh(
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        xlim=(0, 1),
        ylim=(0, 1),
        zlim=(0, 1),
        slab_depth=slab_depth,
    )
