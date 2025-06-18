import numpy as np
import pytest

from superfv.boundary_conditions import BoundaryConditions
from superfv.mesh import UniformFVMesh
from superfv.tools.array_management import VariableIndexMap


@pytest.mark.parametrize("bc_type", ["periodic", "free", "symmetric", "zeros", "ones"])
def test_boundary_conditions_numpy_pad_equivalence(bc_type):
    bc = BoundaryConditions(VariableIndexMap(), bcx=bc_type, bcy=bc_type, bcz=bc_type)

    # apply with BoundaryConditions object
    u = np.random.rand(5, 32, 32, 32)
    u_padded = bc(u, (16, 16, 16))

    # manually pad with numpy
    pad_kwargs = {
        "periodic": {"mode": "wrap"},
        "free": {"mode": "edge"},
        "symmetric": {"mode": "symmetric"},
        "zeros": {"mode": "constant", "constant_values": 0},
        "ones": {"mode": "constant", "constant_values": 1},
    }
    np_padded = np.pad(u, [(0,), (16,), (16,), (16,)], **(pad_kwargs[bc_type]))

    assert np.array_equal(u_padded, np_padded)


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_dirichlet_boundary_condition(dims):

    def sinus(idx, x, y, z, t=None):
        u = np.zeros_like(x)
        if "x" in dims:
            u += x
        if "y" in dims:
            u += y
        if "z" in dims:
            u += z
        return np.expand_dims(np.sin(2 * np.pi * u), axis=0)

    def trivial(f):
        return f

    mesh = UniformFVMesh(
        nx=32 if "x" in dims else 1,
        ny=32 if "y" in dims else 1,
        nz=32 if "z" in dims else 1,
        ignore_x="x" not in dims,
        ignore_y="y" not in dims,
        ignore_z="z" not in dims,
        slab_depth=16,
    )
    megamesh = UniformFVMesh(
        nx=64 if "x" in dims else 1,
        ny=64 if "y" in dims else 1,
        nz=64 if "z" in dims else 1,
        xlim=(-0.5, 1.5) if "x" in dims else (0, 1),
        ylim=(-0.5, 1.5) if "y" in dims else (0, 1),
        zlim=(-0.5, 1.5) if "z" in dims else (0, 1),
        slab_depth=0,
    )
    bc = BoundaryConditions(
        VariableIndexMap(),
        mesh,
        bcx="dirichlet" if "x" in dims else "periodic",
        bcy="dirichlet" if "y" in dims else "periodic",
        bcz="dirichlet" if "z" in dims else "periodic",
        x_dirichlet=sinus if "x" in dims else None,
        y_dirichlet=sinus if "y" in dims else None,
        z_dirichlet=sinus if "z" in dims else None,
        conservatives_wrapper=trivial,
        fv_average_wrapper=trivial,
    )

    # compare dirichlet boundary condition with (-0.5, 1.5) mesh
    arr = sinus(None, mesh.X, mesh.Y, mesh.Z, None)
    padded_arr = bc(
        arr,
        (16 if "x" in dims else 0, 16 if "y" in dims else 0, 16 if "z" in dims else 0),
    )
    megamesh_arr = sinus(None, megamesh.X, megamesh.Y, megamesh.Z, None)
    assert np.array_equal(padded_arr, megamesh_arr)
