import numpy as np
import pytest

from superfv.initial_conditions import sinus, slotted_disk, square
from superfv.mesh import UniformFVMesh
from superfv.tools.norms import linf_norm
from superfv.tools.slicing import VariableIndexMap


def get_core_mesh(dims):
    """Helper function to create a mesh for testing."""
    return UniformFVMesh(
        64 if "x" in dims else 1, 64 if "y" in dims else 1, 64 if "z" in dims else 1
    ).get_cell_centers()


def get_linear_velocity(dims):
    """Helper function to create a linear velocity field."""
    vx = 1 if "x" in dims else 0
    vy = 1 if "y" in dims else 0
    vz = 1 if "z" in dims else 0
    return {"vx": vx, "vy": vy, "vz": vz}


@pytest.fixture
def idx():
    idx = VariableIndexMap({"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4})
    return idx


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_sinus_periodicity(dims, idx):
    """
    Test that the sinus initial condition is periodic in all dimensions.
    """
    u0 = sinus(idx, *get_core_mesh(dims), 0.0, **get_linear_velocity(dims), xp=np)
    u1 = sinus(idx, *get_core_mesh(dims), 1.0, **get_linear_velocity(dims), xp=np)
    assert linf_norm(u0 - u1) < 1e-15


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_square_periodicity(dims, idx):
    """
    Test that the square initial condition is periodic in all dimensions.
    """
    u0 = square(idx, *get_core_mesh(dims), 0.0, xp=np, **get_linear_velocity(dims))
    u1 = square(idx, *get_core_mesh(dims), 1.0, xp=np, **get_linear_velocity(dims))
    assert linf_norm(u0 - u1) < 1e-15


def test_disk_periodicity(idx):
    """
    Test that the slotted disk initial condition is periodic in xy.
    """
    u0 = slotted_disk(idx, *get_core_mesh("xy"), xp=np)
    u1 = slotted_disk(idx, *get_core_mesh("xy"), xp=np)
    assert linf_norm(u0 - u1) < 1e-15
