import numpy as np
import pytest
import wtflux.hydro as hydro

from superfv.hydro import conservatives_from_primitives, primitives_from_conservatives
from superfv.tools.array_management import VariableIndexMap


def l1_norm(a, b):
    return np.mean(np.abs((a - b)))


@pytest.fixture
def euler_slicer():
    _euler_slicer = VariableIndexMap(
        {
            "rho": 0,
            "vx": 1,
            "vy": 2,
            "vz": 3,
            "P": 4,
            "mx": 1,
            "my": 2,
            "mz": 3,
            "E": 4,
            "passive_scalar1": 5,
            "passive_scalar2": 6,
            "passive_scalar3": 7,
        },
    )
    _euler_slicer.add_var_to_group(("vx", "vy", "vz"), "v")
    _euler_slicer.add_var_to_group(("mx", "my", "mz"), "m")
    _euler_slicer.add_var_to_group(
        ("passive_scalar1", "passive_scalar2", "passive_scalar3"),
        "user_defined_passives",
    )
    return _euler_slicer


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
def test_primitive_to_conservative_invertability(trial, gamma, euler_slicer):
    """
    Test that the primitive_to_conservative and conservative_to_primitive functions
    are inverses of each other.
    """
    _slc = euler_slicer
    N = 64

    W = np.empty((8, N, N, N))
    W[...] = np.random.rand(*W.shape)
    W[_slc("rho")] += 1.0
    W[_slc("P")] += 1.0

    # convert to conservative and back to primitive
    U = conservatives_from_primitives(hydro, euler_slicer, W, gamma=1.4)
    W2 = primitives_from_conservatives(hydro, euler_slicer, U, gamma=1.4)

    # check that the primitive values are the same
    assert l1_norm(W, W2) < 1e-15


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
def test_conservative_to_primitive_invertability(trial, gamma, euler_slicer):
    """
    Test that the conservative_to_primitive and primitive_to_conservative functions
    are inverses of each other.
    """
    _slc = euler_slicer
    N = 64

    U = np.empty((8, N, N, N))
    U[...] = np.random.rand(*U.shape)
    U[_slc("rho")] += 1.0
    U[_slc("E")] += 1.0

    # convert to conservative and back to primitive
    W = primitives_from_conservatives(hydro, euler_slicer, U, gamma=1.4)
    U2 = conservatives_from_primitives(hydro, euler_slicer, W, gamma=1.4)

    # check that the primitive values are the same
    assert l1_norm(U, U2) < 1e-15
