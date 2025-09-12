import numpy as np
import pytest

import teyssier
from superfv.hydro import cons_to_prim, fluxes, prim_to_cons
from superfv.tools.norms import l1_norm, linf_norm
from superfv.tools.slicing import VariableIndexMap


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
    _euler_slicer.add_var_to_group("v", ("vx", "vy", "vz"))
    _euler_slicer.add_var_to_group("m", ("mx", "my", "mz"))
    _euler_slicer.add_var_to_group(
        "passives",
        ("passive_scalar1", "passive_scalar2", "passive_scalar3"),
    )
    return _euler_slicer


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
def test_primitive_to_conservative_invertability(trial, gamma, euler_slicer):
    """
    Test that the primitive_to_conservative and conservative_to_primitive functions
    are inverses of each other.
    """
    idx = euler_slicer
    N = 64

    W = np.empty((8, N, N, N))
    W[...] = np.random.rand(*W.shape)
    W[idx("rho")] += 1.0
    W[idx("P")] += 1.0

    # convert to conservative and back to primitive
    U = prim_to_cons(np, idx, W, active_dims="xyz", gamma=gamma)
    W2 = cons_to_prim(np, idx, U, active_dims="xyz", gamma=gamma)

    # check that the primitive values are the same
    assert l1_norm(W - W2) < 1e-15


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
def test_conservative_to_primitive_invertability(trial, gamma, euler_slicer):
    """
    Test that the conservative_to_primitive and primitive_to_conservative functions
    are inverses of each other.
    """
    idx = euler_slicer
    N = 64

    U = np.empty((8, N, N, N))
    U[...] = np.random.rand(*U.shape)
    U[idx("rho")] += 1.0
    U[idx("E")] += 1.0

    # convert to conservative and back to primitive
    W = cons_to_prim(np, idx, U, active_dims="xyz", gamma=gamma)
    U2 = prim_to_cons(np, idx, W, active_dims="xyz", gamma=gamma)

    # check that the primitive values are the same
    assert l1_norm(U - U2) < 1e-15


def test_teyssier_prim_to_cons(euler_slicer):
    idx = euler_slicer
    idx.add_var_to_group("test", ("rho", "vx", "P"))

    N = 64

    w = np.empty((5, N, N, N))
    w[idx("rho")] = 1e-6 * np.random.rand(N, N, N)
    w[idx("vx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    w[idx("P")] = 1e-6 * np.random.rand(N, N, N)

    u1 = prim_to_cons(np, idx, w, active_dims=("x",), gamma=1.4)
    u2 = teyssier.prim_to_cons(w[idx("test")])

    assert linf_norm(u1[idx("rho")] - u2[0]) == 0
    assert linf_norm(u1[idx("mx")] - u2[1]) == 0
    assert linf_norm(u1[idx("E")] - u2[2]) == 0


def test_teyssier_cons_to_prim(euler_slicer):
    idx = euler_slicer
    idx.add_var_to_group("test", ("rho", "mx", "E"))

    N = 64

    u = np.empty((5, N, N, N))
    u[idx("rho")] = 1e-6 * np.random.rand(N, N, N) + 1.0
    u[idx("mx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    u[idx("E")] = 1e-6 * np.random.rand(N, N, N) + 1.0

    w1 = cons_to_prim(np, idx, u, active_dims=("x",), gamma=1.4)
    w2 = teyssier.cons_to_prim(u[idx("test")])

    assert linf_norm(w1[idx("rho")] - w2[0]) == 0
    assert linf_norm(w1[idx("vx")] - w2[1]) == 0
    assert linf_norm(w1[idx("P")] - w2[2]) == 0


def test_teyssier_compute_fluxes(euler_slicer):
    idx = euler_slicer
    idx.add_var_to_group("test", ("rho", "vx", "P"))

    N = 64

    w = np.empty((5, N, N, N))
    w[idx("rho")] = 1e-6 * np.random.rand(N, N, N) + 1.0
    w[idx("vx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    w[idx("P")] = 1e-6 * np.random.rand(N, N, N) + 1.0

    f1 = teyssier.prim_to_flux(w[idx("test")])
    f2 = fluxes(
        np,
        idx,
        w,
        "x",
        active_dims=("x",),
        gamma=1.4,
    )

    assert linf_norm(f1[0] - f2[idx("rho")]) < 1e-15
    assert linf_norm(f1[1] - f2[idx("mx")]) < 1e-15
    assert linf_norm(f1[2] - f2[idx("E")]) < 1e-15
