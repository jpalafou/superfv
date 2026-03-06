import numpy as np
import pytest

import teyssier
from superfv.hydro import cons_to_prim, fluxes, prim_to_cons
from superfv.tools.device_management import CUPY_AVAILABLE, xp
from superfv.tools.norms import l1_norm, linf_norm
from superfv.tools.slicing import VariableIndexMap

if CUPY_AVAILABLE:
    from superfv.hydro import (
        make_cons_to_prim_elementwise_kernel,
        make_prim_to_cons_elementwise_kernel,
    )


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
@pytest.mark.parametrize("cupy", [False, True])
def test_primitive_to_conservative_invertability(trial, gamma, cupy, euler_slicer):
    """
    Test that the primitive_to_conservative and conservative_to_primitive functions
    are inverses of each other.
    """
    if cupy and not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available")

    idx = euler_slicer
    N = 64

    W = xp.empty((8, N, N, N)) if cupy else np.empty((8, N, N, N))
    W[...] = xp.random.rand(*W.shape) if cupy else np.random.rand(*W.shape)
    W[idx("rho")] += 1.0
    W[idx("P")] += 1.0

    # convert to conservative and back to primitive
    if cupy:
        prim_to_cons_elementwise_kernel = make_prim_to_cons_elementwise_kernel(3)
        cons_to_prim_elementwise_kernel = make_cons_to_prim_elementwise_kernel(3)

        U = xp.empty_like(W)
        prim_to_cons_elementwise_kernel(
            W[idx("rho")],
            W[idx("vx")],
            W[idx("vy")],
            W[idx("vz")],
            W[idx("P")],
            gamma,
            *(W[idx(v)] for v in idx.group_var_map.get("passives", [])),
            U[idx("rho")],
            U[idx("mx")],
            U[idx("my")],
            U[idx("mz")],
            U[idx("E")],
            *(U[idx(v)] for v in idx.group_var_map.get("passives", [])),
        )

        W2 = xp.empty_like(W)
        cons_to_prim_elementwise_kernel(
            U[idx("rho")],
            U[idx("mx")],
            U[idx("my")],
            U[idx("mz")],
            U[idx("E")],
            gamma,
            False,
            0.0,
            *(U[idx(v)] for v in idx.group_var_map.get("passives", [])),
            W2[idx("rho")],
            W2[idx("vx")],
            W2[idx("vy")],
            W2[idx("vz")],
            W2[idx("P")],
            *(W2[idx(v)] for v in idx.group_var_map.get("passives", [])),
        )
    else:
        U = prim_to_cons(idx, W, gamma=gamma)
        W2 = cons_to_prim(idx, U, gamma=gamma, isothermal=False)

    # check that the primitive values are the same
    assert l1_norm(W - W2) < 1e-15


@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
@pytest.mark.parametrize("cupy", [False, True])
def test_conservative_to_primitive_invertability(trial, gamma, cupy, euler_slicer):
    """
    Test that the conservative_to_primitive and primitive_to_conservative functions
    are inverses of each other.
    """
    if cupy and not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available")

    idx = euler_slicer
    N = 64

    U = xp.empty((8, N, N, N)) if cupy else np.empty((8, N, N, N))
    U[...] = xp.random.rand(*U.shape) if cupy else np.random.rand(*U.shape)
    U[idx("rho")] += 1.0
    U[idx("E")] += 1.0

    # convert to conservative and back to primitive
    if cupy:
        cons_to_prim_elementwise_kernel = make_cons_to_prim_elementwise_kernel(3)
        prim_to_cons_elementwise_kernel = make_prim_to_cons_elementwise_kernel(3)

        W = xp.empty_like(U)
        cons_to_prim_elementwise_kernel(
            U[idx("rho")],
            U[idx("mx")],
            U[idx("my")],
            U[idx("mz")],
            U[idx("E")],
            gamma,
            False,
            0.0,
            *(U[idx(v)] for v in idx.group_var_map.get("passives", [])),
            W[idx("rho")],
            W[idx("vx")],
            W[idx("vy")],
            W[idx("vz")],
            W[idx("P")],
            *(W[idx(v)] for v in idx.group_var_map.get("passives", [])),
        )

        U2 = xp.empty_like(U)
        prim_to_cons_elementwise_kernel(
            W[idx("rho")],
            W[idx("vx")],
            W[idx("vy")],
            W[idx("vz")],
            W[idx("P")],
            gamma,
            *(W[idx(v)] for v in idx.group_var_map.get("passives", [])),
            U2[idx("rho")],
            U2[idx("mx")],
            U2[idx("my")],
            U2[idx("mz")],
            U2[idx("E")],
            *(U2[idx(v)] for v in idx.group_var_map.get("passives", [])),
        )
    else:
        W = cons_to_prim(idx, U, gamma=gamma)
        U2 = prim_to_cons(idx, W, gamma=gamma)

    # check that the primitive values are the same
    assert l1_norm(U - U2) < 1e-15


def test_teyssier_prim_to_cons(euler_slicer):
    idx = euler_slicer
    idx.add_var_to_group("test", ("rho", "vx", "P"))

    N = 64

    w = np.zeros((5, N, N, N))
    w[idx("rho")] = 1e-6 * np.random.rand(N, N, N)
    w[idx("vx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    w[idx("P")] = 1e-6 * np.random.rand(N, N, N)

    u1 = prim_to_cons(idx, w, gamma=1.4)
    u2 = teyssier.prim_to_cons(w[idx("test")])

    assert linf_norm(u1[idx("rho")] - u2[0]) == 0
    assert linf_norm(u1[idx("mx")] - u2[1]) == 0
    assert linf_norm(u1[idx("E")] - u2[2]) == 0


def test_teyssier_cons_to_prim(euler_slicer):
    idx = euler_slicer
    idx.add_var_to_group("test", ("rho", "mx", "E"))

    N = 64

    u = np.zeros((5, N, N, N))
    u[idx("rho")] = 1e-6 * np.random.rand(N, N, N) + 1.0
    u[idx("mx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    u[idx("E")] = 1e-6 * np.random.rand(N, N, N) + 1.0

    w1 = cons_to_prim(idx, u, gamma=1.4)
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
    f2 = fluxes(idx, w, "x", gamma=1.4)

    assert linf_norm(f1[0] - f2[idx("rho")]) < 1e-15
    assert linf_norm(f1[1] - f2[idx("mx")]) < 1e-15
    assert linf_norm(f1[2] - f2[idx("E")]) < 1e-15
