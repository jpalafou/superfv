import numpy as np
import pytest

import teyssier
from superfv.hydro import cons_to_prim, prim_to_cons, prim_to_flux
from superfv.riemann_solvers import (
    HLLC_RiemannSolver,
    LLF_RiemannSolver,
    UpwindRiemannSolver,
)
from superfv.tools.device_management import xp
from superfv.tools.norms import l1_norm, linf_norm
from superfv.tools.variable_index_map import VariableIndexMap


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
        {
            "v": ["vx", "vy", "vz"],
            "m": ["mx", "my", "mz"],
            "passives": ["passive_scalar1", "passive_scalar2", "passive_scalar3"],
        },
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

    W = xp.empty((8, N, N, N))
    U = xp.empty((8, N, N, N))
    W2 = xp.empty((8, N, N, N))

    W[...] = xp.random.rand(*W.shape)
    W[idx("rho")] += 1.0
    W[idx("P")] += 1.0

    # convert to conservative and back to primitive
    prim_to_cons(W, U, idx, gamma)
    cons_to_prim(U, W2, idx, gamma)

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

    U = xp.empty((8, N, N, N))
    W = xp.empty((8, N, N, N))
    U2 = xp.empty((8, N, N, N))

    U[...] = xp.random.rand(*U.shape)
    U[idx("rho")] += 1.0
    U[idx("E")] += 1.0

    # convert to conservative and back to primitive
    cons_to_prim(U, W, idx, gamma)
    prim_to_cons(W, U2, idx, gamma)

    # check that the primitive values are the same
    assert l1_norm(U - U2) < 1e-15


def test_teyssier_prim_to_cons(euler_slicer):
    idx = euler_slicer

    idx.add_member_to_group("rho", "test")
    idx.add_member_to_group("vx", "test")
    idx.add_member_to_group("P", "test")

    N = 64

    w = np.zeros((5, N, N, N))
    u1 = np.zeros((5, N, N, N))
    u2 = np.zeros((3, N, N, N))

    w[idx("rho")] = 1e-6 * np.random.rand(N, N, N)
    w[idx("vx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    w[idx("P")] = 1e-6 * np.random.rand(N, N, N)

    prim_to_cons(w, u1, idx, gamma=1.4)
    u2[...] = teyssier.prim_to_cons(w[idx("test")])

    assert linf_norm(u1[idx("rho")] - u2[0]) == 0
    assert linf_norm(u1[idx("mx")] - u2[1]) == 0
    assert linf_norm(u1[idx("E")] - u2[2]) == 0


def test_teyssier_cons_to_prim(euler_slicer):
    idx = euler_slicer

    idx.add_member_to_group("rho", "test")
    idx.add_member_to_group("mx", "test")
    idx.add_member_to_group("E", "test")

    N = 64

    u = np.zeros((5, N, N, N))
    w1 = np.zeros((5, N, N, N))
    w2 = np.zeros((3, N, N, N))

    u[idx("rho")] = 1e-6 * np.random.rand(N, N, N) + 1.0
    u[idx("mx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    u[idx("E")] = 1e-6 * np.random.rand(N, N, N) + 1.0

    cons_to_prim(u, w1, idx, gamma=1.4)
    w2[...] = teyssier.cons_to_prim(u[idx("test")])

    assert linf_norm(w1[idx("rho")] - w2[0]) == 0
    assert linf_norm(w1[idx("vx")] - w2[1]) == 0
    assert linf_norm(w1[idx("P")] - w2[2]) == 0


def test_teyssier_compute_fluxes(euler_slicer):
    idx = euler_slicer

    idx.add_member_to_group("rho", "test")
    idx.add_member_to_group("vx", "test")
    idx.add_member_to_group("P", "test")

    N = 64

    w = np.zeros((5, N, N, N))
    f1 = np.zeros((5, N, N, N))
    f2 = np.zeros((3, N, N, N))

    w[idx("rho")] = 1e-6 * np.random.rand(N, N, N) + 1.0
    w[idx("vx")] = 2.0 * np.random.rand(N, N, N) - 1.0
    w[idx("P")] = 1e-6 * np.random.rand(N, N, N) + 1.0

    prim_to_flux(w, f1, idx, "x", gamma=1.4)
    f2[...] = teyssier.prim_to_flux(w[idx("test")])

    assert linf_norm(f1[idx("rho")] - f2[0]) < 1e-15
    assert linf_norm(f1[idx("mx")] - f2[1]) < 1e-15
    assert linf_norm(f1[idx("E")] - f2[2]) < 1e-15


@pytest.mark.parametrize(
    "solver_cls",
    [LLF_RiemannSolver, HLLC_RiemannSolver],
)
@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("isothermal", [False, True])
def test_passive_scalars_are_density_weighted_and_advected(
    euler_slicer, solver_cls, dim, isothermal
):
    idx = euler_slicer

    w = np.zeros((8, 1, 1, 1))
    u = np.zeros((8, 1, 1, 1))
    w2 = np.zeros((8, 1, 1, 1))

    w[idx("rho")] = 2.0
    w[idx("vx")] = 1.5
    w[idx("vy")] = -0.25
    w[idx("vz")] = 0.125
    w[idx("P")] = 3.0
    w[idx("passive_scalar1")] = 0.4
    w[idx("passive_scalar2")] = 0.6
    w[idx("passive_scalar3")] = 0.8

    prim_to_cons(w, u, idx, gamma=1.4)
    assert linf_norm(u[idx("passive_scalar1")] - w[idx("rho")] * w[idx("passive_scalar1")]) == 0
    assert linf_norm(u[idx("passive_scalar2")] - w[idx("rho")] * w[idx("passive_scalar2")]) == 0
    assert linf_norm(u[idx("passive_scalar3")] - w[idx("rho")] * w[idx("passive_scalar3")]) == 0

    cons_to_prim(u, w2, idx, gamma=1.4, isothermal=isothermal)
    assert linf_norm(w2[idx("passive_scalar1")] - w[idx("passive_scalar1")]) == 0
    assert linf_norm(w2[idx("passive_scalar2")] - w[idx("passive_scalar2")]) == 0
    assert linf_norm(w2[idx("passive_scalar3")] - w[idx("passive_scalar3")]) == 0

    flux = np.empty_like(w)
    expected_flux = np.empty_like(w)

    solver_cls(npassives=3)(w, w, flux, dim, idx, gamma=1.4, isothermal=isothermal)
    prim_to_flux(w, expected_flux, idx, dim, gamma=1.4)

    assert linf_norm(flux[idx("passive_scalar1")] - expected_flux[idx("passive_scalar1")]) == 0
    assert linf_norm(flux[idx("passive_scalar2")] - expected_flux[idx("passive_scalar2")]) == 0
    assert linf_norm(flux[idx("passive_scalar3")] - expected_flux[idx("passive_scalar3")]) == 0


@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("npassives", [0, 2])
@pytest.mark.parametrize("pressure", [np.nan, -10.0, 100.0])
def test_upwind_advection(dim, npassives, pressure):
    passives = {f"scalar{i + 1}": i + 5 for i in range(npassives)}
    idx = VariableIndexMap(
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
            **passives,
        },
        {"v": ["vx", "vy", "vz"], **({"passives": list(passives.keys())} if npassives else {})},
    )
    wl = xp.full((idx.nvars, 6), 2.0)
    wr = xp.full_like(wl, 3.0)
    wl[idx(f"v{dim}")] = xp.array([2, -3, 1, -1, 0, 1])
    wr[idx(f"v{dim}")] = xp.array([1, -2, -2, 1, 0, -1])
    wl[idx("P")] = pressure
    wr[idx("P")] = pressure
    if npassives:
        wl[idx("passives")] = xp.array([[5], [7]])
        wr[idx("passives")] = xp.array([[11], [13]])
    flux = xp.full_like(wl, xp.nan)

    UpwindRiemannSolver(npassives)(wl, wr, flux, dim, idx, gamma=np.nan)

    assert linf_norm(flux[idx("rho")] - xp.array([4, -9, -6, 2, 0, -3])) == 0
    assert linf_norm(flux[idx("v")]) == 0
    assert linf_norm(flux[idx("E")]) == 0
    if npassives:
        expected = xp.array([[10, -33, -22, 5, 0, -11], [14, -39, -26, 7, 0, -13]])
        assert linf_norm(flux[idx("passives")] - expected) == 0
