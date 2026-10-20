import numpy as np
import pytest

from superfv import AdvectionSolver
from superfv.advection_driver import (
    compute_advection_dt,
    compute_flux_jvp,
    update_advection_fluxes,
    update_advection_workspace,
)
from superfv.riemann_solvers import UpwindRiemannSolver


@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_flux_jvp_matches_upwind_flux_derivative(dim):
    sim = AdvectionSolver(nx=8, passive_ics={"dye": lambda x, y, z, t, *, xp: x})
    idx = sim.idx
    rng = np.random.default_rng(0)
    u = rng.uniform(1, 2, (idx.nvars, 8))
    vec = rng.uniform(-1, 1, u.shape)
    u[idx("P")] = np.nan
    jvp = np.empty_like(u)
    compute_flux_jvp(u, vec, jvp, idx, dim)

    eps = 1e-5
    fp, fm = np.empty_like(u), np.empty_like(u)
    solver = UpwindRiemannSolver(npassives=1)
    for sign, flux in [(1, fp), (-1, fm)]:
        state = u + sign * eps * vec
        solver(state, state, flux, dim, idx, gamma=np.nan)

    np.testing.assert_allclose(jvp, (fp - fm) / (2 * eps), rtol=1e-9, atol=1e-10)
    assert (jvp[idx("E")] == 0).all()


@pytest.mark.parametrize("dims", ["xy", "xz", "yz"])
@np.errstate(divide="ignore", invalid="ignore", over="ignore")
def test_pp2d_hancock_zero_timestep_limit(dims):
    sim = AdvectionSolver(
        **{f"n{d}": 8 if d in dims else 1 for d in "xyz"},
        p=1,
        use_MUSCL=True,
        MUSCL_limiter="pp2d",
        use_SED=False,
        clip_zero_tol=0.1,
    )
    a, idx = sim.arrays, sim.idx
    u = np.random.default_rng(5).uniform(-0.01, 0.01, a["u"].shape)
    u[idx("v")] = 1
    u[idx("P")] = np.nan
    update_advection_workspace(u, a["_u_"], 0, idx, sim.mesh, sim.params.bc)
    results = []
    for dt in [0, 1e-30]:
        update_advection_fluxes(
            a["_u_"],
            *[a[f"_{f}_"] if d in dims else np.array([]) for d, f in zip("xyz", "FGH")],
            np.array([]),
            np.array([]),
            a["_alpha_"],
            idx,
            tuple(dims),
            sim.mesh,
            sim.params.fv_scheme,
            hancock_dt=dt,
        )
        results.append([a[f].copy() for d, f in zip("xyz", "FGH") if d in dims])
    for ordinary, hancock in zip(*results):
        np.testing.assert_allclose(ordinary, hancock, rtol=0, atol=1e-15)


def test_advection_dt_ignores_dummy_pressure_and_inactive_velocity():
    sim = AdvectionSolver(nx=8, ny=4, zlims=(0, 0.01))
    u, idx = sim.arrays["u"], sim.idx
    u[idx("vx")] = 2
    u[idx("vy")] = -3
    u[idx("vz")] = 1e6
    assert np.isnan(u[idx("P")]).all()
    assert compute_advection_dt(u, idx, sim.mesh, 0.8) == pytest.approx(0.8 / (2 * 8 + 3 * 4))
    u[idx("vx")] = u[idx("vy")] = 0
    assert compute_advection_dt(u, idx, sim.mesh, 0.8) == np.inf
