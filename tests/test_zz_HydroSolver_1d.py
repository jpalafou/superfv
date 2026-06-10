from functools import partial

import numpy as np
import pytest

from superfv import (
    BC,
    CUPY_AVAILABLE,
    FallbackCascade,
    HydroSolver,
    LazyPrimitiveMode,
    TimeIntegrator,
    ics,
)
from superfv.axes import DIM_TO_AXIS
from superfv.configs import FluxRecipe
from superfv.riemann_solvers import RiemannSolver
from superfv.tools.norms import linf_norm
from teyssier import cons_to_prim, weno


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(p=1, use_ZS=True, adaptive_dt=True),
        dict(p=2, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
def test_sedov(scheme):
    sim = HydroSolver(
        ic=partial(ics.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MATCH_P_UP_TO_SSPRK3)


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
    ],
)
def test_sedov_with_passive_scalar(scheme):
    sim = HydroSolver(
        ic=partial(ics.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim_with_passive = HydroSolver(
        ic=partial(ics.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        passive_ics={"passive1": lambda x, y, z, t, xp: xp.where(xp.abs(x - 0.5) < 0.25, 1, 0)},
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )

    sim.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
    )
    sim_with_passive.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
    )

    idx = sim.params.variable_index_map
    assert (
        linf_norm(
            sim.snapshot_history[-1].u[idx("rho")]
            - sim_with_passive.snapshot_history[-1].u[idx("rho")]
        )
        < 1e-14
    )


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(p=1, use_ZS=True, adaptive_dt=True),
        dict(p=2, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FIRST_ORDER, rtol=0, max_revs=1),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, rtol=0, max_revs=2),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, rtol=0, max_revs=64),
    ],
)
def test_preservation_of_maximum_principle(scheme):
    with pytest.warns(
        UserWarning,
        match="PAD lower bound for 'rho' is 1, which is different from the hydro parameter rho_min=1e-12.",
    ):
        sim = HydroSolver(
            ic=partial(ics.square, rho_min=1, rho_max=2, vx=1),
            PAD_bounds={"rho": (1, 2)},
            nx=64,
            cupy=CUPY_AVAILABLE,
            **scheme,
        )
    if scheme.get("use_MUSCL", False):
        sim.run(1.0, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.run(1.0, time_integrator=TimeIntegrator.MATCH_P_UP_TO_SSPRK3)

    assert min(sim.step_history.get_history("rho_min")) > 1 - 1e-14
    assert min(sim.step_history.get_history("rho_min")) < 2 + 1e-14


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(p=3),
        dict(p=7),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.FULL),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.FULL),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_forward_backwards_advection_symmetry(scheme, dim):
    left_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim}": 1}
    right_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim}": -1}
    mesh = dict(nx=1, ny=1, nz=1) | {f"n{dim}": 64}

    sim1 = HydroSolver(
        ic=partial(ics.square, rho_min=1, rho_max=2, **left_vel),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim2 = HydroSolver(
        ic=partial(ics.square, rho_min=1, rho_max=2, **right_vel),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )

    sim1.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
    )
    sim2.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
    )

    idx = sim1.params.variable_index_map
    assert (
        linf_norm(
            sim1.snapshot_history[-1].u[idx("rho", keepdims=True)]
            - np.flip(sim2.snapshot_history[-1].u[idx("rho", keepdims=True)], axis=DIM_TO_AXIS[dim])
        )
        < 1e-14
    )


@pytest.mark.parametrize("p", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "ic_type_t_sim",
    [("sod test", 0.245), ("toro test2", 0.2), ("toro test3", 0.012)],
)
def test_compare_with_teyssier_code(p, ic_type_t_sim):
    ic_type, t_sim = ic_type_t_sim

    N = 400 if ic_type == "shu osher" else 100

    sim = HydroSolver(
        ic={"sod test": ics.sod_shock_tube_1d, "toro test2": ics.toro2, "toro test3": ics.toro3}[
            ic_type
        ],
        force_1st_order_ic_cell_averages=True,
        bcx=(BC.FREE, BC.FREE),
        nx=N,
        p=p,
        riemann_solver=RiemannSolver.HLLC_TEYSSIER,
        flux_recipe=FluxRecipe.PRIM_PRIM_LIM,
        use_ZS=True,
        rho_min=-np.inf,
        P_min=-np.inf,
    )
    sim.run(t_sim, time_integrator=TimeIntegrator.MATCH_P_UP_TO_RK4, allow_overshoot=True)

    _, ut = weno(
        t_sim,
        N,
        ic_type=ic_type,
        bc_type="free",
        riemann_solver="hllc",
        time=p + 1,
        space=p + 1,
    )
    wt = cons_to_prim(ut[-1, :, :])

    diff = np.max(np.abs(sim.snapshot_history[-1].w[sim.idx("rho"), :, 0, 0] - wt[0, :]))
    assert diff < 1e-12
