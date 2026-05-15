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
    ic,
)
from superfv.axes import DIM_TO_AXIS
from superfv.tools.norms import linf_norm


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
def test_sedov(scheme):
    sim = HydroSolver(
        ic=partial(ic.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3)


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
    ],
)
def test_sedov_with_passive_scalar(scheme):
    sim = HydroSolver(
        ic=partial(ic.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim_with_passive = HydroSolver(
        ic=partial(ic.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
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
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FIRST_ORDER, rtol=0, max_revs=1),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, rtol=0, max_revs=2),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, rtol=0, max_revs=16),
    ],
)
def test_preservation_of_maximum_principle(scheme):
    with pytest.warns(
        UserWarning,
        match="PAD lower bound for 'rho' is 1, which is different from the hydro parameter rho_min=1e-12.",
    ):
        sim = HydroSolver(
            ic=partial(ic.square, rho_min=1, rho_max=2, vx=1),
            PAD_bounds={"rho": (1, 2)},
            nx=64,
            cupy=CUPY_AVAILABLE,
            **scheme,
        )
    if scheme.get("use_MUSCL", False):
        sim.run(1.0, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.run(1.0, time_integrator=TimeIntegrator.SSPRK3)

    assert min(sim.step_history.get_history("rho_min")) > 1 - 1e-14
    assert min(sim.step_history.get_history("rho_min")) < 2 + 1e-14


@pytest.mark.parametrize(
    "scheme",
    [
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
        ic=partial(ic.square, rho_min=1, rho_max=2, **left_vel),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim2 = HydroSolver(
        ic=partial(ic.square, rho_min=1, rho_max=2, **right_vel),
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
