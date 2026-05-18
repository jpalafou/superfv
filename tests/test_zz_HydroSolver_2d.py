from functools import partial

import numpy as np
import pytest

from superfv import (
    BC,
    CUPY_AVAILABLE,
    FallbackCascade,
    FluxQuadrature,
    HydroSolver,
    LazyPrimitiveMode,
    MUSCL_SlopeLimiter,
    TimeIntegrator,
    ics,
)
from superfv.axes import DIM_TO_AXIS
from superfv.tools.norms import linf_norm


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(
            p=3,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode=LazyPrimitiveMode.FULL,
            flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE,
        ),
        dict(
            p=7,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode=LazyPrimitiveMode.FULL,
            flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE,
        ),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
def test_sedov(scheme):
    sim = HydroSolver(
        ic=partial(ics.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        bcy=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=64,
        ny=64,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK, print_frequency=1)
    else:
        sim.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3, print_frequency=1)


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True, MUSCL_limiter=MUSCL_SlopeLimiter.PP2D),
        dict(
            p=7,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE,
            flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE,
        ),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, rtol=0, max_revs=64),
    ],
)
def test_preservation_of_maximum_principle(scheme):
    with pytest.warns(
        UserWarning,
        match="PAD lower bound for 'rho' is 1, which is different from the hydro parameter rho_min=1e-12.",
    ):
        sim = HydroSolver(
            ic=partial(ics.square, rho_min=1, rho_max=2, vx=2, vy=1),
            PAD_bounds={"rho": (1, 2)},
            nx=64,
            ny=64,
            cupy=CUPY_AVAILABLE,
            **scheme,
        )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3)

    assert min(sim.step_history.get_history("rho_min")) > 1 - 1e-14
    assert min(sim.step_history.get_history("rho_min")) < 2 + 1e-14


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3),
        dict(p=7),
        dict(p=3, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
    ],
)
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("y", "z"), ("x", "z")])
def test_forward_backwards_advection_symmetry(scheme, dim1_dim2):
    dim1, dim2 = dim1_dim2
    left_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim1}": 1, f"v{dim2}": 1}
    right_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim1}": -1, f"v{dim2}": -1}
    mesh = dict(nx=1, ny=1, nz=1) | {f"n{dim1}": 64, f"n{dim2}": 64}

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

    def double_flip(x, dim1, dim2):
        return np.flip(np.flip(x, axis=DIM_TO_AXIS[dim1]), axis=DIM_TO_AXIS[dim2])

    idx = sim1.params.variable_index_map
    assert (
        linf_norm(
            sim1.snapshot_history[-1].u[idx("rho", keepdims=True)]
            - double_flip(sim2.snapshot_history[-1].u[idx("rho", keepdims=True)], dim1, dim2)
        )
        < 1e-14
    )


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3),
        dict(p=7),
        dict(p=3, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
    ],
)
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("y", "z"), ("x", "z")])
def test_rotational_advection_symmetry(scheme, dim1_dim2):
    dim1, dim2 = dim1_dim2
    mesh = dict(nx=1, ny=1, nz=1) | {f"n{dim1}": 64, f"n{dim2}": 64}

    sim1 = HydroSolver(
        ic=partial(ics.slotted_disk, rho_min=1, rho_max=2),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim2 = HydroSolver(
        ic=partial(ics.slotted_disk, rho_min=1, rho_max=2, rotation="cw"),
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

    def reflect_for_opposite_rotation(x, dim1):
        return np.flip(x, axis=DIM_TO_AXIS[dim1])

    idx = sim1.params.variable_index_map
    assert (
        linf_norm(
            sim1.snapshot_history[-1].u[idx("rho", keepdims=True)]
            - reflect_for_opposite_rotation(
                sim2.snapshot_history[-1].u[idx("rho", keepdims=True)], dim1
            )
        )
        < 1e-14
    )
