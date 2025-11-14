from functools import partial

import pytest

from superfv import EulerSolver
from superfv.initial_conditions import decaying_isotropic_turbulence

rho_PAD = {"rho": (0.0, None)}


@pytest.mark.parametrize(
    "config",
    [
        dict(p=0),
        dict(p=1, MUSCL=True, MUSCL_limiter="moncen"),
        dict(p=3, lazy_primitives="adaptive", ZS=True, PAD=rho_PAD),
    ],
)
@pytest.mark.parametrize("seed", range(1, 31))
def test_1d_isotropic_decaying_turbulence(config: dict, seed: int):
    N = 100
    T = 0.1

    sim = EulerSolver(
        ic=partial(
            decaying_isotropic_turbulence,
            seed=seed,
            M=10,
            slope=-5 / 3,
            solenoidal=False,
        ),
        isothermal=True,
        nx=N,
        SED=False,
        **config,
    )

    sim.run(T, allow_overshoot=True, q_max=2, muscl_hancock=config.get("MUSCL", False))


@pytest.mark.parametrize(
    "config",
    [
        dict(p=0),
        dict(p=1, MUSCL=True, MUSCL_limiter="PP2D"),
        dict(p=3, ZS=True, GL=True, lazy_primitives="adaptive", PAD=rho_PAD),
    ],
)
@pytest.mark.parametrize("seed", range(1, 31))
def test_2d_isotropic_decaying_turbulence(config: dict, seed: int):
    N = 64
    T = 0.01

    sim = EulerSolver(
        ic=partial(
            decaying_isotropic_turbulence,
            seed=seed,
            M=10,
            slope=-5 / 3,
        ),
        isothermal=True,
        nx=N,
        ny=N,
        SED=False,
        **config,
    )

    sim.run(T, allow_overshoot=True, q_max=2, muscl_hancock=config.get("MUSCL", False))
