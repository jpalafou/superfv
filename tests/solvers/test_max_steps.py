from functools import partial

import pytest

from superfv import EulerSolver
from superfv.initial_conditions import square


@pytest.mark.parametrize(
    "config",
    [
        dict(p=0),
        dict(p=1, ZS=True),
        dict(p=2, ZS=True),
        dict(p=3, ZS=True),
        dict(p=1, MUSCL=True),
    ],
)
def test_max_steps(config):
    T = 1.0
    max_steps = 20

    ic = partial(square, bounds=(1e-3, 1), vx=1, P=1e-3)
    PAD = {"rho": (0, None), "P": (0, None)}

    sim1 = EulerSolver(ic=ic, nx=64, PAD=PAD, lazy_primitives="adaptive", **config)
    sim2 = EulerSolver(ic=ic, nx=64, PAD=PAD, lazy_primitives="adaptive", **config)

    sim1.run(T, muscl_hancock=config.get("MUSCL", False))
    sim2.run(T, muscl_hancock=config.get("MUSCL", False), max_steps=max_steps)

    assert sim1.n_steps > max_steps
    assert sim2.n_steps == max_steps
