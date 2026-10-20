from functools import partial

import numpy as np
import pytest

from superfv import AdvectionSolver, ics
from superfv.axes import DIM_TO_AXIS
from superfv.tools.device_management import CUPY_AVAILABLE
from superfv.tools.norms import linf_norm


@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori", "none"])
def test_mpp_advection_of_square(p, limiting):
    if p == 0:
        if limiting != "none":
            pytest.skip("p=0 does not support limiting")
        limiting_config = dict()
    else:
        match limiting:
            case "a priori":
                limiting_config = dict(use_ZS=True, adaptive_dt=True)
            case "a posteriori":
                limiting_config = dict(
                    use_MOOD=True,
                    rtol=0.0,
                    fallback_cascade="muscl0",
                    max_revs=3,
                )
            case "none":
                pytest.skip("p>0 requires limiting")
            case _:
                raise ValueError(f"{limiting=}")

    sim = AdvectionSolver(
        cupy=CUPY_AVAILABLE,
        ic=partial(ics.square, rho_min=0.0, rho_max=1.0, vx=1.0),
        p=p,
        PAD_bounds={"rho": (0.0, 1.0)},
        use_global_bounds=True,
        use_SED=False,
        **limiting_config,
    )
    sim.take_n_steps(10)

    assert min(sim.step_history.get_history("rho_min")) >= 0.0
    assert max(sim.step_history.get_history("rho_max")) <= 1.0


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(p=3),
        dict(p=7),
        dict(p=3, use_ZS=True),
        dict(p=7, use_ZS=True),
        dict(p=7, use_MOOD=True, fallback_cascade="muscl0", max_revs=3),
    ],
)
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_forward_backwards_advection_symmetry(scheme, dim):
    left_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim}": 1}
    right_vel = dict(vx=0, vy=0, vz=0) | {f"v{dim}": -1}
    mesh = dict(nx=1, ny=1, nz=1) | {f"n{dim}": 64}

    sim1 = AdvectionSolver(
        ic=partial(ics.square, rho_min=1, rho_max=2, **left_vel),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )
    sim2 = AdvectionSolver(
        ic=partial(ics.square, rho_min=1, rho_max=2, **right_vel),
        **mesh,
        cupy=CUPY_AVAILABLE,
        **scheme,
    )

    sim1.take_n_steps(
        10,
        time_integrator=("muscl_hancock" if scheme.get("use_MUSCL", False) else "ssprk3"),
    )
    sim2.take_n_steps(
        10,
        time_integrator=("muscl_hancock" if scheme.get("use_MUSCL", False) else "ssprk3"),
    )

    idx = sim1.params.variable_index_map
    assert (
        linf_norm(
            sim1.snapshot_history[-1].u[idx("rho", keepdims=True)]
            - np.flip(sim2.snapshot_history[-1].u[idx("rho", keepdims=True)], axis=DIM_TO_AXIS[dim])
        )
        < 1e-14
    )
