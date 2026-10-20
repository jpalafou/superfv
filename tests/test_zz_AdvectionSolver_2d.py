from functools import partial

import pytest

from superfv import AdvectionSolver, ics
from superfv.tools.device_management import CUPY_AVAILABLE


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
                limiting_config = dict(
                    use_ZS=True, adaptive_dt=True, flux_quadrature="gauss_legendre"
                )
            case "a posteriori":
                limiting_config = dict(
                    use_MOOD=True,
                    rtol=0.0,
                    fallback_cascade="muscl0",
                    MUSCL_limiter="pp2d",
                    max_revs=16,
                )
            case "none":
                pytest.skip("p>0 requires limiting")
            case _:
                raise ValueError(f"{limiting=}")

    sim = AdvectionSolver(
        cupy=CUPY_AVAILABLE,
        ic=partial(ics.square, rho_min=0.0, rho_max=1.0, vx=2.0, vy=1.0),
        nx=32,
        ny=32,
        p=p,
        PAD_bounds={"rho": (0.0, 1.0)},
        use_global_bounds=True,
        use_SED=False,
        **limiting_config,
    )
    sim.take_n_steps(10)

    assert min(sim.step_history.get_history("rho_min")) >= 0.0
    assert max(sim.step_history.get_history("rho_max")) <= 1.0
