from superfv import AdvectionSolver, ics
from functools import partial
import pytest

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