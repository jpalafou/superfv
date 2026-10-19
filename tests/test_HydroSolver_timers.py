from unittest.mock import Mock

import numpy as np
import pytest

from superfv import HydroSolver


@pytest.mark.parametrize("method", ["run", "take_n_steps"])
@pytest.mark.parametrize("mood", [False, True])
def test_profile_preserves_solution_and_timer_counts(method, mood):
    results = []
    for profile in (False, True):
        sim = HydroSolver(
            nx=12,
            ny=8,
            p=3,
            use_MOOD=mood,
            max_revs=2,
            fallback_cascade="muscl0",
            positivity_guard=False,
            profile=profile,
        )
        if method == "run":
            sim.run([0.01, 0.02], time_integrator="ssprk3", print_update=False)
        else:
            sim.take_n_steps(2, time_integrator="ssprk3", snapshot_mode="every", print_update=False)
        results.append(sim.arrays["u"].copy())
        for step in sim.step_history.steps[1:]:
            expected = {
                "take_step": 1,
                "take_snapshot": 1,
                "compute_dt": 1,
                "update_unew": 1,
                "f": 3,
                "compute_dudt": 3,
                "mood_loop": 3 if mood else 0,
            }
            for category, count in expected.items():
                assert step.timer[category].n_calls == (
                    count if profile or category == "take_step" else 0
                )
            assert not any(timer.is_timing for timer in step.timer.timers.values())
    np.testing.assert_array_equal(*results)


def test_failed_snapshot_closes_timer():
    sim = HydroSolver(nx=8, profile=True)
    sim.bonus_snapshot_routine = Mock(side_effect=RuntimeError("snapshot callback failed"))
    with pytest.raises(RuntimeError, match="snapshot callback failed"):
        sim._take_snapshot()
    assert not sim.step_summary.timer["take_snapshot"].is_timing


def test_failed_step_closes_timers(monkeypatch):
    sim = HydroSolver(nx=8, profile=True)
    monkeypatch.setattr(
        "superfv.hydro_solver.compute_hydro_dt",
        Mock(side_effect=RuntimeError("timestep failed")),
    )
    with pytest.raises(RuntimeError, match="timestep failed"):
        sim.take_n_steps(1, print_update=False)
    assert not any(timer.is_timing for timer in sim.step_summary.timer.timers.values())
