from functools import partial

import numpy as np
import pandas as pd
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, TimeIntegrator, ics
from superfv.hydro_solver import SnapshotMode
from superfv.tools.device_management import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    pass

base_directory = "/scratch/gpfs/jp7427/FVvsSD/sinus-timing/"

UNLIMITED_ROUTINES = (
    "compute_dt",
    "update_unew",
    "primitive_conservative",
    "boundary_conditions",
    "stencil_sweep",
    "einsum",
    "transpose",
    "riemann_solver",
)

MOOD_ROUTINES = (
    "candidate_solution",
    "detect_troubles",
    "fallback_fluxes",
    "assign_fluxes",
)


def run_superfv_sim(p, N, nsteps, **kwargs):

    sim = HydroSolver(
        ic=partial(ics.sinus, vx=2.0, vy=1.0),
        rho_min=1e-10,
        P_min=1e-10,
        nx=N,
        ny=N,
        p=p,
        use_MOOD=False,
        rtol=0.0,
        use_SED=False,
        cupy=True,
        profile=True,
        **kwargs,
    )

    # prime solver with untimed step
    sim.take_n_steps(
        nsteps + 1, time_integrator=TimeIntegrator.SSPRK3, snapshot_mode=SnapshotMode.NONE
    )

    step_history = sim.step_history[2:]
    assert len(step_history) == nsteps

    def get_total_time(routine):
        if not step_history or routine not in step_history[0].timer.timers:
            return 0.0
        return sum(step.timer[routine].cum_time for step in step_history)

    take_step_time = get_total_time("take_step")

    report = dict(
        cell_updates_per_second=nsteps * N**2 / take_step_time,
        total_time_per_step=take_step_time / nsteps,
        mood_loop_time_per_step=get_total_time("mood_loop") / nsteps,
        **{f"{cat}_per_step": get_total_time(cat) / nsteps for cat in UNLIMITED_ROUTINES},
        **{f"{cat}_per_step": get_total_time(cat) / nsteps for cat in MOOD_ROUTINES},
    )

    return report


def time_spd_sim(p, NDOF, nsteps, **kwargs):
    def sine_wave(xy: np.ndarray, case: int, vx=1, vy=1, P=1):
        x = xy[0]
        y = xy[1]
        if case == 0:
            # density
            return 1.5 + 0.5 * np.sin(2 * np.pi * (x + y))
        elif case == 1:
            # vx
            return vx * np.ones(x.shape)
        elif case == 2:
            # vy
            return vy * np.ones(x.shape)
        elif case == 4:
            # Pressure
            return P * np.ones(x.shape)
        else:
            return np.zeros(x.shape)

    Nelements = NDOF // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(Nelements, Nelements),
        init_fct=partial(sine_wave, vx=2.0, vy=1.0),
        use_cupy=True,
        time_integrator="rk3",
        scheme="SD",
        FB=False,
        # scheme="SDFB",
        # fallback="MUSCL",
        # slope_limiter="moncen",
        # potential=True,
        # limiting_variables=[0, 1, 2, 4],
        # tolerance=0.0,
        # PAD=True,
        # SED=True,
        # blending=False,
        riemann_solver_sd="hllc",
        riemann_solver_fv="hllc",
        **kwargs,
    )

    # take nsteps and time the simulation
    sim.perform_iterations(1)  # warm-up / compile
    step0 = sim.n_step
    sim.perform_iterations(nsteps)  # resets timers and uses synced take_step timing
    assert sim.n_step - step0 == nsteps

    def get_total_time(routine):
        if routine not in sim.timer.timers:
            return 0.0
        return sim.timer[routine].cum_time

    take_step_time = get_total_time("take_step")

    report = dict(
        DOF_updates_per_second=sim.domain_size * nsteps / take_step_time,
        total_time_per_step=take_step_time / nsteps,
        mood_loop_time_per_step=get_total_time("mood_loop") / nsteps,
        **{f"{cat}_per_step": get_total_time(cat) / nsteps for cat in UNLIMITED_ROUTINES},
        **{f"{cat}_per_step": get_total_time(cat) / nsteps for cat in MOOD_ROUTINES},
    )

    return report


if __name__ == "__main__":
    data = []
    for NDOF in [64, 128, 256, 512, 1024, 2048, 3072]:
        for p in [3, 7]:
            print(f"Running FV simulation with NDOF={NDOF}, p={p}")
            fv_report = run_superfv_sim(p, NDOF, nsteps=10)
            update_rate = fv_report["cell_updates_per_second"]
            data.append(
                dict(
                    NDOF=NDOF,
                    p=p,
                    scheme="FV",
                    update_rate=update_rate,
                    total_per_step=fv_report["total_time_per_step"],
                    rs_per_step=fv_report["riemann_solver_per_step"],
                    mood_loop_per_step=fv_report["mood_loop_time_per_step"],
                    **{
                        f"{cat}_per_step": fv_report[f"{cat}_per_step"]
                        for cat in UNLIMITED_ROUTINES
                    },
                    **{f"{cat}_per_step": fv_report[f"{cat}_per_step"] for cat in MOOD_ROUTINES},
                )
            )
            print(f"Measured update rate: {update_rate:.2e} DOF updates per second\n")

            print(f"Running SD simulation with NDOF={NDOF}, p={p}")
            spd_report = time_spd_sim(p, NDOF, nsteps=10)
            update_rate = spd_report["DOF_updates_per_second"]
            data.append(
                dict(
                    NDOF=NDOF,
                    p=p,
                    scheme="SD",
                    update_rate=update_rate,
                    total_per_step=spd_report["total_time_per_step"],
                    rs_per_step=spd_report["riemann_solver_per_step"],
                    mood_loop_per_step=spd_report["mood_loop_time_per_step"],
                    **{
                        f"{cat}_per_step": spd_report[f"{cat}_per_step"]
                        for cat in UNLIMITED_ROUTINES
                    },
                    **{f"{cat}_per_step": spd_report[f"{cat}_per_step"] for cat in MOOD_ROUTINES},
                )
            )
            print(f"Measured update rate: {update_rate:.2e} DOF updates per second\n")
    df = pd.DataFrame(data)

    # write to csv
    df.to_csv(base_directory + "timing_results.csv", index=False)
