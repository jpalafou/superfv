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


def run_superfv_sim(p, N, nsteps, **kwargs):

    sim = HydroSolver(
        ic=partial(ics.sinus, vx=2.0, vy=1.0),
        rho_min=1e-10,
        P_min=1e-10,
        nx=N,
        ny=N,
        p=p,
        use_MOOD=True,
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
    assert len(sim.step_history[2:]) == nsteps

    total_time = sum(x.timer["take_step"].cum_time for x in step_history)
    riemann_solver_time = sum(x.timer["riemann_solver"].cum_time for x in step_history)
    limiter_time = sum(x.timer["mood_loop"].cum_time for x in step_history)

    report = dict(
        cell_updates_per_second=nsteps * N**2 / total_time,
        riemann_solver_time_per_step=riemann_solver_time / nsteps,
        limiter_time_per_step=limiter_time / nsteps,
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
        # scheme="SD",
        # FB=False,
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
        potential=True,
        limiting_variables=[0, 1, 2, 4],
        tolerance=0.0,
        PAD=True,
        SED=True,
        blending=False,
        riemann_solver_sd="hllc",
        riemann_solver_fv="hllc",
        **kwargs,
    )

    # take nsteps and time the simulation
    sim.perform_iterations(1)  # warm-up / compile
    step0 = sim.n_step
    sim.perform_iterations(nsteps)  # use synced take_step timing
    assert sim.n_step - step0 == nsteps

    report = dict(
        DOF_updates_per_second=sim.domain_size * nsteps / sim.execution_times["take_step"],
        riemann_solver_time_per_step=sim.execution_times["riemann_solver_sd"] / nsteps,
        limiter_time_per_step=sim.execution_times["mood_loop"] / nsteps,
    )

    return report


if __name__ == "__main__":
    data = []
    for NDOF in [64, 128, 256, 512, 1024, 2048, 3000]:
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
                    rs_per_step=fv_report["riemann_solver_time_per_step"],
                    limiter_per_step=fv_report["limiter_time_per_step"],
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
                    rs_per_step=spd_report["riemann_solver_time_per_step"],
                    limiter_per_step=spd_report["limiter_time_per_step"],
                )
            )
            print(f"Measured update rate: {update_rate:.2e} DOF updates per second\n")
    df = pd.DataFrame(data)

    # write to csv
    df.to_csv(base_directory + "timing_results.csv", index=False)
