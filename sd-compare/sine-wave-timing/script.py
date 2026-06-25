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
        cupy=True,
        **kwargs,
    )

    # prime solver with untimed step
    sim.take_n_steps(
        nsteps + 1, time_integrator=TimeIntegrator.SSPRK3, snapshot_mode=SnapshotMode.NONE
    )
    assert len(sim.step_history) == nsteps + 2
    ellapsed_time = sum(x.timer["take_step"].cum_time for x in sim.step_history[2:])
    cell_updates_per_second = nsteps * N**2 / ellapsed_time

    return cell_updates_per_second


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
        riemann_solver_sd="hllc",
        riemann_solver_fv="hllc",
        **kwargs,
    )

    # take nsteps and time the simulation
    sim.perform_iterations(1)  # warm-up / compile
    step0 = sim.n_step
    sim.perform_iterations(nsteps)  # use sim.execution_time
    assert sim.n_step - step0 == nsteps

    DOF_updates_per_second = sim.domain_size * nsteps / sim.execution_time
    return DOF_updates_per_second


if __name__ == "__main__":
    data = []
    for NDOF in [64, 128, 256, 512, 1024, 2048, 3000]:
        for p in [3, 7]:
            print(f"Running FV simulation with NDOF={NDOF}, p={p}")
            superfv_update_rate = run_superfv_sim(p, NDOF, nsteps=10)
            data.append(dict(NDOF=NDOF, p=p, scheme="FV", update_rate=superfv_update_rate))

            print(f"Running SD simulation with NDOF={NDOF}, p={p}")
            spd_time = time_spd_sim(p, NDOF, nsteps=10)
            data.append(dict(NDOF=NDOF, p=p, scheme="SD", update_rate=spd_time))
    df = pd.DataFrame(data)

    # write to csv
    df.to_csv(base_directory + "timing_results.csv", index=False)
