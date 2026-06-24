from functools import partial
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, TimeIntegrator, ics
from superfv.hydro_solver import SnapshotMode
from superfv.tools.device_management import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

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
    sim.take_n_steps(1, time_integrator=TimeIntegrator.SSPRK3, snapshot_mode=SnapshotMode.NONE)

    # take nsteps and time the simulation
    t0 = timer()
    sim.take_n_steps(nsteps, time_integrator=TimeIntegrator.SSPRK3, snapshot_mode=SnapshotMode.NONE)
    elapsed = timer() - t0

    cell_updates_per_second = nsteps * N**2 / elapsed
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

    # prime solver with untimed step
    sim.compute_dt()
    sim.perform_update()

    # take nsteps and time the simulation
    t0 = timer()
    for _ in range(nsteps):
        sim.compute_dt()
        sim.perform_update()
    if CUPY_AVAILABLE and sim.use_cupy:
        cp.cuda.Device().synchronize()
    elapsed = timer() - t0

    DOF_updates_per_second = nsteps * NDOF**2 / elapsed
    return DOF_updates_per_second


if __name__ == "__main__":
    data = []
    for NDOF in [64, 128, 256, 512, 1024]:
        for p in [3, 7]:
            superfv_update_rate = run_superfv_sim(p, NDOF, nsteps=10)
            data.append(dict(NDOF=NDOF, p=p, scheme="FV", update_rate=superfv_update_rate))

            spd_time = time_spd_sim(p, NDOF, nsteps=10)
            data.append(dict(NDOF=NDOF, p=p, scheme="SD", update_rate=spd_time))
    df = pd.DataFrame(data)

    # write to csv
    df.to_csv(base_directory + "timing_results.csv", index=False)
