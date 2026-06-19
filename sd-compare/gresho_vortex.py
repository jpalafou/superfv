import os
import shutil
from functools import partial

import spd.initial_conditions as ic
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput, ics

base_directory = "/scratch/gpfs/jp7427/FVvsSD/gresho/"


def run_superfv_sim(name, p, N, v0=5.0, M_max=0.1, **kwargs):
    path = base_directory + f"FV_{name}_{v0=}_{M_max=}_{N=}_{p=}"

    try:
        out = HydroSolverOutput(path)
        print(f"Loaded output from '{path}'")
        return out
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        pass

    if os.path.exists(path):
        print(f"Removing bad output at '{path}'")
        shutil.rmtree(path)

    sim = HydroSolver(
        ic=partial(ics.gresho_vortex, gamma=5 / 3, v0=v0, M_max=M_max),
        gamma=5 / 3,
        rho_min=1e-10,
        P_min=1e-10,
        nx=N,
        ny=N,
        p=p,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(1.0)
    return sim


def run_spd_sim(name, p, N, v0=5.0, M_max=0.1, **kwargs):
    path = base_directory + f"SD_{name}_{v0=}_{M_max=}_{N=}_{p=}"

    sim = SPD_Simulator(
        p=p,
        N=(N, N),
        init_fct=ic.gresho_vortex(gamma=5 / 3, v0=v0, M_max=M_max),
        gamma=5 / 3,
        cfl_coeff=0.4,
        use_cupy=True,
        time_integrator="rk3",
        scheme="SD",
        FB=False,
        riemann_solver_sd="hllc",  # MUSCL fallback flux
        folder=path,
        **kwargs,
    )

    try:
        sim.load_output()
        print(f"Loaded output from '{path}'")
        return sim
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        pass

    sim.perform_time_evolution(1.0)
    sim.output()
    return sim


NDOF = 96

FV4_1 = run_superfv_sim("p=3", 3, NDOF, v0=5.0, M_max=0.1)
FV4_2 = run_superfv_sim("p=3", 3, NDOF, v0=5.0, M_max=0.01)
FV4_3 = run_superfv_sim("p=3", 3, NDOF, v0=5.0, M_max=0.001)
FV8_1 = run_superfv_sim("p=7", 7, NDOF, v0=5.0, M_max=0.1)
FV8_2 = run_superfv_sim("p=7", 7, NDOF, v0=5.0, M_max=0.01)
FV8_3 = run_superfv_sim("p=7", 7, NDOF, v0=5.0, M_max=0.001)

SD4_1 = run_spd_sim("p=3", 3, NDOF // (3 + 1), v0=5.0, M_max=0.1)
SD4_2 = run_spd_sim("p=3", 3, NDOF // (3 + 1), v0=5.0, M_max=0.01)
SD4_3 = run_spd_sim("p=3", 3, NDOF // (3 + 1), v0=5.0, M_max=0.001)
SD8_1 = run_spd_sim("p=7", 7, NDOF // (7 + 1), v0=5.0, M_max=0.1)
SD8_2 = run_spd_sim("p=7", 7, NDOF // (7 + 1), v0=5.0, M_max=0.01)
SD8_3 = run_spd_sim("p=7", 7, NDOF // (7 + 1), v0=5.0, M_max=0.001)
