import os
import shutil
from functools import partial

import numpy as np
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput, RiemannSolver, TimeIntegrator, ics

base_directory = "/scratch/gpfs/jp7427/FVvsSD/sinus/"


def reduce_CFL(N: int, p: int, initial_CFL: float = 0.8, qmax: int = 3) -> float:
    if p > qmax:
        return initial_CFL * ((1 / N) ** ((p - qmax) / (qmax + 1)))
    else:
        return initial_CFL


def run_superfv_sim(name, p, N, **kwargs):
    path = base_directory + f"FV_{name}_{N=}_{p=}"

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
        ic=partial(ics.sinus, vx=2.0, vy=1.0),
        rho_min=1e-10,
        P_min=1e-10,
        nx=N,
        ny=N,
        p=p,
        CFL=reduce_CFL(N, p),
        riemann_solver=RiemannSolver.LLF,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(1.0, time_integrator=TimeIntegrator.RK4)
    return sim


def run_spd_sim(name, p, NDOF, **kwargs):
    path = base_directory + f"SD_{name}_{NDOF=}_{p=}"

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
        cfl_coeff=reduce_CFL(NDOF, p, 0.8),
        use_cupy=True,
        time_integrator="rk4",
        scheme="SD",
        FB=False,
        riemann_solver_sd="llf",  # MUSCL fallback flux
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


for NDOF in [16, 32, 64]:
    for p in [3, 7]:
        run_superfv_sim("LLF", p, NDOF)
        run_spd_sim("LLF", p, NDOF)
