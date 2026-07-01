import os
import shutil
from functools import partial
from itertools import product

import spd.initial_conditions as ic
from spd.sdfb_simulator import SPD_Simulator

from superfv import (
    BC,
    HydroSolver,
    HydroSolverOutput,
    MUSCL_SlopeLimiter,
    RiemannSolver,
    TimeIntegrator,
    ics,
)

base_directory = "/scratch/gpfs/jp7427/FVvsSD/rti/"

gamma = 5.0 / 3.0
P0 = 1.0
target_time = 1.95


def gravity(idx, u, *, xp):
    gx = 0.0
    gy = 1.0

    out = xp.zeros_like(u)
    out[idx("mx")] = u[idx("rho")] * gx
    out[idx("my")] = u[idx("rho")] * gy
    out[idx("E")] = u[idx("mx")] * gx + u[idx("my")] * gy
    return out


def run_superfv_sim(name, p, NDOF, **kwargs):
    path = base_directory + f"FV_{name}_{P0=}_{NDOF=}_{p=}"

    try:
        out = HydroSolverOutput(path)
        print(f"Loaded output from '{path}'")
        return out
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")

    if os.path.exists(path):
        print(f"Removing bad output at '{path}'")
        shutil.rmtree(path)

    sim = HydroSolver(
        ic=partial(ics.rayleigh_taylor, gamma=gamma, P0=P0),
        gamma=gamma,
        rho_min=1e-10,
        P_min=1e-10,
        source=gravity,
        nx=NDOF // 4,
        ny=NDOF,
        xlims=(0.0, 0.25),
        ylims=(0.0, 1.0),
        p=p,
        bcy=(BC.REFLECTIVE, BC.REFLECTIVE),
        use_MOOD=True,
        use_NAD=True,
        use_SED=True,
        blend_troubles=False,
        MUSCL_limiter=MUSCL_SlopeLimiter.MONCEN,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(target_time, time_integrator=TimeIntegrator.SSPRK3)
    return sim


def run_spd_sim(name, p, NDOF, **kwargs):
    path = base_directory + f"SD_{name}_{P0=}_{NDOF=}_{p=}"
    Nelements = NDOF // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(Nelements // 4, Nelements),
        xlim=(0.0, 0.25),
        ylim=(0.0, 1.0),
        gamma=gamma,
        BC=(
            ("periodic", "periodic"),
            ("reflective", "reflective"),
        ),
        init_fct=ic.RTI(P0=P0, gamma=gamma),
        cfl_coeff={3: 0.4, 7: 0.2}[p],
        use_cupy=True,
        time_integrator="rk3",
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
        potential=True,
        limiting_variables=[0, 1, 2, 4],
        PAD=True,
        SED=True,
        NAD="",
        blending=False,
        folder=path,
        **kwargs,
    )

    try:
        sim.load_output()
        print(f"Loaded output from '{path}'")
        return sim
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        if os.path.exists(path):
            print(f"Removing bad output at '{path}'")
            shutil.rmtree(path)

    sim.output()
    sim.perform_time_evolution(target_time)
    sim.output()
    return sim


if __name__ == "__main__":
    NDOF = 768

    for p, riemann_solver in product([3, 7], ["llf", "hllc"]):
        print(f"Running FV and SD simulations for p={p}, riemann_solver={riemann_solver}")
        run_superfv_sim(
            riemann_solver + "_rtol=0",
            p,
            NDOF,
            rtol=0,
            riemann_solver=dict(llf=RiemannSolver.LLF, hllc=RiemannSolver.HLLC)[riemann_solver],
        )
        run_spd_sim(
            riemann_solver + "_rtol=0",
            p,
            NDOF,
            tolerance=0,
            riemann_solver_sd=riemann_solver,
            riemann_solver_fv=riemann_solver,
        )
