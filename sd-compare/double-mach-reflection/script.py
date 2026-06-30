import os
import shutil
from itertools import product

import numpy as np
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
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.tools.slicing import crop

base_directory = "/scratch/gpfs/jp7427/FVvsSD/double-mach-reflection/"

gamma = 1.4
shock_foot = 1.0 / 6.0
shock_angle = np.pi / 3.0
target_time = 0.2


def post_shock_state(idx, x, y, z, t, *, xp):
    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = 8.0
    out[idx("vx")] = 8.25 * np.cos(np.pi / 6.0)
    out[idx("vy")] = -8.25 * np.sin(np.pi / 6.0)
    out[idx("P")] = 116.5
    return out


def moving_shock_state(idx, x, y, z, t, *, xp):
    shock_x = 10.0 * t / np.sin(shock_angle) + shock_foot + y / np.tan(shock_angle)
    behind_shock = x < shock_x

    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = xp.where(behind_shock, 8.0, gamma)
    out[idx("vx")] = xp.where(behind_shock, 8.25 * np.cos(np.pi / 6.0), 0.0)
    out[idx("vy")] = xp.where(behind_shock, -8.25 * np.sin(np.pi / 6.0), 0.0)
    out[idx("P")] = xp.where(behind_shock, 116.5, 1.0)
    return out


def bottom_boundary(u, context):
    if context.mesh is None:
        raise ValueError("The Double Mach Reflection patch requires a mesh.")

    x = context.mesh.centers[0]
    split = int((x < shock_foot).sum().item()) + context.nghost - 1

    inflow_section = crop(1, (None, split), ndim=4)
    wall_section = crop(1, (split, None), ndim=4)

    apply_free_bc(u[inflow_section], context)
    apply_reflective_bc(u[wall_section], context)


def run_superfv_sim(name, p, NDOF, **kwargs):
    path = base_directory + f"FV_{name}_{NDOF=}_{p=}"

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
        ic=ics.double_mach_reflection,
        gamma=gamma,
        rho_min=1e-10,
        P_min=1e-10,
        nx=NDOF,
        ny=NDOF // 4,
        xlims=(0.0, 4.0),
        ylims=(0.0, 1.0),
        p=p,
        bcx=(BC.DIRICHLET, BC.FREE),
        bcy=(BC.PATCH, BC.DIRICHLET),
        bcx_callable_lower=post_shock_state,
        bcy_callable_lower=bottom_boundary,
        bcy_callable_upper=moving_shock_state,
        use_MOOD=True,
        MUSCL_limiter=MUSCL_SlopeLimiter.MONCEN,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(target_time, time_integrator=TimeIntegrator.SSPRK3)
    return sim


def run_spd_sim(name, p, NDOF, **kwargs):
    path = base_directory + f"SD_{name}_{NDOF=}_{p=}"
    Nelements = NDOF // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(Nelements, Nelements // 4),
        xlim=(0.0, 4.0),
        ylim=(0.0, 1.0),
        gamma=gamma,
        BC=(("doublemach", "doublemach"), ("doublemach", "doublemach")),
        init_fct=ic.double_mach_reflection(),
        cfl_coeff={3: 0.4, 7: 0.2}[p],
        use_cupy=True,
        time_integrator="rk3",
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
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
    NDOF = 3200

    for p, riemann_solver in product([3, 7], ["llf", "hllc"]):
        print(f"Running FV and SD simulations for p={p}, riemann_solver={riemann_solver}")
        run_superfv_sim(
            riemann_solver + "_rtol=1e-1",
            p,
            NDOF,
            rtol=1e-1,
            riemann_solver=dict(llf=RiemannSolver.LLF, hllc=RiemannSolver.HLLC)[riemann_solver],
        )
        run_spd_sim(
            riemann_solver + "_rtol=1e-1",
            p,
            NDOF,
            tolerance=1e-1,
            riemann_solver_sd=riemann_solver,
            riemann_solver_fv=riemann_solver,
        )
