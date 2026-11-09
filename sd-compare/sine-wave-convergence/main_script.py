import csv
import shutil
from functools import partial
from pathlib import Path

import numpy as np
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput
from superfv import ics as superfv_ics

gamma = 5 / 3
base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/sine-wave-convergence")

superfv_density_wave = partial(
    superfv_ics.sinus,
    rho_min=1.0,
    rho_max=2.0,
    vx=2.0,
    vy=1.0,
    vz=0.0,
    P=1.0,
)


def spd_density_wave(xy, case):
    """Return the 2D density-wave primitive state using SPD's field indexing."""
    x, y = xy
    if case == 0:
        return 1.5 + 0.5 * np.sin(2 * np.pi * (x + y))
    if case == 1:
        return np.full_like(x, 2.0)
    if case in (2, 4):
        return np.ones_like(x)
    return np.zeros_like(x)


def choose_cfl(p, n, time_integrator="rk4"):
    """Scale RK4's time step so its temporal error converges like order ``p + 1``."""
    cfl = 0.8
    if time_integrator.lower() == "rk4" and p > 3:
        h = 1 / n
        cfl *= h ** ((p - 3) / 4)
    return cfl


def run_fv_simulation(
    t,
    n,
    p,
    riemann_solver="hllc",
    time_integrator="rk4",
    rtol=1e-5,
    sed=True,
    nad_cfl=True,
    cfl=None,
):
    time_integrator = time_integrator.lower()
    riemann_solver = riemann_solver.lower()
    cfl = choose_cfl(p, n, time_integrator) if cfl is None else cfl
    params = (
        f"t={t}_n={n}_p={p}_riemann={riemann_solver}_integrator={time_integrator}_"
        f"rtol={rtol}_sed={sed}_nad_cfl={nad_cfl}_"
        f"cfl={cfl:.12g}"
    )
    path = base_output_path / ("fv_" + params)

    try:
        sim = HydroSolverOutput(path)
    except Exception:
        if path.exists():
            print(f"Removing bad superfv output at '{path}'")
            shutil.rmtree(path)
        print(f"Running superfv simulation with output path '{path}'")
    else:
        print(f"Loaded superfv output from '{path}'")
        return sim

    sim = HydroSolver(
        ic=superfv_density_wave,
        gamma=gamma,
        xlims=(0, 1),
        ylims=(0, 1),
        nx=n,
        ny=n,
        bcx=("periodic", "periodic"),
        bcy=("periodic", "periodic"),
        p=p,
        CFL=cfl,
        riemann_solver=riemann_solver,
        use_MOOD=True,
        positivity_guard=False,
        use_SED=sed,
        rtol=rtol,
        scale_by_cfl=nad_cfl,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t, time_integrator)
    return sim


def run_sd_simulation(
    t,
    n,
    p,
    riemann_solver="hllc",
    time_integrator="rk4",
    rtol=1e-5,
    sed=True,
    nad_cfl=True,
    cfl=None,
):
    if n % (p + 1):
        raise ValueError(f"n={n} must be divisible by p + 1={p + 1}.")

    time_integrator = time_integrator.lower()
    riemann_solver = riemann_solver.lower()
    cfl = choose_cfl(p, n, time_integrator) if cfl is None else cfl
    params = (
        f"t={t}_n={n}_p={p}_riemann={riemann_solver}_integrator={time_integrator}_"
        f"rtol={rtol}_sed={sed}_nad_cfl={nad_cfl}_"
        f"cfl={cfl:.12g}"
    )
    path = base_output_path / ("sd_" + params)
    nelems = n // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(nelems, nelems),
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        gamma=gamma,
        min_rho=1e-12,
        min_P=1e-12,
        BC=(("periodic", "periodic"), ("periodic", "periodic")),
        init_fct=spd_density_wave,
        cfl_coeff=cfl,
        use_cupy=True,
        time_integrator=time_integrator,
        scheme="SDFB",
        fallback="MUSCL",
        riemann_solver_sd=riemann_solver,
        riemann_solver_fv=riemann_solver,
        slope_limiter="moncen",
        NAD="",
        tolerance=rtol,
        NAD_cfl=nad_cfl,
        limiting_variables=[0, 1, 2, 4],
        PAD=True,
        SED=sed,
        blending=False,
        folder=str(path),
    )

    try:
        sim.load_output()
    except Exception:
        if path.exists():
            print(f"Removing bad spd output at '{path}'")
            shutil.rmtree(path)
        print(f"Running spd simulation with output path '{path}'")
    else:
        print(f"Loaded spd output from '{path}'")
        return sim

    sim.output()
    sim.perform_time_evolution(t)
    sim.output()
    return sim


def density_error(sim, solver):
    if solver == "fv":
        irho = sim.params.variable_index_map("rho")
        rho0 = sim.snapshot_history[0].u[irho]
        rho1 = sim.snapshot_history[-1].u[irho]
    elif solver == "sd":
        rho0 = sim.dm.asnumpy(sim.W_init_cv[0])
        rho1 = sim.dm.asnumpy(sim.dm.W_cv[0])
    else:
        raise ValueError(f"Unknown solver '{solver}'.")

    error = np.abs(rho1 - rho0)
    return float(np.mean(error)), float(np.max(error))


if __name__ == "__main__":
    t = 1.0
    resolutions = [32, 64, 128]
    polynomial_degrees = [3, 7]
    riemann_solvers = ["hllc"]
    time_integrator = "rk4"

    output_dir = Path("out/sine-wave-convergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "errors.csv"
    fieldnames = ["solver", "p", "n", "riemann_solver", "L1", "L1_rate", "Linf", "Linf_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    previous = {}
    for riemann_solver in riemann_solvers:
        for solver in ("fv", "sd"):
            for p in polynomial_degrees:
                for n in resolutions:
                    runner = run_fv_simulation if solver == "fv" else run_sd_simulation
                    sim = runner(
                        t=t,
                        n=n,
                        p=p,
                        riemann_solver=riemann_solver,
                        time_integrator=time_integrator,
                    )
                    L1, Linf = density_error(sim, solver)
                    key = (solver, p, riemann_solver)
                    row = dict(
                        solver=solver,
                        p=p,
                        n=n,
                        riemann_solver=riemann_solver,
                        L1=L1,
                        L1_rate="",
                        Linf=Linf,
                        Linf_rate="",
                    )
                    if key in previous:
                        old = previous[key]
                        scale = np.log(n / old["n"])
                        if L1 > 0 and old["L1"] > 0:
                            row["L1_rate"] = np.log(old["L1"] / L1) / scale
                        if Linf > 0 and old["Linf"] > 0:
                            row["Linf_rate"] = np.log(old["Linf"] / Linf) / scale
                    previous[key] = row
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
