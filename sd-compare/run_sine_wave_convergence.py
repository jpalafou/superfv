import csv
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import generate_fv_runner, generate_sd_runner

from superfv import ics as superfv_ics

gamma = 5 / 3

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/sinus")

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
    x, y = xy
    if case == 0:
        return 1.5 + 0.5 * np.sin(2 * np.pi * (x + y))
    if case == 1:
        return np.full_like(x, 2.0)
    if case in (2, 4):
        return np.ones_like(x)
    return np.zeros_like(x)


def choose_cfl(p, n):
    """Scale RK4's time step to retain spatial convergence of order p + 1."""
    return 0.8 * (1 / n) ** (max(p - 3, 0) / 4)


generic_run_fv_simulation = generate_fv_runner(
    base_output_path,
    time_integrator="rk4",
    ic=superfv_density_wave,
    gamma=gamma,
    use_MOOD=True,
    positivity_guard=False,
    cupy=True,
)


def run_fv_simulation(t, *, n, p, cfl=None, **solver_config):
    cfl = choose_cfl(p, n) if cfl is None else cfl
    return generic_run_fv_simulation(t, nx=n, ny=n, p=p, CFL=cfl, **solver_config)


generic_run_sd_simulation = generate_sd_runner(
    base_output_path,
    time_integrator="rk4",
    init_fct=spd_density_wave,
    gamma=gamma,
    use_cupy=True,
)


def run_sd_simulation(t, *, n, p, riemann_solver="hllc", cfl=None, **solver_config):
    if n % (p + 1):
        raise ValueError(f"n={n} must be divisible by p + 1={p + 1}.")
    nelems = n // (p + 1)
    cfl = choose_cfl(p, n) if cfl is None else cfl
    return generic_run_sd_simulation(
        t,
        N=(nelems, nelems),
        p=p,
        cfl_coeff=cfl,
        riemann_solver_sd=riemann_solver,
        riemann_solver_fv=riemann_solver,
        **solver_config,
    )


def density_error(sim, solver):
    weights = None
    if solver == "fv":
        irho = sim.params.variable_index_map("rho")
        rho0 = sim.snapshot_history[0].u[irho]
        rho1 = sim.snapshot_history[-1].u[irho]
    elif solver == "sd":
        rho0 = sim.dm.asnumpy(sim.W_init_cv[0])
        rho1 = sim.dm.asnumpy(sim.dm.W_cv[0])
        dx = np.diff(sim.fp["x"]) * sim.h["x"]
        dy = np.diff(sim.fp["y"]) * sim.h["y"]
        # SD arrays end in (subcell_y, subcell_x), repeated over elements.
        weights = np.broadcast_to(dy[:, None] * dx[None, :], rho0.shape)
    else:
        raise ValueError(f"Unknown solver '{solver}'.")
    error = np.abs(rho1 - rho0)
    return float(np.average(error, weights=weights)), float(np.max(error))


if __name__ == "__main__":
    t = 1.0

    configs = []
    for riemann_solver, resolutions in (("hllc", (32, 64, 128)), ("llf", (32, 64))):
        configs.extend(product(("fv", "sd"), (riemann_solver,), (3, 7), resolutions))

    output_dir = Path("out/sinus")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "errors.csv"
    plot_path = output_dir / "errors.pdf"
    fieldnames = ["solver", "p", "n", "riemann_solver", "L1", "L1_rate", "Linf", "Linf_rate"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    data = []
    previous = {}
    for solver, riemann_solver, p, n in configs:
        runner = run_fv_simulation if solver == "fv" else run_sd_simulation
        if solver == "fv":
            sim = runner(t, n=n, p=p, riemann_solver=riemann_solver, scale_by_cfl=True)
        else:
            sim = runner(t, n=n, p=p, riemann_solver=riemann_solver, NAD_cfl=True)
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
        data.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    fig, ax = plt.subplots(figsize=(8, 8))
    for solver, riemann_solver, p in product(("fv", "sd"), ("hllc", "llf"), (3, 7)):
        rows = [
            row
            for row in data
            if (row["solver"], row["riemann_solver"], row["p"]) == (solver, riemann_solver, p)
        ]
        ax.plot(
            [row["n"] for row in rows],
            [row["Linf"] for row in rows],
            marker="s" if p == 3 else "o",
            linestyle="-" if riemann_solver == "hllc" else "--",
            label=f"{solver.upper()}, p={p}, {riemann_solver}",
        )
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\mathcal{L}^\infty$ error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
