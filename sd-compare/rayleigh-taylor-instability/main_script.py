import shutil
from functools import partial
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import spd.initial_conditions as spd_ics
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput
from superfv import ics as superfv_ics

gamma = 5 / 3
t = 1.95

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/rti2")
output_plot = Path("sd-compare/rayleigh-taylor-instability/results/test.png")
output_plot.unlink(missing_ok=True)  # remove the output file if it exists


def nu_from_Re(Re, P_background):
    L = 1.0
    yc = 0.5
    rho1 = 2.0

    dv = np.sqrt(gamma * (P_background + rho1 * yc - P_background + 1) / rho1)
    vmag = 0.025 * dv

    nu = vmag * L / Re
    return nu


def run_muscl_hancock_simulation(t, ny, sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ny=}_{sed=}_{P_background=}_Re={Re}"
    path = base_output_path / ("muscl_hancock_" + params)

    try:
        sim = HydroSolverOutput(path)
        print(f"Loaded superfv output from '{path}'")
        return sim
    except Exception:
        if path.exists():
            print(f"Removing bad superfv output at '{path}'")
            shutil.rmtree(path)
        print(f"Running superfv simulation with output path '{path}'")

    def gravity(idx, u, *, xp):
        gx = 0.0
        gy = 1.0
        out = xp.zeros_like(u)
        out[idx("mx")] = u[idx("rho")] * gx
        out[idx("my")] = u[idx("rho")] * gy
        out[idx("E")] = u[idx("mx")] * gx + u[idx("my")] * gy
        return out

    nu = nu_from_Re(Re, P_background) if Re is not None else 0.0

    sim = HydroSolver(
        ic=partial(superfv_ics.rayleigh_taylor, gamma=gamma, P0=P_background),
        gamma=gamma,
        source=gravity,
        nu=nu,
        Chi=nu,
        xlims=(0, 0.25),
        ylims=(0, 1),
        nx=ny // 4,
        ny=ny,
        bcy=("reflective", "reflective"),
        p=1,
        CFL=0.8,
        use_MUSCL=True,
        use_SED=sed,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t, "muscl_hancock")

    return sim


def run_fv_simulation(t, ny, p, rtol, vx_vy_nad=True, sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ny=}_{p=}_{rtol=}_{vx_vy_nad=}_{sed=}_{P_background=}_Re={Re}"
    path = base_output_path / ("fv_" + params)

    try:
        sim = HydroSolverOutput(path)
        print(f"Loaded superfv output from '{path}'")
        return sim
    except Exception:
        if path.exists():
            print(f"Removing bad superfv output at '{path}'")
            shutil.rmtree(path)
        print(f"Running superfv simulation with output path '{path}'")

    def gravity(idx, u, *, xp):
        gx = 0.0
        gy = 1.0
        out = xp.zeros_like(u)
        out[idx("mx")] = u[idx("rho")] * gx
        out[idx("my")] = u[idx("rho")] * gy
        out[idx("E")] = u[idx("mx")] * gx + u[idx("my")] * gy
        return out

    nu = nu_from_Re(Re, P_background) if Re is not None else 0.0

    sim = HydroSolver(
        ic=partial(superfv_ics.rayleigh_taylor, gamma=gamma, P0=P_background),
        gamma=gamma,
        source=gravity,
        nu=nu,
        Chi=nu,
        xlims=(0, 0.25),
        ylims=(0, 1),
        nx=ny // 4,
        ny=ny,
        bcy=("reflective", "reflective"),
        p=p,
        CFL=0.8,
        use_MOOD=True,
        positivity_guard=False,
        omit_vars=None if vx_vy_nad else ["vx", "vy"],
        use_SED=sed,
        rtol=rtol,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t)

    return sim


def run_sd_simulation(t, ndofy, p, rtol, vx_vy_nad=True, sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ndofy=}_{p=}_{rtol=}_{vx_vy_nad=}_{sed=}_{P_background=}_Re={Re}"
    path = base_output_path / ("sd_" + params)

    nelemsy = ndofy // (p + 1)
    nu = nu_from_Re(Re, P_background) if Re is not None else 0.0

    sim = SPD_Simulator(
        p=p,
        N=(nelemsy // 4, nelemsy),
        xlim=(0.0, 0.25),
        ylim=(0.0, 1.0),
        gamma=gamma,
        viscosity=bool(nu),
        nu=nu,
        chi=nu,
        min_rho=1e-12,
        min_P=1e-12,
        BC=(
            ("periodic", "periodic"),
            ("reflective", "reflective"),
        ),
        init_fct=spd_ics.RTI(P0=P_background, gamma=gamma),
        cfl_coeff={3: 0.8, 7: 0.5}[p],
        use_cupy=True,
        time_integrator="rk3",
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
        potential=True,
        NAD="",
        tolerance=rtol,
        limiting_variables=[0, 1, 2, 4] if vx_vy_nad else [0, 4],
        PAD=True,
        SED=sed,
        blending=False,
        folder=str(path),
    )

    try:
        sim.load_output()
        print(f"Loaded spd output from '{path}'")
        return sim
    except Exception:
        if path.exists():
            print(f"Removing bad spd output at '{path}'")
            shutil.rmtree(path)
        print(f"Running spd simulation with output path '{path}'")

    sim.output()
    sim.perform_time_evolution(t)
    sim.output()
    return sim


def plot_fv_density(ax, sim, i=-1, **kwargs):
    x = sim.mesh.faces[0]
    y = sim.mesh.faces[1]
    rho = sim.snapshot_history[i].u[sim.params.variable_index_map("rho"), :, :, 0]

    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    if isinstance(y, cp.ndarray):
        y = cp.asnumpy(y)

    ax.pcolormesh(x, y, rho.T, shading="flat", **kwargs)
    ax.set_aspect("equal")


def get_sd_density_array(sim, uniform=False):
    rho = sim.dm.W_cv[0:1]
    if uniform:
        x, y = sim.regular_faces()
        if sim.p > 1:
            # regular_mesh expects point values rather than cell averages.
            rho = sim.regular_mesh(sim.compute_sp_from_cv(rho))
    else:
        ngh = sim.Nghc
        x = sim.faces["x"][ngh:-ngh]
        y = sim.faces["y"][ngh:-ngh]
    rho = sim.transpose_to_fv(rho)[0]
    return rho, x, y


def plot_sd_density(ax, sim, uniform=False, **kwargs):
    rho, x, y = get_sd_density_array(sim, uniform=uniform)

    x, y, rho = (sim.dm.asnumpy(a) for a in (x, y, rho))
    ax.pcolormesh(x, y, rho, shading="flat", **kwargs)
    ax.set_aspect("equal")


def get_fv_rho_maxmin(sim):
    rho = sim.snapshot_history[-1].u[sim.params.variable_index_map("rho"), :, :, 0]
    return (rho.max(), rho.min())


def get_sd_rho_maxmin(sim, uniform=False):
    rho, _, _ = get_sd_density_array(sim, uniform=uniform)
    return (rho.max(), rho.min())


def get_simulation_rho_maxmin(sims, uniform_sd=False):
    maxmins = [
        (
            get_fv_rho_maxmin(sim)
            if isinstance(sim, HydroSolver) or isinstance(sim, HydroSolverOutput)
            else get_sd_rho_maxmin(sim, uniform_sd)
        )
        for sim in sims
    ]
    rhomax = max(x[0] for x in maxmins)
    rhomin = min(x[1] for x in maxmins)
    return rhomax, rhomin


if __name__ == "__main__":
    N = 768

    sim1 = run_muscl_hancock_simulation(t=t, ny=N)
    sim2 = run_fv_simulation(t=t, ny=N, p=3, rtol=1e-5)
    sim3 = run_fv_simulation(t=t, ny=N, p=3, rtol=1e-3)
    sim4 = run_fv_simulation(t=t, ny=N, p=3, rtol=1e-1)
    sim5 = run_sd_simulation(t=t, ndofy=N, p=3, rtol=1e-5)
    sim6 = run_sd_simulation(t=t, ndofy=N, p=3, rtol=1e-3)
    sim7 = run_sd_simulation(t=t, ndofy=N, p=3, rtol=1e-1)
    sim_list = [sim1, sim2, sim3, sim4, sim5, sim6, sim7]

    uniform_sd = True

    print("Computing global min/max density for color scaling...")
    rhomax, rhomin = get_simulation_rho_maxmin(sim_list, uniform_sd=uniform_sd)
    print(f"Global density range: {rhomin:.3f} to {rhomax:.3f}")

    print("Plotting results...")
    fig, axs = plt.subplots(2, 4, figsize=(8, 16), sharex=True, sharey=True)

    axs[0, 0].set_title("MUSCL-Hancock")
    plot_fv_density(axs[0, 0], sim1, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[0, 1].set_title(r"FV4, $\epsilon=10^{-5}$")
    plot_fv_density(axs[0, 1], sim2, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[0, 2].set_title(r"FV4, $\epsilon=10^{-3}$")
    plot_fv_density(axs[0, 2], sim3, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[0, 3].set_title(r"FV4, $\epsilon=10^{-1}$")
    plot_fv_density(axs[0, 3], sim4, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[1, 1].set_title(r"SD4, $\epsilon=10^{-5}$")
    plot_sd_density(axs[1, 1], sim5, uniform=uniform_sd, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[1, 2].set_title(r"SD4, $\epsilon=10^{-3}$")
    plot_sd_density(axs[1, 2], sim6, uniform=uniform_sd, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    axs[1, 3].set_title(r"SD4, $\epsilon=10^{-1}$")
    plot_sd_density(axs[1, 3], sim7, uniform=uniform_sd, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    print("Saving figure...")
    fig.savefig(output_plot, dpi=300)

    print("Done!")
