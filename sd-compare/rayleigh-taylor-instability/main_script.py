import os
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

base_output_path = Path("/scratch/gpfs/jp7427/scratch/gpfs/jp7427/FVvsSD/rti2")
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

    if path.exists():
        return HydroSolverOutput(path)

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


def run_fv_simulation(t, ny, p, rtol, omit_vars=None, sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ny=}_{p=}_{rtol=}_{omit_vars=}_{sed=}_{P_background=}_Re={Re}"
    path = base_output_path / ("fv_" + params)

    if path.exists():
        return HydroSolverOutput(path)

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
        omit_vars=omit_vars,
        use_SED=sed,
        rtol=rtol,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t)

    return sim


def run_sd_simulation(
    t, ndofy, p, rtol, limiting_variables=[0, 1, 2, 4], sed=True, P_background=1.0, Re=None
):
    params = f"{t=}_{ndofy=}_{p=}_{rtol=}_{sed=}_{P_background=}_Re={Re}"
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
        limiting_variables=limiting_variables,
        PAD=True,
        SED=sed,
        blending=False,
        folder=path,
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


def plot_sd_density(ax, sim, uniform=False, **kwargs):
    """Plot SD cell averages, or reconstructed density on a uniform grid."""
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

    x, y, rho = (sim.dm.asnumpy(a) for a in (x, y, rho))
    ax.pcolormesh(x, y, rho, shading="flat", **kwargs)
    ax.set_aspect("equal")


if __name__ == "__main__":
    sim1 = run_muscl_hancock_simulation(t=t, ny=768)
    sim2 = run_fv_simulation(t=t, ny=768, p=3, rtol=1e-5)
    sim3 = run_fv_simulation(t=t, ny=768, p=3, rtol=1e-3)
    sim4 = run_fv_simulation(t=t, ny=768, p=3, rtol=1e-1)

    rhomin = min(
        sim.snapshot_history[-1].u[sim1.params.variable_index_map("rho")].min()
        for sim in [sim1, sim2, sim3, sim4]
    )
    rhomax = max(
        sim.snapshot_history[-1].u[sim1.params.variable_index_map("rho")].max()
        for sim in [sim1, sim2, sim3, sim4]
    )

    fig, axs = plt.subplots(1, 4, figsize=(16, 16), sharex=True, sharey=True)
    plot_fv_density(axs[0], sim1, cmap="afmhot", vmin=rhomin, vmax=rhomax)
    plot_fv_density(axs[1], sim2, cmap="afmhot", vmin=rhomin, vmax=rhomax)
    plot_fv_density(axs[2], sim3, cmap="afmhot", vmin=rhomin, vmax=rhomax)
    plot_fv_density(axs[3], sim4, cmap="afmhot", vmin=rhomin, vmax=rhomax)

    fig.savefig(output_plot, dpi=300)
