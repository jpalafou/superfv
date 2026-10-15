from functools import partial
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from superfv import HydroSolver, HydroSolverOutput, ics

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


def run_muscl_hancock_simulation(t, ny, limiter="pp2d", sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ny=}_{limiter=}_{sed=}_{P_background=}_Re={Re}"
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
        ic=partial(ics.rayleigh_taylor, gamma=gamma, P0=P_background),
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
        MUSCL_limiter=limiter,
        cupy=True,
        output_path=path,
    )
    sim.run(t, "muscl_hancock")

    return sim


def run_fv_simulation(t, ny, p, rtol, MUSCL_limiter="pp2d", sed=True, P_background=1.0, Re=None):
    params = f"{t=}_{ny=}_{p=}_{rtol=}_{MUSCL_limiter=}_{sed=}_{P_background=}_Re={Re}"
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
        ic=partial(ics.rayleigh_taylor, gamma=gamma, P0=P_background),
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
        use_SED=sed,
        rtol=rtol,
        MUSCL_limiter=MUSCL_limiter,
        cupy=True,
        output_path=path,
    )
    sim.run(t)

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
