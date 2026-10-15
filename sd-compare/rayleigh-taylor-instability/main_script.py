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
    params = f"{t=}_{ny=}_Re={Re}_P_background={P_background}_limiter={limiter}_sed={sed}"
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
        ic=partial(ics.rayleigh_taylor, gamma=5 / 3, P0=P_background),
        gamma=5 / 3,
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
    sim = run_muscl_hancock_simulation(t=t, ny=768)

    fig, ax = plt.subplots(figsize=(4, 16))
    plot_fv_density(ax, sim, cmap="magma")

    fig.savefig(output_plot, dpi=300)
