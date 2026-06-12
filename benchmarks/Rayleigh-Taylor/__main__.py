from functools import partial

import cupy as cp
import matplotlib.pyplot as plt

from superfv import BC, HydroSolver, MUSCL_SlopeLimiter, TimeIntegrator, ics
from superfv.tools.run_helper import run_multiple_simulations

N = 768
P0 = 1.0

base_path = f"/scratch/gpfs/jp7427/superfv/Rayleigh-Taylor/{P0=}/{N=}/"
overwrite = False


def gravity(idx, u, *, xp):
    gx = 0.0
    gy = 1.0
    out = xp.zeros_like(u)
    out[idx("mx")] = u[idx("rho")] * gx
    out[idx("my")] = u[idx("rho")] * gy
    out[idx("E")] = u[idx("mx")] * gx + u[idx("my")] * gy
    return out


init_params = dict(
    ic=partial(ics.rayleigh_taylor, gamma=5 / 3, P0=P0),
    gamma=5 / 3,
    source=gravity,
    xlims=(0, 0.25),
    ylims=(0, 1),
    nx=N // 4,
    ny=N,
    bcy=(BC.REFLECTIVE, BC.REFLECTIVE),
    cupy=True,
)
run_params = dict(t=1.95)

schemes = {
    "MUSCL-Hancock": dict(p=1, use_MUSCL=True, MUSCL_limiter=MUSCL_SlopeLimiter.PP2D),
    "MM4_rtol=0": dict(p=3, use_MOOD=True, rtol=0),
    "MM8_rtol=0": dict(p=7, use_MOOD=True, rtol=0),
    "MM4_rtol=1e-7": dict(p=3, use_MOOD=True, rtol=1e-7),
    "MM8_rtol=1e-7": dict(p=7, use_MOOD=True, rtol=1e-7),
    "MM4_rtol=1e-5": dict(p=3, use_MOOD=True, rtol=1e-5),
    "MM8_rtol=1e-5": dict(p=7, use_MOOD=True, rtol=1e-5),
    "MM4_rtol=1e-3": dict(p=3, use_MOOD=True, rtol=1e-3),
    "MM8_rtol=1e-3": dict(p=7, use_MOOD=True, rtol=1e-3),
    "MM4_rtol=1e-1": dict(p=3, use_MOOD=True, rtol=1e-1),
    "MM8_rtol=1e-1": dict(p=7, use_MOOD=True, rtol=1e-1),
}


def plot_density(name: str, sim: HydroSolver):
    if sim.params.output_path is None:
        return

    print(f"Plotting density for simulation: {name}")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    x_faces, y_faces, _ = sim.mesh.faces

    ax.pcolormesh(
        cp.asnumpy(x_faces),
        cp.asnumpy(y_faces),
        sim.snapshot_history[-1].u[sim.idx("rho"), :, :, 0].T,
    )
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label(r"$\rho$")

    fig.savefig(f"{sim.params.output_path}/rho.png", dpi=300, bbox_inches="tight")


run_multiple_simulations(
    configs={
        name: (
            scheme_params | init_params,
            run_params
            | dict(
                time_integrator=(
                    TimeIntegrator.MUSCL_HANCOCK
                    if name == "MUSCL-Hancock"
                    else TimeIntegrator.SSPRK3
                )
            ),
        )
        for name, scheme_params in schemes.items()
    },
    base_path=base_path,
    overwrite=overwrite,
    postprocess=plot_density,
)
