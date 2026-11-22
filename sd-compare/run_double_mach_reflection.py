from dataclasses import replace
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import spd.initial_conditions as spd_ics
from common import (
    generate_fv_runner,
    generate_muscl_runner,
    generate_sd_runner,
    plot_fv_2d_density_map,
    plot_sd_2d_density_map,
    update_maxmin,
)

from superfv import ics as superfv_ics
from superfv.boundary_conditions import apply_dirichlet_bc, apply_reflective_bc
from superfv.tools.slicing import crop

gamma = 1.4

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/dmr")


def dirichlet_x0(idx, x, y, z, t, xp):
    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = 8.0
    out[idx("vx")] = 8.25 * np.cos(np.pi / 6)
    out[idx("vy")] = -4.125
    out[idx("P")] = 116.5
    return out


def dirichlet_y1(idx, x, y, z, t, xp):
    shock_x = 10 * t / xp.sin(np.pi / 3) + 1 / 6 + y / xp.tan(np.pi / 3)
    behind_shock = x < shock_x

    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = xp.where(behind_shock, 8.0, gamma)
    out[idx("vx")] = xp.where(behind_shock, 8.25 * np.cos(np.pi / 6), 0.0)
    out[idx("vy")] = xp.where(behind_shock, -4.125, 0.0)
    out[idx("P")] = xp.where(behind_shock, 116.5, 1.0)
    return out


def patch_bc(u, context):
    x = cp.asnumpy(context.mesh.Centers[0][:, 0, 0])
    i = np.max(np.where(x < 1 / 6)[0]).item() + 1 + context.nghost

    apply_dirichlet_bc(u, replace(context, f=dirichlet_x0))
    apply_reflective_bc(u[crop(1, (i, None), ndim=4)], context)


generic_run_muscl_hancock_simulation = generate_muscl_runner(
    base_output_path,
    ic=superfv_ics.double_mach_reflection,
    gamma=gamma,
    xlims=(0, 4),
    ylims=(0, 1),
    bcx=("dirichlet", "free"),
    bcy=("patch", "dirichlet"),
    bcx_callable_lower=dirichlet_x0,
    bcy_callable_lower=patch_bc,
    bcy_callable_upper=dirichlet_y1,
    cupy=True,
)


def run_muscl_hancock_simulation(t, *, nx, **solver_config):
    return generic_run_muscl_hancock_simulation(t, nx=nx, ny=nx // 4, **solver_config)


generic_run_fv_simulation = generate_fv_runner(
    base_output_path,
    ic=superfv_ics.double_mach_reflection,
    gamma=gamma,
    xlims=(0, 4),
    ylims=(0, 1),
    bcx=("dirichlet", "free"),
    bcy=("patch", "dirichlet"),
    bcx_callable_lower=dirichlet_x0,
    bcy_callable_lower=patch_bc,
    bcy_callable_upper=dirichlet_y1,
    use_MOOD=True,
    positivity_guard=False,
    cupy=True,
)


def run_fv_simulation(t, *, nx, **solver_config):
    return generic_run_fv_simulation(t, nx=nx, ny=nx // 4, **solver_config)


generic_run_sd_simulation = generate_sd_runner(
    base_output_path,
    xlim=(0.0, 4.0),
    gamma=gamma,
    BC=(
        ("doublemach", "doublemach"),
        ("doublemach", "doublemach"),
    ),
    init_fct=spd_ics.double_mach_reflection(gamma=gamma),
    use_cupy=True,
)


def run_sd_simulation(t, *, ndofx, p, **solver_config):
    nelemsx = ndofx // (p + 1)
    return generic_run_sd_simulation(t, N=(nelemsx, nelemsx // 4), p=p, **solver_config)


if __name__ == "__main__":
    t = 0.2
    N = 3200
    p = 3
    uniform_sd = True

    # run simulations

    sim1 = run_fv_simulation(t=t, nx=N, p=p, scale_by_cfl=True, rtol=1e-5)
    sim2 = run_sd_simulation(t=t, ndofx=N, p=p, NAD_cfl=True, tolerance=1e-5)

    rhomax, rhomin = update_maxmin(sim1)
    rhomax, rhomin = update_maxmin(sim2, rhomax, rhomin, uniform_sd=uniform_sd)
    # plot them

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
    axs[1].set_xlim(0, 3)
    plot_configs = dict(cmap="viridis", vmin=rhomin, vmax=rhomax)

    plot_fv_2d_density_map(axs[0], sim1, **plot_configs)

    plot_sd_2d_density_map(axs[1], sim2, uniform=uniform_sd, **plot_configs)

    fig.savefig("out/dmr/compare.png", dpi=600)
    print("Done!")
