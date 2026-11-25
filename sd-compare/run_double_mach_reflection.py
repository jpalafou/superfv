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
    plot_fv_fallback_history,
    plot_sd_fallback_history,
)

from superfv import ics as superfv_ics
from superfv.boundary_conditions import apply_dirichlet_bc, apply_reflective_bc
from superfv.tools.slicing import crop

gamma = 1.4

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/dmr2")


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
    cfl_nad = True
    sd_cfl_lookup = {3: 0.7, 7: 0.4}  # {3: 0.4, 7: 0.2}
    # p = 3
    # rtol=1e-3
    # uniform_sd = True
    # sd_cfl=0.4

    output_dir = Path("out/dmr")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(
        output_dir
        / f"fallback_compare_{cfl_nad=}_sd_cfl_p=3_{sd_cfl_lookup[3]}_p=7_{sd_cfl_lookup[7]}_volume_fraction.pdf"
    )

    history_fig, history_axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 4))
    history_axs[0, 0].set_title("FV")
    history_axs[0, 1].set_title("SD")
    history_axs[0, 0].set_ylabel(r"Fallback fraction/n$p=3$")
    history_axs[1, 0].set_ylabel(r"Fallback fraction/n$p=7$")
    for ax in history_axs[1, :]:
        ax.set_xlabel(r"$t$")
    for ax in history_axs[:, 0]:
        ax.set_yscale("log")

    for i, p in enumerate([3, 7]):
        for tol in [1e-5, 1e-4, 1e-3]:
            color = {1e-5: "tab:blue", 1e-4: "tab:orange", 1e-3: "tab:green"}[tol]

            sim = run_fv_simulation(
                t=t, nx=N, p=p, scale_by_cfl=cfl_nad, rtol=tol, omit_vars=["vx", "vy", "vz"]
            )
            plot_fv_fallback_history(
                history_axs[i, 0], sim, n_avg=100, label=f"{tol=}", color=color
            )

            sim = run_sd_simulation(
                t=t,
                ndofx=N,
                p=p,
                cfl_coeff=sd_cfl_lookup[p],
                NAD_cfl=cfl_nad,
                tolerance=tol,
                limiting_variables=[0, 4],
            )
            plot_sd_fallback_history(
                history_axs[i, 1],
                sim,
                n_avg=100,
                label=f"{tol=:.0e} not volume weighted",
                volume_weighted=False,
                color=color,
            )
            plot_sd_fallback_history(
                history_axs[i, 1], sim, n_avg=100, label=f"{tol=:.0e}", color=color, linestyle="--"
            )

    history_axs[0, 0].legend()
    for ax in history_axs.flatten():
        ax.grid()
    history_fig.tight_layout()
    history_fig.savefig(plot_path)
    plt.close(history_fig)

    # # run simulations

    # sim1 = run_fv_simulation(t=t, nx=N, p=p, scale_by_cfl=True, rtol=rtol, omit_vars=["vx", "vy", "vz"])
    # sim2 = run_sd_simulation(t=t, ndofx=N, p=p, NAD_cfl=True, tolerance=rtol, cfl_coeff=sd_cfl,limiting_variables=[0, 4])

    # rhomax, rhomin = update_maxmin(sim1)
    # rhomax, rhomin = update_maxmin(sim2, rhomax, rhomin, uniform_sd=uniform_sd)
    # # plot them

    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
    # axs[1].set_xlim(0, 3)
    # plot_configs = dict(cmap="viridis", vmin=rhomin, vmax=rhomax)

    # plot_fv_2d_density_map(axs[0], sim1, **plot_configs)

    # plot_sd_2d_density_map(axs[1], sim2, uniform=uniform_sd, **plot_configs)

    # output_dir = Path("out/dmr")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output_dir / "compare.png", dpi=600)

    # history_fig, history_ax = plt.subplots(figsize=(8, 4))
    # plot_fv_fallback_history(history_ax, sim1, n_avg=100, label="FV")
    # plot_sd_fallback_history(history_ax, sim2, n_avg=100, label="SD")
    # history_ax.set_xlabel(r"$t$")
    # history_ax.set_ylabel("Fallback fraction")
    # history_ax.set_yscale("log")
    # history_ax.legend()
    # history_fig.tight_layout()
    # history_fig.savefig(output_dir / "fallback_compare.pdf")
    # plt.close(history_fig)

    # trouble_fig, trouble_axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
    # trouble_axs[1].set_xlim(0, 3)
    # trouble_configs = dict(cmap="binary", vmin=0, vmax=1)
    # plot_fv_2d_trouble_map(trouble_axs[0], sim1, **trouble_configs)
    # plot_sd_2d_trouble_map(trouble_axs[1], sim2, uniform=False, **trouble_configs)
    # trouble_fig.savefig(output_dir / "trouble.png", dpi=600)
    # plt.close(trouble_fig)
    # print("Done!")
