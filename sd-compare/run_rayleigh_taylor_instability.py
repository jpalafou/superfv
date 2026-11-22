import gc
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spd.initial_conditions as spd_ics
from common import (
    generate_fv_runner,
    generate_muscl_runner,
    generate_sd_runner,
    plot_fv_2d_density_map,
    plot_fv_fallback_history,
    plot_sd_2d_density_map,
    plot_sd_fallback_history,
    update_maxmin,
)
from matplotlib.colors import Normalize

from superfv import ics as superfv_ics

gamma = 5 / 3

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/rti")


def gravity(idx, u, *, xp):
    out = xp.zeros_like(u)
    out[idx("my")] = u[idx("rho")]
    out[idx("E")] = u[idx("my")]
    return out


def run_muscl_hancock_simulation(t, *, ny, P_background=1.0, **solver_config):
    generic_run_muscl_hancock_simulation = generate_muscl_runner(
        base_output_path / f"{P_background=}",
        ic=partial(superfv_ics.rayleigh_taylor, gamma=gamma, P0=P_background),
        gamma=gamma,
        source=gravity,
        xlims=(0, 0.25),
        bcy=("reflective", "reflective"),
        cupy=True,
    )
    return generic_run_muscl_hancock_simulation(t, nx=ny // 4, ny=ny, **solver_config)


def run_fv_simulation(t, *, ny, P_background=1.0, **solver_config):
    generic_run_fv_simulation = generate_fv_runner(
        base_output_path / f"{P_background=}",
        ic=partial(superfv_ics.rayleigh_taylor, gamma=gamma, P0=P_background),
        gamma=gamma,
        source=gravity,
        xlims=(0, 0.25),
        bcy=("reflective", "reflective"),
        use_MOOD=True,
        positivity_guard=False,
        cupy=True,
    )
    return generic_run_fv_simulation(t, nx=ny // 4, ny=ny, **solver_config)


def run_sd_simulation(t, *, ndofy, p, P_background=1.0, **solver_config):
    generic_run_sd_simulation = generate_sd_runner(
        base_output_path / f"{P_background=}",
        xlim=(0, 0.25),
        gamma=gamma,
        BC=(
            ("periodic", "periodic"),
            ("reflective", "reflective"),
        ),
        init_fct=spd_ics.RTI(P0=P_background, gamma=gamma),
        potential=True,
        use_cupy=True,
    )
    nelemsy = ndofy // (p + 1)
    return generic_run_sd_simulation(t, N=(nelemsy // 4, nelemsy), p=p, **solver_config)


if __name__ == "__main__":
    t = 1.95
    N = 192
    degrees = (3, 7)
    rtols = (1e-5, 1e-3, 1e-1)
    pressures = (1.0, 100.0)
    sd_cfl = {3: 0.4, 7: 0.2}
    nad_cfl = True
    uniform_sd = True

    # Share the density color scale and finalize its limits before saving.
    density_norm = Normalize()
    density_figures = {}
    rhomax, rhomin = -np.inf, np.inf
    plot_configs = dict(cmap="afmhot", norm=density_norm)
    plot_output_path = Path("out/rti")
    plot_output_path.mkdir(parents=True, exist_ok=True)
    n_avg = 100

    for P_background in pressures:
        fig, axs = plt.subplots(2, 6, figsize=(15, 15), sharex=True, sharey=True)
        axs[0, 0].set_ylabel("FV")
        axs[1, 0].set_ylabel("SD")

        history_fig, history_axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
        history_axs[0, 0].set_title("FV")
        history_axs[0, 1].set_title("SD")
        history_axs[1, 0].set_xlabel(r"$t$")
        history_axs[1, 1].set_xlabel(r"$t$")
        history_axs[0, 0].set_yscale("log")

        for i, p in enumerate(degrees):
            history_axs[i, 0].set_ylabel(rf"$p={p}$")
            for j, rtol in enumerate(rtols):
                col = i * len(rtols) + j
                exponent = int(np.log10(rtol))
                axs[0, col].set_title(rf"$p={p}$, $\epsilon=10^{{{exponent}}}$")
                for row, solver in enumerate(("fv", "sd")):
                    if solver == "fv":
                        sim = run_fv_simulation(
                            t,
                            ny=N,
                            p=p,
                            P_background=P_background,
                            rtol=rtol,
                            scale_by_cfl=nad_cfl,
                        )
                        plot_fv_2d_density_map(axs[row, col], sim, **plot_configs)
                        plot_history = plot_fv_fallback_history
                    else:
                        sim = run_sd_simulation(
                            t,
                            ndofy=N,
                            p=p,
                            P_background=P_background,
                            tolerance=rtol,
                            NAD_cfl=nad_cfl,
                            cfl_coeff=sd_cfl[p],
                        )
                        plot_sd_2d_density_map(
                            axs[row, col], sim, uniform=uniform_sd, **plot_configs
                        )
                        plot_history = plot_sd_fallback_history
                    rhomax, rhomin = update_maxmin(sim, rhomax, rhomin, uniform_sd=uniform_sd)
                    plot_history(
                        history_axs[i, row],
                        sim,
                        n_avg=n_avg,
                        label=rf"$\epsilon=10^{{{exponent}}}$",
                    )
                    del sim
                    gc.collect()

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        density_figures[P_background] = fig
        history_axs[0, 0].legend()
        history_fig.savefig(
            plot_output_path / f"fallback_history_{N=}_{P_background=}_{nad_cfl=}.png",
            dpi=300,
        )
        plt.close(history_fig)

    print(f"Global density range: {rhomin:.3f} to {rhomax:.3f}")
    density_norm.vmin = rhomin
    density_norm.vmax = rhomax
    for P_background, fig in density_figures.items():
        fig.savefig(plot_output_path / f"{N=}_{P_background=}_{nad_cfl=}.png", dpi=300)
        plt.close(fig)

    print("Done!")
