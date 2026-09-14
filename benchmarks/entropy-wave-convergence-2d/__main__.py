import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv import run_multiple_simulations
from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence-2d/"
plot_path = "benchmarks/entropy-wave-convergence-2d/entropy-wave-convergence-2d.pdf"
overwrite = False

gamma = 5 / 3
run_params = dict(t=1.0)
init_params = dict(
    ic=partial(entropy_wave, gamma=gamma),
    gamma=gamma,
    use_SED=True,
    skip_trouble_counts=True,
    cupy=True,
)

# Loop parameters
resolutions = [32, 64, 128]

musclhancock = dict(p=1, use_MUSCL=True, MUSCL_limiter="pp2d")
apriori = dict(use_ZS=True, lazy_primitive_mode="adaptive", adaptive_dt=True)
aposteriori = dict(use_MOOD=True, lazy_primitive_mode="full", MUSCL_limiter="pp2d")
aposteriori_1rev = dict(fallback_cascade="muscl", max_revs=1, **aposteriori)
aposteriori_2revs = dict(fallback_cascade="muscl0", max_revs=2, **aposteriori)
aposteriori_3revs = dict(fallback_cascade="muscl0", max_revs=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock,
    "ZS3": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "ZS3lazy": dict(
        p=3, flux_quadrature="gauss_legendre", **(apriori | dict(lazy_primitive_mode="full"))
    ),
    "ZS7lazy": dict(
        p=7, flux_quadrature="gauss_legendre", **(apriori | dict(lazy_primitive_mode="full"))
    ),
    "MM3/1rev/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_0": dict(p=3, rtol=0, **aposteriori_1rev),
    "MM7/1rev/rtol_0": dict(p=7, rtol=0, **aposteriori_1rev),
}

markersize = 8
styles = {
    "MUSCL-RK3": dict(color="grey", marker="s", mfc="none", markersize=markersize),
    "MUSCL-Hancock": dict(color="grey", marker="o", mfc="none", markersize=markersize),
    "ZS3": dict(color="blue", marker="o", mfc="none", markersize=markersize, label="ZS4"),
    "ZS3lazy": dict(
        color="green", marker="o", mfc="none", markersize=markersize, label="ZS4, lazy"
    ),
    "MM3/1rev/rtol_1e-1": dict(
        color="blue",
        marker="s",
        mfc="none",
        markersize=markersize,
        label=r"MM4, $\epsilon=10^{-1}$",
    ),
    "MM3/1rev/rtol_0": dict(
        color="green",
        marker="s",
        mfc="none",
        markersize=markersize,
        label=r"MM4, $\epsilon=0$",
    ),
    "ZS7": dict(color="red", marker="o", mfc="none", markersize=markersize, label="ZS8"),
    "ZS7lazy": dict(
        color="purple",
        linestyle="--",
        marker="o",
        mfc="none",
        markersize=markersize,
        label="ZS8, lazy",
    ),
    "MM7/1rev/rtol_1e-1": dict(
        color="red",
        marker="s",
        mfc="none",
        markersize=markersize,
        label=r"MM8, $\epsilon=10^{-1}$",
    ),
    "MM7/1rev/rtol_0": dict(
        color="purple",
        linestyle="--",
        marker="s",
        mfc="none",
        markersize=markersize,
        label=r"MM8, $\epsilon=0$",
    ),
}


data = []


def plot_error(name, sim):
    # parse name without resolution
    name = name.split("/N_")[0]

    # measure error
    idx = sim.params.variable_index_map
    vz0 = sim.snapshot_history[0].w[idx("vz")]
    vz1 = sim.snapshot_history[-1].w[idx("vz")]
    error = linf_norm(vz1 - vz0)

    # update dataframe
    data.append(dict(name=name, N=sim.mesh.nx, error=error))
    df = pd.DataFrame(data)

    # plot error curves of p over N
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\mathcal{L}^\infty$ error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid()

    for name in configs.keys():
        df_name = df[df["name"] == name]

        if df_name.empty:
            continue

        style = styles.get(name, dict())
        if "label" not in style:
            style["label"] = name
        ax.plot(
            df_name["N"],
            df_name["error"],
            **(dict(markersize=5, linewidth=2, alpha=0.7) | style),
        )
    ax.legend()
    fig.savefig(plot_path, bbox_inches="tight")


# remove old output
if os.path.exists(plot_path):
    os.remove(plot_path)

# loop over all configs and resolutions
run_multiple_simulations(
    {
        f"{name}/N_{N}/": (
            dict(nx=N, ny=N, **init_params, **config),
            dict(
                time_integrator=(
                    "ssprk3"
                    if "RK3" in name
                    else (
                        "muscl_hancock"
                        if config.get("use_MUSCL", False)
                        else "match_p_up_to_ssprk3"
                    )
                ),
                **run_params,
            ),
        )
        for (name, config), N in product(configs.items(), resolutions)
    },
    base_path,
    overwrite=overwrite,
    postprocess=plot_error,
)
