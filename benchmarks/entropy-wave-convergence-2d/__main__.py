import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm
from superfv.tools.run_helper import run_multiple_simulations

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence-2d/"
plot_path = "benchmarks/entropy-wave-convergence-2d/entropy-wave-convergence-2d.pdf"
overwrite = False

gamma = 5 / 3
run_params = dict(T=1.0, reduce_CFL=True, q_max=3)
init_params = dict(
    ic=partial(entropy_wave, gamma=gamma),
    gamma=gamma,
    PAD={"rho": (0, None), "P": (0, None)},
    SED=True,
    log_limiter_scalars=False,
    skip_trouble_counts=True,
    cupy=True,
)

# Loop parameters
resolutions = [32, 64, 128]

musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive")
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")
aposteriori_1rev = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori_2revs = dict(cascade="muscl0", max_MOOD_iters=2, **aposteriori)
aposteriori_3revs = dict(cascade="muscl0", max_MOOD_iters=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock,
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3lazy": dict(p=3, GL=True, **(apriori | dict(lazy_primitives="full"))),
    "ZS7lazy": dict(p=7, GL=True, **(apriori | dict(lazy_primitives="full"))),
    "MM3/1rev/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM3/1rev/rtol_0": dict(p=3, NAD_rtol=0, **aposteriori_1rev),
    "MM7/1rev/rtol_0": dict(p=7, NAD_rtol=0, **aposteriori_1rev),
}

styles = {
    "MUSCL-RK3": dict(color="grey", marker="^", mfc="none"),
    "MUSCL-Hancock": dict(color="grey", marker="o", mfc="none"),
    "ZS3": dict(color="blue", marker="o", mfc="none"),
    "ZS3lazy": dict(color="blue", linestyle="--", marker="*", mfc="none"),
    "MM3/1rev/rtol_1e-3": dict(color="blue", marker="s", mfc="none"),
    "MM3/1rev/rtol_0": dict(color="blue", linestyle="--", marker="+", mfc="none"),
    "ZS7": dict(color="red", marker="o", mfc="none"),
    "ZS7lazy": dict(color="red", linestyle="--", marker="*", mfc="none"),
    "MM7/1rev/rtol_1e-3": dict(color="red", marker="s", mfc="none"),
    "MM7/1rev/rtol_0": dict(color="red", linestyle="--", marker="+", mfc="none"),
}


data = []


def plot_error(name, sim):
    # parse name without resolution
    name = name.split("/N_")[0]

    # measure error
    idx = sim.variable_index_map
    vz0 = sim.snapshots[0]["wcc"][idx("vz")]
    vz1 = sim.snapshots[-1]["wcc"][idx("vz")]
    error = linf_norm(vz1 - vz0)

    # update dataframe
    data.append(dict(name=name, N=sim.mesh.nx, error=error))
    df = pd.DataFrame(data)

    # plot error curves of p over N
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("N")
    ax.set_ylabel("Linf error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid()

    for name in configs.keys():
        df_name = df[df["name"] == name]

        if df_name.empty:
            continue

        ax.plot(
            df_name["N"],
            df_name["error"],
            label=name,
            **(dict(markersize=5, linewidth=2, alpha=0.7) | styles.get(name, dict())),
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
                muscl_hancock=False if name == "MUSCL-RK3" else True,
                time_degree=2 if name == "MUSCL-RK3" else None,
                **run_params,
            ),
        )
        for (name, config), N in product(configs.items(), resolutions)
    },
    base_path,
    overwrite=overwrite,
    postprocess=plot_error,
)
