import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm
from superfv.tools.run_helper import run_multiple_simulations

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence-2d/"
plot_path = "benchmarks/entropy-wave-convergence-2d/plot.png"
overwrite = False

gamma = 5 / 3
run_params = dict(T=1.0, reduce_CFL=True)
init_params = dict(
    ic=partial(entropy_wave, gamma=gamma),
    gamma=gamma,
    PAD={"rho": (0, None), "P": (0, None)},
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
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p3": dict(p=3),
    "p7": dict(p=7),
    "MUSCL-Hancock": musclhancock,
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori_1rev),
}

styles = {
    "p0": dict(color="k"),
    "p1": dict(color="green"),
    "p3": dict(color="blue"),
    "p7": dict(color="red"),
    "MUSCL-Hancock": dict(color="green", linestyle="--", marker="o", mfc="none"),
    "ZS3": dict(color="blue", linestyle="--", marker="o", mfc="none"),
    "ZS7": dict(color="red", linestyle="--", marker="o", mfc="none"),
    "ZS3t": dict(color="blue", linestyle="--", marker="+", mfc="none"),
    "ZS7t": dict(color="red", linestyle="--", marker="+", mfc="none"),
    "MM3/1rev/rtol_1e-3": dict(color="blue", linestyle="--", marker="s", mfc="none"),
    "MM7/1rev/rtol_1e-3": dict(color="red", linestyle="--", marker="s", mfc="none"),
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
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title("Convergence of 2D entropy wave")
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
            **styles.get(name, dict()),
        )
    ax.legend()
    fig.savefig(plot_path, dpi=300)


# remove old output
if os.path.exists(plot_path):
    os.remove(plot_path)

# loop over all configs and resolutions
run_multiple_simulations(
    {
        f"{name}/N_{N}/": (dict(nx=N, ny=N, **init_params, **config), run_params)
        for (name, config), N in product(configs.items(), resolutions)
    },
    base_path,
    overwrite=overwrite,
    postprocess=plot_error,
)
