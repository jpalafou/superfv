import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence-2d/"
plot_path = "benchmarks/entropy-wave-convergence/2d/plot.png"
overwrite = False

common = dict(PAD={"rho": (0, None), "P": (0, None)})
musclhancock = dict(p=1, MUSCL=True, **common)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    **common,
)
aposteriori1 = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori2 = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori)
aposteriori3 = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori)

configs = {
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p3": dict(p=3),
    "p7": dict(p=7),
    "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/3revs/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori3),
    "MM7/3revs/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori3),
    "MM3/3revs/rtol_1e0": dict(p=3, NAD_rtol=1e0, **aposteriori3),
    "MM7/3revs/rtol_1e0": dict(p=7, NAD_rtol=1e0, **aposteriori3),
    "MM3/3revs/no_NAD": dict(p=3, NAD=False, SED=False, **aposteriori3),
    "MM7/3revs/no_NAD": dict(p=7, NAD=False, SED=False, **aposteriori3),
}

styles = {
    "p0": dict(color="k"),
    "p1": dict(color="blue"),
    "p3": dict(color="green"),
    "p7": dict(color="red"),
    "MUSCL-Hancock": dict(color="blue", linestyle="--", marker="o", mfc="none"),
    "ZS3": dict(color="green", linestyle="--", marker="o", mfc="none"),
    "ZS7": dict(color="red", linestyle="--", marker="o", mfc="none"),
    "ZS3t": dict(color="green", linestyle="--", marker="+", mfc="none"),
    "ZS7t": dict(color="red", linestyle="--", marker="+", mfc="none"),
    "MM3/3revs/rtol_1e-1": dict(color="green", linestyle="--", marker="s", mfc="none"),
    "MM7/3revs/rtol_1e-1": dict(color="red", linestyle="--", marker="s", mfc="none"),
    "MM3/3revs/rtol_1e0": dict(color="green", linestyle="--", marker="^", mfc="none"),
    "MM7/3revs/rtol_1e0": dict(color="red", linestyle="--", marker="^", mfc="none"),
    "MM3/3revs/no_NAD": dict(color="green", linestyle="--", marker="v", mfc="none"),
    "MM7/3revs/no_NAD": dict(color="red", linestyle="--", marker="v", mfc="none"),
}

N_values = [16, 32, 64, 128, 256]


gamma = 5 / 3
T = 1.0


def plot_convergence(df):
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

        style = styles[name]

        ax.plot(
            df_name["N"],
            df_name["error"],
            label=name,
            **style,
        )
    ax.legend()
    fig.savefig(plot_path, dpi=300)


# remove old output
if os.path.exists(plot_path):
    os.remove(plot_path)

# loop over all combinations of N and p
data = []
for (name, config), N in product(configs.items(), N_values):
    sim_path = base_path + f"{name}/N_{N}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for N={N}, config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"Running simulation for N={N}, config={name}...")
        sim = EulerSolver(
            ic=partial(entropy_wave, gamma=gamma),
            nx=N,
            ny=N,
            gamma=gamma,
            cupy=True,
            **config,
        )
        try:
            sim.run(
                T,
                reduce_CFL=True,
                muscl_hancock=name == "MUSCL-Hancock",
                path=sim_path,
                overwrite=True,
            )
            sim.write_timings()

            # clean up error file if it exists
            if os.path.exists(f"{sim_path}error.txt"):
                os.remove(f"{sim_path}error.txt")
        except RuntimeError as e:
            print(f"  Failed: {e}")
            with open(f"{sim_path}error.txt", "w") as f:
                f.write(str(e))
            continue

    # measure error
    idx = sim.variable_index_map
    mesh = sim.mesh
    vz0 = sim.snapshots[0]["wcc"][idx("vz")]
    vz1 = sim.snapshots[-1]["wcc"][idx("vz")]
    error = linf_norm(vz1 - vz0)
    data.append(dict(N=N, name=name, p=getattr(sim, "p", None), error=error))

    df = pd.DataFrame(data)
    plot_convergence(df)
