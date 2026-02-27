import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver, OutputLoader
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/hydro-advection-of-a-square/"
plot_path = "benchmarks/hydro-advection-of-a-square/plot.png"
overwrite = True

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
    # "p0": dict(p=0),
    # "p1": dict(p=1),
    # "p3": dict(p=3),
    # "p7": dict(p=7),
    # "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    # "ZS3": dict(p=3, GL=True, **apriori),
    # "ZS7": dict(p=7, GL=True, **apriori),
    # "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    # "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    # "MM3/1rev/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori1),
    # "MM7/1rev/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori1),
    # "MM3/3revs/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori3),
    # "MM7/3revs/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori3),
    "p0gl": dict(p=0, GL=True),
    "p1gl": dict(p=1, GL=True),
    "p2gl": dict(p=2, GL=True),
    "p3gl": dict(p=3, GL=True),
    "p4gl": dict(p=4, GL=True),
    "p5gl": dict(p=5, GL=True),
    "p6gl": dict(p=6, GL=True),
    "p7gl": dict(p=7, GL=True),
    "p0t": dict(p=0),
    "p1t": dict(p=1),
    "p2t": dict(p=2),
    "p3t": dict(p=3),
    "p4t": dict(p=4),
    "p5t": dict(p=5),
    "p6t": dict(p=6),
    "p7t": dict(p=7),
}

styles = {
    # "p0": dict(color="k"),
    # "p1": dict(color="blue"),
    # "p3": dict(color="green"),
    # "p7": dict(color="red"),
    # "MUSCL-Hancock": dict(color="blue", linestyle="--", marker="o", mfc="none"),
    # "ZS3": dict(color="green", linestyle="--", marker="o", mfc="none"),
    # "ZS7": dict(color="red", linestyle="--", marker="o", mfc="none"),
    # "ZS3t": dict(color="green", linestyle="--", marker="+", mfc="none"),
    # "ZS7t": dict(color="red", linestyle="--", marker="+", mfc="none"),
    # "MM3/1rev/rtol_1e-3": dict(color="green", linestyle="--", marker="s", mfc="none"),
    # "MM7/1rev/rtol_1e-3": dict(color="red", linestyle="--", marker="s", mfc="none"),
    # "MM3/3revs/rtol_1e-3": dict(color="green", linestyle="--", marker="^", mfc="none"),
    # "MM7/3revs/rtol_1e-3": dict(color="red", linestyle="--", marker="^", mfc="none"),
    "p0gl": dict(color=plt.get_cmap("tab20")(0)),
    "p1gl": dict(color=plt.get_cmap("tab20")(2)),
    "p2gl": dict(color=plt.get_cmap("tab20")(4)),
    "p3gl": dict(color=plt.get_cmap("tab20")(6)),
    "p4gl": dict(color=plt.get_cmap("tab20")(8)),
    "p5gl": dict(color=plt.get_cmap("tab20")(10)),
    "p6gl": dict(color=plt.get_cmap("tab20")(12)),
    "p7gl": dict(color=plt.get_cmap("tab20")(14)),
    "p0t": dict(color=plt.get_cmap("tab20")(1), linestyle="--"),
    "p1t": dict(color=plt.get_cmap("tab20")(3), linestyle="--"),
    "p2t": dict(color=plt.get_cmap("tab20")(5), linestyle="--"),
    "p3t": dict(color=plt.get_cmap("tab20")(7), linestyle="--"),
    "p4t": dict(color=plt.get_cmap("tab20")(9), linestyle="--"),
    "p5t": dict(color=plt.get_cmap("tab20")(11), linestyle="--"),
    "p6t": dict(color=plt.get_cmap("tab20")(13), linestyle="--"),
    "p7t": dict(color=plt.get_cmap("tab20")(15), linestyle="--"),
}

N_values = [32, 64, 128]


gamma = 1.4
T = 1.0


def sinus(idx, x, y, z, t, *, xp):
    out = xp.zeros((idx.nvars, *x.shape))

    out[idx("rho")] = 0.5 * xp.sin(2 * xp.pi * (x)) + 1.5
    out[idx("vx")] = 1.0
    out[idx("vy")] = 0.0
    out[idx("P")] = 1.0

    return out


def plot_convergence(df):
    # plot error curves of p over N
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title("Convergence of 2D advection of a square in a single direction")
    ax.set_xlabel("N")
    ax.set_ylabel("Linf error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid()

    for name in configs.keys():
        df_name = df[df["name"] == name]

        if df_name.empty:
            continue

        style = styles[name] if name in styles else {}

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

# Run simulations
data = []
for (name, config), N in product(configs.items(), N_values):
    sim_path = f"{base_path}{name}/N_{N}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"- - - Starting simulation: {name}, N={N} - - -")
        print(f"\tRunning config with name '{name}' and writing to path '{sim_path}'.")

        sim = EulerSolver(ic=sinus, nx=N, ny=N, gamma=gamma, cupy=N >= 64, **config)

        try:
            sim.run(
                T,
                reduce_CFL=True,
                muscl_hancock=config.get("MUSCL", False),
                path=sim_path,
                overwrite=True,
            )
            sim.write_timings()

            print("\tSuccess!\n")

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
        rho0 = sim.snapshots[0]["wcc"][idx("rho")]
        rho1 = sim.snapshots[-1]["wcc"][idx("rho")]
        error = linf_norm(rho1 - rho0)
        data.append(dict(N=N, name=name, p=getattr(sim, "p", None), error=error))

        # update plot
        df = pd.DataFrame(data)
        plot_convergence(df)
