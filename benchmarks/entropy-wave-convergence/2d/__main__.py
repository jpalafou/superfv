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

common = dict(PAD={"rho": (0, None), "P": (0, None)})
apriori = dict(ZS=True, GL=True, **common)
aposteriori = dict(
    MOOD=True, MUSCL_limiter="PP2D", lazy_primitives="full", NAD_atol=1e-3, **common
)

configs = {
    "p0": dict(p=0),
    "MH": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", **common),
    "p3": dict(p=3),
    "ZS3/wa": dict(p=3, lazy_primitives="adaptive", **apriori),
    "MM3": dict(p=3, **aposteriori),
    "p7": dict(p=7),
    "ZS7/wa": dict(p=7, lazy_primitives="adaptive", **apriori),
    "MM7": dict(p=7, **aposteriori),
}

styles = {
    "p0": dict(color="k"),
    "p1": dict(color="darkgray"),
    "MH/none": dict(color="darkgray", linestyle=".."),
    "MH": dict(color="darkgray", linestyle="--"),
    "p3": dict(color="tab:blue"),
    "ZS3/wa": dict(color="tab:blue", linestyle="--", marker="^"),
    "MM3": dict(color="tab:blue", linestyle="--", marker="o", mfc="none"),
    "p7": dict(color="tab:green"),
    "ZS7/wa": dict(color="tab:green", linestyle="--", marker="^"),
    "MM7": dict(color="tab:green", linestyle="--", marker="o", mfc="none"),
}

N_values = [16, 32, 64, 128, 256]


gamma = 5 / 3
T = 1.0

# remove old output
if os.path.exists(plot_path):
    os.remove(plot_path)

# loop over all combinations of N and p
data = []
for (name, config), N in product(configs.items(), N_values):
    sim_path = base_path + f"{name}/N_{N}/"

    # print status
    print(f"Running N={N}, config={name}")

    # run solver
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
            muscl_hancock=name == "MUSCL-Hancock" or "MH" in name,
            path=sim_path,
        )
        sim.write_timings()
    except FileExistsError:
        sim = OutputLoader(sim_path)
    except RuntimeError as e:
        print(f"  Failed: {e}")
        continue

    # measure error
    idx = sim.variable_index_map
    mesh = sim.mesh
    vz0 = sim.snapshots[0]["wcc"][idx("vz")]
    vz1 = sim.snapshots[-1]["wcc"][idx("vz")]
    error = linf_norm(vz1 - vz0)
    data.append(dict(N=N, name=name, p=getattr(sim, "p", None), error=error))
df = pd.DataFrame(data)

# plot error curves of p over N
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_title("Entropy wave convergence")
ax.set_xlabel("N")
ax.set_ylabel("Linf error")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.grid()


for name in configs.keys():
    df_name = df[df["name"] == name]
    p = df_name["p"].values[0]
    style = styles[name]

    ax.plot(
        df_name["N"],
        df_name["error"],
        label=name,
        **style,
    )
ax.legend()
fig.savefig(plot_path, dpi=300)
