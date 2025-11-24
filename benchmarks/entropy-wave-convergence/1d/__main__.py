import os
from functools import partial
from itertools import product
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence-1d/"
plot_path = "benchmarks/entropy-wave-convergence/1d/plot.png"

common = dict(PAD=None)
apriori = dict(ZS=True, adaptive_dt=False, **common)
aposteriori = dict(
    MOOD=True, MUSCL_limiter="moncen", lazy_primitives="full", NAD_atol=1e-3, **common
)

configs = {
    # "p0": dict(p=0),
    # "MUSCL-Hancock": dict(p=1, MUSCL=True, **common),
    # "ZS2": dict(p=2, **apriori),
    # "MM2": dict(p=2, **aposteriori),
    "ZS3-w1": dict(p=3, lazy_primitives="full", **apriori),
    "ZS3-adaptive/eta_max_0.025": dict(
        p=3, lazy_primitives="adaptive", eta_max=0.025, **apriori
    ),
    "ZS3-adaptive/eta_max_0.05": dict(
        p=3, lazy_primitives="adaptive", eta_max=0.05, **apriori
    ),
    "ZS3-adaptive/eta_max_0.01": dict(
        p=3, lazy_primitives="adaptive", eta_max=0.1, **apriori
    ),
    "ZS3-wp": dict(p=3, lazy_primitives="none", **apriori),
    # "MM3": dict(p=3, **aposteriori),
    # "ZS7": dict(p=7, **apriori),
    # "MM7": dict(p=7, **aposteriori),
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
        ic=partial(entropy_wave, gamma=gamma), nx=N, gamma=gamma, **config
    )

    try:
        sim.run(
            T, reduce_CFL=True, muscl_hancock=config.get("MUSCL", False), path=sim_path
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


hits: Dict[int, int] = {}


def style_resolver(p):
    nhits = hits.get(p, 0)
    hits[p] = nhits + 1

    if nhits == 0:
        style = dict(linestyle="-", marker="o")
    elif nhits == 1:
        style = dict(linestyle="--", marker="o")
    elif nhits == 2:
        style = dict(linestyle="-", marker="*")
    elif nhits == 3:
        style = dict(linestyle="--", marker="*")
    elif nhits == 4:
        style = dict(linestyle="-", marker="s")
    elif nhits == 5:
        style = dict(linestyle="--", marker="s")
    else:
        raise ValueError("Too many hits for p={p}")

    return style


for name in configs.keys():
    df_name = df[df["name"] == name]
    p = df_name["p"].values[0]
    style = style_resolver(p)

    ax.plot(
        df_name["N"],
        df_name["error"],
        label=name,
        **style,
    )
ax.legend()
fig.savefig(plot_path, dpi=300)
