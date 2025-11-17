import os
from functools import partial
from itertools import product
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver
from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/entropy-wave-convergence/"
plot_path = base_path + "plot.png"

PAD = {"rho": (0, None), "P": (0, None)}
apriori = dict(ZS=True, lazy_primitives="adaptive", PAD=PAD)
aposteriori = dict(MOOD=True, lazy_primitives="adaptive", PAD=PAD, NAD_atol=1e-3)

configs = {
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="moncen"),
    "ZS3": dict(p=3, **apriori),
    "MM3": dict(p=3, **aposteriori),
    "ZS7": dict(p=7, **apriori),
    "MM7": dict(p=7, **aposteriori),
}

N_values = [32, 64, 128, 256, 512]


gamma = 5 / 3
T = 1.0

# remove old output
if os.path.exists(plot_path):
    os.remove(plot_path)

# loop over all combinations of N and p
data = []
for N, (name, config) in product(N_values, configs.items()):
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
    except RuntimeError as e:
        print(f"  Failed: {e}")
        continue

    # measure error
    idx = sim.variable_index_map
    mesh = sim.mesh
    vz0 = sim.snapshots[0]["wcc"][idx("vz")]
    vz1 = sim.snapshots[-1]["wcc"][idx("vz")]
    error = linf_norm(vz1 - vz0)
    data.append(dict(N=N, name=name, p=sim.p, error=error))
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


def style_resolver(p, base_linestyle="-", base_marker="o"):
    nhits = hits.get(p, 0)
    hits[(p)] = nhits + 1

    if nhits == 0:
        style = dict(linestyle=base_linestyle, marker=base_marker)
    elif nhits == 1:
        style = dict(linestyle="--", marker=base_marker)
    elif nhits == 2:
        style = dict(linestyle=base_linestyle, marker="*")
    elif nhits == 3:
        style = dict(linestyle=base_linestyle, marker="s")
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
