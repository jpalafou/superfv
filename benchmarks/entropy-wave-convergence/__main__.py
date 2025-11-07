import os
from functools import partial
from itertools import product
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver
from superfv.initial_conditions import entropy_wave
from superfv.tools.norms import linf_norm

gamma = 5 / 3
T = 1.0

OUTPUT_NAME = "benchmarks/entropy-wave-convergence/" + "plot.png"
N_LIST = [32, 64, 128]
configs = {
    "p0": dict(riemann_solver="hllc", p=0),
    "MUSCL-Hancock": dict(
        riemann_solver="hllc",
        p=1,
        MUSCL=True,
        MUSCL_limiter="moncen",
        flux_recipe=2,
        SED=True,
    ),
    "ZS3": dict(
        riemann_solver="hllc",
        p=3,
        ZS=True,
        flux_recipe=2,
        lazy_primitives="adaptive",
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    # "ZS7": dict(
    #     riemann_solver="hllc",
    #     p=7,
    #     ZS=True,
    #     flux_recipe=2,
    #     lazy_primitives="adaptive",
    #     PAD={"rho": (0, None), "P": (0, None)},
    #     SED=True,
    # ),
    "MM3-nolazy": dict(
        riemann_solver="hllc",
        p=3,
        flux_recipe=2,
        lazy_primitives="none",
        MOOD=True,
        limiting_vars="actives",
        cascade="muscl",
        MUSCL_limiter="moncen",
        max_MOOD_iters=1,
        blend=True,
        NAD=True,
        NAD_rtol=1e-2,
        NAD_atol=1e-3,
        SED=True,
    ),
    "MM3-fulllazy": dict(
        riemann_solver="hllc",
        p=3,
        flux_recipe=2,
        lazy_primitives="full",
        MOOD=True,
        limiting_vars="actives",
        cascade="muscl",
        MUSCL_limiter="moncen",
        max_MOOD_iters=1,
        blend=True,
        NAD=True,
        NAD_rtol=1e-2,
        NAD_atol=1e-3,
        SED=True,
    ),
    # "MM7": dict(
    #     riemann_solver="hllc",
    #     p=7,
    #     flux_recipe=2,
    #     lazy_primitives="none",
    #     MOOD=True,
    #     cascade="muscl",
    #     MUSCL_limiter="moncen",
    #     max_MOOD_iters=1,
    #     blend=True,
    #     NAD=True,
    #     NAD_rtol=1e-2,
    #     NAD_atol=1e-8,
    #     SED=True,
    # ),
}

# remove old output
if os.path.exists(OUTPUT_NAME):
    os.remove(OUTPUT_NAME)

# loop over all combinations of N and p
data = []
for N, (name, config) in product(N_LIST, configs.items()):
    # print status
    print(f"Running N={N}, config={name}")

    # run solver
    sim = EulerSolver(
        ic=partial(entropy_wave, gamma=gamma), nx=N, gamma=gamma, **config
    )
    try:
        sim.run(T, reduce_CFL=True, muscl_hancock=config.get("MUSCL", False))
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
fig.savefig(OUTPUT_NAME)
