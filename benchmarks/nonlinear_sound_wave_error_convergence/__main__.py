import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superfv import EulerSolver
from superfv.tools.norms import linf_norm

gamma = 1.4
rho0 = 1
cs0 = 1
P0 = 1 / gamma
A = 1e-5
T = 1.0


def nonlinear_sound_wave(idx, x, y, z, t, *, xp):

    def vp(x, t):
        return A * xp.sin(2 * np.pi * (x - t)) - A**2 * (
            gamma + 1
        ) / 4 * 2 * np.pi * t * xp.sin(4 * np.pi * (x - t))

    out = xp.zeros((idx.nvars, *x.shape))
    out[idx("rho")] = rho0 + rho0 * vp(x, 0)  # only valid for t=0
    out[idx("vx")] = vp(x, t)
    out[idx("P")] = P0 + gamma * P0 * vp(x, 0)  # only valid for t=0
    return out


# problem inputs
OUTPUT_NAME = "benchmarks/nonlinear_sound_wave_error_convergence/" + "plot.png"
N_LIST = [16, 32, 64, 128, 256, 512]
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
    "ZS1": dict(
        riemann_solver="hllc",
        p=1,
        ZS=True,
        flux_recipe=2,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "ZS3-FR1-lazy": dict(
        riemann_solver="hllc",
        p=3,
        ZS=True,
        flux_recipe=2,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "ZS3-FR2-lazy": dict(
        riemann_solver="hllc",
        p=3,
        ZS=True,
        flux_recipe=1,
        lazy_primitives=True,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "ZS3-FR3-lazy": dict(
        riemann_solver="hllc",
        p=3,
        ZS=True,
        flux_recipe=2,
        lazy_primitives=True,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "ZS3-FR3": dict(
        riemann_solver="hllc",
        p=3,
        ZS=True,
        flux_recipe=3,
        lazy_primitives=False,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
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
    sim = EulerSolver(ic=nonlinear_sound_wave, nx=N, **config)
    try:
        if config.get("MUSCL", False):
            sim.musclhancock(T)
        else:
            sim.run(T)
    except RuntimeError as e:
        print(f"  Failed: {e}")
        continue

    # measure error
    idx = sim.variable_index_map
    mesh = sim.mesh
    vx_numerical = sim.snapshots[-1]["wcc"][idx("vx")]
    vx_exact = nonlinear_sound_wave(idx, *mesh.get_cell_centers(), T, xp=np)[idx("vx")]
    error = linf_norm(vx_numerical - vx_exact)
    data.append(dict(N=N, name=name, p=sim.p, error=error))
df = pd.DataFrame(data)

# plot error curves of p over N
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_title("Nonlinear sound wave error convergence")
ax.set_xlabel("N")
ax.set_ylabel("Linf error")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.grid()


hits = {}


def style_resolver(p, base_linestyle="-", base_marker="o"):
    color = plt.get_cmap("viridis")(p / 3)

    nhits = hits.get(color, 0)
    hits[(color)] = nhits + 1

    if nhits == 0:
        style = dict(color=color, linestyle=base_linestyle, marker=base_marker)
    elif nhits == 1:
        style = dict(color=color, linestyle="--", marker=base_marker)
    elif nhits == 2:
        style = dict(color=color, linestyle=base_linestyle, marker="*")
    elif nhits == 3:
        style = dict(color=color, linestyle=base_linestyle, marker="s")
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
