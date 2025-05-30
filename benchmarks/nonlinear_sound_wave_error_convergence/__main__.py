import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superfv import EulerSolver
from superfv.tools.array_management import l1_norm

gamma = 1.4
rho0 = 1
cs0 = 1
P0 = 1 / gamma
A = 1e-5
T = 1.0


def nonlinear_sound_wave(array_slicer, x, y, z, t):
    _slc = array_slicer

    def vp(x, t):
        return A * np.sin(2 * np.pi * (x - t)) - A**2 * (
            gamma + 1
        ) / 4 * 2 * np.pi * t * np.sin(4 * np.pi * (x - t))

    out = np.zeros((5, *x.shape))
    out[_slc("rho")] = rho0 + rho0 * vp(x, 0)  # only valid for t=0
    out[_slc("vx")] = vp(x, t)
    out[_slc("P")] = P0 + gamma * P0 * vp(x, 0)  # only valid for t=0
    return out


# problem inputs
OUTPUT_NAME = "benchmarks/nonlinear_sound_wave_error_convergence/" + "plot.png"
N_LIST = [16, 32, 64, 128, 256, 512, 1024]
P_LIST = [0, 1, 2, 3]
OTHER_INPUTS = dict(
    gamma=gamma,
    riemann_solver="llf",
    flux_recipe=3,
    lazy_primitives=False,
    ZS=True,
    adaptive_timestepping=False,
    SED=True,
)

# remove old output
if os.path.exists(OUTPUT_NAME):
    os.remove(OUTPUT_NAME)

# loop over all combinations of N and p
data = []
for N, p in product(N_LIST, P_LIST):
    # print status
    print(f"Running N={N}, p={p}")

    # run solver
    solver = EulerSolver(
        ic=partial(nonlinear_sound_wave, t=0), nx=N, p=p, **OTHER_INPUTS
    )
    solver.run(T)

    # measure error
    _slc = solver.array_slicer
    if "wcc" in solver.snapshots[-1]:
        vx_numerical = solver.snapshots[-1]["wcc"][_slc("vx")]
    else:
        vx_numerical = solver.primitive_cell_centers(solver.snapshots[-1]["u"])[
            _slc("vx")
        ]
    vx_exact = nonlinear_sound_wave(_slc, solver.X, solver.Y, solver.Z, T)[_slc("vx")]
    error = l1_norm(vx_numerical - vx_exact)
    data.append(dict(N=N, p=p, error=error))
df = pd.DataFrame(data)

# plot error curves of p over N
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")

for p in P_LIST:
    df_p = df[df["p"] == p]
    ax.plot(
        df_p["N"],
        df_p["error"],
        label=f"p={p}",
        marker="o",
        linestyle="-",
        color=cmap(p / max(P_LIST)),
    )
ax.set_title("Nonlinear sound wave error convergence")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("L1 error")
ax.legend()
fig.savefig(OUTPUT_NAME)
