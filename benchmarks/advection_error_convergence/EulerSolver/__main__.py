import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superfv import EulerSolver, initial_conditions
from superfv.tools.norms import linf_norm

# problem inputs
OUTPUT_NAME = "benchmarks/advection_error_convergence/EulerSolver/" + "plot.png"
DIMS = "x"
N_LIST = [16, 32, 64, 128, 256]
P_LIST = [-1, 0, 1, 2, 3]
OTHER_INPUTS = dict(
    flux_recipe=1,
    lazy_primitives=False,
    SED=True,
    ZS=True,
    adaptive_dt=False,
    # MOOD=True,
    # NAD=True,
    # limiting_vars=("rho",),
)
MUSCL_INPUTS = dict(flux_recipe=2, MUSCL=True, MUSCL_limiter="moncen", SED=True)

# remove old output
if os.path.exists(OUTPUT_NAME):
    os.remove(OUTPUT_NAME)

# loop over all combinations of N and p
data = []
for N, p in product(N_LIST, P_LIST):
    # print status
    print(f"Running N={N}, MUSCL-Hancock" if p == -1 else f"Running N={N}, p={p}")

    def analytical_solution(idx, x, y, z, t, xp):
        return initial_conditions.sinus(
            idx,
            x,
            y,
            z,
            t,
            xp=xp,
            **{"v" + dim: len(DIMS) - i for i, dim in enumerate(DIMS)},
            bounds=(1e-8, 1.0),
            P=1e-8,
        )

    # run solver
    solver = EulerSolver(
        ic=analytical_solution,
        nx=N if "x" in DIMS else 1,
        ny=N if "y" in DIMS else 1,
        nz=N if "z" in DIMS else 1,
        p=1 if p == -1 else p,
        **(MUSCL_INPUTS if p == -1 else OTHER_INPUTS),
    )
    if p == -1:
        solver.musclhancock(1.0, log_freq=10)
    else:
        solver.run(1.0, log_freq=10)
    print()

    # measure error
    idx = solver.variable_index_map
    rho_numerical = solver.snapshots(1.0)["wcc"][idx("rho")]
    rho_analytical = analytical_solution(
        idx, solver.mesh.X, solver.mesh.Y, solver.mesh.Z, 1.0, xp=np
    )[idx("rho")]
    error = linf_norm(rho_numerical - rho_analytical)
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
        label="MUSCL-Hancock" if p == -1 else f"p={p}",
        marker="o",
        color="red" if p == -1 else cmap((p) / max(P_LIST)),
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel(r"$L_\infty$")
ax.legend()
fig.savefig(OUTPUT_NAME)
