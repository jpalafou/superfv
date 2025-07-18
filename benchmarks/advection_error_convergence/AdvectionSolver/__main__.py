import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import superfv.initial_conditions as initial_conditions
from superfv import AdvectionSolver
from superfv.tools.norms import l1_norm

# problem inputs
OUTPUT_NAME = "benchmarks/advection_error_convergence/AdvectionSolver/" + "plot.png"
DIMS = "x"
N_LIST = [16, 32, 64, 128, 256]
P_LIST = [0, 1, 2, 3]
OTHER_INPUTS = dict(
    interpolation_scheme="transverse",
    cupy=False,
    ZS=True,
    adaptive_dt=False,
    SED=True,
    lazy_primitives=False,
)

# remove old output
if os.path.exists(OUTPUT_NAME):
    os.remove(OUTPUT_NAME)

# loop over all combinations of N and p
data = []
for N, p in product(N_LIST, P_LIST):
    # print status
    print(f"Running N={N}, p={p}")

    def analytical_solution(idx, x, y, z, t, xp):
        return initial_conditions.sinus(
            idx,
            x,
            y,
            z,
            t,
            xp=xp,
            **{"v" + dim: len(DIMS) - i for i, dim in enumerate(DIMS)},
        )

    # run solver
    solver = AdvectionSolver(
        ic=analytical_solution,
        nx=N if "x" in DIMS else 1,
        ny=N if "y" in DIMS else 1,
        nz=N if "z" in DIMS else 1,
        p=p,
        **OTHER_INPUTS,
    )
    solver.run(1.0, progress_bar=False)

    # measure error
    idx = solver.variable_index_map
    rho_numerical = solver.snapshots(1.0)["wcc"][idx("rho")]
    rho_analytical = analytical_solution(
        idx, solver.mesh.X, solver.mesh.Y, solver.mesh.Z, 1.0, xp=np
    )[idx("rho")]
    error = l1_norm(rho_numerical - rho_analytical)
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
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("L1 error")
ax.legend()
fig.savefig(OUTPUT_NAME)
