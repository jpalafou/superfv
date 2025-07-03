import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver, initial_conditions
from superfv.tools.norms import linf_norm

# problem inputs
OUTPUT_NAME = "benchmarks/advection_error_convergence/EulerSolver/" + "plot.png"
DIMS = "x"
N_LIST = [16, 32, 64, 128, 256, 512, 1024]
P_LIST = [0, 1, 2, 3]
OTHER_INPUTS = dict(
    interpolation_scheme="transverse",
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
        ic=partial(
            initial_conditions.sinus,
            **{"v" + dim: len(DIMS) - i for i, dim in enumerate(DIMS)},
            P=0,
        ),
        nx=N if "x" in DIMS else 1,
        ny=N if "y" in DIMS else 1,
        nz=N if "z" in DIMS else 1,
        p=p,
        **OTHER_INPUTS,
    )
    solver.run(1.0)

    # measure error
    _slc = solver.array_slicer
    rho_numerical = solver.snapshots(1.0)["wcc"][_slc("rho")]
    rho_analytical = initial_conditions.sinus(_slc, solver.X, solver.Y, solver.Z, P=0)[
        _slc("rho")
    ]
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
        label=f"p={p}",
        marker="o",
        linestyle="-",
        color=cmap(p / max(P_LIST)),
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("Linf error")
ax.legend()
fig.savefig(OUTPUT_NAME)
