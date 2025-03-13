from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

import superfv.initial_conditions as initial_conditions
from superfv import AdvectionSolver
from superfv.tools.array_management import linf_norm
from superfv.visualization import plot_power_law_fit

# problem inputs
OUTPUT_NAME = "benchmarks/AdvectionSolver_error_convergence/plot"
DIMS = "x"
N_LIST = [16, 32, 64, 128, 256]
P_LIST = [0, 1, 2, 3]
OTHER_INPUTS = dict(
    interpolation_scheme="transverse",
    MOOD=True,
    NAD=1e-5,
    SED=True,
)

# loop over all combinations of N and p
data = []
for N, p in product(N_LIST, P_LIST):
    # print status
    print(f"Running N={N}, p={p}")

    # run solver
    solver = AdvectionSolver(
        ic=partial(
            initial_conditions.sinus,
            **{"v" + dim: len(DIMS) - i for i, dim in enumerate(DIMS)},
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
    rho0 = solver.snapshots(0.0)["u"][_slc("rho")]
    rho1 = solver.snapshots(1.0)["u"][_slc("rho")]
    data.append(dict(N=N, p=p, error=linf_norm(rho1 - rho0)))
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
        linestyle="none",
        color=cmap(p / max(P_LIST)),
    )
    plot_power_law_fit(
        ax, df_p["N"].to_numpy(), df_p["error"].to_numpy(), color="grey", linestyle="--"
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("Linf error")
ax.legend()
fig.savefig(OUTPUT_NAME + ".png")
