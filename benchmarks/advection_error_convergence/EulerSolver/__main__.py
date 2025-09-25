import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superfv import EulerSolver, initial_conditions
from superfv.tools.norms import linf_norm as norm

# solver parameters
OUTPUT_NAME = "benchmarks/advection_error_convergence/EulerSolver/" + "plot.png"
DIMS = "xy"
N_LIST = [16, 32, 64, 128, 256]
P_LIST = [0, 1, 2, 3]
Q_MAX = 3
MUSCL_CONFIG = dict(
    MUSCL=True,
    MUSCL_limiter="moncen",
    SED=True,
    riemann_solver="hllc",
)
APRIORI_CONFIG = dict(
    ZS=True,
    include_corners=True,
    adaptive_dt=True,
    PAD={"rho": (0, None), "P": (0, None)},
    SED=True,
    riemann_solver="hllc",
)
APOSTERIORI_CONFIG = dict(
    MOOD=True,
    cascade="first-order",
    max_MOOD_iters=1,
    limiting_vars="actives",
    NAD=True,
    include_corners=True,
    NAD_rtol=1e-2,
    NAD_atol=1e-2,
    SED=True,
    riemann_solver="hllc",
)


def sinus(idx, x, y, z, t, xp):
    return initial_conditions.sinus(
        idx,
        x,
        y,
        z,
        t,
        xp=xp,
        **{"v" + dim: 1.0 for dim in DIMS},
        bounds=(1, 2),
        P=1e-5,
    )


# clean up old output
if os.path.exists(OUTPUT_NAME):
    os.remove(OUTPUT_NAME)

# loop over configurations
data = []
for (i, flux_recipe), (j, (config, config_name)) in product(
    enumerate([1, 2, 3]),
    enumerate(
        [
            (APRIORI_CONFIG, "a priori"),
            (APOSTERIORI_CONFIG, "a posteriori"),
            (MUSCL_CONFIG, "muscl-hancock"),
        ]
    ),
):
    for N, p in product(N_LIST, P_LIST):
        # MUSCL-Hancock is picky
        if config_name == "muscl-hancock" and (p != 1 or flux_recipe != 2):
            continue

        print(
            f"Running N={N}, p={p}, flux_recipe={flux_recipe}, config_name={config_name}"
        )

        # init solver
        sim = EulerSolver(
            ic=sinus,
            nx=N if "x" in DIMS else 1,
            ny=N if "y" in DIMS else 1,
            nz=N if "z" in DIMS else 1,
            p=p,
            flux_recipe=flux_recipe,
            **config,
        )

        # run solver
        try:
            if config.get("MUSCL", False):
                sim.musclhancock(1.0, verbose=False)
            else:
                sim.run(1.0, verbose=False, q_max=Q_MAX)
        except RuntimeError as e:
            print(f"  -> simulation failed: {e}\n")
            error = np.nan
            data.append(
                dict(N=N, p=p, flux_recipe=flux_recipe, config=config_name, error=error)
            )
            continue
        print()

        # compute error
        idx = sim.variable_index_map
        rho_numerical = sim.snapshots(1.0)["wcc"][idx("rho")]
        rho_analytical = sinus(idx, sim.mesh.X, sim.mesh.Y, sim.mesh.Z, 1.0, xp=np)[
            idx("rho")
        ]
        error = norm(rho_numerical - rho_analytical)

        data.append(
            dict(N=N, p=p, flux_recipe=flux_recipe, config=config_name, error=error)
        )
df = pd.DataFrame(data)

# plot results
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 8))

axs[0, 0].set_ylabel("flux recipe 1")
axs[1, 0].set_ylabel("flux recipe 2")
axs[2, 0].set_ylabel("flux recipe 3")
axs[0, 0].set_title(r"a priori")
axs[0, 1].set_title(r"a posteriori")
for i, j in product(range(3), range(2)):
    ax = axs[i, j]

    ax.grid()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)


for (i, flux_recipe), config, p in product(
    enumerate([1, 2, 3]), ["a priori", "a posteriori", "muscl-hancock"], P_LIST
):
    if config == "muscl-hancock" and (flux_recipe != 2 or p != 1):
        continue
    j = {"a priori": 0, "a posteriori": 1, "muscl-hancock": 0}[config]
    style = (
        dict(
            color=plt.get_cmap("tab10")(4),
            label="MUSCL-Hancock",
            marker="o",
            mfc="none",
            linestyle="--",
        )
        if config == "muscl-hancock"
        else dict(
            color=plt.get_cmap("tab10")(p),
            label=f"$p={p}$",
            marker="o",
            mfc="none",
            linestyle="-",
        )
    )
    df_sub = df[(df.flux_recipe == flux_recipe) & (df.config == config) & (df.p == p)]
    df_sub = df_sub.sort_values("N")
    axs[i, j].plot(df_sub.N, df_sub.error, **style)

axs[1, 0].legend()

fig.savefig(OUTPUT_NAME, dpi=300)
