import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from superfv import EulerSolver, OutputLoader
from superfv.tools.norms import linf_norm

base_path = "/scratch/gpfs/jp7427/out/hydro-advection-of-a-square/"
plot_path = "benchmarks/hydro-advection-of-a-square/plot.png"

common = dict(PAD={"rho": (0, None), "P": (0, None)})
apriori = dict(ZS=True, GL=True, **common)
aposteriori = dict(
    MOOD=True, MUSCL_limiter="PP2D", lazy_primitives="full", NAD_atol=1e-3, **common
)

configs = {
    "p0": dict(p=0),
    "MH/minmod": dict(p=1, MUSCL=True, MUSCL_limiter="minmod", **common),
    "MH/moncen": dict(p=1, MUSCL=True, MUSCL_limiter="moncen", **common),
    "MH/PP2D": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", **common),
    "MH/none": dict(p=1, MUSCL=True, MUSCL_limiter=None, **common),
    "ZS3/wp": dict(p=3, lazy_primitives="none", **apriori),
    "MM3": dict(p=3, **aposteriori),
    "p3": dict(p=3),
}

styles = {
    "p0": dict(color="k"),
    "MH/none": dict(color="darkgray"),
    "MH/PP2D": dict(color="darkgray", marker="o", mfc="none"),
    "MH/minmod": dict(color="darkgray", linestyle="--", marker="*", mfc="none"),
    "MH/moncen": dict(color="darkgray", linestyle="--", marker="^", mfc="none"),
    "ZS3/wp": dict(color="tab:green", marker="o", mfc="none"),
    "MM3": dict(color="tab:orange", marker="o", mfc="none"),
    "p3": dict(color="tab:blue"),
}

N_values = [16, 32, 64, 128, 256]


gamma = 1.4
T = 1.0


def sinus(idx, x, y, z, t, *, xp):
    out = xp.zeros((idx.nvars, *x.shape))

    out[idx("rho")] = 0.5 * xp.sin(2 * xp.pi * (x)) + 1.5
    out[idx("vx")] = 1.0
    out[idx("vy")] = 0.0
    out[idx("P")] = 1.0

    return out


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
    sim = EulerSolver(ic=sinus, nx=N, ny=N, gamma=gamma, cupy=True, **config)

    try:
        sim.run(
            T,
            reduce_CFL=True,
            muscl_hancock=name == "MUSCL-Hancock" or "MH" in name,
            path=sim_path,
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
    rho0 = sim.snapshots[0]["wcc"][idx("rho")]
    rho1 = sim.snapshots[-1]["wcc"][idx("rho")]
    error = linf_norm(rho1 - rho0)
    data.append(dict(N=N, name=name, p=getattr(sim, "p", None), error=error))
df = pd.DataFrame(data)

# plot error curves of p over N
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_title("Convergence of 2D advection of a square in a single direction")
ax.set_xlabel("N")
ax.set_ylabel("Linf error")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.grid()


for name in configs.keys():
    df_name = df[df["name"] == name]
    p = df_name["p"].values[0]
    style = styles[name]

    ax.plot(
        df_name["N"],
        df_name["error"],
        label=name,
        **style,
    )
ax.legend()
fig.savefig(plot_path, dpi=300)
