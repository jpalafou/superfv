import os
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from superfv import OutputLoader, plot_2d_slice
from superfv.initial_conditions import gresho_vortex
from superfv.tools.run_helper import run_multiple_simulations

base_path = "/scratch/gpfs/jp7427/out/gresho-vortex/"

N = 96
gamma = 5 / 3
init_params = dict(
    gamma=gamma,
    PAD={"rho": (0, None), "P": (0, None)},
    nx=N,
    ny=N,
    cupy=True,
)

run_params = dict(T=[0.2, 0.4, 0.6, 0.8, 1.0])

# loop parameters
v0_values = [5.0]

M_max_values = [0.1, 0.01, 0.001]

musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive")
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")
aposteriori_1rev = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori_2revs = dict(cascade="muscl0", max_MOOD_iters=2, **aposteriori)
aposteriori_3revs = dict(cascade="muscl0", max_MOOD_iters=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock | dict(CFL=0.5),
    "ZS3/no_v": dict(p=3, GL=True, limiting_vars=("rho",), **apriori),
    "ZS7/no_v": dict(p=7, GL=True, limiting_vars=("rho",), **apriori),
    "MM3/1rev/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_1rev),
}


def compute_M(idx, mesh, w, v0):
    x, y, _ = mesh.get_cell_centers()
    rho = w[idx("rho")]
    vx = w[idx("vx")] - v0
    vy = w[idx("vy")]
    P = w[idx("P")]

    xc = x - 0.5
    yc = y - 0.5
    r = np.sqrt(xc**2 + yc**2)

    v_phi = vx * (-yc / r) + vy * (xc / r)

    cs2 = gamma * P / rho
    cs = np.sqrt(np.maximum(cs2, 0.0))

    M = np.abs(v_phi) / cs

    return M


def compute_M_func(M_max, v0):
    def f(idx, mesh, w):
        M = compute_M(idx, mesh, w, v0)
        return M / M_max

    return f


def makeplot(name, _):
    plot_path = f"out/gresho-vortex-plots/{name}.pdf"
    dir_name = os.path.dirname(plot_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fig, ax = plt.subplots(figsize=(6, 6))

    # parse v0 and M_max from name
    v0 = float(name.split("v0_")[1].split("/")[0])
    M_max = float(name.split("M_max_")[1].rstrip("/"))

    # load saved data
    sim = OutputLoader(os.path.join(base_path, name))

    plot_2d_slice(
        sim,
        ax,
        "w",
        multivar_func=compute_M_func(M_max, v0),
        cmap="jet",
        vmin=0,
        vmax=1,
        colorbar=True,
    )
    fig.savefig(plot_path, bbox_inches="tight")


run_multiple_simulations(
    {
        f"v0_{v0}/{name}/M_max_{M_max}": (
            dict(
                ic=partial(gresho_vortex, gamma=gamma, M_max=M_max, v0=v0),
                **init_params,
                **config,
            ),
            dict(
                muscl_hancock=False if name == "MUSCL-RK3" else True,
                time_degree=2 if name == "MUSCL-RK3" else None,
                **run_params,
            ),
        )
        for (name, config), v0, M_max in product(
            configs.items(), v0_values, M_max_values
        )
    },
    base_path=base_path,
    postprocess=makeplot,
)
