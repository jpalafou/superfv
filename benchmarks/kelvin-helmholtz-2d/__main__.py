import os

import matplotlib.pyplot as plt
import numpy as np

from superfv import plot_2d_slice
from superfv.initial_conditions import kelvin_helmholtz_2d
from superfv.tools.run_helper import run_multiple_simulations

# simulation parameters
N = 2048
init_params = dict(
    ic=kelvin_helmholtz_2d,
    gamma=1.4,
    nx=N,
    ny=N,
    PAD={"rho": (0, None), "P": (0, None)},
    cupy=True,
)
run_params = dict(T=np.linspace(0, 0.8, 9).tolist(), allow_overshoot=True)

# loop parameters
musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive")
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")
aposteriori_1rev = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori_2revs = dict(cascade="muscl0", max_MOOD_iters=2, **aposteriori)
aposteriori_3revs = dict(cascade="muscl0", max_MOOD_iters=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock | dict(CFL=0.5),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_1rev),
}


def makeplot(name, sim):
    plot_path = f"out/kelvin-helmholtz-plots/{name}.png"
    dir_name = os.path.dirname(plot_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fig, ax = plt.subplots(figsize=(6, 6))

    plot_2d_slice(sim, ax, "rho", colorbar=False)
    fig.savefig(plot_path, dpi=300)


run_multiple_simulations(
    {
        name: (
            init_params | config,
            dict(
                muscl_hancock=False if name == "MUSCL-RK3" else True,
                time_degree=2 if name == "MUSCL-RK3" else None,
                **run_params,
            ),
        )
        for name, config in configs.items()
    },
    "/scratch/gpfs/jp7427/out/kelvin-helmholtz-2d/",
    overwrite=False,
    postprocess=makeplot,
)
