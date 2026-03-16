import argparse
from functools import partial
from itertools import product

import numpy as np

from superfv import EulerSolver
from superfv.initial_conditions import decaying_isotropic_turbulence
from superfv.tools.run_helper import run_multiple_simulations

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, required=True)
parser.add_argument("--cupy", action="store_true", help="Use CuPy for GPU acceleration")
args = parser.parse_args()

cupy = args.cupy
N = args.N

overwrite = False
base_path = f"/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/{N}x{N}/"
if cupy:
    base_path += "cupy/"

run_params = dict(allow_overshoot=True)
init_params = dict(
    isothermal=True,
    PAD={"rho": (0, None)},
    SED=False,
    nx=N,
    ny=N,
    cupy=cupy,
)

# Loop parameters
M_max_values = [0.01, 0.1, 1, 10, 20, 30, 40, 50]
seeds = range(1, 31)

musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive")
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")
aposteriori.update(dict(limiting_vars=("rho", "vx", "vy")))
aposteriori_1rev = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori_2revs = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori)
aposteriori_3revs = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori)


configs = {
    "MUSCL-Hancock": musclhancock,
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/3revs/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_3revs),
    "MM7/3revs/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_3revs),
    "MM3/3revs/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_3revs),
    "MM7/3revs/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_3revs),
    "MM3/2revs/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_2revs),
    "MM7/2revs/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_2revs),
    "MM3/2revs/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_2revs),
    "MM7/2revs/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_2revs),
    "MM3/1rev/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_1rev),
}


def compute_velocity_rms(sim):
    idx = sim.variable_index_map
    xp = sim.xp

    u = sim.arrays["u"]
    w = xp.empty_like(u)
    sim.conservatives_to_primitives(u, w)

    v = xp.sqrt(xp.mean(xp.sum(xp.square(w[idx("v")]), axis=0))).item()

    return v


def compute_turbulence_crossing_time(sim):
    mesh = sim.mesh

    Lx = mesh.xlim[1] - mesh.xlim[0]
    Ly = mesh.ylim[1] - mesh.ylim[0]
    Lz = mesh.zlim[1] - mesh.zlim[0]
    L = max(Lx, Ly, Lz)

    sigma = compute_velocity_rms(sim)

    return L / sigma


def compute_reference_dt(sim):
    mesh = sim.mesh

    h = min(mesh.hx, mesh.hy, mesh.hz)
    sigma = compute_velocity_rms(sim)

    return h / (3 * sigma)


# precompute crossing times and max_steps
simtimes = {}
for (name, config), M_max, seed in product(configs.items(), M_max_values, seeds):
    dummy_sim = EulerSolver(
        ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=-5 / 3),
        **init_params,
        **config,
    )
    t_cross = compute_turbulence_crossing_time(dummy_sim)
    dt_ref = compute_reference_dt(dummy_sim)
    max_steps = 10 * int(t_cross / dt_ref) if M_max > 1 else None

    key = f"{name}_{M_max}_{seed}"
    simtimes[key] = (t_cross, max_steps)


run_multiple_simulations(
    {
        f"{name}/M_max_{M_max}/seed_{seed:02d}/": (
            dict(
                ic=partial(
                    decaying_isotropic_turbulence, seed=seed, M=M_max, slope=-5 / 3
                ),
                **config,
                **init_params,
            ),
            dict(
                T=np.linspace(0, simtimes[f"{name}_{M_max}_{seed}"][0], 4).tolist(),
                max_steps=simtimes[f"{name}_{M_max}_{seed}"][1] if M_max >= 1 else None,
                **run_params,
            ),
        )
        for (name, config), M_max, seed in product(configs.items(), M_max_values, seeds)
        if M_max >= 1 or seed == 1
    },
    base_path,
    overwrite=overwrite,
)
