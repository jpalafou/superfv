from functools import partial
from itertools import product
from typing import Tuple

import numpy as np

from superfv import HydroSolver, run_multiple_simulations
from superfv.hydro import cons_to_prim
from superfv.initial_conditions import decaying_isotropic_turbulence

N = 128
fine_factor = 8  # how much larger is the reference solution

overwrite = False
base_path = "/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/"

run_params = dict(allow_overshoot=True)
init_params = dict(
    isothermal=True,
    PAD_bounds={"rho": (0.0, None)},
    use_SED=False,
    cupy=True,
)

# Loop parameters
M_max_values = [0.01, 0.1, 1, 10, 20, 30, 40, 50]
seeds = range(1, 31)

musclhancock = dict(p=1, use_MUSCL=True, MUSCL_limiter="pp2d")
apriori = dict(use_ZS=True, lazy_primitive_mode="adaptive", adaptive_dt=True)
aposteriori = dict(
    use_MOOD=True,
    lazy_primitive_mode="full",
    MUSCL_limiter="pp2d",
    omit_vars=["vz", "P"],
    positivity_guard=False,
)
aposteriori_1rev = dict(fallback_cascade="muscl", max_revs=1, **aposteriori)
aposteriori_2revs = dict(fallback_cascade="muscl0", max_revs=2, **aposteriori)
aposteriori_3revs = dict(fallback_cascade="muscl0", max_revs=3, **aposteriori)


configs = {
    "ref": musclhancock,
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock | dict(CFL=0.5),
    "ZS3": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "ZS3t": dict(p=3, **(apriori | dict(adaptive_dt=False))),
    "ZS7t": dict(p=7, **(apriori | dict(adaptive_dt=False))),
    "MM3/3revs/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_3revs),
    "MM7/3revs/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_3revs),
    "MM3/3revs/rtol_1e-3": dict(p=3, rtol=1e-3, **aposteriori_3revs),
    "MM7/3revs/rtol_1e-3": dict(p=7, rtol=1e-3, **aposteriori_3revs),
    "MM3/3revs/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_3revs),
    "MM7/3revs/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_3revs),
    "MM3/2revs/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_2revs),
    "MM7/2revs/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_2revs),
    "MM3/2revs/rtol_1e-3": dict(p=3, rtol=1e-3, **aposteriori_2revs),
    "MM7/2revs/rtol_1e-3": dict(p=7, rtol=1e-3, **aposteriori_2revs),
    "MM3/2revs/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_2revs),
    "MM7/2revs/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_2revs),
    "MM3/1rev/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-3": dict(p=3, rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, rtol=1e-3, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_1rev),
    "MM3/1rev/rtol_0": dict(p=3, rtol=0, **aposteriori_1rev),
    "MM7/1rev/rtol_0": dict(p=7, rtol=0, **aposteriori_1rev),
    "MUSCL-RK4": musclhancock | dict(CFL=0.5),
    "ZS3-RK4": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7-RK4": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "MM3-RK4/3revs/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_3revs),
    "MM7-RK4/3revs/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_3revs),
    "MM3-RK4/3revs/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_3revs),
    "MM7-RK4/3revs/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_3revs),
    "MM3-RK4/1rev/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_1rev),
    "MM7-RK4/1rev/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_1rev),
    "MM3-RK4/1rev/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_1rev),
    "MM7-RK4/1rev/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_1rev),
}


def compute_velocity_rms(sim):
    idx = sim.idx
    xp = sim.xp

    u = sim.arrays["u"]
    w = xp.empty_like(u)
    hp = sim.params.hydro
    cons_to_prim(u, w, idx, hp.gamma, hp.isothermal, hp.iso_cs)

    v = xp.sqrt(xp.mean(xp.sum(xp.square(w[idx("v")]), axis=0))).item()

    return v


def compute_turbulence_crossing_time(sim):
    mesh = sim.mesh

    Lx = mesh.xlims[1] - mesh.xlims[0]
    Ly = mesh.ylims[1] - mesh.ylims[0]
    Lz = mesh.zlims[1] - mesh.zlims[0]
    L = max(Lx, Ly, Lz)

    sigma = compute_velocity_rms(sim)

    return L / sigma


def compute_reference_dt(sim):
    mesh = sim.mesh

    h = min(mesh.hx, mesh.hy, mesh.hz)
    sigma = compute_velocity_rms(sim)

    return h / (3 * sigma)


# precompute crossing times and max_steps
def precompute_simulation_times(M_max: float, seed: int, name: str, **kwargs) -> Tuple[float, int]:
    dummy_sim = HydroSolver(
        ic=partial(
            decaying_isotropic_turbulence,
            seed=seed,
            M=M_max,
            slope=-5 / 3,
            fine_factor=8 if name == "ref" else 1,
            seed_fine=seed + 1,
        ),
        nx=N * fine_factor if name == "ref" else N,
        ny=N * fine_factor if name == "ref" else N,
        **init_params,
        **kwargs,
    )
    t_cross = compute_turbulence_crossing_time(dummy_sim)
    dt_ref = compute_reference_dt(dummy_sim)
    max_steps = 10 * int(t_cross / dt_ref) if M_max > 1 else None

    return t_cross, max_steps


sim_configs = {}
for (name, config), M_max, seed in product(configs.items(), M_max_values, seeds):
    if M_max < 1 and seed > 1:
        continue

    t_cross, max_steps = precompute_simulation_times(M_max, seed, name, **config)

    key = f"{name}/M_max_{M_max}/seed_{seed:02d}/"
    sim_init_params = dict(
        ic=partial(
            decaying_isotropic_turbulence,
            seed=seed,
            M=M_max,
            slope=-5 / 3,
            fine_factor=8 if name == "ref" else 1,
            seed_fine=seed + 1,
        ),
        nx=N * fine_factor if name == "ref" else N,
        ny=N * fine_factor if name == "ref" else N,
        **init_params,
        **config,
    )
    sim_run_params = dict(
        t=np.linspace(0, t_cross, 4)[1:].tolist(),
        time_integrator=(
            "rk4"
            if "RK4" in name
            else (
                "ssprk3"
                if "RK3" in name
                else ("muscl_hancock" if config.get("use_MUSCL", False) else "match_p_up_to_ssprk3")
            )
        ),
        max_steps=max_steps,
        **run_params,
    )

    sim_configs[key] = (sim_init_params, sim_run_params)


run_multiple_simulations(
    sim_configs, base_path=base_path, overwrite=overwrite, skip_errors=not overwrite
)
