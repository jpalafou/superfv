from functools import partial
from itertools import product

import numpy as np

from superfv import EulerSolver
from superfv.initial_conditions import decaying_isotropic_turbulence

base_path = "/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/"

# Loop parameters
M_max_values = [1, 2.5, 5, 10, 25, 50]

seeds = range(1, 31)

PAD = {"rho": (0, None)}
apriori = dict(ZS=True, lazy_primitives="adaptive", PAD=PAD)
aposteriori = dict(
    MOOD=True, lazy_primitives="adaptive", limiting_vars=("rho", "vx", "vy"), PAD=PAD
)

configs = {
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D"),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3": dict(p=3, **aposteriori),
    "MM7": dict(p=7, **aposteriori),
    "MM3b": dict(p=3, blend=True, **aposteriori),
    "MM7b": dict(p=7, blend=True, **aposteriori),
    "MM3-2": dict(p=3, cascade="muscl1", max_MOOD_iters=2, **aposteriori),
    "MM7-2": dict(p=7, cascade="muscl1", max_MOOD_iters=2, **aposteriori),
}

# Simulation parameters
N = 64
slope = -5 / 3


def compute_velocity_rms(sim):
    idx = sim.variable_index_map
    xp = sim.xp

    u = sim.arrays["u"]
    w = sim.primitives_from_conservatives(u)

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


# Run simulations
for (name, config), seed, M_max in product(configs.items(), seeds, M_max_values):
    print(f"- - - Starting simulation: {name}, seed={seed}, M_max={M_max} - - -")

    sim_path = f"{base_path}/{name}/M_max_{M_max}/seed_{seed}/"

    # dummy sim used purely for computing T and dt_ref
    dummy_sim = EulerSolver(
        ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=slope),
        isothermal=True,
        nx=N,
        ny=N,
        SED=False,
        **config,
    )

    t_cross = compute_turbulence_crossing_time(dummy_sim)
    dt_ref = compute_reference_dt(dummy_sim)
    print(f"\tTurbulent crossing time = {t_cross:.4f}, dt_ref = {dt_ref:.2e}")

    sim = EulerSolver(
        ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=slope),
        isothermal=True,
        nx=N,
        ny=N,
        SED=False,
        dt_min=0.1 * dt_ref,
        **config,
    )

    try:
        sim.run(
            [t.item() for t in np.linspace(0, t_cross, 4)[1:]],
            allow_overshoot=True,
            log_freq=100,
            path=sim_path,
            muscl_hancock=config.get("MUSCL", False),
        )
        sim.write_timings()

    except Exception as e:
        print(f"\nFailed: {e}\n")

        # write error
        with open(f"{sim_path}error.txt", "w") as f:
            f.write(str(e))

        continue
