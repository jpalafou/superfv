from functools import partial
from itertools import product

import numpy as np

from superfv import EulerSolver
from superfv.initial_conditions import decaying_isotropic_turbulence

base_path = "/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/"

# Loop parameters
M_max_values = [1, 2.5, 5, 10, 25, 50]

seeds = range(1, 31)

common = dict(PAD={"rho": (0, None)}, SED=False)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(MOOD=True, limiting_vars=("rho", "vx", "vy"), **common)
adaptive = dict(lazy_primitives="adaptive")
lazy = dict(lazy_primitives="full")

configs = {
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", **common),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/adaptive": dict(p=3, **aposteriori, **adaptive),
    "MM7/adaptive": dict(p=7, **aposteriori, **adaptive),
    "MM3/w1": dict(p=3, **aposteriori, **lazy),
    "MM7/w1": dict(p=7, **aposteriori, **lazy),
    "MM3b/adaptive": dict(p=3, blend=True, **aposteriori, **adaptive),
    "MM7b/adaptive": dict(p=7, blend=True, **aposteriori, **adaptive),
    "MM3b/w1": dict(p=3, blend=True, **aposteriori, **lazy),
    "MM7b/w1": dict(p=7, blend=True, **aposteriori, **lazy),
    "MM3-2/adaptive": dict(
        p=3, cascade="muscl1", max_MOOD_iters=2, **aposteriori, **adaptive
    ),
    "MM7-2/adaptive": dict(
        p=7, cascade="muscl1", max_MOOD_iters=2, **aposteriori, **adaptive
    ),
    "MM3-2/w1": dict(p=3, cascade="muscl1", max_MOOD_iters=2, **aposteriori, **lazy),
    "MM7-2/w1": dict(p=7, cascade="muscl1", max_MOOD_iters=2, **aposteriori, **lazy),
    "MM3-3/w1": dict(p=3, cascade="muscl1", max_MOOD_iters=3, **aposteriori, **lazy),
    "MM7-3/w1": dict(p=7, cascade="muscl1", max_MOOD_iters=3, **aposteriori, **lazy),
}

# Simulation parameters
N = 64
slope = -5 / 3
dt_min_rel = 1e-4


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
for (name, config), M_max, seed in product(configs.items(), M_max_values, seeds):
    print(f"- - - Starting simulation: {name}, seed={seed}, M_max={M_max} - - -")

    sim_path = f"{base_path}{name}/M_max_{M_max}/seed_{seed:02d}/"

    # dummy sim used purely for computing T and dt_ref
    dummy_sim = EulerSolver(
        ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=slope),
        isothermal=True,
        nx=N,
        ny=N,
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
        dt_min=dt_min_rel * dt_ref,
        **config,
    )

    try:
        sim.run(
            [t.item() for t in np.linspace(0, t_cross, 4)[1:]],
            allow_overshoot=True,
            q_max=2,
            muscl_hancock=config.get("MUSCL", False),
            log_freq=1000,
            path=sim_path,
        )
        sim.write_timings()

    except FileExistsError:
        print("\nSimulation already completed, skipping.\n")
        continue

    except Exception as e:

        if name in ("p0", "MUSCL-Hancock", "ZS3", "ZS7"):
            raise RuntimeError(f"Simulation {name} failed unexpectedly.") from e

        print(f"\nFailed: {e}\n")

        # write error
        with open(f"{sim_path}error.txt", "w") as f:
            f.write(str(e))

        continue
