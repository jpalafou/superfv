from functools import partial
from itertools import product

import numpy as np

from superfv import EulerSolver
from superfv.initial_conditions import square

base_path = (
    "/scratch/gpfs/jp7427/out/varying-min-density-for-hydro-advection-of-a-square/"
)

# Loop parameters
rho_min_values = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]

common = dict(PAD={"rho": (0, None), "P": (0, None)}, SED=False)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D", **common)

configs = {
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", **common),
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
    "MM3-3": dict(p=3, cascade="muscl1", max_MOOD_iters=3, **aposteriori),
    "MM7-3": dict(p=7, cascade="muscl1", max_MOOD_iters=3, **aposteriori),
}

# Simulation parameters
N = 64
T = 1.0
dt_min_rel = 1e-4

# Run simulations
for (name, config), rho_min in product(configs.items(), rho_min_values):
    print(f"- - - Starting simulation: {name} with {rho_min=} - - -")

    sim_path = f"{base_path}/{name}/rho_min_{rho_min}/"

    # get reference dt
    cmax = np.sqrt(1.4 * 1e-5 / rho_min).item()
    h = 1 / N
    dt_ref = 0.8 / ((2 + cmax) / h + (1 + cmax) / h)

    print(f"\t{dt_ref=:.2e}")

    sim = EulerSolver(
        ic=partial(square, bounds=(rho_min, 1), vx=2, vy=1, P=rho_min),
        nx=N,
        ny=N,
        dt_min=dt_min_rel * dt_ref,
        **config,
    )

    try:
        sim.run(
            T,
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
