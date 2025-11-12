from functools import partial
from itertools import product

from superfv import EulerSolver
from superfv.initial_conditions import decaying_isotropic_turbulence

base_path = "/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/"

# Loop parameters
M_max_values = [1, 2.5, 5, 10, 25, 50]

seeds = range(1, 11)

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
T = [0.08, 0.16, 0.24]
N = 64
slope = -5 / 3

# Run simulations
for (name, config), seed, M_max in product(configs.items(), seeds, M_max_values):
    print(f"- - - Starting simulation: {name}, seed={seed}, M_max={M_max} - - -")

    sim_path = f"{base_path}/{name}/M_max_{M_max}/seed_{seed}/"

    sim = EulerSolver(
        ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=slope),
        isothermal=True,
        nx=N,
        ny=N,
        **config,
    )

    try:
        sim.run(
            T,
            allow_overshoot=True,
            log_freq=1000,
            path=sim_path,
            muscl_hancock=config.get("MUSCL", False),
        )
        sim.write_timings()

        with open(sim_path + "status.txt", "w") as f:
            f.write("passed")

    except Exception as e:
        print(f"\nFailed: {e}\n")

        with open(sim_path + "status.txt", "w") as f:
            f.write("failed")

        continue
