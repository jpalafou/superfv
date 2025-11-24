from functools import partial
from itertools import product

from superfv import EulerSolver
from superfv.initial_conditions import gresho_vortex

N = 96
T = [0.2, 0.4, 0.6, 0.8, 1.0]
gamma = 5 / 3
v0 = 5.0
M_max_values = [1e-1, 1e-2, 1e-3]
base_path = "/scratch/gpfs/jp7427/out/gresho-vortex/"

configs = {
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p2": dict(p=2),
    "p3": dict(p=3),
    "p4": dict(p=4),
    "p5": dict(p=5),
}

for (name, config), M_max in product(configs.items(), M_max_values):
    print(f"Running {name} with M_max = {M_max}")

    sim_path = base_path + f"{name}/M_max_{M_max}/"

    sim = EulerSolver(
        ic=partial(gresho_vortex, gamma=gamma, M_max=M_max, v0=v0),
        gamma=gamma,
        nx=N,
        ny=N,
        **config,
    )

    try:
        sim.run(
            T,
            q_max=2,
            muscl_hancock=config.get("MUSCL", False),
            allow_overshoot=True,
            log_freq=20,
            path=sim_path,
        )
    except FileExistsError as e:
        print(f"File exists for simulation {name}, skipping: {e}")
    except RuntimeError as e:
        print(f"Simulation '{name}' failed: {e}")
