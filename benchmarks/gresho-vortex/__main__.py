from functools import partial
from itertools import product

from superfv import EulerSolver
from superfv.initial_conditions import gresho_vortex

N = 96
T = [0.2, 0.4, 0.6, 0.8, 1.0]
gamma = 5 / 3
base_path = "/scratch/gpfs/jp7427/out/gresho-vortex/"

v0_values = [5.0]
M_max_values = [1e-1, 1e-2, 1e-3]

common = dict(PAD={"rho": (0, None), "P": (0, None)})
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    **common,
)
aposteriori1 = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori2 = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori)
aposteriori3 = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori)

configs = {
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p2": dict(p=2),
    "p3": dict(p=3),
    "p4": dict(p=4),
    "p5": dict(p=5),
    "p7": dict(p=7),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/rtol_1e-2": dict(p=3, NAD_rtol=1e-2, **aposteriori1),
    "MM7/1rev/rtol_1e-2": dict(p=7, NAD_rtol=1e-2, **aposteriori1),
    "MM3/1rev/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori1),
    "MM7/1rev/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori1),
    "MM3/1rev/rtol_1e0": dict(p=3, NAD_rtol=1e0, **aposteriori1),
    "MM7/1rev/rtol_1e0": dict(p=7, NAD_rtol=1e0, **aposteriori1),
}

for v0, (name, config), M_max in product(v0_values, configs.items(), M_max_values):
    print(f"Running {name} with M_max = {M_max} and v0 = {v0}")

    sim_path = base_path + f"v0_{v0}/{name}/M_max_{M_max}/"

    sim = EulerSolver(
        ic=partial(gresho_vortex, gamma=gamma, M_max=M_max, v0=v0),
        gamma=gamma,
        nx=N,
        ny=N,
        cupy=True,
        **config,
    )

    try:
        sim.run(
            T,
            allow_overshoot=True,
            q_max=2,
            muscl_hancock=config.get("MUSCL", False),
            log_freq=100,
            path=sim_path,
        )
        sim.write_timings()
    except FileExistsError as e:
        print(f"File exists for simulation {name}, skipping: {e}")
    except RuntimeError as e:
        print(f"Simulation '{name}' failed: {e}")
