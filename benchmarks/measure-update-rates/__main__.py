from functools import partial
from itertools import product

from superfv import EulerSolver
from superfv.initial_conditions import square

base_path = "/scratch/gpfs/jp7427/out/profiling-2d-square/"

common = dict(PAD={"rho": (0, None), "P": (0, None)})
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    NAD_rtol=0,
    NAD_atol=0,
    skip_trouble_counts=True,
    detect_closing_troubles=False,
    **common,
)

configs = {
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", **common),
    "MUSCL3": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", CFL=0.5, **common),
    "p3t": dict(p=3),
    "p7t": dict(p=7),
    "p3gl": dict(p=3, GL=True),
    "p7gl": dict(p=7, GL=True),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3": dict(p=3, **aposteriori),
    "MM7": dict(p=7, **aposteriori),
    "MM3-2": dict(p=3, cascade="muscl1", max_MOOD_iters=2, **aposteriori),
    "MM7-2": dict(p=7, cascade="muscl1", max_MOOD_iters=2, **aposteriori),
    "MM3-3": dict(p=3, cascade="muscl1", max_MOOD_iters=3, **aposteriori),
    "MM7-3": dict(p=7, cascade="muscl1", max_MOOD_iters=3, **aposteriori),
}

n_steps = 10
N_values = [64, 128, 256, 512, 1024, 2048]
devices = ["gpu", "cpu"]
max_cpu_resolution = 512

for device, (name, config), N in product(devices, configs.items(), N_values):
    if device == "cpu" and N > max_cpu_resolution:
        continue

    print(f"Running {name} with N={N} on {device}")

    sim_path = f"{base_path}{device}/{name}/N_{N}/"

    sim = EulerSolver(
        ic=partial(square, bounds=(1, 2), P=1, vx=1, vy=1),
        nx=N,
        ny=N,
        cupy=device == "gpu",
        **config,
    )
    sim.run(
        n=n_steps,
        q_max=2,
        muscl_hancock=name == "MUSCL-Hancock",
        time_degree=2 if name == "MUSCL3" else None,
        verbose=False,
        path=sim_path,
        overwrite=True,
    )
    sim.write_timings()

    # ensure that all MOOD iterations reached max iters
    if config.get("MOOD", False):
        max_iters = sim.MOOD_config.max_iters
        iters = sim.minisnapshots["nfine_MOOD_iters"][1:]
        if not all(all(substep == max_iters for substep in step) for step in iters):
            raise ValueError(f"Not all MOOD loops maxed out in {name} with N={N}")
