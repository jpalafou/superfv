import argparse
import os
from functools import partial
from itertools import product

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import sinus

parser = argparse.ArgumentParser()
parser.add_argument("--cupy", action="store_true", help="Use CuPy for GPU acceleration")
args = parser.parse_args()

base_path = "/scratch/gpfs/jp7427/out/timing-of-2d-sine-wave/"
if args.cupy:
    base_path += "cupy/"
overwrite = True

# loop parameters
N_values = [32, 64, 128, 256, 512, 1024, 2048]

common = dict(
    PAD={"rho": (0, None), "P": (0, None)},
    log_limiter_scalars=False,
    skip_trouble_counts=True,
)
musclhancock = dict(p=1, MUSCL=True, **common)
apriori = dict(ZS=True, lazy_primitives="adaptive", adaptive_dt=False, **common)
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
    # "p0": dict(p=0),
    # "p1": dict(p=1),
    # "p3": dict(p=3),
    # "p7": dict(p=7),
    # "p3/GL": dict(p=3, GL=True),
    "p7/GL": dict(p=7, GL=True),
    # "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    # "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    # "ZS3t": dict(p=3, **apriori),
    # "ZS7t": dict(p=7, **apriori),
    # "MM3/1rev/no_delta/rtol_1e-5": dict(
    #     p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    # ),
    # "MM7/1rev/no_delta/rtol_1e-5": dict(
    #     p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    # ),
    # "MM3/2revs/no_delta/rtol_1e-5": dict(
    #     p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori2
    # ),
    # "MM7/2revs/no_delta/rtol_1e-5": dict(
    #     p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori2
    # ),
    # "MM3/3revs/no_delta/rtol_1e-5": dict(
    #     p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori3
    # ),
    # "MM7/3revs/no_delta/rtol_1e-5": dict(
    #     p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori3
    # ),
}

# simulation parameters
n_steps = 10
gamma = 1.4

for (name, config), N in product(configs.items(), N_values):
    sim_path = f"{base_path}/{name}/N_{N}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"Running {name} with N={N}...")

        sim = EulerSolver(
            ic=partial(sinus, bounds=(1, 2), vx=2, vy=1, P=1),
            gamma=gamma,
            nx=N,
            ny=N,
            cupy=args.cupy,
            **config,
        )

        try:
            sim.run(
                n=n_steps,
                q_max=2,
                muscl_hancock=config.get("MUSCL", False),
                path=sim_path,
                overwrite=True,
            )
            sim.write_timings()

            # clean up error file if it exists
            if os.path.exists(f"{sim_path}error.txt"):
                os.remove(f"{sim_path}error.txt")

        except RuntimeError as e:
            print(f"  Failed: {e}")
            with open(f"{sim_path}error.txt", "w") as f:
                f.write(str(e))

            continue
