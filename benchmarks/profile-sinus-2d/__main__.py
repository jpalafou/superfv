import argparse
from functools import partial
from itertools import product

from superfv.initial_conditions import sinus
from superfv.tools.run_helper import run_multiple_simulations

parser = argparse.ArgumentParser()
parser.add_argument("--cupy", action="store_true", help="Use CuPy for GPU acceleration")
cupy = parser.parse_args().cupy

# loop parameters
N_values = [32, 64, 128, 256, 512, 1024, 2048]

common = dict(
    ic=partial(sinus, bounds=(1, 2), vx=2, vy=1, P=1),
    gamma=1.4,
    PAD={"rho": (0, None), "P": (0, None)},
    log_limiter_scalars=False,
    skip_trouble_counts=True,
    cupy=cupy,
)

musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive", adaptive_dt=False)
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")

configs = {
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p3": dict(p=3),
    "p7": dict(p=7),
    "p3/GL": dict(p=3, GL=True),
    "p7/GL": dict(p=7, GL=True),
    "MUSCL-Hancock": musclhancock,
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, **apriori),
    "ZS7t": dict(p=7, **apriori),
    "MM3/1rev/rtol_0": dict(p=3, max_MOOD_iters=1, NAD_rtol=0, **aposteriori),
    "MM7/1rev/rtol_0": dict(p=7, max_MOOD_iters=1, NAD_rtol=0, **aposteriori),
    "MM3/2revs/rtol_0": dict(p=3, max_MOOD_iters=2, NAD_rtol=0, **aposteriori),
    "MM7/2revs/rtol_0": dict(p=7, max_MOOD_iters=2, NAD_rtol=0, **aposteriori),
    "MM3/3revs/rtol_0": dict(p=3, max_MOOD_iters=3, NAD_rtol=0, **aposteriori),
    "MM7/3revs/rtol_0": dict(p=7, max_MOOD_iters=3, NAD_rtol=0, **aposteriori),
}

run_multiple_simulations(
    "euler",
    {
        f"{name}/N_{N}": {**config, **dict(nx=N, ny=N), **common}
        for (name, config), N in product(configs.items(), N_values)
    },
    "/scratch/gpfs/jp7427/out/timing-of-2d-sine-wave/" + ("cupy/" if cupy else ""),
    snapshot_mode="none",
    overwrite=True,
)
