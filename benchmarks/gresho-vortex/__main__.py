import os
from functools import partial
from itertools import product

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import gresho_vortex

N = 96
T = [0.2, 0.4, 0.6, 0.8, 1.0]
gamma = 5 / 3
base_path = "/scratch/gpfs/jp7427/out/gresho-vortex/"
overwrite = False

# N = 96
# T = [0.02, 0.04, 0.06, 0.08, 0.1]  # [0.2, 0.4, 0.6, 0.8, 1.0]
# gamma = 5 / 3
# base_path = "/scratch/gpfs/jp7427/out/gresho-vortex/short-debug/"
# overwrite = False

v0_values = [5.0]
M_max_values = [1e-1, 1e-2, 1e-3]

common = dict(PAD={"rho": (0, None), "P": (0, None)})
musclhancock = dict(p=1, MUSCL=True, **common)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    **common,
)
aposteriori1 = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori2 = dict(cascade="muscl0", max_MOOD_iters=2, **aposteriori)
aposteriori3 = dict(cascade="muscl0", max_MOOD_iters=3, **aposteriori)

no_v = dict(limiting_vars=("rho", "P"))

configs = {
    "p3": dict(p=3),
    "p7": dict(p=7),
    "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3/no_v": dict(p=3, GL=True, **apriori, **no_v),
    "ZS7/no_v": dict(p=7, GL=True, **apriori, **no_v),
    "MM3/1rev/no_v/rtol_1e-2": dict(p=3, NAD_rtol=1e-2, **aposteriori1, **no_v),
    "MM7/1rev/no_v/rtol_1e-2": dict(p=7, NAD_rtol=1e-2, **aposteriori1, **no_v),
    "MM3/1rev/no_delta/rtol_1e-5": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-5": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
}


for v0, (name, config), M_max in product(v0_values, configs.items(), M_max_values):
    sim_path = f"{base_path}v0_{v0}/{name}/M_max_{M_max}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"Running {name} with M_max = {M_max} and v0 = {v0}")

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
